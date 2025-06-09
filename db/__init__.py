"""
Database Module for Agentical

This module provides database connectivity, session management, and optimization
features for the Agentical framework. It includes:

- Connection pooling with SQLAlchemy
- Configurable database engines for different environments
- Async and sync database session management
- Query performance monitoring and optimization
- Integration with Logfire for observability

The module is designed to be used with SQLAlchemy ORM models and follows
DevQ.ai best practices for database interaction patterns.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional, Dict, Any, Union, List, Type
import logging
import time
import os
from functools import wraps

import logfire
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import Executable
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy Base Model
Base = declarative_base()

# Environment variables and configuration
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./agentical.db")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"
DB_ECHO_POOL = os.getenv("DB_ECHO_POOL", "false").lower() == "true"
SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "0.5"))  # seconds

# Determine async DB URL from regular URL if not provided
ASYNC_DB_URL = os.getenv("ASYNC_DATABASE_URL")
if not ASYNC_DB_URL and DB_URL.startswith("sqlite:"):
    ASYNC_DB_URL = DB_URL.replace("sqlite:", "sqlite+aiosqlite:")
elif not ASYNC_DB_URL and DB_URL.startswith("postgresql:"):
    ASYNC_DB_URL = DB_URL.replace("postgresql:", "postgresql+asyncpg:")

# Create SQLAlchemy engines with connection pooling
engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_recycle=DB_POOL_RECYCLE,
    echo=DB_ECHO,
    echo_pool=DB_ECHO_POOL
)

# Create async engine if async URL is available
if ASYNC_DB_URL:
    async_engine = create_async_engine(
        ASYNC_DB_URL,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_recycle=DB_POOL_RECYCLE,
        echo=DB_ECHO,
        echo_pool=DB_ECHO_POOL
    )

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async session factory if async engine is available
if ASYNC_DB_URL:
    AsyncSessionLocal = async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=async_engine,
        expire_on_commit=False
    )

# Query execution time tracking
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    # Store execution start time in context
    context._query_start_time = time.monotonic()
    
    # Log query for debugging if echo is enabled
    if DB_ECHO:
        logger.debug(f"SQL Query: {statement}")
        logger.debug(f"Parameters: {parameters}")

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    # Calculate query execution time
    execution_time = time.monotonic() - context._query_start_time
    
    # Log slow queries
    if execution_time > SLOW_QUERY_THRESHOLD:
        formatted_statement = statement.replace('\n', ' ').replace('\t', ' ')
        logfire.warning(
            "Slow database query detected",
            execution_time=execution_time,
            query=formatted_statement[:1000],  # Truncate very long queries
            query_type=context.execution_options.get('query_type', 'unknown'),
            threshold=SLOW_QUERY_THRESHOLD
        )
        
        # Add detailed logging
        logger.warning(
            f"Slow query ({execution_time:.4f}s): {formatted_statement[:1000]}"
        )


# Connection management functions
@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get a database session with automatic closing.
    
    Usage:
        with get_db() as db:
            result = db.query(Model).first()
    """
    db = SessionLocal()
    start_time = time.monotonic()
    try:
        with logfire.span("Database Session"):
            yield db
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        logfire.error(
            "Database session error",
            error=str(e),
            error_type=type(e).__name__,
            session_duration=time.monotonic() - start_time
        )
        raise
    finally:
        session_duration = time.monotonic() - start_time
        if session_duration > SLOW_QUERY_THRESHOLD:
            logfire.warning(
                "Long-running database session",
                session_duration=session_duration,
                threshold=SLOW_QUERY_THRESHOLD
            )
        db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session with automatic closing.
    
    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(Model))
            model = result.scalars().first()
    """
    if not ASYNC_DB_URL:
        raise RuntimeError("Async database URL not configured")
        
    db = AsyncSessionLocal()
    start_time = time.monotonic()
    try:
        with logfire.span("Async Database Session"):
            yield db
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Async database error: {e}")
        logfire.error(
            "Async database session error",
            error=str(e),
            error_type=type(e).__name__,
            session_duration=time.monotonic() - start_time
        )
        raise
    finally:
        session_duration = time.monotonic() - start_time
        if session_duration > SLOW_QUERY_THRESHOLD:
            logfire.warning(
                "Long-running async database session",
                session_duration=session_duration,
                threshold=SLOW_QUERY_THRESHOLD
            )
        await db.close()


def get_db_dependency():
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db_dependency())):
            return db.query(Item).all()
    """
    def _get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    return _get_db


def get_async_db_dependency():
    """
    FastAPI dependency for async database sessions.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db_dependency())):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    if not ASYNC_DB_URL:
        raise RuntimeError("Async database URL not configured")
        
    async def _get_async_db():
        db = AsyncSessionLocal()
        try:
            yield db
        finally:
            await db.close()
    
    return _get_async_db


# Database utility functions
def optimize_query(query: Executable, hints: Optional[Dict[str, Any]] = None) -> Executable:
    """
    Apply optimization hints to a SQLAlchemy query.
    
    Args:
        query: SQLAlchemy query to optimize
        hints: Dictionary of optimization hints (e.g., {'indexHint': 'myindex'})
    
    Returns:
        Optimized query
    """
    if not hints:
        return query
        
    execution_options = {}
    
    # Add query type for logging
    if "query_type" in hints:
        execution_options["query_type"] = hints["query_type"]
    
    # Apply execution options
    if execution_options:
        query = query.execution_options(**execution_options)
    
    return query


def initialize_database(drop_all: bool = False) -> None:
    """
    Initialize database schema.
    
    Args:
        drop_all: If True, drop all tables before creating them
    """
    with logfire.span("Database Initialization"):
        if drop_all:
            logger.warning("Dropping all database tables")
            Base.metadata.drop_all(bind=engine)
            
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=engine)
        
        # Create indexes that aren't part of model definitions
        with get_db() as db:
            # Execute raw SQL for complex index creation if needed
            # Example: db.execute(text("CREATE INDEX IF NOT EXISTS..."))
            db.commit()
        
        logfire.info("Database initialized successfully")


async def initialize_async_database(drop_all: bool = False) -> None:
    """
    Initialize database schema asynchronously.
    
    Args:
        drop_all: If True, drop all tables before creating them
    """
    if not ASYNC_DB_URL:
        raise RuntimeError("Async database URL not configured")
        
    with logfire.span("Async Database Initialization"):
        if drop_all:
            logger.warning("Dropping all database tables asynchronously")
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                
        logger.info("Creating database tables asynchronously")
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        # Create indexes that aren't part of model definitions
        async with get_async_db() as db:
            # Execute raw SQL for complex index creation if needed
            # Example: await db.execute(text("CREATE INDEX IF NOT EXISTS..."))
            await db.commit()
        
        logfire.info("Async database initialized successfully")


def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connected, False otherwise
    """
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def check_async_database_connection() -> bool:
    """
    Check if async database connection is working.
    
    Returns:
        True if connected, False otherwise
    """
    if not ASYNC_DB_URL:
        return False
        
    try:
        async with get_async_db() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Async database connection check failed: {e}")
        return False


# Export common database components
__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_dependency",
    "optimize_query",
    "initialize_database",
    "check_database_connection"
]

# Add async components if available
if ASYNC_DB_URL:
    __all__.extend([
        "async_engine",
        "AsyncSessionLocal",
        "get_async_db",
        "get_async_db_dependency",
        "initialize_async_database",
        "check_async_database_connection"
    ])