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

# Import logfire with fallback
try:
    import logfire
except ImportError:
    class MockLogfire:
        @staticmethod
        def span(name, **kwargs):
            class MockSpan:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return MockSpan()
        @staticmethod
        def info(*args, **kwargs): pass
        @staticmethod
        def error(*args, **kwargs): pass
        @staticmethod
        def warning(*args, **kwargs): pass
    logfire = MockLogfire()
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import Executable
from sqlalchemy.exc import SQLAlchemyError

# Import SurrealDB integration
from .surrealdb_client import (
    SurrealDBManager,
    get_surrealdb_manager,
    execute_surreal_query,
    check_surrealdb_health,
    shutdown_surrealdb,
    get_surrealdb_dependency
)

# Import graph operations
from .graph_operations import (
    GraphOperations,
    GraphNode,
    GraphRelationship,
    GraphTraversalPath,
    VectorSearchResult,
    NodeType,
    RelationshipType,
    create_graph_operations,
    graph_transaction
)

# Import knowledge schemas
from .knowledge_schemas import (
    AgentKnowledgeSchema,
    CodeAgentSchema,
    DataScienceAgentSchema,
    SuperAgentSchema,
    KnowledgeEntity,
    AgentCapability,
    ToolUsagePattern,
    LearningRecord,
    AgentDomain,
    KnowledgeType,
    CapabilityLevel,
    create_agent_schema,
    merge_knowledge_schemas,
    extract_learning_insights
)

# Import vector search
from .vector_search import (
    VectorSearchEngine,
    VectorSearchConfig,
    VectorSearchResult as VectorResult,
    EmbeddingModel,
    SimilarityMetric,
    create_vector_search_engine,
    vector_search_session,
    quick_similarity_search,
    batch_similarity_matrix
)

# Import data synchronization
from .graph_sync import (
    DataSynchronizer,
    SyncConfig,
    ChangeRecord,
    SyncResult,
    SyncDirection,
    SyncStrategy,
    ChangeType,
    ConflictResolution,
    create_data_synchronizer,
    sync_session,
    quick_sync_entity,
    bulk_sync_entities
)

# Import SurrealDB components
from .surrealdb_client import (
    shutdown_surrealdb,
    SURREALDB_AVAILABLE
)

# Import Redis integration
from .redis_client import (
    get_redis_manager,
    get_session_manager,
    check_redis_health,
    shutdown_redis,
    REDIS_AVAILABLE,
    cached
)

# Import backup system
from .backup import (
    get_backup_manager,
    create_database_backup,
    restore_database_backup,
    get_backup_system_status,
    start_backup_scheduler,
    stop_backup_scheduler
)

# Import middleware
from .middleware import (
    database_middleware,
    transaction_middleware,
    database_health_middleware,
    get_database_metrics
)

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


class DatabaseManager:
    """Comprehensive database manager for all database systems."""

    def __init__(self):
        self.started_at = time.time()
        self.query_count = 0
        self.error_count = 0
        self.connection_stats = {
            'sqlalchemy': {'active': 0, 'total': 0},
            'surrealdb': {'active': 0, 'total': 0},
            'redis': {'active': 0, 'total': 0}
        }
        self.backup_manager = None
        self.redis_manager = None
        self.graph_operations = None
        self.vector_search_engine = None
        self.data_synchronizer = None

    async def initialize_all_databases(self, drop_all: bool = False):
        """Initialize SQLAlchemy, SurrealDB, and Redis."""
        with logfire.span("Database Layer Initialization"):
            # Initialize SQLAlchemy
            if ASYNC_DB_URL:
                await initialize_async_database(drop_all)
            else:
                initialize_database(drop_all)

            # Initialize SurrealDB
            if SURREALDB_AVAILABLE:
                try:
                    await get_surrealdb_manager()
                    logfire.info("SurrealDB initialized successfully")
                except Exception as e:
                    logfire.warning("SurrealDB initialization failed", error=str(e))

            # Initialize Redis
            if REDIS_AVAILABLE:
                try:
                    self.redis_manager = await get_redis_manager()
                    logfire.info("Redis initialized successfully")
                except Exception as e:
                    logfire.warning("Redis initialization failed", error=str(e))

            # Initialize backup system
            try:
                self.backup_manager = await get_backup_manager()
                await start_backup_scheduler()
                logfire.info("Backup system initialized successfully")
            except Exception as e:
                logfire.warning("Backup system initialization failed", error=str(e))

    async def health_check_all(self) -> Dict[str, Any]:
        """Comprehensive health check for all database systems."""
        health_status = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.started_at,
            "databases": {}
        }

        # SQLAlchemy health check
        try:
            if ASYNC_DB_URL:
                sqlalchemy_healthy = await check_async_database_connection()
            else:
                sqlalchemy_healthy = check_database_connection()

            health_status["databases"]["sqlalchemy"] = {
                "status": "healthy" if sqlalchemy_healthy else "unhealthy",
                "type": "relational",
                "url": DB_URL,
                "async_available": bool(ASYNC_DB_URL),
                "pool_size": DB_POOL_SIZE,
                "max_overflow": DB_MAX_OVERFLOW
            }
        except Exception as e:
            health_status["databases"]["sqlalchemy"] = {
                "status": "error",
                "error": str(e)
            }

        # SurrealDB health check
        try:
            surrealdb_health = await check_surrealdb_health()
            health_status["databases"]["surrealdb"] = surrealdb_health
        except Exception as e:
            health_status["databases"]["surrealdb"] = {
                "status": "error",
                "available": SURREALDB_AVAILABLE,
                "error": str(e)
            }

        # Redis health check
        try:
            redis_health = await check_redis_health()
            health_status["databases"]["redis"] = redis_health
        except Exception as e:
            health_status["databases"]["redis"] = {
                "status": "error",
                "available": REDIS_AVAILABLE,
                "error": str(e)
            }

        # Overall status
        all_healthy = all(
            db.get("status") == "healthy"
            for db in health_status["databases"].values()
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"

        return health_status

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics."""
        stats = {
            "uptime_seconds": time.time() - self.started_at,
            "query_count": self.query_count,
            "error_count": self.error_count,
            "sqlalchemy": {
                "pool_size": DB_POOL_SIZE,
                "max_overflow": DB_MAX_OVERFLOW,
                "pool_timeout": DB_POOL_TIMEOUT,
                "echo": DB_ECHO
            }
        }

        # Add SurrealDB stats if available
        try:
            surrealdb_manager = await get_surrealdb_manager()
            stats["surrealdb"] = surrealdb_manager.get_stats()
        except Exception:
            stats["surrealdb"] = {"status": "unavailable"}

        # Add Redis stats if available
        try:
            redis_health = await check_redis_health()
            stats["redis"] = redis_health.get("stats", {"status": "unavailable"})
        except Exception:
            stats["redis"] = {"status": "unavailable"}

        # Add backup system stats
        try:
            backup_status = await get_backup_system_status()
            stats["backup_system"] = backup_status
        except Exception:
            stats["backup_system"] = {"status": "unavailable"}

        # Add middleware metrics
        try:
            middleware_metrics = get_database_metrics()
            stats["middleware"] = middleware_metrics
        except Exception:
            stats["middleware"] = {"status": "unavailable"}

        return stats

    async def shutdown_all(self):
        """Shutdown all database connections."""
        with logfire.span("Database Layer Shutdown"):
            # Stop backup scheduler
            try:
                await stop_backup_scheduler()
            except Exception as e:
                logfire.warning("Error stopping backup scheduler", error=str(e))

            # Shutdown Redis
            try:
                await shutdown_redis()
            except Exception as e:
                logfire.warning("Error shutting down Redis", error=str(e))

            # Close SQLAlchemy engines
            if 'async_engine' in globals():
                await async_engine.dispose()
            engine.dispose()

            # Shutdown SurrealDB
            await shutdown_surrealdb()

            logfire.info("All database connections closed",
                        uptime_seconds=time.time() - self.started_at,
                        total_queries=self.query_count)


# Global database manager instance
database_manager = DatabaseManager()


async def initialize_database_layer(drop_all: bool = False):
    """Initialize all database systems."""
    await database_manager.initialize_all_databases(drop_all)


async def get_database_health() -> Dict[str, Any]:
    """Get comprehensive database health status."""
    return await database_manager.health_check_all()


async def get_database_stats() -> Dict[str, Any]:
    """Get detailed database statistics."""
    return await database_manager.get_connection_stats()


async def shutdown_database_layer():
    """Shutdown all database systems."""
    await database_manager.shutdown_all()


# Export common database components
__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_dependency",
    "optimize_query",
    "initialize_database",
    "check_database_connection",
    "DatabaseManager",
    "database_manager",
    "initialize_database_layer",
    "get_database_health",
    "get_database_stats",
    "shutdown_database_layer",
    "execute_surreal_query",
    "check_surrealdb_health",
    "SURREALDB_AVAILABLE",
    # Redis components
    "get_redis_manager",
    "get_session_manager",
    "check_redis_health",
    "shutdown_redis",
    "REDIS_AVAILABLE",
    "cached",
    # Backup components
    "get_backup_manager",
    "create_database_backup",
    "restore_database_backup",
    "get_backup_system_status",
    "start_backup_scheduler",
    "stop_backup_scheduler",
    # Middleware components
    "database_middleware",
    "transaction_middleware",
    "database_health_middleware",
    "get_database_metrics"
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
