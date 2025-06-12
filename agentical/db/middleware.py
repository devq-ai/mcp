"""
Database Middleware for Agentical

This module provides comprehensive database middleware for FastAPI request processing,
monitoring, and optimization. It integrates with the structured logging system and
provides database-specific middleware for connection management, query monitoring,
and performance optimization.

Features:
- Database connection management per request
- Query performance monitoring and logging
- Automatic transaction management
- Database health checks and circuit breaker
- Connection pooling optimization
- Request-scoped database sessions
- Error handling and retry logic
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Callable, List
import logging

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

from fastapi import Request, Response
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.orm import Session

from . import (
    get_db, get_async_db, check_database_connection,
    check_async_database_connection, DatabaseManager
)
from .redis_client import get_redis_manager, REDIS_AVAILABLE
from ..core.structured_logging import StructuredLogger, LogLevel, OperationType

# Configure logging
logger = logging.getLogger(__name__)

# Global middleware configuration
SLOW_QUERY_THRESHOLD = 0.5  # seconds
MAX_RETRY_ATTEMPTS = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 30  # seconds


class DatabaseMetrics:
    """Track database middleware metrics"""

    def __init__(self):
        self.requests_processed = 0
        self.database_errors = 0
        self.slow_queries = 0
        self.connection_retries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        self.last_error = None
        self.error_count_window = []

    def record_request(self, execution_time: float, success: bool):
        """Record request metrics"""
        self.requests_processed += 1

        if not success:
            self.database_errors += 1
            self.error_count_window.append(time.time())
            # Keep only errors from last 5 minutes
            cutoff = time.time() - 300
            self.error_count_window = [t for t in self.error_count_window if t > cutoff]

        if execution_time > SLOW_QUERY_THRESHOLD:
            self.slow_queries += 1

    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should activate"""
        return len(self.error_count_window) >= CIRCUIT_BREAKER_THRESHOLD

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "requests_processed": self.requests_processed,
            "database_errors": self.database_errors,
            "slow_queries": self.slow_queries,
            "connection_retries": self.connection_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_rate": self.database_errors / max(self.requests_processed, 1),
            "avg_requests_per_second": self.requests_processed / max(uptime, 1),
            "circuit_breaker_active": self.should_circuit_break()
        }


class DatabaseConnectionManager:
    """Manage database connections for middleware"""

    def __init__(self):
        self.metrics = DatabaseMetrics()
        self.circuit_breaker_until = 0
        self.structured_logger = StructuredLogger("db_middleware")

    async def get_connection_with_retry(self, request: Request) -> Optional[Session]:
        """Get database connection with retry logic"""
        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Check circuit breaker
                if time.time() < self.circuit_breaker_until:
                    raise Exception("Database circuit breaker active")

                with get_db() as db:
                    # Test connection
                    db.execute("SELECT 1")
                    return db

            except Exception as e:
                self.metrics.connection_retries += 1
                self.structured_logger.log_database_operation(
                    message=f"Database connection attempt {attempt + 1} failed",
                    table="connection",
                    operation="connect",
                    level=LogLevel.WARNING,
                    error=str(e),
                    attempt=attempt + 1,
                    correlation_id=correlation_id
                )

                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    # Activate circuit breaker
                    self.circuit_breaker_until = time.time() + CIRCUIT_BREAKER_TIMEOUT
                    self.structured_logger.log_database_operation(
                        message="Database circuit breaker activated",
                        table="circuit_breaker",
                        operation="activate",
                        level=LogLevel.ERROR,
                        correlation_id=correlation_id
                    )
                    return None

                # Exponential backoff
                await asyncio.sleep(0.1 * (2 ** attempt))

        return None

    def reset_circuit_breaker(self):
        """Reset circuit breaker if enough time has passed"""
        if time.time() >= self.circuit_breaker_until:
            self.circuit_breaker_until = 0
            self.structured_logger.log_database_operation(
                message="Database circuit breaker reset",
                table="circuit_breaker",
                operation="reset",
                level=LogLevel.INFO
            )


class DatabaseCacheManager:
    """Manage database query caching"""

    def __init__(self):
        self.metrics = DatabaseMetrics()
        self.cache_enabled = REDIS_AVAILABLE
        self.structured_logger = StructuredLogger("db_cache")

    async def get_cached_query(self, query_hash: str) -> Optional[Any]:
        """Get cached query result"""
        if not self.cache_enabled:
            return None

        try:
            redis_manager = await get_redis_manager()
            result = await redis_manager.get(f"query:{query_hash}")

            if result is not None:
                self.metrics.cache_hits += 1
                self.structured_logger.log_database_operation(
                    message="Query cache hit",
                    table="cache",
                    operation="hit",
                    level=LogLevel.DEBUG,
                    query_hash=query_hash
                )
                return result
            else:
                self.metrics.cache_misses += 1
                return None

        except Exception as e:
            self.structured_logger.log_database_operation(
                message="Cache get failed",
                table="cache",
                operation="get",
                level=LogLevel.WARNING,
                error=str(e)
            )
            return None

    async def cache_query_result(self, query_hash: str, result: Any, ttl: int = 300):
        """Cache query result"""
        if not self.cache_enabled:
            return

        try:
            redis_manager = await get_redis_manager()
            await redis_manager.set(f"query:{query_hash}", result, ttl=ttl)

            self.structured_logger.log_database_operation(
                message="Query result cached",
                table="cache",
                operation="set",
                level=LogLevel.DEBUG,
                query_hash=query_hash,
                ttl=ttl
            )

        except Exception as e:
            self.structured_logger.log_database_operation(
                message="Cache set failed",
                table="cache",
                operation="set",
                level=LogLevel.WARNING,
                error=str(e)
            )


# Global instances
_db_connection_manager = DatabaseConnectionManager()
_db_cache_manager = DatabaseCacheManager()


async def database_middleware(request: Request, call_next):
    """
    Database middleware for FastAPI applications.

    Provides:
    - Request-scoped database sessions
    - Query performance monitoring
    - Connection health checks
    - Error handling and retry logic
    - Circuit breaker pattern
    - Query result caching
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    structured_logger = StructuredLogger("db_middleware")

    with structured_logger.correlation_context(
        structured_logger._get_current_context() or
        type('CorrelationContext', (), {
            'request_id': correlation_id,
            'trace_id': correlation_id,
            'session_id': None,
            'user_id': None,
            'agent_id': None,
            'workflow_id': None,
            'parent_operation_id': None
        })()
    ):
        # Reset circuit breaker if needed
        _db_connection_manager.reset_circuit_breaker()

        # Check if circuit breaker is active
        if _db_connection_manager.metrics.should_circuit_break():
            structured_logger.log_database_operation(
                message="Request blocked by circuit breaker",
                table="circuit_breaker",
                operation="block",
                level=LogLevel.WARNING,
                correlation_id=correlation_id
            )

            return Response(
                content="Database temporarily unavailable",
                status_code=503,
                headers={"Retry-After": str(CIRCUIT_BREAKER_TIMEOUT)}
            )

        try:
            # Pre-request database health check for critical endpoints
            if request.url.path.startswith("/api/") and request.method in ["POST", "PUT", "DELETE"]:
                healthy = check_database_connection()
                if not healthy:
                    structured_logger.log_database_operation(
                        message="Database health check failed",
                        table="health_check",
                        operation="check",
                        level=LogLevel.ERROR,
                        correlation_id=correlation_id
                    )

                    return Response(
                        content="Database health check failed",
                        status_code=503
                    )

            # Add database session to request state
            request.state.db_session = None
            request.state.db_queries = []
            request.state.db_start_time = start_time

            # Process request
            response = await call_next(request)

            # Calculate metrics
            execution_time = time.time() - start_time
            success = response.status_code < 400

            # Record metrics
            _db_connection_manager.metrics.record_request(execution_time, success)

            # Log request completion
            structured_logger.log_database_operation(
                message="Database middleware request completed",
                table="middleware",
                operation="request",
                level=LogLevel.INFO,
                correlation_id=correlation_id,
                execution_time_ms=execution_time * 1000,
                status_code=response.status_code,
                queries_executed=len(getattr(request.state, 'db_queries', [])),
                success=success
            )

            # Add performance headers
            response.headers["X-DB-Execution-Time"] = f"{execution_time:.4f}"
            response.headers["X-DB-Queries"] = str(len(getattr(request.state, 'db_queries', [])))
            response.headers["X-DB-Correlation-ID"] = correlation_id

            return response

        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            _db_connection_manager.metrics.record_request(execution_time, False)
            _db_connection_manager.metrics.last_error = str(e)

            structured_logger.log_database_operation(
                message="Database error in middleware",
                table="middleware",
                operation="error",
                level=LogLevel.ERROR,
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time * 1000
            )

            return Response(
                content="Database error occurred",
                status_code=500,
                headers={"X-DB-Error": "true", "X-DB-Correlation-ID": correlation_id}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            _db_connection_manager.metrics.record_request(execution_time, False)

            structured_logger.log_database_operation(
                message="Unexpected error in database middleware",
                table="middleware",
                operation="error",
                level=LogLevel.ERROR,
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time * 1000
            )

            return Response(
                content="Internal server error",
                status_code=500,
                headers={"X-DB-Error": "true", "X-DB-Correlation-ID": correlation_id}
            )

        finally:
            # Cleanup request-scoped resources
            if hasattr(request.state, 'db_session') and request.state.db_session:
                try:
                    request.state.db_session.close()
                except Exception as e:
                    logger.warning(f"Error closing database session: {e}")


async def transaction_middleware(request: Request, call_next):
    """
    Transaction middleware for automatic transaction management.

    Automatically wraps requests in database transactions for write operations
    and provides rollback on errors.
    """
    # Only apply to write operations
    if request.method not in ["POST", "PUT", "DELETE", "PATCH"]:
        return await call_next(request)

    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    structured_logger = StructuredLogger("db_transaction")

    try:
        with get_db() as db:
            request.state.db_session = db

            # Begin transaction
            transaction = db.begin()

            structured_logger.log_database_operation(
                message="Transaction started",
                table="transaction",
                operation="begin",
                level=LogLevel.DEBUG,
                correlation_id=correlation_id
            )

            try:
                # Process request within transaction
                response = await call_next(request)

                if response.status_code < 400:
                    # Commit on success
                    transaction.commit()
                    structured_logger.log_database_operation(
                        message="Transaction committed",
                        table="transaction",
                        operation="commit",
                        level=LogLevel.DEBUG,
                        correlation_id=correlation_id,
                        status_code=response.status_code
                    )
                else:
                    # Rollback on client errors
                    transaction.rollback()
                    structured_logger.log_database_operation(
                        message="Transaction rolled back due to client error",
                        table="transaction",
                        operation="rollback",
                        level=LogLevel.WARNING,
                        correlation_id=correlation_id,
                        status_code=response.status_code
                    )

                return response

            except Exception as e:
                # Rollback on exceptions
                transaction.rollback()
                structured_logger.log_database_operation(
                    message="Transaction rolled back due to exception",
                    table="transaction",
                    operation="rollback",
                    level=LogLevel.ERROR,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    except Exception as e:
        structured_logger.log_database_operation(
            message="Transaction middleware error",
            table="transaction",
            operation="error",
            level=LogLevel.ERROR,
            correlation_id=correlation_id,
            error=str(e)
        )
        raise


async def database_health_middleware(request: Request, call_next):
    """
    Database health monitoring middleware.

    Adds database health information to responses and monitors
    database performance across requests.
    """
    if request.url.path == "/health" or request.url.path == "/infrastructure-status":
        # Add database health info to health check responses
        try:
            response = await call_next(request)

            # Get database health
            db_manager = DatabaseManager()
            health_info = await db_manager.health_check_all()

            # Add health headers
            response.headers["X-DB-Health"] = health_info["overall_status"]
            response.headers["X-DB-Uptime"] = str(int(health_info["uptime_seconds"]))

            # Add metrics headers
            metrics = _db_connection_manager.metrics.get_summary()
            response.headers["X-DB-Requests"] = str(metrics["requests_processed"])
            response.headers["X-DB-Error-Rate"] = f"{metrics['error_rate']:.3f}"

            return response

        except Exception as e:
            logger.error(f"Health middleware error: {e}")
            return await call_next(request)

    return await call_next(request)


def get_database_metrics() -> Dict[str, Any]:
    """Get current database middleware metrics"""
    connection_metrics = _db_connection_manager.metrics.get_summary()
    cache_metrics = _db_cache_manager.metrics.get_summary() if _db_cache_manager.cache_enabled else {}

    return {
        "connection": connection_metrics,
        "cache": cache_metrics,
        "middleware_version": "1.0.0",
        "cache_enabled": _db_cache_manager.cache_enabled,
        "timestamp": time.time()
    }


def reset_database_metrics():
    """Reset database middleware metrics"""
    global _db_connection_manager, _db_cache_manager
    _db_connection_manager.metrics = DatabaseMetrics()
    _db_cache_manager.metrics = DatabaseMetrics()


# Context manager for manual database session management
@asynccontextmanager
async def database_session(request: Request = None):
    """
    Context manager for manual database session management.

    Usage:
        async with database_session(request) as db:
            # Use db session
            result = db.query(Model).all()
    """
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4())) if request else str(uuid.uuid4())
    structured_logger = StructuredLogger("db_session")

    try:
        with get_db() as db:
            structured_logger.log_database_operation(
                message="Manual database session started",
                table="session",
                operation="start",
                level=LogLevel.DEBUG,
                correlation_id=correlation_id
            )

            yield db

    except Exception as e:
        structured_logger.log_database_operation(
            message="Manual database session error",
            table="session",
            operation="error",
            level=LogLevel.ERROR,
            correlation_id=correlation_id,
            error=str(e)
        )
        raise

    finally:
        structured_logger.log_database_operation(
            message="Manual database session closed",
            table="session",
            operation="close",
            level=LogLevel.DEBUG,
            correlation_id=correlation_id
        )


# Export all middleware and utilities
__all__ = [
    "database_middleware",
    "transaction_middleware",
    "database_health_middleware",
    "get_database_metrics",
    "reset_database_metrics",
    "database_session",
    "DatabaseMetrics",
    "DatabaseConnectionManager",
    "DatabaseCacheManager"
]
