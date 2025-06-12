"""
SurrealDB Connection Manager

Async SurrealDB client for the Agentical framework with connection pooling,
health monitoring, and comprehensive error handling.

Features:
- Async connection management with retry logic
- Connection pooling and lifecycle management
- Query execution with logging and monitoring
- Health checks and connectivity verification
- Integration with Logfire observability
- Graceful error handling and recovery
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    import surrealdb
    from surrealdb import Surreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False
    # Mock Surreal class for development without surrealdb installed
    class Surreal:
        async def connect(self, *args, **kwargs):
            pass
        async def use(self, *args, **kwargs):
            pass
        async def query(self, *args, **kwargs):
            return []
        async def close(self):
            pass

# Configure logging
logger = logging.getLogger(__name__)

# Mock logfire if not available
try:
    import logfire
except ImportError:
    class MockLogfire:
        @staticmethod
        def span(name, **kwargs):
            class MockSpan:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def set_attribute(self, key, value): pass
            return MockSpan()
        @staticmethod
        def info(*args, **kwargs): pass
        @staticmethod
        def error(*args, **kwargs): pass
        @staticmethod
        def warning(*args, **kwargs): pass
    logfire = MockLogfire()


@dataclass
class SurrealDBConfig:
    """SurrealDB connection configuration."""
    url: str = "ws://localhost:8000/rpc"
    username: str = "root"
    password: str = "root"
    namespace: str = "devq"
    database: str = "main"
    max_connections: int = 10
    connection_timeout: float = 30.0
    query_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    active_connections: int = 0
    total_connections: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    uptime: float = 0.0
    query_times: List[float] = field(default_factory=list)


class SurrealDBConnection:
    """Individual SurrealDB connection wrapper."""

    def __init__(self, config: SurrealDBConfig):
        self.config = config
        self.client: Optional[Surreal] = None
        self.connected = False
        self.created_at = time.time()
        self.last_used = time.time()
        self.query_count = 0
        self.connection_id = id(self)

    async def connect(self) -> bool:
        """Establish connection to SurrealDB."""
        try:
            with logfire.span("SurrealDB Connection") as span:
                span.set_attribute("connection_id", self.connection_id)
                span.set_attribute("url", self.config.url)

                self.client = Surreal()

                # Connect with timeout
                await asyncio.wait_for(
                    self.client.connect(self.config.url),
                    timeout=self.config.connection_timeout
                )

                # Set namespace and database
                await self.client.use(self.config.namespace, self.config.database)

                self.connected = True
                self.last_used = time.time()

                logfire.info("SurrealDB connection established",
                           connection_id=self.connection_id,
                           namespace=self.config.namespace,
                           database=self.config.database)

                return True

        except Exception as e:
            self.connected = False
            logger.error(f"Failed to connect to SurrealDB: {e}")
            logfire.error("SurrealDB connection failed",
                         connection_id=self.connection_id,
                         error=str(e),
                         url=self.config.url)
            return False

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query with monitoring and error handling."""
        if not self.connected or not self.client:
            raise RuntimeError("Connection not established")

        start_time = time.time()
        query_hash = hash(query)

        try:
            with logfire.span("SurrealDB Query") as span:
                span.set_attribute("connection_id", self.connection_id)
                span.set_attribute("query_hash", query_hash)
                span.set_attribute("query_preview", query[:100])

                # Execute query with timeout
                if params:
                    result = await asyncio.wait_for(
                        self.client.query(query, params),
                        timeout=self.config.query_timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        self.client.query(query),
                        timeout=self.config.query_timeout
                    )

                execution_time = time.time() - start_time
                self.last_used = time.time()
                self.query_count += 1

                span.set_attribute("execution_time_ms", execution_time * 1000)
                span.set_attribute("result_count", len(result) if result else 0)

                logfire.info("SurrealDB query executed",
                           connection_id=self.connection_id,
                           execution_time_ms=execution_time * 1000,
                           query_hash=query_hash,
                           result_count=len(result) if result else 0)

                return result or []

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Query timeout after {execution_time:.2f}s: {query[:100]}")
            logfire.error("SurrealDB query timeout",
                         connection_id=self.connection_id,
                         execution_time_ms=execution_time * 1000,
                         query_hash=query_hash,
                         timeout=self.config.query_timeout)
            raise

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed: {e}")
            logfire.error("SurrealDB query failed",
                         connection_id=self.connection_id,
                         execution_time_ms=execution_time * 1000,
                         query_hash=query_hash,
                         error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check connection health."""
        try:
            if not self.connected or not self.client:
                return False

            # Simple health check query
            result = await self.execute_query("SELECT 1 as health_check")
            return bool(result)

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.connected = False
            return False

    async def close(self):
        """Close the connection."""
        if self.client:
            try:
                await self.client.close()
                logfire.info("SurrealDB connection closed",
                           connection_id=self.connection_id,
                           query_count=self.query_count,
                           uptime=time.time() - self.created_at)
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self.client = None
                self.connected = False


class SurrealDBManager:
    """SurrealDB connection pool manager."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config or self._load_config()
        self.connections: List[SurrealDBConnection] = []
        self.stats = ConnectionStats()
        self.started_at = time.time()
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @classmethod
    def _load_config(cls) -> SurrealDBConfig:
        """Load configuration from environment variables."""
        return SurrealDBConfig(
            url=os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc"),
            username=os.getenv("SURREALDB_USERNAME", "root"),
            password=os.getenv("SURREALDB_PASSWORD", "root"),
            namespace=os.getenv("SURREALDB_NAMESPACE", "devq"),
            database=os.getenv("SURREALDB_DATABASE", "main"),
            max_connections=int(os.getenv("SURREALDB_MAX_CONNECTIONS", "10")),
            connection_timeout=float(os.getenv("SURREALDB_CONNECTION_TIMEOUT", "30.0")),
            query_timeout=float(os.getenv("SURREALDB_QUERY_TIMEOUT", "60.0")),
            retry_attempts=int(os.getenv("SURREALDB_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("SURREALDB_RETRY_DELAY", "1.0")),
            health_check_interval=float(os.getenv("SURREALDB_HEALTH_CHECK_INTERVAL", "30.0"))
        )

    async def start(self):
        """Start the connection manager."""
        with logfire.span("SurrealDB Manager Startup"):
            logger.info("Starting SurrealDB connection manager")

            if not SURREALDB_AVAILABLE:
                logger.warning("SurrealDB client not available, running in mock mode")
                logfire.warning("SurrealDB not available", mode="mock")

            # Initialize connection pool
            await self._initialize_pool()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())

            logfire.info("SurrealDB manager started",
                        max_connections=self.config.max_connections,
                        namespace=self.config.namespace,
                        database=self.config.database)

    async def stop(self):
        """Stop the connection manager."""
        with logfire.span("SurrealDB Manager Shutdown"):
            logger.info("Stopping SurrealDB connection manager")

            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Close all connections
            async with self._lock:
                for connection in self.connections:
                    await connection.close()
                self.connections.clear()

            logfire.info("SurrealDB manager stopped",
                        uptime=time.time() - self.started_at,
                        total_queries=self.stats.successful_queries + self.stats.failed_queries)

    async def _initialize_pool(self):
        """Initialize connection pool."""
        async with self._lock:
            for i in range(min(2, self.config.max_connections)):  # Start with 2 connections
                connection = SurrealDBConnection(self.config)
                if await connection.connect():
                    self.connections.append(connection)
                    self.stats.total_connections += 1
                    self.stats.active_connections += 1

    async def _get_connection(self) -> SurrealDBConnection:
        """Get an available connection from the pool."""
        async with self._lock:
            # Find available connection
            for connection in self.connections:
                if connection.connected:
                    return connection

            # Create new connection if under limit
            if len(self.connections) < self.config.max_connections:
                connection = SurrealDBConnection(self.config)
                if await connection.connect():
                    self.connections.append(connection)
                    self.stats.total_connections += 1
                    self.stats.active_connections += 1
                    return connection

            # All connections busy, wait and retry
            raise RuntimeError("No available connections")

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query with retry logic."""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                start_time = time.time()
                connection = await self._get_connection()
                result = await connection.execute_query(query, params)

                # Update statistics
                execution_time = time.time() - start_time
                self.stats.successful_queries += 1
                self.stats.query_times.append(execution_time)

                # Keep only last 1000 query times for average calculation
                if len(self.stats.query_times) > 1000:
                    self.stats.query_times = self.stats.query_times[-1000:]

                self.stats.average_query_time = sum(self.stats.query_times) / len(self.stats.query_times)

                return result

            except Exception as e:
                last_exception = e
                self.stats.failed_queries += 1

                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Query attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Query failed after {self.config.retry_attempts} attempts: {e}")

        raise last_exception or RuntimeError("Query execution failed")

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test basic connectivity
            test_result = await self.execute_query("SELECT 1 as health")

            health_status = {
                "status": "healthy" if test_result else "unhealthy",
                "available": SURREALDB_AVAILABLE,
                "connected_instances": sum(1 for c in self.connections if c.connected),
                "total_instances": len(self.connections),
                "configuration": {
                    "url": self.config.url,
                    "namespace": self.config.namespace,
                    "database": self.config.database,
                    "max_connections": self.config.max_connections
                },
                "statistics": {
                    "successful_queries": self.stats.successful_queries,
                    "failed_queries": self.stats.failed_queries,
                    "average_query_time_ms": self.stats.average_query_time * 1000,
                    "uptime_seconds": time.time() - self.started_at
                },
                "last_check": datetime.utcnow().isoformat()
            }

            self.stats.last_health_check = datetime.utcnow()
            self.stats.health_status = health_status["status"]

            return health_status

        except Exception as e:
            error_status = {
                "status": "error",
                "available": SURREALDB_AVAILABLE,
                "error": str(e),
                "connected_instances": 0,
                "total_instances": len(self.connections),
                "last_check": datetime.utcnow().isoformat()
            }

            self.stats.health_status = "error"
            logger.error(f"SurrealDB health check failed: {e}")

            return error_status

    async def _health_monitor(self):
        """Background health monitoring task."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Remove dead connections
                async with self._lock:
                    alive_connections = []
                    for connection in self.connections:
                        if await connection.health_check():
                            alive_connections.append(connection)
                        else:
                            await connection.close()
                            self.stats.active_connections -= 1

                    self.connections = alive_connections

                # Log health status
                health = await self.health_check()
                if health["status"] != "healthy":
                    logfire.warning("SurrealDB health check warning", **health)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[SurrealDBConnection, None]:
        """Context manager for getting a connection."""
        connection = await self._get_connection()
        try:
            yield connection
        finally:
            # Connection is returned to pool automatically
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "active_connections": self.stats.active_connections,
            "total_connections": self.stats.total_connections,
            "successful_queries": self.stats.successful_queries,
            "failed_queries": self.stats.failed_queries,
            "average_query_time_ms": self.stats.average_query_time * 1000,
            "health_status": self.stats.health_status,
            "last_health_check": self.stats.last_health_check.isoformat() if self.stats.last_health_check else None,
            "uptime_seconds": time.time() - self.started_at,
            "configuration": {
                "max_connections": self.config.max_connections,
                "namespace": self.config.namespace,
                "database": self.config.database,
                "url": self.config.url
            }
        }


# Global SurrealDB manager instance
surrealdb_manager: Optional[SurrealDBManager] = None


async def get_surrealdb_manager() -> SurrealDBManager:
    """Get or create the global SurrealDB manager."""
    global surrealdb_manager
    if surrealdb_manager is None:
        surrealdb_manager = SurrealDBManager()
        await surrealdb_manager.start()
    return surrealdb_manager


async def execute_surreal_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a SurrealDB query using the global manager."""
    manager = await get_surrealdb_manager()
    return await manager.execute_query(query, params)


async def check_surrealdb_health() -> Dict[str, Any]:
    """Check SurrealDB health using the global manager."""
    try:
        manager = await get_surrealdb_manager()
        return await manager.health_check()
    except Exception as e:
        return {
            "status": "error",
            "available": SURREALDB_AVAILABLE,
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }


async def shutdown_surrealdb():
    """Shutdown the global SurrealDB manager."""
    global surrealdb_manager
    if surrealdb_manager:
        await surrealdb_manager.stop()
        surrealdb_manager = None


# FastAPI dependency
async def get_surrealdb_dependency():
    """FastAPI dependency for SurrealDB access."""
    manager = await get_surrealdb_manager()
    async with manager.get_connection() as connection:
        yield connection


__all__ = [
    "SurrealDBConfig",
    "SurrealDBManager",
    "SurrealDBConnection",
    "get_surrealdb_manager",
    "execute_surreal_query",
    "check_surrealdb_health",
    "shutdown_surrealdb",
    "get_surrealdb_dependency",
    "SURREALDB_AVAILABLE"
]
