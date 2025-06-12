"""
Database Tool for Agentical

This module provides comprehensive database operations and query execution
capabilities supporting multiple database types with connection management,
transaction handling, and integration with the Agentical framework.

Features:
- Multi-database support (SQLite, PostgreSQL, MySQL, SurrealDB)
- Connection pooling and management
- Transaction handling and rollback
- Query execution with parameter binding
- Schema introspection and management
- Data migration utilities
- Performance monitoring and query optimization
- Integration with SQLAlchemy and native drivers
- Async support for high-performance operations
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from contextlib import asynccontextmanager
import tempfile

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, MetaData, Table
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import surrealdb
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError
)
from ...core.logging import log_operation


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SURREALDB = "surrealdb"


class TransactionContext:
    """Context manager for database transactions."""

    def __init__(self, connection, database_type: DatabaseType):
        self.connection = connection
        self.database_type = database_type
        self.in_transaction = False

    async def __aenter__(self):
        if self.database_type == DatabaseType.SQLITE:
            await self.connection.execute("BEGIN")
        elif self.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            await self.connection.begin()
        elif self.database_type == DatabaseType.SURREALDB:
            # SurrealDB handles transactions differently
            pass

        self.in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.in_transaction:
            if exc_type is None:
                await self.commit()
            else:
                await self.rollback()

    async def commit(self):
        """Commit the transaction."""
        if self.database_type == DatabaseType.SQLITE:
            await self.connection.execute("COMMIT")
        elif self.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            await self.connection.commit()
        elif self.database_type == DatabaseType.SURREALDB:
            pass

        self.in_transaction = False

    async def rollback(self):
        """Rollback the transaction."""
        if self.database_type == DatabaseType.SQLITE:
            await self.connection.execute("ROLLBACK")
        elif self.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            await self.connection.rollback()
        elif self.database_type == DatabaseType.SURREALDB:
            pass

        self.in_transaction = False


class QueryResult:
    """Result of database query execution."""

    def __init__(
        self,
        query_id: str,
        query: str,
        success: bool,
        rows: Optional[List[Dict[str, Any]]] = None,
        affected_rows: int = 0,
        execution_time: float = 0.0,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.query_id = query_id
        self.query = query
        self.success = success
        self.rows = rows or []
        self.affected_rows = affected_rows
        self.execution_time = execution_time
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "success": self.success,
            "rows": self.rows,
            "row_count": len(self.rows),
            "affected_rows": self.affected_rows,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class DatabaseConnection:
    """Database connection wrapper with connection pooling."""

    def __init__(
        self,
        database_type: DatabaseType,
        connection_string: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.database_type = database_type
        self.connection_string = connection_string
        self.config = config or {}
        self.connection = None
        self.engine = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        """Establish database connection."""
        try:
            if self.database_type == DatabaseType.SQLITE:
                await self._connect_sqlite()
            elif self.database_type == DatabaseType.POSTGRESQL:
                await self._connect_postgresql()
            elif self.database_type == DatabaseType.MYSQL:
                await self._connect_mysql()
            elif self.database_type == DatabaseType.SURREALDB:
                await self._connect_surrealdb()
            else:
                raise ToolValidationError(f"Unsupported database type: {self.database_type}")

            self.is_connected = True
            self.logger.info(f"Connected to {self.database_type.value} database")

        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise ToolExecutionError(f"Failed to connect to database: {e}")

    async def _connect_sqlite(self) -> None:
        """Connect to SQLite database."""
        if SQLALCHEMY_AVAILABLE:
            self.engine = create_async_engine(
                self.connection_string,
                echo=self.config.get("echo", False),
                poolclass=QueuePool,
                pool_size=self.config.get("pool_size", 5),
                max_overflow=self.config.get("max_overflow", 10)
            )
        else:
            # Fallback to native sqlite3
            db_path = self.connection_string.replace("sqlite:///", "")
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = sqlite3.Row

    async def _connect_postgresql(self) -> None:
        """Connect to PostgreSQL database."""
        if not POSTGRESQL_AVAILABLE:
            raise ToolExecutionError("PostgreSQL driver not available. Install psycopg2-binary.")

        if SQLALCHEMY_AVAILABLE:
            self.engine = create_async_engine(
                self.connection_string,
                echo=self.config.get("echo", False),
                poolclass=QueuePool,
                pool_size=self.config.get("pool_size", 5),
                max_overflow=self.config.get("max_overflow", 10)
            )
        else:
            # Native psycopg2 connection would go here
            raise NotImplementedError("Native PostgreSQL connection not implemented")

    async def _connect_mysql(self) -> None:
        """Connect to MySQL database."""
        if not MYSQL_AVAILABLE:
            raise ToolExecutionError("MySQL driver not available. Install PyMySQL.")

        if SQLALCHEMY_AVAILABLE:
            self.engine = create_async_engine(
                self.connection_string,
                echo=self.config.get("echo", False),
                poolclass=QueuePool,
                pool_size=self.config.get("pool_size", 5),
                max_overflow=self.config.get("max_overflow", 10)
            )
        else:
            # Native PyMySQL connection would go here
            raise NotImplementedError("Native MySQL connection not implemented")

    async def _connect_surrealdb(self) -> None:
        """Connect to SurrealDB database."""
        if not SURREALDB_AVAILABLE:
            raise ToolExecutionError("SurrealDB driver not available. Install surrealdb.")

        # SurrealDB connection implementation
        self.connection = surrealdb.Surreal()
        await self.connection.connect(self.connection_string)

        # Authenticate if credentials provided
        if "username" in self.config and "password" in self.config:
            await self.connection.signin({
                "user": self.config["username"],
                "pass": self.config["password"]
            })

        # Use namespace and database if specified
        if "namespace" in self.config and "database" in self.config:
            await self.connection.use(self.config["namespace"], self.config["database"])

    async def disconnect(self) -> None:
        """Close database connection."""
        try:
            if self.engine:
                await self.engine.dispose()
            elif self.connection:
                if self.database_type == DatabaseType.SURREALDB:
                    await self.connection.close()
                else:
                    self.connection.close()

            self.is_connected = False
            self.logger.info(f"Disconnected from {self.database_type.value} database")

        except Exception as e:
            self.logger.error(f"Error disconnecting from database: {e}")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


class DatabaseTool:
    """
    Comprehensive database tool supporting multiple database types
    with connection management, query execution, and transaction handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database tool.

        Args:
            config: Configuration for database operations
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.connection_timeout = self.config.get("connection_timeout", 30)
        self.query_timeout = self.config.get("query_timeout", 300)
        self.max_connections = self.config.get("max_connections", 10)
        self.enable_ssl = self.config.get("enable_ssl", True)

        # Active connections cache
        self.connections: Dict[str, DatabaseConnection] = {}

        # Supported database configurations
        self.database_configs = {
            DatabaseType.SQLITE: {
                "driver": "sqlite3",
                "connection_string_template": "sqlite:///{database}",
                "supports_transactions": True,
                "supports_schemas": False,
                "supports_async": True
            },
            DatabaseType.POSTGRESQL: {
                "driver": "psycopg2",
                "connection_string_template": "postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}",
                "supports_transactions": True,
                "supports_schemas": True,
                "supports_async": True
            },
            DatabaseType.MYSQL: {
                "driver": "pymysql",
                "connection_string_template": "mysql+aiomysql://{user}:{password}@{host}:{port}/{database}",
                "supports_transactions": True,
                "supports_schemas": True,
                "supports_async": True
            },
            DatabaseType.SURREALDB: {
                "driver": "surrealdb",
                "connection_string_template": "ws://{host}:{port}/rpc",
                "supports_transactions": True,
                "supports_schemas": True,
                "supports_async": True
            }
        }

    @log_operation("database_query")
    async def execute_query(
        self,
        query: str,
        connection_config: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True,
        use_transaction: bool = False,
        timeout_override: Optional[int] = None
    ) -> QueryResult:
        """
        Execute database query with comprehensive error handling.

        Args:
            query: SQL query to execute
            connection_config: Database connection configuration
            parameters: Query parameters for parameter binding
            fetch_results: Whether to fetch and return results
            use_transaction: Whether to wrap in transaction
            timeout_override: Override default query timeout

        Returns:
            QueryResult: Query execution result with details
        """
        query_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # Validate query
            if not query.strip():
                raise ToolValidationError("Query cannot be empty")

            # Get or create database connection
            connection = await self._get_connection(connection_config)

            # Set timeout
            timeout = timeout_override or self.query_timeout

            # Execute query with optional transaction
            if use_transaction:
                async with self._create_transaction(connection) as tx:
                    result = await self._execute_query_internal(
                        query_id, query, connection, parameters, fetch_results, timeout
                    )
            else:
                result = await self._execute_query_internal(
                    query_id, query, connection, parameters, fetch_results, timeout
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Query execution timed out after {timeout} seconds")
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                query_id=query_id,
                query=query,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    async def _execute_query_internal(
        self,
        query_id: str,
        query: str,
        connection: DatabaseConnection,
        parameters: Optional[Dict[str, Any]],
        fetch_results: bool,
        timeout: int
    ) -> QueryResult:
        """Internal query execution method."""

        if connection.database_type == DatabaseType.SQLITE:
            return await self._execute_sqlite_query(
                query_id, query, connection, parameters, fetch_results, timeout
            )
        elif connection.database_type == DatabaseType.POSTGRESQL:
            return await self._execute_postgresql_query(
                query_id, query, connection, parameters, fetch_results, timeout
            )
        elif connection.database_type == DatabaseType.MYSQL:
            return await self._execute_mysql_query(
                query_id, query, connection, parameters, fetch_results, timeout
            )
        elif connection.database_type == DatabaseType.SURREALDB:
            return await self._execute_surrealdb_query(
                query_id, query, connection, parameters, fetch_results, timeout
            )
        else:
            raise ToolValidationError(f"Unsupported database type: {connection.database_type}")

    async def _execute_sqlite_query(
        self,
        query_id: str,
        query: str,
        connection: DatabaseConnection,
        parameters: Optional[Dict[str, Any]],
        fetch_results: bool,
        timeout: int
    ) -> QueryResult:
        """Execute SQLite query."""

        if connection.engine and SQLALCHEMY_AVAILABLE:
            # Use SQLAlchemy async engine
            async with connection.engine.begin() as conn:
                if parameters:
                    result = await conn.execute(text(query), parameters)
                else:
                    result = await conn.execute(text(query))

                rows = []
                affected_rows = result.rowcount

                if fetch_results and result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchall()]

                return QueryResult(
                    query_id=query_id,
                    query=query,
                    success=True,
                    rows=rows,
                    affected_rows=affected_rows
                )
        else:
            # Use native sqlite3
            cursor = connection.connection.cursor()

            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            rows = []
            affected_rows = cursor.rowcount

            if fetch_results and cursor.description:
                rows = [dict(row) for row in cursor.fetchall()]

            connection.connection.commit()

            return QueryResult(
                query_id=query_id,
                query=query,
                success=True,
                rows=rows,
                affected_rows=affected_rows
            )

    async def _execute_postgresql_query(
        self,
        query_id: str,
        query: str,
        connection: DatabaseConnection,
        parameters: Optional[Dict[str, Any]],
        fetch_results: bool,
        timeout: int
    ) -> QueryResult:
        """Execute PostgreSQL query."""

        if connection.engine and SQLALCHEMY_AVAILABLE:
            # Use SQLAlchemy async engine
            async with connection.engine.begin() as conn:
                if parameters:
                    result = await conn.execute(text(query), parameters)
                else:
                    result = await conn.execute(text(query))

                rows = []
                affected_rows = result.rowcount

                if fetch_results and result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchall()]

                return QueryResult(
                    query_id=query_id,
                    query=query,
                    success=True,
                    rows=rows,
                    affected_rows=affected_rows
                )
        else:
            raise NotImplementedError("Native PostgreSQL execution not implemented")

    async def _execute_mysql_query(
        self,
        query_id: str,
        query: str,
        connection: DatabaseConnection,
        parameters: Optional[Dict[str, Any]],
        fetch_results: bool,
        timeout: int
    ) -> QueryResult:
        """Execute MySQL query."""

        if connection.engine and SQLALCHEMY_AVAILABLE:
            # Use SQLAlchemy async engine
            async with connection.engine.begin() as conn:
                if parameters:
                    result = await conn.execute(text(query), parameters)
                else:
                    result = await conn.execute(text(query))

                rows = []
                affected_rows = result.rowcount

                if fetch_results and result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchall()]

                return QueryResult(
                    query_id=query_id,
                    query=query,
                    success=True,
                    rows=rows,
                    affected_rows=affected_rows
                )
        else:
            raise NotImplementedError("Native MySQL execution not implemented")

    async def _execute_surrealdb_query(
        self,
        query_id: str,
        query: str,
        connection: DatabaseConnection,
        parameters: Optional[Dict[str, Any]],
        fetch_results: bool,
        timeout: int
    ) -> QueryResult:
        """Execute SurrealDB query."""

        # SurrealDB uses different query methods
        if query.strip().upper().startswith('SELECT'):
            result = await connection.connection.query(query, parameters)
        else:
            result = await connection.connection.query(query, parameters)

        rows = []
        affected_rows = 0

        if fetch_results and result:
            if isinstance(result, list):
                rows = result
            elif hasattr(result, 'result'):
                rows = result.result if isinstance(result.result, list) else [result.result]

        return QueryResult(
            query_id=query_id,
            query=query,
            success=True,
            rows=rows,
            affected_rows=affected_rows
        )

    async def _get_connection(self, connection_config: Dict[str, Any]) -> DatabaseConnection:
        """Get or create database connection."""

        # Create connection key for caching
        connection_key = self._create_connection_key(connection_config)

        # Return existing connection if available
        if connection_key in self.connections:
            connection = self.connections[connection_key]
            if connection.is_connected:
                return connection

        # Create new connection
        database_type = DatabaseType(connection_config["type"])
        connection_string = self._build_connection_string(database_type, connection_config)

        connection = DatabaseConnection(
            database_type=database_type,
            connection_string=connection_string,
            config=connection_config.get("config", {})
        )

        await connection.connect()

        # Cache connection
        self.connections[connection_key] = connection

        return connection

    def _create_connection_key(self, connection_config: Dict[str, Any]) -> str:
        """Create unique key for connection caching."""
        key_parts = [
            connection_config.get("type", ""),
            connection_config.get("host", ""),
            connection_config.get("port", ""),
            connection_config.get("database", ""),
            connection_config.get("username", "")
        ]
        return "|".join(str(part) for part in key_parts)

    def _build_connection_string(
        self,
        database_type: DatabaseType,
        connection_config: Dict[str, Any]
    ) -> str:
        """Build connection string from configuration."""

        config = self.database_configs[database_type]
        template = config["connection_string_template"]

        if database_type == DatabaseType.SQLITE:
            database_path = connection_config.get("database", ":memory:")
            return template.format(database=database_path)

        elif database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            return template.format(
                user=connection_config.get("username", ""),
                password=connection_config.get("password", ""),
                host=connection_config.get("host", "localhost"),
                port=connection_config.get("port", 5432 if database_type == DatabaseType.POSTGRESQL else 3306),
                database=connection_config.get("database", "")
            )

        elif database_type == DatabaseType.SURREALDB:
            return template.format(
                host=connection_config.get("host", "localhost"),
                port=connection_config.get("port", 8000)
            )

        else:
            raise ToolValidationError(f"Unsupported database type: {database_type}")

    def _create_transaction(self, connection: DatabaseConnection) -> TransactionContext:
        """Create transaction context."""
        return TransactionContext(connection, connection.database_type)

    async def get_schema_info(self, connection_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get database schema information."""

        connection = await self._get_connection(connection_config)

        schema_info = {
            "database_type": connection.database_type.value,
            "tables": [],
            "views": [],
            "indexes": [],
            "constraints": []
        }

        try:
            if connection.database_type == DatabaseType.SQLITE:
                # Get table information
                result = await self.execute_query(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                    connection_config,
                    fetch_results=True
                )
                schema_info["tables"] = [row["name"] for row in result.rows]

                # Get view information
                result = await self.execute_query(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                    connection_config,
                    fetch_results=True
                )
                schema_info["views"] = [row["name"] for row in result.rows]

            elif connection.database_type == DatabaseType.POSTGRESQL:
                # Get table information
                result = await self.execute_query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                    connection_config,
                    fetch_results=True
                )
                schema_info["tables"] = [row["table_name"] for row in result.rows]

            # Additional schema introspection would go here for other database types

        except Exception as e:
            self.logger.error(f"Schema introspection failed: {e}")
            schema_info["error"] = str(e)

        return schema_info

    async def test_connection(self, connection_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test database connection."""

        test_result = {
            "success": False,
            "database_type": connection_config.get("type"),
            "response_time": 0.0,
            "error_message": None
        }

        start_time = datetime.now()

        try:
            connection = await self._get_connection(connection_config)

            # Execute simple test query
            if connection.database_type == DatabaseType.SQLITE:
                test_query = "SELECT 1 as test"
            elif connection.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                test_query = "SELECT 1 as test"
            elif connection.database_type == DatabaseType.SURREALDB:
                test_query = "SELECT 1 as test"

            result = await self.execute_query(test_query, connection_config, fetch_results=True)

            test_result["success"] = result.success
            test_result["response_time"] = (datetime.now() - start_time).total_seconds()

            if not result.success:
                test_result["error_message"] = result.error_message

        except Exception as e:
            test_result["error_message"] = str(e)
            test_result["response_time"] = (datetime.now() - start_time).total_seconds()

        return test_result

    async def close_all_connections(self) -> None:
        """Close all cached database connections."""

        for connection_key, connection in list(self.connections.items()):
            try:
                await connection.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing connection {connection_key}: {e}")

        self.connections.clear()

    def get_supported_databases(self) -> List[str]:
        """Get list of supported database types."""
        return [db_type.value for db_type in DatabaseType]

    def get_database_info(self, database_type: Union[DatabaseType, str]) -> Dict[str, Any]:
        """Get information about a specific database type."""

        if isinstance(database_type, str):
            database_type = DatabaseType(database_type.lower())

        if database_type not in self.database_configs:
            raise ToolValidationError(f"Database type {database_type.value} is not supported")

        return self.database_configs[database_type].copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database tool."""

        health_status = {
            "status": "healthy",
            "supported_databases": self.get_supported_databases(),
            "active_connections": len(self.connections),
            "configuration": {
                "connection_timeout": self.connection_timeout,
                "query_timeout": self.query_timeout,
                "max_connections": self.max_connections,
                "enable_ssl": self.enable_ssl
            },
            "dependencies": {
                "sqlalchemy": SQLALCHEMY_AVAILABLE,
                "postgresql": POSTGRESQL_AVAILABLE,
                "mysql": MYSQL_AVAILABLE,
                "surrealdb": SURREALDB_AVAILABLE
            }
        }

        # Test basic functionality with SQLite
        try:
            test_config = {
                "type": "sqlite",
                "database": ":memory:"
            }

            test_result = await self.test_connection(test_config)
            health_status["basic_connectivity"] = test_result["success"]

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_connectivity"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating database tool
def create_database_tool(config: Optional[Dict[str, Any]] = None) -> DatabaseTool:
    """
    Create a database tool with specified configuration.

    Args:
        config: Configuration for database operations

    Returns:
        DatabaseTool: Configured database tool instance
    """
    return DatabaseTool(config=config)
