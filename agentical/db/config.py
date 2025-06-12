"""
Database Configuration Module for Agentical

This module provides comprehensive database configuration management with
environment variable handling, validation, and multi-database support.

Features:
- Environment-based configuration
- Multiple database support (SQLAlchemy, SurrealDB, Redis)
- Configuration validation and defaults
- Connection string building
- Performance tuning parameters
- Security configuration
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SQLAlchemyConfig:
    """SQLAlchemy database configuration"""

    # Basic connection settings
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./agentical.db"))
    async_database_url: Optional[str] = field(default=None)

    # Connection pool settings
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "20")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "1800")))
    pool_pre_ping: bool = field(default_factory=lambda: os.getenv("DB_POOL_PRE_PING", "true").lower() == "true")

    # Performance settings
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true")
    echo_pool: bool = field(default_factory=lambda: os.getenv("DB_ECHO_POOL", "false").lower() == "true")
    slow_query_threshold: float = field(default_factory=lambda: float(os.getenv("SLOW_QUERY_THRESHOLD", "0.5")))

    # Connection retry settings
    connect_retries: int = field(default_factory=lambda: int(os.getenv("DB_CONNECT_RETRIES", "3")))
    retry_interval: float = field(default_factory=lambda: float(os.getenv("DB_RETRY_INTERVAL", "1.0")))

    # Migration settings
    migrations_dir: str = field(default_factory=lambda: os.getenv("MIGRATIONS_DIR", "alembic"))
    auto_migrate: bool = field(default_factory=lambda: os.getenv("AUTO_MIGRATE", "false").lower() == "true")

    def __post_init__(self):
        """Post-initialization processing"""
        # Auto-generate async URL if not provided
        if not self.async_database_url:
            if self.database_url.startswith("sqlite:"):
                self.async_database_url = self.database_url.replace("sqlite:", "sqlite+aiosqlite:")
            elif self.database_url.startswith("postgresql:"):
                self.async_database_url = self.database_url.replace("postgresql:", "postgresql+asyncpg:")
            elif self.database_url.startswith("mysql:"):
                self.async_database_url = self.database_url.replace("mysql:", "mysql+aiomysql:")

    def get_engine_options(self) -> Dict[str, Any]:
        """Get SQLAlchemy engine options"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "echo": self.echo,
            "echo_pool": self.echo_pool
        }


@dataclass
class SurrealDBConfig:
    """SurrealDB database configuration"""

    # Connection settings
    url: str = field(default_factory=lambda: os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc"))
    host: str = field(default_factory=lambda: os.getenv("SURREALDB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("SURREALDB_PORT", "8000")))

    # Authentication
    username: str = field(default_factory=lambda: os.getenv("SURREALDB_USERNAME", "root"))
    password: str = field(default_factory=lambda: os.getenv("SURREALDB_PASSWORD", "root"))

    # Database/Namespace
    namespace: str = field(default_factory=lambda: os.getenv("SURREALDB_NAMESPACE", "devq"))
    database: str = field(default_factory=lambda: os.getenv("SURREALDB_DATABASE", "main"))

    # Connection pool settings
    max_connections: int = field(default_factory=lambda: int(os.getenv("SURREALDB_MAX_CONNECTIONS", "10")))
    connection_timeout: float = field(default_factory=lambda: float(os.getenv("SURREALDB_CONNECTION_TIMEOUT", "10.0")))
    idle_timeout: float = field(default_factory=lambda: float(os.getenv("SURREALDB_IDLE_TIMEOUT", "300.0")))

    # Performance settings
    query_timeout: float = field(default_factory=lambda: float(os.getenv("SURREALDB_QUERY_TIMEOUT", "30.0")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("SURREALDB_BATCH_SIZE", "1000")))

    # Health monitoring
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("SURREALDB_HEALTH_CHECK_INTERVAL", "30")))

    def get_connection_string(self) -> str:
        """Build SurrealDB connection string"""
        if self.url:
            return self.url

        protocol = "ws" if not os.getenv("SURREALDB_TLS", "false").lower() == "true" else "wss"
        return f"{protocol}://{self.host}:{self.port}/rpc"


@dataclass
class RedisConfig:
    """Redis database configuration"""

    # Connection settings
    url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_REST_URL"))
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))

    # Authentication
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD") or os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    username: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_USERNAME"))

    # SSL/TLS settings
    ssl: bool = field(default_factory=lambda: os.getenv("REDIS_SSL", "false").lower() == "true")
    ssl_cert_reqs: Optional[str] = field(default=None)

    # Connection pool settings
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "20")))
    socket_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")))
    socket_connect_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0")))
    retry_on_timeout: bool = field(default_factory=lambda: os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true")

    # Health monitoring
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")))

    # Cache settings
    default_ttl: int = field(default_factory=lambda: int(os.getenv("REDIS_DEFAULT_TTL", "3600")))
    key_prefix: str = field(default_factory=lambda: os.getenv("REDIS_KEY_PREFIX", "agentical:"))

    def __post_init__(self):
        """Post-initialization processing"""
        if self.ssl:
            self.ssl_cert_reqs = "required"


@dataclass
class BackupConfig:
    """Database backup configuration"""

    # Basic backup settings
    enabled: bool = field(default_factory=lambda: os.getenv("BACKUP_ENABLED", "true").lower() == "true")
    backup_dir: str = field(default_factory=lambda: os.getenv("BACKUP_DIR", "./backups"))
    retention_days: int = field(default_factory=lambda: int(os.getenv("BACKUP_RETENTION_DAYS", "30")))

    # Backup types and scheduling
    auto_backup_enabled: bool = field(default_factory=lambda: os.getenv("AUTO_BACKUP_ENABLED", "true").lower() == "true")
    backup_interval_hours: int = field(default_factory=lambda: int(os.getenv("BACKUP_INTERVAL_HOURS", "24")))
    incremental_backup_enabled: bool = field(default_factory=lambda: os.getenv("INCREMENTAL_BACKUP", "true").lower() == "true")
    incremental_interval_hours: int = field(default_factory=lambda: int(os.getenv("INCREMENTAL_INTERVAL_HOURS", "6")))

    # Compression and encryption
    compression_enabled: bool = field(default_factory=lambda: os.getenv("BACKUP_COMPRESSION", "true").lower() == "true")
    encryption_enabled: bool = field(default_factory=lambda: os.getenv("BACKUP_ENCRYPTION", "false").lower() == "true")
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv("BACKUP_ENCRYPTION_KEY"))

    # Cloud storage
    cloud_storage_enabled: bool = field(default_factory=lambda: os.getenv("CLOUD_BACKUP_ENABLED", "false").lower() == "true")
    cloud_provider: str = field(default_factory=lambda: os.getenv("CLOUD_BACKUP_PROVIDER", "s3").lower())
    cloud_bucket: Optional[str] = field(default_factory=lambda: os.getenv("CLOUD_BACKUP_BUCKET"))
    cloud_region: str = field(default_factory=lambda: os.getenv("CLOUD_BACKUP_REGION", "us-east-1"))

    # Verification and testing
    verify_backups: bool = field(default_factory=lambda: os.getenv("VERIFY_BACKUPS", "true").lower() == "true")
    test_recovery: bool = field(default_factory=lambda: os.getenv("TEST_RECOVERY", "false").lower() == "true")

    def __post_init__(self):
        """Post-initialization processing"""
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class PerformanceConfig:
    """Database performance configuration"""

    # Query performance
    slow_query_threshold: float = field(default_factory=lambda: float(os.getenv("SLOW_QUERY_THRESHOLD", "0.5")))
    query_cache_enabled: bool = field(default_factory=lambda: os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true")
    query_cache_ttl: int = field(default_factory=lambda: int(os.getenv("QUERY_CACHE_TTL", "300")))

    # Connection management
    connection_pool_monitoring: bool = field(default_factory=lambda: os.getenv("CONNECTION_POOL_MONITORING", "true").lower() == "true")
    circuit_breaker_enabled: bool = field(default_factory=lambda: os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true")
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")))
    circuit_breaker_timeout: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30")))

    # Retry configuration
    max_retry_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_RETRY_ATTEMPTS", "3")))
    retry_backoff_factor: float = field(default_factory=lambda: float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")))
    retry_max_delay: float = field(default_factory=lambda: float(os.getenv("RETRY_MAX_DELAY", "60.0")))

    # Monitoring and metrics
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("DB_METRICS_ENABLED", "true").lower() == "true")
    metrics_interval: int = field(default_factory=lambda: int(os.getenv("DB_METRICS_INTERVAL", "60")))
    detailed_logging: bool = field(default_factory=lambda: os.getenv("DB_DETAILED_LOGGING", "false").lower() == "true")


@dataclass
class SecurityConfig:
    """Database security configuration"""

    # SSL/TLS settings
    require_ssl: bool = field(default_factory=lambda: os.getenv("DB_REQUIRE_SSL", "false").lower() == "true")
    ssl_ca_cert: Optional[str] = field(default_factory=lambda: os.getenv("DB_SSL_CA_CERT"))
    ssl_client_cert: Optional[str] = field(default_factory=lambda: os.getenv("DB_SSL_CLIENT_CERT"))
    ssl_client_key: Optional[str] = field(default_factory=lambda: os.getenv("DB_SSL_CLIENT_KEY"))

    # Connection security
    allowed_hosts: List[str] = field(default_factory=lambda: os.getenv("DB_ALLOWED_HOSTS", "").split(",") if os.getenv("DB_ALLOWED_HOSTS") else [])
    max_connections_per_host: int = field(default_factory=lambda: int(os.getenv("DB_MAX_CONNECTIONS_PER_HOST", "100")))

    # Authentication
    auth_timeout: float = field(default_factory=lambda: float(os.getenv("DB_AUTH_TIMEOUT", "30.0")))
    password_rotation_enabled: bool = field(default_factory=lambda: os.getenv("DB_PASSWORD_ROTATION", "false").lower() == "true")

    # Audit logging
    audit_enabled: bool = field(default_factory=lambda: os.getenv("DB_AUDIT_ENABLED", "true").lower() == "true")
    audit_log_queries: bool = field(default_factory=lambda: os.getenv("DB_AUDIT_LOG_QUERIES", "false").lower() == "true")
    audit_log_connections: bool = field(default_factory=lambda: os.getenv("DB_AUDIT_LOG_CONNECTIONS", "true").lower() == "true")


class DatabaseConfig:
    """Main database configuration container"""

    def __init__(self):
        self.environment = os.getenv("AGENTICAL_ENV", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Initialize all configuration sections
        self.sqlalchemy = SQLAlchemyConfig()
        self.surrealdb = SurrealDBConfig()
        self.redis = RedisConfig()
        self.backup = BackupConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration settings"""
        errors = []

        # Validate SQLAlchemy configuration
        if not self.sqlalchemy.database_url:
            errors.append("DATABASE_URL is required")

        if self.sqlalchemy.pool_size <= 0:
            errors.append("DB_POOL_SIZE must be positive")

        if self.sqlalchemy.pool_timeout <= 0:
            errors.append("DB_POOL_TIMEOUT must be positive")

        # Validate backup configuration
        if self.backup.enabled and not self.backup.backup_dir:
            errors.append("BACKUP_DIR is required when backups are enabled")

        if self.backup.cloud_storage_enabled and not self.backup.cloud_bucket:
            errors.append("CLOUD_BACKUP_BUCKET is required when cloud storage is enabled")

        # Validate performance configuration
        if self.performance.circuit_breaker_threshold <= 0:
            errors.append("CIRCUIT_BREAKER_THRESHOLD must be positive")

        if self.performance.max_retry_attempts < 0:
            errors.append("MAX_RETRY_ATTEMPTS must be non-negative")

        # Log validation errors
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")

            if self.environment == "production":
                raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    def get_database_urls(self) -> Dict[str, str]:
        """Get all database URLs"""
        urls = {
            "sqlalchemy": self.sqlalchemy.database_url,
            "sqlalchemy_async": self.sqlalchemy.async_database_url,
            "surrealdb": self.surrealdb.get_connection_string()
        }

        if self.redis.url:
            urls["redis"] = self.redis.url
        else:
            urls["redis"] = f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"

        return urls

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() in ["development", "dev"]

    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment.lower() in ["testing", "test"]

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "databases": {
                "sqlalchemy": {
                    "url": self.sqlalchemy.database_url.split("@")[-1] if "@" in self.sqlalchemy.database_url else self.sqlalchemy.database_url,
                    "pool_size": self.sqlalchemy.pool_size,
                    "echo": self.sqlalchemy.echo
                },
                "surrealdb": {
                    "host": self.surrealdb.host,
                    "port": self.surrealdb.port,
                    "namespace": self.surrealdb.namespace,
                    "database": self.surrealdb.database
                },
                "redis": {
                    "host": self.redis.host,
                    "port": self.redis.port,
                    "db": self.redis.db
                }
            },
            "features": {
                "backup_enabled": self.backup.enabled,
                "auto_backup": self.backup.auto_backup_enabled,
                "cloud_backup": self.backup.cloud_storage_enabled,
                "query_cache": self.performance.query_cache_enabled,
                "circuit_breaker": self.performance.circuit_breaker_enabled,
                "ssl_required": self.security.require_ssl,
                "audit_enabled": self.security.audit_enabled
            }
        }


# Global configuration instance
_config_instance = None


def get_database_config() -> DatabaseConfig:
    """Get global database configuration instance"""
    global _config_instance

    if _config_instance is None:
        _config_instance = DatabaseConfig()

    return _config_instance


def reload_database_config():
    """Reload database configuration from environment variables"""
    global _config_instance
    _config_instance = None
    return get_database_config()


# Export configuration classes and functions
__all__ = [
    "SQLAlchemyConfig",
    "SurrealDBConfig",
    "RedisConfig",
    "BackupConfig",
    "PerformanceConfig",
    "SecurityConfig",
    "DatabaseConfig",
    "get_database_config",
    "reload_database_config"
]
