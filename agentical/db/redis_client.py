"""
Redis Client for Agentical

This module provides Redis connectivity and caching functionality for the Agentical framework.
It includes connection pooling, async operations, and comprehensive caching patterns.

Features:
- Redis connection pooling with health monitoring
- Async and sync Redis operations
- Caching patterns (get/set, hash operations, lists, sets)
- Session management and temporary data storage
- Integration with Logfire for observability
- Automatic retry logic with exponential backoff
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Generator
import logging
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

# Redis imports with fallback
try:
    import redis
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None
    RedisError = Exception
    RedisConnectionError = Exception

# Configure logging
logger = logging.getLogger(__name__)

# Redis Configuration
class RedisConfig:
    """Redis connection configuration"""

    def __init__(self):
        self.url = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.password = os.getenv("REDIS_PASSWORD") or os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.username = os.getenv("REDIS_USERNAME")
        self.ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
        self.ssl_cert_reqs = "required" if self.ssl else None

        # Connection pool settings
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.retry_on_timeout = True
        self.socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
        self.socket_connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))
        self.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

        # Cache settings
        self.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))  # 1 hour
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "agentical:")


# Connection Statistics
class RedisConnectionStats:
    """Track Redis connection statistics"""

    def __init__(self):
        self.connections_created = 0
        self.connections_closed = 0
        self.commands_executed = 0
        self.errors_count = 0
        self.last_error = None
        self.uptime_start = time.time()


# Mock Redis for testing when Redis is not available
class MockRedis:
    """Mock Redis client for development/testing"""

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._ttl = {}

    async def get(self, key: str) -> Optional[bytes]:
        # Check TTL
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._data[key]
            del self._ttl[key]
            return None
        return self._data.get(key)

    async def set(self, key: str, value: Union[str, bytes], ex: Optional[int] = None) -> bool:
        self._data[key] = value.encode() if isinstance(value, str) else value
        if ex:
            self._ttl[key] = time.time() + ex
        return True

    async def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            if key in self._ttl:
                del self._ttl[key]
            return 1
        return 0

    async def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    async def hget(self, name: str, key: str) -> Optional[bytes]:
        hash_data = self._data.get(name, {})
        if isinstance(hash_data, dict):
            return hash_data.get(key)
        return None

    async def hset(self, name: str, key: str, value: Union[str, bytes]) -> int:
        if name not in self._data:
            self._data[name] = {}
        if not isinstance(self._data[name], dict):
            self._data[name] = {}
        self._data[name][key] = value.encode() if isinstance(value, str) else value
        return 1

    async def ping(self) -> bool:
        return True

    async def close(self):
        pass


class RedisConnection:
    """Redis connection wrapper with health monitoring"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.stats = RedisConnectionStats()
        self._client = None
        self._async_client = None
        self._pool = None
        self._async_pool = None
        self._health_check_task = None

    async def connect(self) -> bool:
        """Establish Redis connections"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using mock client")
            self._async_client = MockRedis()
            return True

        try:
            with logfire.span("Redis Connection Setup"):
                # Create connection pool
                if self.config.url.startswith("redis://") or self.config.url.startswith("rediss://"):
                    # Use URL-based connection
                    self._async_pool = aioredis.ConnectionPool.from_url(
                        self.config.url,
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout
                    )
                else:
                    # Use individual parameters
                    self._async_pool = aioredis.ConnectionPool(
                        host=self.config.host,
                        port=self.config.port,
                        db=self.config.db,
                        password=self.config.password,
                        username=self.config.username,
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout,
                        ssl=self.config.ssl,
                        ssl_cert_reqs=self.config.ssl_cert_reqs
                    )

                # Create async client
                self._async_client = aioredis.Redis(connection_pool=self._async_pool)

                # Test connection
                await self._async_client.ping()

                self.stats.connections_created += 1
                logfire.info("Redis connection established",
                           host=self.config.host, port=self.config.port)

                # Start health monitoring
                self._health_check_task = asyncio.create_task(self._health_monitor())

                return True

        except Exception as e:
            self.stats.errors_count += 1
            self.stats.last_error = str(e)
            logfire.error("Redis connection failed", error=str(e))

            # Fall back to mock client
            logger.warning(f"Redis connection failed, using mock client: {e}")
            self._async_client = MockRedis()
            return False

    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command with error handling and monitoring"""
        if not self._async_client:
            raise RuntimeError("Redis client not connected")

        start_time = time.time()
        try:
            with logfire.span("Redis Command", command=command):
                method = getattr(self._async_client, command.lower())
                result = await method(*args, **kwargs)

                self.stats.commands_executed += 1

                execution_time = time.time() - start_time
                if execution_time > 1.0:  # Log slow commands
                    logfire.warning("Slow Redis command",
                                  command=command,
                                  execution_time=execution_time)

                return result

        except Exception as e:
            self.stats.errors_count += 1
            self.stats.last_error = str(e)
            logfire.error("Redis command failed",
                         command=command, error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        try:
            start_time = time.time()
            await self._async_client.ping()
            ping_time = time.time() - start_time

            info = await self._async_client.info() if hasattr(self._async_client, 'info') else {}

            return {
                "status": "healthy",
                "ping_time_ms": round(ping_time * 1000, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "stats": {
                    "connections_created": self.stats.connections_created,
                    "commands_executed": self.stats.commands_executed,
                    "errors_count": self.stats.errors_count,
                    "client_uptime": time.time() - self.stats.uptime_start
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_error": self.stats.last_error,
                "stats": {
                    "connections_created": self.stats.connections_created,
                    "commands_executed": self.stats.commands_executed,
                    "errors_count": self.stats.errors_count
                }
            }

    async def _health_monitor(self):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                health = await self.health_check()

                if health["status"] != "healthy":
                    logfire.warning("Redis health check failed", health=health)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("Health monitor error", error=str(e))

    async def close(self):
        """Close Redis connections"""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            if self._async_client and hasattr(self._async_client, 'close'):
                await self._async_client.close()

            if self._async_pool and hasattr(self._async_pool, 'disconnect'):
                await self._async_pool.disconnect()

            self.stats.connections_closed += 1
            logfire.info("Redis connections closed")

        except Exception as e:
            logfire.error("Error closing Redis connections", error=str(e))


class RedisCacheManager:
    """High-level Redis cache management"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.connection = RedisConnection(config)
        self._client = None

    async def start(self):
        """Start Redis cache manager"""
        await self.connection.connect()
        self._client = self.connection._async_client

    async def stop(self):
        """Stop Redis cache manager"""
        await self.connection.close()

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.config.key_prefix}{key}"

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with JSON deserialization"""
        try:
            cache_key = self._make_key(key)
            value = await self.connection.execute_command("get", cache_key)

            if value is None:
                return default

            # Try to deserialize JSON
            try:
                return json.loads(value.decode() if isinstance(value, bytes) else value)
            except (json.JSONDecodeError, AttributeError):
                # Return raw value if not JSON
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logfire.error("Cache get failed", key=key, error=str(e))
            return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with JSON serialization"""
        try:
            cache_key = self._make_key(key)
            ttl = ttl or self.config.default_ttl

            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            return await self.connection.execute_command("set", cache_key, serialized_value, ex=ttl)

        except Exception as e:
            logfire.error("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            cache_key = self._make_key(key)
            result = await self.connection.execute_command("delete", cache_key)
            return result > 0

        except Exception as e:
            logfire.error("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            cache_key = self._make_key(key)
            result = await self.connection.execute_command("exists", cache_key)
            return result > 0

        except Exception as e:
            logfire.error("Cache exists check failed", key=key, error=str(e))
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key"""
        try:
            cache_key = self._make_key(key)
            return await self.connection.execute_command("expire", cache_key, ttl)

        except Exception as e:
            logfire.error("Cache expire failed", key=key, error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value"""
        try:
            cache_key = self._make_key(key)
            if amount == 1:
                return await self.connection.execute_command("incr", cache_key)
            else:
                return await self.connection.execute_command("incrby", cache_key, amount)

        except Exception as e:
            logfire.error("Cache increment failed", key=key, error=str(e))
            return 0

    # Hash operations
    async def hget(self, name: str, key: str, default: Any = None) -> Any:
        """Get value from hash"""
        try:
            cache_name = self._make_key(name)
            value = await self.connection.execute_command("hget", cache_name, key)

            if value is None:
                return default

            try:
                return json.loads(value.decode() if isinstance(value, bytes) else value)
            except (json.JSONDecodeError, AttributeError):
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logfire.error("Hash get failed", name=name, key=key, error=str(e))
            return default

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Set value in hash"""
        try:
            cache_name = self._make_key(name)

            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.connection.execute_command("hset", cache_name, key, serialized_value)
            return result >= 0

        except Exception as e:
            logfire.error("Hash set failed", name=name, key=key, error=str(e))
            return False

    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all values from hash"""
        try:
            cache_name = self._make_key(name)
            result = await self.connection.execute_command("hgetall", cache_name)

            if not result:
                return {}

            # Process hash result
            hash_data = {}
            if isinstance(result, dict):
                for k, v in result.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    value = v.decode() if isinstance(v, bytes) else v
                    try:
                        hash_data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        hash_data[key] = value

            return hash_data

        except Exception as e:
            logfire.error("Hash getall failed", name=name, error=str(e))
            return {}

    # List operations
    async def lpush(self, key: str, *values) -> int:
        """Push values to left of list"""
        try:
            cache_key = self._make_key(key)
            serialized_values = [
                json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)
                for v in values
            ]
            return await self.connection.execute_command("lpush", cache_key, *serialized_values)

        except Exception as e:
            logfire.error("List push failed", key=key, error=str(e))
            return 0

    async def rpop(self, key: str) -> Any:
        """Pop value from right of list"""
        try:
            cache_key = self._make_key(key)
            value = await self.connection.execute_command("rpop", cache_key)

            if value is None:
                return None

            try:
                return json.loads(value.decode() if isinstance(value, bytes) else value)
            except (json.JSONDecodeError, AttributeError):
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logfire.error("List pop failed", key=key, error=str(e))
            return None


# Session Management
class RedisSessionManager:
    """Redis-based session management"""

    def __init__(self, cache_manager: RedisCacheManager):
        self.cache = cache_manager
        self.session_prefix = "session:"
        self.default_session_ttl = 86400  # 24 hours

    async def create_session(self, user_id: str, data: Dict[str, Any] = None) -> str:
        """Create new user session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "data": data or {}
        }

        await self.cache.set(
            f"{self.session_prefix}{session_id}",
            session_data,
            ttl=self.default_session_ttl
        )

        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return await self.cache.get(f"{self.session_prefix}{session_id}")

    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False

        session_data["last_activity"] = datetime.utcnow().isoformat()
        session_data["data"].update(data)

        return await self.cache.set(
            f"{self.session_prefix}{session_id}",
            session_data,
            ttl=self.default_session_ttl
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        return await self.cache.delete(f"{self.session_prefix}{session_id}")

    async def extend_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """Extend session expiration"""
        ttl = ttl or self.default_session_ttl
        return await self.cache.expire(f"{self.session_prefix}{session_id}", ttl)


# Global Redis manager
_redis_manager = None
_redis_session_manager = None


async def get_redis_manager() -> RedisCacheManager:
    """Get global Redis cache manager"""
    global _redis_manager

    if _redis_manager is None:
        config = RedisConfig()
        _redis_manager = RedisCacheManager(config)
        await _redis_manager.start()

    return _redis_manager


async def get_session_manager() -> RedisSessionManager:
    """Get global Redis session manager"""
    global _redis_session_manager

    if _redis_session_manager is None:
        redis_manager = await get_redis_manager()
        _redis_session_manager = RedisSessionManager(redis_manager)

    return _redis_session_manager


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis health status"""
    try:
        manager = await get_redis_manager()
        return await manager.connection.health_check()
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "available": REDIS_AVAILABLE
        }


async def shutdown_redis():
    """Shutdown Redis connections"""
    global _redis_manager, _redis_session_manager

    try:
        if _redis_manager:
            await _redis_manager.stop()
            _redis_manager = None

        _redis_session_manager = None
        logfire.info("Redis connections shutdown completed")

    except Exception as e:
        logfire.error("Error shutting down Redis", error=str(e))


# Cache decorator
def cached(key_template: str, ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_template.format(*args, **kwargs)

            try:
                manager = await get_redis_manager()

                # Try to get from cache
                cached_result = await manager.get(cache_key)
                if cached_result is not None:
                    logfire.info("Cache hit", key=cache_key, function=func.__name__)
                    return cached_result

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                await manager.set(cache_key, result, ttl)
                logfire.info("Cache miss - result cached", key=cache_key, function=func.__name__)

                return result

            except Exception as e:
                logfire.error("Cache decorator error", error=str(e), function=func.__name__)
                # Execute function without caching on error
                return await func(*args, **kwargs)

        return wrapper
    return decorator


# Export availability
__all__ = [
    "REDIS_AVAILABLE",
    "RedisConfig",
    "RedisCacheManager",
    "RedisSessionManager",
    "get_redis_manager",
    "get_session_manager",
    "check_redis_health",
    "shutdown_redis",
    "cached"
]
