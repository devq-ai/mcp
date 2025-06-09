"""
Caching System for Database Operations

This module provides caching functionality to optimize database performance by
reducing database load for frequently accessed data. It includes:

- In-memory caching using LRU cache
- Redis caching integration (optional)
- Configurable cache invalidation strategies
- Integration with Logfire for cache hit/miss monitoring
- Cache statistics and performance tracking
"""

import time
import functools
import logging
import json
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
from datetime import datetime, timedelta
from enum import Enum
import os
from threading import Lock

import logfire

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Type variables
T = TypeVar('T')
CacheKey = Union[str, Tuple[Any, ...]]
CacheValue = Any

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration from environment variables
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TYPE = os.getenv("CACHE_TYPE", "memory").lower()
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "agentical:")

# Enum for cache types
class CacheType(str, Enum):
    """Cache storage types."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    NONE = "none"

# Cache configuration
cache_config = {
    "enabled": CACHE_ENABLED,
    "type": CACHE_TYPE,
    "ttl": CACHE_TTL,
    "max_size": CACHE_MAX_SIZE,
    "redis_url": REDIS_URL,
    "redis_prefix": REDIS_PREFIX
}

# Cache statistics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "sets": 0,
    "invalidations": 0,
    "errors": 0
}
cache_stats_lock = Lock()

# Create Redis client if available
redis_client = None
if REDIS_AVAILABLE and CACHE_TYPE in ["redis", "hybrid"]:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        # Test connection
        redis_client.ping()
        logger.info(f"Redis cache connected at {REDIS_URL}")
    except Exception as e:
        redis_client = None
        logger.warning(f"Failed to connect to Redis: {e}")
        cache_stats["errors"] += 1

# In-memory LRU cache
class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to store in cache
        """
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, expiry, last_used)
        self.max_size = max_size
        self.lock = Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None
                
            value, expiry, _ = self.cache[key]
            
            # Check if expired
            current_time = time.time()
            if expiry and current_time > expiry:
                del self.cache[key]
                return None
                
            # Update last_used time
            self.cache[key] = (value, expiry, current_time)
            
            return value
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self.lock:
            current_time = time.time()
            expiry = current_time + ttl if ttl else None
            
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict least recently used item
                self._evict_lru()
                
            # Store value with expiry and last_used time
            self.cache[key] = (value, expiry, current_time)
            
    def delete(self, key: str) -> None:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            
    def _evict_lru(self) -> None:
        """Evict least recently used item from cache."""
        if not self.cache:
            return
            
        # Find key with minimum last_used time
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
        del self.cache[lru_key]

# Create memory cache
memory_cache = LRUCache(max_size=CACHE_MAX_SIZE)

def serialize_cache_key(key: CacheKey) -> str:
    """Serialize cache key to string.
    
    Args:
        key: Cache key (string or tuple)
        
    Returns:
        Serialized string key
    """
    if isinstance(key, str):
        return key
        
    # For tuples and other types, use hash
    key_str = str(key)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_from_cache(key: CacheKey) -> Tuple[bool, Any]:
    """Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Tuple of (hit, value) where hit is True if value was found
    """
    if not CACHE_ENABLED:
        return False, None
        
    # Serialize key
    serialized_key = serialize_cache_key(key)
    
    # Check memory cache first
    if CACHE_TYPE in ["memory", "hybrid"]:
        value = memory_cache.get(serialized_key)
        if value is not None:
            with cache_stats_lock:
                cache_stats["hits"] += 1
            return True, value
    
    # Check Redis cache if available
    if redis_client and CACHE_TYPE in ["redis", "hybrid"]:
        try:
            redis_key = f"{REDIS_PREFIX}{serialized_key}"
            value = redis_client.get(redis_key)
            if value:
                try:
                    # Deserialize JSON
                    deserialized = json.loads(value)
                    with cache_stats_lock:
                        cache_stats["hits"] += 1
                    return True, deserialized
                except json.JSONDecodeError:
                    # If not JSON, return as is
                    with cache_stats_lock:
                        cache_stats["hits"] += 1
                    return True, value
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            with cache_stats_lock:
                cache_stats["errors"] += 1
    
    # Cache miss
    with cache_stats_lock:
        cache_stats["misses"] += 1
    return False, None

def set_in_cache(key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
    """Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds (overrides default TTL)
    """
    if not CACHE_ENABLED:
        return
        
    # Use default TTL if not specified
    ttl = ttl if ttl is not None else CACHE_TTL
    
    # Serialize key
    serialized_key = serialize_cache_key(key)
    
    # Set in memory cache
    if CACHE_TYPE in ["memory", "hybrid"]:
        memory_cache.set(serialized_key, value, ttl)
    
    # Set in Redis cache if available
    if redis_client and CACHE_TYPE in ["redis", "hybrid"]:
        try:
            redis_key = f"{REDIS_PREFIX}{serialized_key}"
            # Serialize value to JSON if possible
            try:
                serialized_value = json.dumps(value)
                redis_client.setex(redis_key, ttl, serialized_value)
            except (TypeError, OverflowError):
                # If not JSON serializable, store as string
                redis_client.setex(redis_key, ttl, str(value))
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            with cache_stats_lock:
                cache_stats["errors"] += 1
    
    with cache_stats_lock:
        cache_stats["sets"] += 1

def invalidate_cache(key: CacheKey) -> None:
    """Invalidate cache entry.
    
    Args:
        key: Cache key to invalidate
    """
    if not CACHE_ENABLED:
        return
        
    # Serialize key
    serialized_key = serialize_cache_key(key)
    
    # Invalidate memory cache
    if CACHE_TYPE in ["memory", "hybrid"]:
        memory_cache.delete(serialized_key)
    
    # Invalidate Redis cache if available
    if redis_client and CACHE_TYPE in ["redis", "hybrid"]:
        try:
            redis_key = f"{REDIS_PREFIX}{serialized_key}"
            redis_client.delete(redis_key)
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            with cache_stats_lock:
                cache_stats["errors"] += 1
    
    with cache_stats_lock:
        cache_stats["invalidations"] += 1

def invalidate_cache_pattern(pattern: str) -> None:
    """Invalidate all cache entries matching pattern.
    
    Args:
        pattern: Pattern to match cache keys (glob style for Redis,
                string.contains for memory cache)
    """
    if not CACHE_ENABLED:
        return
    
    # Invalidate memory cache
    if CACHE_TYPE in ["memory", "hybrid"]:
        with memory_cache.lock:
            keys_to_delete = [k for k in memory_cache.cache.keys() if pattern in k]
            for key in keys_to_delete:
                memory_cache.delete(key)
    
    # Invalidate Redis cache if available
    if redis_client and CACHE_TYPE in ["redis", "hybrid"]:
        try:
            redis_key_pattern = f"{REDIS_PREFIX}{pattern}*"
            keys = redis_client.keys(redis_key_pattern)
            if keys:
                redis_client.delete(*keys)
                with cache_stats_lock:
                    cache_stats["invalidations"] += len(keys)
        except Exception as e:
            logger.error(f"Redis cache pattern invalidation error: {e}")
            with cache_stats_lock:
                cache_stats["errors"] += 1

def clear_cache() -> None:
    """Clear entire cache."""
    if not CACHE_ENABLED:
        return
    
    # Clear memory cache
    if CACHE_TYPE in ["memory", "hybrid"]:
        memory_cache.clear()
    
    # Clear Redis cache if available
    if redis_client and CACHE_TYPE in ["redis", "hybrid"]:
        try:
            # Only clear keys with our prefix
            keys = redis_client.keys(f"{REDIS_PREFIX}*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            with cache_stats_lock:
                cache_stats["errors"] += 1
    
    logger.info("Cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Dictionary of cache statistics
    """
    with cache_stats_lock:
        stats = cache_stats.copy()
    
    # Calculate hit rate
    total_requests = stats["hits"] + stats["misses"]
    hit_rate = (stats["hits"] / total_requests) * 100 if total_requests > 0 else 0
    
    return {
        **stats,
        "hit_rate": hit_rate,
        "total_requests": total_requests,
        "config": cache_config
    }

def cached(ttl: Optional[int] = None, key_func: Optional[Callable[..., CacheKey]] = None):
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (overrides default TTL)
        key_func: Custom function to generate cache key from function arguments
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not CACHE_ENABLED:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: function name + args + kwargs
                cache_key = (func.__name__, str(args), str(sorted(kwargs.items())))
            
            # Check cache
            hit, cached_value = get_from_cache(cache_key)
            if hit:
                return cast(T, cached_value)
            
            # Call function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            set_in_cache(cache_key, result, ttl)
            
            # Log slow functions for optimization
            if execution_time > 0.1:  # 100ms threshold
                logfire.info(
                    "Slow function cached",
                    function=func.__name__,
                    execution_time=execution_time,
                    cache_key=str(cache_key)[:100]  # Truncate long keys
                )
            
            return result
        
        return wrapper
    
    return decorator

def async_cached(ttl: Optional[int] = None, key_func: Optional[Callable[..., CacheKey]] = None):
    """Decorator for caching async function results.
    
    Args:
        ttl: Time to live in seconds (overrides default TTL)
        key_func: Custom function to generate cache key from function arguments
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not CACHE_ENABLED:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: function name + args + kwargs
                cache_key = (func.__name__, str(args), str(sorted(kwargs.items())))
            
            # Check cache
            hit, cached_value = get_from_cache(cache_key)
            if hit:
                return cached_value
            
            # Call function
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            set_in_cache(cache_key, result, ttl)
            
            # Log slow functions for optimization
            if execution_time > 0.1:  # 100ms threshold
                logfire.info(
                    "Slow async function cached",
                    function=func.__name__,
                    execution_time=execution_time,
                    cache_key=str(cache_key)[:100]  # Truncate long keys
                )
            
            return result
        
        return wrapper
    
    return decorator

# Export public API
__all__ = [
    "CacheType",
    "get_from_cache",
    "set_in_cache",
    "invalidate_cache",
    "invalidate_cache_pattern",
    "clear_cache",
    "get_cache_stats",
    "cached",
    "async_cached",
    "CACHE_ENABLED",
    "CACHE_TYPE",
    "CACHE_TTL",
    "CACHE_MAX_SIZE",
    "redis_client"
]