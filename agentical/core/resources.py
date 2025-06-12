"""
Connection Pooling and Resource Management Configuration

This module provides comprehensive resource management including database
connection pooling, external service connection management, memory optimization,
and resource monitoring for the Agentical framework.

Features:
- Database connection pool configuration
- External service connection management
- Resource lifecycle management
- Memory optimization patterns
- Resource monitoring and alerting
- Graceful shutdown procedures
- Connection health checking
"""

import asyncio
import gc
import os
import psutil
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import logfire
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import event, text

from agentical.core.exceptions import DatabaseError, ExternalServiceError, ConfigurationError


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pools"""
    database_url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False
    echo_pool: bool = False


@dataclass
class ResourceLimits:
    """System resource limits and thresholds"""
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_connections: int = 200
    max_threads: int = 50
    max_file_descriptors: int = 1000
    connection_timeout: float = 30.0
    request_timeout: float = 300.0


@dataclass
class ResourceMetrics:
    """Current resource usage metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    active_connections: int = 0
    active_threads: int = 0
    open_file_descriptors: int = 0
    database_connections: int = 0
    http_connections: int = 0


class DatabaseConnectionManager:
    """Manage database connection pools with health monitoring"""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._connection_count = 0
        self._failed_connections = 0
        self._last_health_check = None
        self._health_check_interval = 60  # seconds
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            # Create async engine with connection pool
            pool_kwargs = {
                'poolclass': QueuePool,
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout,
                'pool_recycle': self.config.pool_recycle,
                'pool_pre_ping': self.config.pool_pre_ping,
            }
            
            self.engine = create_async_engine(
                self.config.database_url,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                **pool_kwargs
            )
            
            # Set up event listeners for connection monitoring
            event.listen(self.engine.sync_engine, 'connect', self._on_connect)
            event.listen(self.engine.sync_engine, 'checkout', self._on_checkout)
            event.listen(self.engine.sync_engine, 'checkin', self._on_checkin)
            event.listen(self.engine.sync_engine, 'invalidate', self._on_invalidate)
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self.health_check()
            
            logfire.info("Database connection pool initialized", 
                        pool_size=self.config.pool_size,
                        max_overflow=self.config.max_overflow)
            
        except Exception as e:
            logfire.error("Failed to initialize database connection pool", error=str(e))
            raise DatabaseError(f"Database connection pool initialization failed: {e}")
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Handle new database connection"""
        self._connection_count += 1
        logfire.debug("New database connection established", 
                     total_connections=self._connection_count)
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout from pool"""
        logfire.debug("Database connection checked out from pool")
    
    def _on_checkin(self, dbapi_connection, connection_record):
        """Handle connection checkin to pool"""
        logfire.debug("Database connection checked in to pool")
    
    def _on_invalidate(self, dbapi_connection, connection_record, exception):
        """Handle connection invalidation"""
        self._failed_connections += 1
        logfire.warning("Database connection invalidated", 
                       failed_connections=self._failed_connections,
                       error=str(exception) if exception else "Unknown")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        if not self.session_factory:
            raise DatabaseError("Database connection pool not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logfire.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()
    
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None):
        """Execute a database query with connection pooling"""
        async with self.get_session() as session:
            result = await session.execute(text(query), parameters or {})
            return result.fetchall()
    
    async def health_check(self) -> bool:
        """Perform database health check"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            self._last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            logfire.error("Database health check failed", error=str(e))
            return False
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        if not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": self._connection_count,
            "failed_connections": self._failed_connections,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    async def close(self):
        """Close database connection pool"""
        if self.engine:
            await self.engine.dispose()
            logfire.info("Database connection pool closed")


class HTTPConnectionManager:
    """Manage HTTP client connections with pooling"""
    
    def __init__(self, max_connections: int = 100, max_keepalive: int = 20):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_usage: Dict[str, int] = {}
        self._client_created: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize HTTP connection manager"""
        # Start periodic cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logfire.info("HTTP connection manager initialized",
                    max_connections=self.max_connections,
                    max_keepalive=self.max_keepalive)
    
    async def get_client(self, 
                        base_url: str = None,
                        timeout: float = 30.0,
                        follow_redirects: bool = True,
                        verify: bool = True) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        client_key = f"{base_url}:{timeout}:{follow_redirects}:{verify}"
        
        async with self._lock:
            if client_key not in self._clients:
                # Create new client with optimized settings
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive
                )
                
                timeout_config = httpx.Timeout(timeout)
                
                self._clients[client_key] = httpx.AsyncClient(
                    base_url=base_url,
                    limits=limits,
                    timeout=timeout_config,
                    follow_redirects=follow_redirects,
                    verify=verify,
                    http2=True,  # Enable HTTP/2
                )
                
                self._client_usage[client_key] = 0
                self._client_created[client_key] = datetime.utcnow()
                
                logfire.debug("Created new HTTP client", client_key=client_key)
            
            self._client_usage[client_key] += 1
            return self._clients[client_key]
    
    async def _periodic_cleanup(self):
        """Periodically clean up unused HTTP clients"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_unused_clients()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("HTTP client cleanup error", error=str(e))
    
    async def _cleanup_unused_clients(self):
        """Clean up unused HTTP clients"""
        async with self._lock:
            current_time = datetime.utcnow()
            to_remove = []
            
            for client_key, usage_count in self._client_usage.items():
                created_time = self._client_created[client_key]
                age = (current_time - created_time).total_seconds()
                
                # Remove clients that are old and unused
                if usage_count == 0 and age > 600:  # 10 minutes
                    to_remove.append(client_key)
            
            for client_key in to_remove:
                await self._clients[client_key].aclose()
                del self._clients[client_key]
                del self._client_usage[client_key]
                del self._client_created[client_key]
                
                logfire.debug("Cleaned up unused HTTP client", client_key=client_key)
    
    def release_client(self, client: httpx.AsyncClient):
        """Release HTTP client (decrease usage count)"""
        # Find client key and decrease usage
        for client_key, stored_client in self._clients.items():
            if stored_client is client:
                self._client_usage[client_key] = max(0, self._client_usage[client_key] - 1)
                break
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get HTTP connection statistics"""
        async with self._lock:
            total_clients = len(self._clients)
            active_clients = sum(1 for usage in self._client_usage.values() if usage > 0)
            
            return {
                "total_clients": total_clients,
                "active_clients": active_clients,
                "idle_clients": total_clients - active_clients,
                "max_connections": self.max_connections,
                "max_keepalive": self.max_keepalive
            }
    
    async def close_all(self):
        """Close all HTTP clients"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()
            self._client_usage.clear()
            self._client_created.clear()
            
        logfire.info("All HTTP clients closed")


class MemoryManager:
    """Manage memory usage and optimization"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._memory_warnings = 0
        self._last_gc_run = datetime.utcnow()
        self._gc_threshold_mb = limits.max_memory_mb * 0.8  # 80% threshold
        
    def get_memory_usage(self) -> ResourceMetrics:
        """Get current memory usage metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return ResourceMetrics(
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            memory_percent=system_memory.percent,
            cpu_percent=process.cpu_percent(),
            active_threads=process.num_threads(),
            open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0
        )
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is high"""
        metrics = self.get_memory_usage()
        
        if metrics.memory_usage_mb > self._gc_threshold_mb:
            self._memory_warnings += 1
            logfire.warning("High memory usage detected",
                           memory_mb=metrics.memory_usage_mb,
                           threshold_mb=self._gc_threshold_mb,
                           warnings=self._memory_warnings)
            return True
        
        return False
    
    async def optimize_memory(self):
        """Optimize memory usage"""
        if self.check_memory_pressure():
            # Force garbage collection
            collected = gc.collect()
            self._last_gc_run = datetime.utcnow()
            
            logfire.info("Memory optimization performed",
                        objects_collected=collected,
                        memory_after_mb=self.get_memory_usage().memory_usage_mb)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        metrics = self.get_memory_usage()
        gc_stats = gc.get_stats()
        
        return {
            "memory_usage_mb": round(metrics.memory_usage_mb, 2),
            "memory_percent": round(metrics.memory_percent, 2),
            "memory_limit_mb": self.limits.max_memory_mb,
            "memory_warnings": self._memory_warnings,
            "last_gc_run": self._last_gc_run.isoformat(),
            "gc_collections": [stat["collections"] for stat in gc_stats],
            "gc_collected": [stat["collected"] for stat in gc_stats],
            "gc_uncollectable": [stat["uncollectable"] for stat in gc_stats]
        }


class ResourceMonitor:
    """Monitor system resources and enforce limits"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.memory_manager = MemoryManager(limits)
        self._monitoring_active = False
        self._monitor_task = None
        self._resource_history: List[ResourceMetrics] = []
        self._max_history = 100
        
    async def start_monitoring(self, interval: float = 30.0):
        """Start resource monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logfire.info("Resource monitoring started", interval=interval)
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logfire.info("Resource monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                await self._check_resources()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("Resource monitoring error", error=str(e))
                await asyncio.sleep(interval)
    
    async def _check_resources(self):
        """Check system resources and take action if needed"""
        metrics = self.memory_manager.get_memory_usage()
        
        # Add to history
        self._resource_history.append(metrics)
        if len(self._resource_history) > self._max_history:
            self._resource_history.pop(0)
        
        # Check limits and take action
        violations = []
        
        if metrics.memory_usage_mb > self.limits.max_memory_mb:
            violations.append(f"Memory usage: {metrics.memory_usage_mb:.1f}MB > {self.limits.max_memory_mb}MB")
            await self.memory_manager.optimize_memory()
        
        if metrics.cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"CPU usage: {metrics.cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
        
        if metrics.active_threads > self.limits.max_threads:
            violations.append(f"Thread count: {metrics.active_threads} > {self.limits.max_threads}")
        
        if violations:
            logfire.warning("Resource limit violations detected", violations=violations)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        current_metrics = self.memory_manager.get_memory_usage()
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Calculate averages from history
        if self._resource_history:
            avg_memory = sum(m.memory_usage_mb for m in self._resource_history) / len(self._resource_history)
            avg_cpu = sum(m.cpu_percent for m in self._resource_history) / len(self._resource_history)
        else:
            avg_memory = current_metrics.memory_usage_mb
            avg_cpu = current_metrics.cpu_percent
        
        return {
            "current": {
                "memory_mb": round(current_metrics.memory_usage_mb, 2),
                "memory_percent": round(current_metrics.memory_percent, 2),
                "cpu_percent": round(current_metrics.cpu_percent, 2),
                "threads": current_metrics.active_threads,
                "file_descriptors": current_metrics.open_file_descriptors
            },
            "averages": {
                "memory_mb": round(avg_memory, 2),
                "cpu_percent": round(avg_cpu, 2)
            },
            "limits": {
                "memory_mb": self.limits.max_memory_mb,
                "cpu_percent": self.limits.max_cpu_percent,
                "threads": self.limits.max_threads,
                "file_descriptors": self.limits.max_file_descriptors
            },
            "memory_stats": memory_stats,
            "monitoring_active": self._monitoring_active,
            "history_samples": len(self._resource_history)
        }


class ResourceManager:
    """Main resource management coordinator"""
    
    def __init__(self, 
                 db_config: ConnectionPoolConfig = None,
                 resource_limits: ResourceLimits = None):
        self.db_config = db_config or ConnectionPoolConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
        )
        self.resource_limits = resource_limits or ResourceLimits()
        
        self.db_manager = DatabaseConnectionManager(self.db_config)
        self.http_manager = HTTPConnectionManager()
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        
        self._initialized = False
        self._shutdown_callbacks: List[Callable] = []
    
    async def initialize(self):
        """Initialize all resource managers"""
        if self._initialized:
            return
        
        try:
            await self.db_manager.initialize()
            await self.http_manager.initialize()
            await self.resource_monitor.start_monitoring()
            
            self._initialized = True
            logfire.info("Resource manager initialized successfully")
            
        except Exception as e:
            logfire.error("Resource manager initialization failed", error=str(e))
            await self.shutdown()
            raise
    
    def add_shutdown_callback(self, callback: Callable):
        """Add callback to be called during shutdown"""
        self._shutdown_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown all resource managers gracefully"""
        logfire.info("Starting resource manager shutdown")
        
        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logfire.error("Shutdown callback failed", error=str(e))
        
        # Shutdown components
        await self.resource_monitor.stop_monitoring()
        await self.http_manager.close_all()
        await self.db_manager.close()
        
        self._initialized = False
        logfire.info("Resource manager shutdown complete")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all resources"""
        return {
            "database": await self.db_manager.get_pool_status(),
            "http_connections": await self.http_manager.get_connection_stats(),
            "system_resources": self.resource_monitor.get_resource_summary(),
            "resource_limits": {
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_cpu_percent": self.resource_limits.max_cpu_percent,
                "max_connections": self.resource_limits.max_connections,
                "max_threads": self.resource_limits.max_threads,
                "connection_timeout": self.resource_limits.connection_timeout,
                "request_timeout": self.resource_limits.request_timeout
            },
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global resource manager instance
resource_manager = ResourceManager()

# Convenience functions
async def get_database_session():
    """Get database session from global resource manager"""
    return resource_manager.db_manager.get_session()

async def get_http_client(base_url: str = None, **kwargs) -> httpx.AsyncClient:
    """Get HTTP client from global resource manager"""
    return await resource_manager.http_manager.get_client(base_url, **kwargs)

async def get_resource_status() -> Dict[str, Any]:
    """Get resource status from global resource manager"""
    return await resource_manager.get_comprehensive_status()