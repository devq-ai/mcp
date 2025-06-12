"""
Async Optimization and Performance Module

This module provides async optimization patterns, response improvements,
and performance monitoring for the Agentical framework.

Features:
- Async request processing optimization
- Background task management
- Response compression and optimization
- Connection pooling and resource management
- Performance monitoring and metrics
- Async database operations
- Non-blocking I/O patterns
"""

import asyncio
import gzip
import json
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Union
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

try:
    import psutil
except ImportError:
    psutil = None

import httpx
import logfire
from fastapi import Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .exceptions import TimeoutError, ExternalServiceError
from .structured_logging import StructuredLogger, LogLevel


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    request_count: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = None


class AsyncConnectionPool:
    """Async HTTP connection pool manager"""

    def __init__(self, max_connections: int = 100, max_keepalive: int = 20):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_usage: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_client(self, base_url: str = None, timeout: float = 30.0) -> httpx.AsyncClient:
        """Get or create an async HTTP client"""
        async with self._lock:
            client_key = f"{base_url}:{timeout}"

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
                    http2=True,  # Enable HTTP/2
                )
                self._client_usage[client_key] = 0

            self._client_usage[client_key] += 1
            return self._clients[client_key]

    async def close_all(self):
        """Close all HTTP clients"""
        async with self._lock:
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()
            self._client_usage.clear()

    async def cleanup_unused(self, max_idle_time: int = 300):
        """Clean up unused clients"""
        async with self._lock:
            current_time = time.time()
            to_remove = []

            for client_key, usage_count in self._client_usage.items():
                if usage_count == 0:  # No active usage
                    to_remove.append(client_key)

            for client_key in to_remove:
                await self._clients[client_key].aclose()
                del self._clients[client_key]
                del self._client_usage[client_key]


class BackgroundTaskManager:
    """Manage background tasks for heavy operations"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: weakref.WeakSet = weakref.WeakSet()
        self._task_metrics = {
            'total_started': 0,
            'total_completed': 0,
            'total_failed': 0,
            'currently_running': 0
        }

    async def submit_task(self, func: Callable, *args, **kwargs) -> asyncio.Task:
        """Submit a background task"""
        self._task_metrics['total_started'] += 1
        self._task_metrics['currently_running'] += 1

        async def task_wrapper():
            try:
                with logfire.span("Background task", task_name=func.__name__):
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await run_in_threadpool(func, *args, **kwargs)

                self._task_metrics['total_completed'] += 1
                return result
            except Exception as e:
                self._task_metrics['total_failed'] += 1
                logfire.error(f"Background task failed: {func.__name__}", error=str(e))
                raise
            finally:
                self._task_metrics['currently_running'] -= 1

        task = asyncio.create_task(task_wrapper())
        self.active_tasks.add(task)
        return task

    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all active tasks to complete"""
        if not self.active_tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*list(self.active_tasks), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logfire.warning("Background tasks did not complete within timeout")

    def get_metrics(self) -> Dict[str, int]:
        """Get background task metrics"""
        return self._task_metrics.copy()

    async def shutdown(self):
        """Shutdown the task manager"""
        await self.wait_for_completion(timeout=30.0)
        self.executor.shutdown(wait=True)


class ResponseOptimizer:
    """Optimize response generation and compression"""

    def __init__(self):
        self.compression_threshold = 1024  # 1KB
        self.json_encoder_options = {
            'ensure_ascii': False,
            'separators': (',', ':'),  # Compact JSON
            'default': self._json_serializer
        }

    def _json_serializer(self, obj):
        """Custom JSON serializer for common objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def should_compress(self, content: bytes, accept_encoding: str = "") -> bool:
        """Determine if response should be compressed"""
        if len(content) < self.compression_threshold:
            return False

        if 'gzip' not in accept_encoding:
            return False

        return True

    def compress_response(self, content: bytes) -> bytes:
        """Compress response content with gzip"""
        return gzip.compress(content, compresslevel=6)

    async def optimize_json_response(
        self,
        data: Any,
        request: Request = None
    ) -> Response:
        """Create optimized JSON response"""
        # Serialize to JSON with optimized settings
        json_content = json.dumps(data, **self.json_encoder_options)
        content_bytes = json_content.encode('utf-8')

        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Cache-Control': 'no-cache',  # Default, can be overridden
        }

        # Check if compression should be applied
        if request and self.should_compress(content_bytes,
                                          request.headers.get('accept-encoding', '')):
            content_bytes = self.compress_response(content_bytes)
            headers['Content-Encoding'] = 'gzip'

        return Response(
            content=content_bytes,
            media_type='application/json',
            headers=headers
        )

    async def stream_large_response(
        self,
        data_generator: AsyncGenerator[Dict[str, Any], None]
    ) -> StreamingResponse:
        """Stream large responses for better performance"""
        async def generate_json_stream():
            yield b'{"items":['
            first_item = True

            async for item in data_generator:
                if not first_item:
                    yield b','

                json_item = json.dumps(item, **self.json_encoder_options)
                yield json_item.encode('utf-8')
                first_item = False

            yield b']}'

        return StreamingResponse(
            generate_json_stream(),
            media_type='application/json'
        )


class AsyncDatabaseOperations:
    """Async database operation patterns"""

    def __init__(self):
        self.connection_pool = None
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000

    async def setup_connection_pool(self, database_url: str, **kwargs):
        """Setup async database connection pool"""
        # This would be implemented based on the specific database
        # For now, we'll create a mock implementation
        self.database_url = database_url
        logfire.info("Database connection pool configured", url=database_url)

    async def execute_query_cached(
        self,
        query: str,
        parameters: tuple = (),
        cache_key: str = None
    ) -> List[Dict[str, Any]]:
        """Execute database query with caching"""
        if cache_key is None:
            cache_key = f"{hash(query)}:{hash(parameters)}"

        # Check cache
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logfire.debug("Query cache hit", cache_key=cache_key)
                return cached_result

        # Execute query (mock implementation)
        with logfire.span("Database query", query=query[:100]):
            # In real implementation, this would execute the actual query
            await asyncio.sleep(0.01)  # Simulate query time
            result = [{"mock": "data"}]  # Mock result

        # Cache result
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.query_cache.keys(),
                           key=lambda k: self.query_cache[k][1])
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = (result, time.time())
        return result

    async def execute_batch_operations(
        self,
        operations: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute multiple database operations efficiently"""
        results = []

        # Group operations by type for batch processing
        grouped_ops = {}
        for op in operations:
            op_type = op.get('type', 'unknown')
            if op_type not in grouped_ops:
                grouped_ops[op_type] = []
            grouped_ops[op_type].append(op)

        # Execute grouped operations
        for op_type, ops in grouped_ops.items():
            with logfire.span("Batch database operation", operation_type=op_type, count=len(ops)):
                # Mock batch execution
                await asyncio.sleep(0.05 * len(ops))
                for op in ops:
                    results.append({"operation": op_type, "status": "completed"})

        return results


class PerformanceMonitor:
    """Monitor and track performance metrics with enhanced system monitoring"""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._request_times = []
        self._max_samples = 1000
        self._lock = threading.Lock()
        self._logger = StructuredLogger("performance_monitor")
        self._start_time = time.time()
        self._window_start = time.time()
        self._window_requests = 0
        self._error_count = 0

    def record_request(self, response_time: float, status_code: int, request_size: int = 0, response_size: int = 0):
        """Record request performance metrics with enhanced tracking"""
        with self._lock:
            self.metrics.request_count += 1
            self.metrics.total_requests += 1
            self._window_requests += 1

            # Track errors
            if status_code >= 400:
                self._error_count += 1
                self.metrics.failed_requests += 1

            # Update response times
            self._request_times.append(response_time)

            # Maintain sliding window of request times
            if len(self._request_times) > self._max_samples:
                self._request_times = self._request_times[-self._max_samples:]

            # Update metrics
            self._update_metrics()

            # Log performance data periodically
            if self.metrics.request_count % 100 == 0:
                self._log_performance_summary()

    def _update_metrics(self):
        """Update calculated metrics"""
        if not self._request_times:
            return

        # Basic statistics
        self.metrics.avg_response_time = sum(self._request_times) / len(self._request_times)
        self.metrics.min_response_time = min(self._request_times)
        self.metrics.max_response_time = max(self._request_times)

        # Percentiles
        self.metrics.p95_response_time = self._calculate_percentile(95)
        self.metrics.p99_response_time = self._calculate_percentile(99)

        # Error rate
        self.metrics.error_rate = self._error_count / self.metrics.total_requests if self.metrics.total_requests > 0 else 0

        # Throughput (requests per second in current window)
        current_time = time.time()
        window_duration = current_time - self._window_start
        if window_duration >= 60:  # Reset window every minute
            self.metrics.throughput = self._window_requests / window_duration
            self._window_start = current_time
            self._window_requests = 0

        # System metrics
        try:
            process = psutil.Process()
            self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics.cpu_usage = process.cpu_percent()
        except Exception:
            pass  # Ignore system metric errors

        self.metrics.last_updated = datetime.utcnow()

    def _log_performance_summary(self):
        """Log performance summary using structured logging"""
        summary = self.get_performance_summary()
        self._logger.log_performance_metric(
            message="Performance summary",
            metric_name="system_performance",
            metric_value=self.metrics.avg_response_time,
            metric_unit="milliseconds",
            level=LogLevel.INFO,
            tags=summary
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            if not self._request_times:
                return {"status": "no_data"}

            uptime = time.time() - self._start_time

            return {
                "request_count": self.metrics.request_count,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "avg_response_time": round(self.metrics.avg_response_time, 3),
                "min_response_time": round(self.metrics.min_response_time, 3),
                "max_response_time": round(self.metrics.max_response_time, 3),
                "p95_response_time": round(self.metrics.p95_response_time, 3),
                "p99_response_time": round(self.metrics.p99_response_time, 3),
                "error_rate": round(self.metrics.error_rate * 100, 2),
                "throughput": round(self.metrics.throughput, 2),
                "memory_usage_mb": round(self.metrics.memory_usage, 2),
                "cpu_usage_percent": round(self.metrics.cpu_usage, 2),
                "uptime_seconds": round(uptime, 2),
                "active_connections": self.metrics.active_connections,
                "last_updated": self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
            }

    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate response time percentile"""
        if not self._request_times:
            return 0.0

        sorted_times = sorted(self._request_times)
        index = int((percentile / 100) * len(sorted_times)) - 1
        index = max(0, min(index, len(sorted_times) - 1))
        return sorted_times[index]

    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics = PerformanceMetrics()
            self._request_times = []
            self._error_count = 0
            self._window_requests = 0
            self._start_time = time.time()
            self._window_start = time.time()

    def create_performance_middleware(self):
        """Create FastAPI middleware for performance monitoring"""
        async def middleware(request: Request, call_next):
            start_time = time.time()

            # Get request size
            request_size = 0
            if request.headers.get("content-length"):
                try:
                    request_size = int(request.headers["content-length"])
                except ValueError:
                    pass

            # Process request
            response = await call_next(request)

            # Calculate metrics
            response_time = (time.time() - start_time) * 1000  # milliseconds
            response_size = 0
            if hasattr(response, 'headers') and response.headers.get("content-length"):
                try:
                    response_size = int(response.headers["content-length"])
                except ValueError:
                    pass

            # Record metrics
            self.record_request(
                response_time=response_time,
                status_code=response.status_code,
                request_size=request_size,
                response_size=response_size
            )

            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            response.headers["X-Request-Count"] = str(self.metrics.total_requests)

            return response

        return middleware


class AsyncOptimizationManager:
    """Main manager for async optimization features"""

    def __init__(self):
        self.connection_pool = AsyncConnectionPool()
        self.task_manager = BackgroundTaskManager()
        self.response_optimizer = ResponseOptimizer()
        self.db_operations = AsyncDatabaseOperations()
        self.performance_monitor = PerformanceMonitor()
        self._initialized = False

    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize async optimization components"""
        if self._initialized:
            return

        config = config or {}

        # Initialize database connection pool
        db_url = config.get('database_url', 'sqlite:///./app.db')
        await self.db_operations.setup_connection_pool(db_url)

        self._initialized = True
        logfire.info("Async optimization manager initialized")

    async def shutdown(self):
        """Shutdown all async components"""
        await self.task_manager.shutdown()
        await self.connection_pool.close_all()
        self._initialized = False
        logfire.info("Async optimization manager shutdown complete")

    @asynccontextmanager
    async def performance_tracking(self, operation_name: str):
        """Context manager for performance tracking"""
        start_time = time.time()

        with logfire.span("Performance tracking", operation=operation_name):
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.performance_monitor.record_request(duration, 200)  # Assume success

    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry"""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                with logfire.span("Retry operation", attempt=attempt, operation=operation.__name__):
                    if asyncio.iscoroutinefunction(operation):
                        return await operation(*args, **kwargs)
                    else:
                        return await run_in_threadpool(operation, *args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    wait_time = delay * (backoff_factor ** attempt)
                    logfire.warning(
                        f"Operation failed, retrying in {wait_time}s",
                        attempt=attempt,
                        error=str(e),
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logfire.error(
                        f"Operation failed after {max_retries} retries",
                        operation=operation.__name__,
                        error=str(e)
                    )

        raise last_exception

    async def parallel_execution(
        self,
        operations: List[Callable],
        max_concurrency: int = 10
    ) -> List[Any]:
        """Execute operations in parallel with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(operation):
            async with semaphore:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return await run_in_threadpool(operation)

        with logfire.span("Parallel execution", operation_count=len(operations)):
            tasks = [execute_with_semaphore(op) for op in operations]
            return await asyncio.gather(*tasks, return_exceptions=True)


# Global async optimization manager
async_optimizer = AsyncOptimizationManager()


# Performance middleware
async def performance_middleware(request: Request, call_next):
    """Middleware for performance monitoring and optimization"""
    start_time = time.time()

    # Record request start
    with logfire.span("Request processing",
                      method=request.method,
                      path=request.url.path):

        response = await call_next(request)

        # Record performance metrics
        duration = time.time() - start_time
        async_optimizer.performance_monitor.record_request(duration, response.status_code)

        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Request-ID"] = str(id(request))

        return response


# Utility functions for async optimization
async def optimize_json_response(data: Any, request: Request = None) -> Response:
    """Create optimized JSON response"""
    return await async_optimizer.response_optimizer.optimize_json_response(data, request)


async def execute_background_task(func: Callable, *args, **kwargs) -> asyncio.Task:
    """Execute function as background task"""
    return await async_optimizer.task_manager.submit_task(func, *args, **kwargs)


async def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    return async_optimizer.performance_monitor.get_performance_summary()
