"""
Performance Monitoring Module

Comprehensive performance monitoring system for the Agentical framework,
providing request timing, resource usage, agent performance metrics, and alerting.

Features:
- HTTP request performance monitoring with detailed metrics
- System resource monitoring (CPU, memory, disk)
- Agent execution performance tracking
- Tool usage analytics and optimization insights
- Intelligent alerting system with configurable thresholds
- Real-time performance health checks
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import psutil
import logfire
from fastapi import Request, Response
from collections import defaultdict, deque
import threading
import json


class PerformanceMetrics:
    """Container for performance metrics data."""

    def __init__(self):
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.error_counts = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'min_time': float('inf'),
            'max_time': 0
        })
        self.agent_metrics = defaultdict(lambda: {
            'executions': 0,
            'total_time': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_tokens': 0,
            'tool_usage': defaultdict(int)
        })
        self.tool_metrics = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0,
            'success_count': 0,
            'error_count': 0
        })
        self.resource_history = deque(maxlen=100)  # Last 100 resource snapshots

    def add_request_metric(self, endpoint: str, method: str, duration: float,
                          status_code: int, request_size: int = 0, response_size: int = 0):
        """Add HTTP request performance metric."""
        key = f"{method} {endpoint}"

        # Add to request times for percentile calculations
        self.request_times.append({
            'timestamp': time.time(),
            'duration': duration,
            'endpoint': key,
            'status_code': status_code
        })

        # Update endpoint-specific metrics
        metrics = self.endpoint_metrics[key]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)

        if status_code >= 400:
            metrics['errors'] += 1
            self.error_counts[status_code] += 1

    def add_agent_metric(self, agent_type: str, duration: float, success: bool,
                        tokens_used: int = 0, tools_called: List[str] = None):
        """Add agent execution performance metric."""
        metrics = self.agent_metrics[agent_type]
        metrics['executions'] += 1
        metrics['total_time'] += duration

        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1

        if tokens_used > 0:
            # Update rolling average for tokens
            total_executions = metrics['executions']
            current_avg = metrics['avg_tokens']
            metrics['avg_tokens'] = ((current_avg * (total_executions - 1)) + tokens_used) / total_executions

        if tools_called:
            for tool in tools_called:
                metrics['tool_usage'][tool] += 1

    def add_tool_metric(self, tool_name: str, duration: float, success: bool):
        """Add tool usage performance metric."""
        metrics = self.tool_metrics[tool_name]
        metrics['calls'] += 1
        metrics['total_time'] += duration

        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1

    def add_resource_snapshot(self, resource_data: Dict[str, Any]):
        """Add system resource snapshot."""
        if isinstance(resource_data, dict):
            if 'timestamp' not in resource_data:
                resource_data['timestamp'] = time.time()
            self.resource_history.append(resource_data)
        else:
            # Legacy support for individual parameters
            self.resource_history.append({
                'timestamp': time.time(),
                'cpu_percent': resource_data if isinstance(resource_data, (int, float)) else 0,
                'memory_percent': 0,
                'memory_mb': 0,
                'disk_percent': 0
            })

    def get_response_time_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.request_times:
            return {'p50': 0, 'p95': 0, 'p99': 0}

        times = sorted([req['duration'] for req in self.request_times])
        n = len(times)

        return {
            'p50': times[int(n * 0.5)] * 1000,  # Convert to ms
            'p95': times[int(n * 0.95)] * 1000,
            'p99': times[int(n * 0.99)] * 1000
        }

    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive current metrics summary."""
        percentiles = self.get_response_time_percentiles()

        # Calculate error rate (last 5 minutes)
        current_time = time.time()
        recent_requests = [req for req in self.request_times
                          if current_time - req['timestamp'] <= 300]

        total_recent = len(recent_requests)
        error_recent = sum(1 for req in recent_requests if req['status_code'] >= 400)
        error_rate = error_recent / total_recent if total_recent > 0 else 0

        # Resource metrics
        latest_resource = self.resource_history[-1] if self.resource_history else {}

        return {
            'response_times': percentiles,
            'error_rate': error_rate,
            'total_requests': len(self.request_times),
            'requests_last_5min': total_recent,
            'resource_usage': latest_resource,
            'top_endpoints': self._get_top_endpoints(),
            'agent_performance': self._get_agent_summary(),
            'tool_performance': self._get_tool_summary()
        }

    def _get_top_endpoints(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top endpoints by request count."""
        sorted_endpoints = sorted(
            self.endpoint_metrics.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        result = []
        for endpoint, metrics in sorted_endpoints[:limit]:
            avg_time = metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0
            error_rate = metrics['errors'] / metrics['count'] if metrics['count'] > 0 else 0

            result.append({
                'endpoint': endpoint,
                'count': metrics['count'],
                'avg_response_time_ms': avg_time * 1000,
                'error_rate': error_rate,
                'min_time_ms': metrics['min_time'] * 1000 if metrics['min_time'] != float('inf') else 0,
                'max_time_ms': metrics['max_time'] * 1000
            })

        return result

    def _get_agent_summary(self) -> List[Dict[str, Any]]:
        """Get agent performance summary."""
        result = []
        for agent_type, metrics in self.agent_metrics.items():
            avg_time = metrics['total_time'] / metrics['executions'] if metrics['executions'] > 0 else 0
            success_rate = metrics['success_count'] / metrics['executions'] if metrics['executions'] > 0 else 0

            result.append({
                'agent_type': agent_type,
                'executions': metrics['executions'],
                'avg_execution_time_ms': avg_time * 1000,
                'success_rate': success_rate,
                'avg_tokens': metrics['avg_tokens'],
                'most_used_tools': dict(sorted(metrics['tool_usage'].items(),
                                             key=lambda x: x[1], reverse=True)[:3])
            })

        return sorted(result, key=lambda x: x['executions'], reverse=True)

    def _get_tool_summary(self) -> List[Dict[str, Any]]:
        """Get tool performance summary."""
        result = []
        for tool_name, metrics in self.tool_metrics.items():
            avg_time = metrics['total_time'] / metrics['calls'] if metrics['calls'] > 0 else 0
            success_rate = metrics['success_count'] / metrics['calls'] if metrics['calls'] > 0 else 0

            result.append({
                'tool_name': tool_name,
                'calls': metrics['calls'],
                'avg_execution_time_ms': avg_time * 1000,
                'success_rate': success_rate
            })

        return sorted(result, key=lambda x: x['calls'], reverse=True)


class PerformanceAlertManager:
    """Intelligent alerting system for performance monitoring."""

    def __init__(self):
        self.alert_thresholds = {
            'response_time_p95_ms': 1000,    # 1 second
            'response_time_p99_ms': 2000,    # 2 seconds
            'error_rate_5min': 0.05,         # 5%
            'error_rate_1min': 0.10,         # 10% (shorter window, higher threshold)
            'memory_usage_percent': 85,       # 85%
            'cpu_usage_percent': 80,         # 80%
            'agent_error_rate': 0.15,        # 15%
            'agent_avg_time_ms': 10000,      # 10 seconds
            'tool_error_rate': 0.20,         # 20%
        }

        self.alert_history = deque(maxlen=1000)
        self.alert_cooldowns = {}  # Prevent alert spam
        self.cooldown_duration = 300  # 5 minutes between same alerts

    async def check_performance_thresholds(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Check all performance metrics against thresholds."""
        alerts = []
        current_summary = metrics.get_current_metrics_summary()
        current_time = time.time()

        # Response time alerts
        response_times = current_summary['response_times']
        alerts.extend(self._check_response_time_alerts(response_times, current_time))

        # Error rate alerts
        error_rate = current_summary['error_rate']
        alerts.extend(self._check_error_rate_alerts(error_rate, current_time))

        # Resource usage alerts
        resource_usage = current_summary.get('resource_usage', {})
        alerts.extend(self._check_resource_alerts(resource_usage, current_time))

        # Agent performance alerts
        agent_performance = current_summary['agent_performance']
        alerts.extend(self._check_agent_alerts(agent_performance, current_time))

        # Tool performance alerts
        tool_performance = current_summary['tool_performance']
        alerts.extend(self._check_tool_alerts(tool_performance, current_time))

        # Send and log alerts
        for alert in alerts:
            await self._send_alert(alert)

        return alerts

    def _check_response_time_alerts(self, response_times: Dict[str, float], current_time: float) -> List[Dict[str, Any]]:
        """Check response time thresholds."""
        alerts = []

        for percentile, threshold_key in [('p95', 'response_time_p95_ms'), ('p99', 'response_time_p99_ms')]:
            if response_times[percentile] > self.alert_thresholds[threshold_key]:
                alert_key = f"response_time_{percentile}"

                if not self._is_in_cooldown(alert_key, current_time):
                    alerts.append({
                        'type': 'performance',
                        'subtype': 'response_time',
                        'severity': 'warning' if percentile == 'p95' else 'critical',
                        'message': f'High {percentile.upper()} response time: {response_times[percentile]:.1f}ms',
                        'value': response_times[percentile],
                        'threshold': self.alert_thresholds[threshold_key],
                        'percentile': percentile,
                        'timestamp': current_time
                    })
                    self._set_cooldown(alert_key, current_time)

        return alerts

    def _check_error_rate_alerts(self, error_rate: float, current_time: float) -> List[Dict[str, Any]]:
        """Check error rate thresholds."""
        alerts = []

        if error_rate > self.alert_thresholds['error_rate_5min']:
            alert_key = "error_rate_5min"

            if not self._is_in_cooldown(alert_key, current_time):
                severity = 'critical' if error_rate > 0.10 else 'warning'
                alerts.append({
                    'type': 'reliability',
                    'subtype': 'error_rate',
                    'severity': severity,
                    'message': f'High error rate: {error_rate*100:.1f}% (last 5 minutes)',
                    'value': error_rate,
                    'threshold': self.alert_thresholds['error_rate_5min'],
                    'timestamp': current_time
                })
                self._set_cooldown(alert_key, current_time)

        return alerts

    def _check_resource_alerts(self, resource_usage: Dict[str, Any], current_time: float) -> List[Dict[str, Any]]:
        """Check system resource usage thresholds."""
        alerts = []

        # Memory usage alert
        memory_percent = resource_usage.get('memory_percent', 0)
        if memory_percent > self.alert_thresholds['memory_usage_percent']:
            alert_key = "memory_usage"

            if not self._is_in_cooldown(alert_key, current_time):
                alerts.append({
                    'type': 'resource',
                    'subtype': 'memory',
                    'severity': 'warning',
                    'message': f'High memory usage: {memory_percent:.1f}%',
                    'value': memory_percent,
                    'threshold': self.alert_thresholds['memory_usage_percent'],
                    'timestamp': current_time
                })
                self._set_cooldown(alert_key, current_time)

        # CPU usage alert
        cpu_percent = resource_usage.get('cpu_percent', 0)
        if cpu_percent > self.alert_thresholds['cpu_usage_percent']:
            alert_key = "cpu_usage"

            if not self._is_in_cooldown(alert_key, current_time):
                alerts.append({
                    'type': 'resource',
                    'subtype': 'cpu',
                    'severity': 'warning',
                    'message': f'High CPU usage: {cpu_percent:.1f}%',
                    'value': cpu_percent,
                    'threshold': self.alert_thresholds['cpu_usage_percent'],
                    'timestamp': current_time
                })
                self._set_cooldown(alert_key, current_time)

        return alerts

    def _check_agent_alerts(self, agent_performance: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
        """Check agent performance thresholds."""
        alerts = []

        for agent_data in agent_performance:
            agent_type = agent_data['agent_type']

            # Check error rate
            if 1 - agent_data['success_rate'] > self.alert_thresholds['agent_error_rate']:
                alert_key = f"agent_error_rate_{agent_type}"

                if not self._is_in_cooldown(alert_key, current_time):
                    alerts.append({
                        'type': 'agent',
                        'subtype': 'error_rate',
                        'severity': 'warning',
                        'message': f'High error rate for {agent_type}: {(1-agent_data["success_rate"])*100:.1f}%',
                        'agent_type': agent_type,
                        'value': 1 - agent_data['success_rate'],
                        'threshold': self.alert_thresholds['agent_error_rate'],
                        'timestamp': current_time
                    })
                    self._set_cooldown(alert_key, current_time)

            # Check execution time
            if agent_data['avg_execution_time_ms'] > self.alert_thresholds['agent_avg_time_ms']:
                alert_key = f"agent_time_{agent_type}"

                if not self._is_in_cooldown(alert_key, current_time):
                    alerts.append({
                        'type': 'agent',
                        'subtype': 'execution_time',
                        'severity': 'info',
                        'message': f'Slow execution for {agent_type}: {agent_data["avg_execution_time_ms"]:.0f}ms',
                        'agent_type': agent_type,
                        'value': agent_data['avg_execution_time_ms'],
                        'threshold': self.alert_thresholds['agent_avg_time_ms'],
                        'timestamp': current_time
                    })
                    self._set_cooldown(alert_key, current_time)

        return alerts

    def _check_tool_alerts(self, tool_performance: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
        """Check tool performance thresholds."""
        alerts = []

        for tool_data in tool_performance:
            tool_name = tool_data['tool_name']

            # Only alert for tools with significant usage
            if tool_data['calls'] >= 10:
                error_rate = 1 - tool_data['success_rate']

                if error_rate > self.alert_thresholds['tool_error_rate']:
                    alert_key = f"tool_error_rate_{tool_name}"

                    if not self._is_in_cooldown(alert_key, current_time):
                        alerts.append({
                            'type': 'tool',
                            'subtype': 'error_rate',
                            'severity': 'info',
                            'message': f'High error rate for tool {tool_name}: {error_rate*100:.1f}%',
                            'tool_name': tool_name,
                            'value': error_rate,
                            'threshold': self.alert_thresholds['tool_error_rate'],
                            'timestamp': current_time
                        })
                        self._set_cooldown(alert_key, current_time)

        return alerts

    def _is_in_cooldown(self, alert_key: str, current_time: float) -> bool:
        """Check if an alert type is in cooldown period."""
        last_alert_time = self.alert_cooldowns.get(alert_key, 0)
        return current_time - last_alert_time < self.cooldown_duration

    def _set_cooldown(self, alert_key: str, current_time: float):
        """Set cooldown for an alert type."""
        self.alert_cooldowns[alert_key] = current_time

    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels."""
        with logfire.span("Performance Alert") as span:
            span.set_attribute("alert_type", alert["type"])
            span.set_attribute("alert_subtype", alert.get("subtype", ""))
            span.set_attribute("severity", alert["severity"])
            span.set_attribute("value", alert["value"])
            span.set_attribute("threshold", alert["threshold"])

            # Log based on severity
            if alert["severity"] == "critical":
                logfire.error("Critical Performance Alert", **alert)
            elif alert["severity"] == "warning":
                logfire.warning("Performance Warning", **alert)
            else:
                logfire.info("Performance Notice", **alert)

            # Store in history
            self.alert_history.append(alert)


class ResourceMonitor:
    """System resource monitoring with periodic collection."""

    def __init__(self, metrics: PerformanceMetrics, collection_interval: float = 30.0):
        self.metrics = metrics
        self.collection_interval = collection_interval
        self.running = False
        self.task = None

    def start(self):
        """Start periodic resource monitoring."""
        if not self.running:
            self.running = True
            try:
                # Try to create task if event loop is running
                self.task = asyncio.create_task(self._monitor_loop())
                logfire.info("Resource monitoring started", interval=self.collection_interval)
            except RuntimeError:
                # No event loop running, will start when one becomes available
                self.task = None
                logfire.info("Resource monitoring queued (no event loop)", interval=self.collection_interval)

    async def async_start(self):
        """Start periodic resource monitoring in async context."""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._monitor_loop())
            logfire.info("Resource monitoring started async", interval=self.collection_interval)

    def stop(self):
        """Stop resource monitoring."""
        self.running = False
        if self.task:
            self.task.cancel()
            logfire.info("Resource monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("Resource monitoring error", error=str(e))
                await asyncio.sleep(self.collection_interval)

    async def collect_metrics(self):
        """Collect current system resource metrics."""
        with logfire.span("Resource Metrics Collection"):
            try:
                # System-wide metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)  # Reduce blocking time

                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()

                # Disk usage for root partition
                try:
                    disk_usage = psutil.disk_usage('/').percent
                except (OSError, AttributeError):
                    disk_usage = 0

                # Load average (Unix-like systems)
                try:
                    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                except (OSError, AttributeError):
                    load_avg = 0

                # Active processes count
                try:
                    active_processes = len([p for p in psutil.process_iter(['status'])
                                          if p.info['status'] != psutil.STATUS_ZOMBIE])
                except (psutil.Error, AttributeError):
                    active_processes = 0

                # Create metrics dict
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_mb": memory_info.available / (1024**2),
                    "disk_usage_percent": disk_usage,
                    "load_average": load_avg,
                    "active_processes": active_processes,
                    "process_memory_mb": process_memory.rss / (1024**2),
                    "process_cpu_percent": process_cpu
                }

                # Log to Logfire
                logfire.info("System Resource Metrics",
                           memory_total_gb=memory_info.total / (1024**3),
                           memory_used_gb=memory_info.used / (1024**3),
                           memory_available_gb=memory_info.available / (1024**3),
                           memory_percent=memory_info.percent,
                           cpu_percent=cpu_percent,
                           disk_percent=disk_usage,
                           load_average=load_avg)

                logfire.info("Process Resource Metrics",
                           process_memory_mb=process_memory.rss / (1024**2),
                           process_cpu_percent=process_cpu,
                           process_threads=process.num_threads(),
                           active_processes=active_processes)

                # Add to metrics collection
                self.metrics.add_resource_snapshot(metrics)

                return metrics

            except Exception as e:
                logfire.error("Failed to collect resource metrics", error=str(e))
                return {}


class PerformanceMonitor:
    """Main performance monitoring coordinator."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.alert_manager = PerformanceAlertManager()
        self.resource_monitor = ResourceMonitor(self.metrics)
        self.monitoring_active = False

    def start_monitoring(self):
        """Start all monitoring components."""
        if not self.monitoring_active:
            self.resource_monitor.start()
            self.monitoring_active = True
            logfire.info("Performance monitoring system started")

    async def async_start_monitoring(self):
        """Start all monitoring components in async context."""
        if not self.monitoring_active:
            await self.resource_monitor.async_start()
            self.monitoring_active = True
            logfire.info("Performance monitoring system started async")

    def stop_monitoring(self):
        """Stop all monitoring components."""
        if self.monitoring_active:
            self.resource_monitor.stop()
            self.monitoring_active = False
            logfire.info("Performance monitoring system stopped")

    async def create_performance_middleware(self):
        """Create FastAPI middleware for request performance monitoring."""
        async def performance_middleware(request: Request, call_next):
            start_time = time.time()

            # Extract request metadata
            method = request.method
            endpoint = request.url.path
            user_agent = request.headers.get("user-agent", "unknown")

            # Get request size
            request_size = 0
            if hasattr(request, '_body'):
                request_size = len(request._body)

            with logfire.span(
                "HTTP Request Performance",
                method=method,
                endpoint=endpoint,
                user_agent=user_agent
            ) as span:
                try:
                    # Process request
                    response = await call_next(request)

                    # Calculate metrics
                    duration = time.time() - start_time
                    status_code = response.status_code
                    response_size = int(response.headers.get("content-length", 0))

                    # Set span attributes
                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("status_code", status_code)
                    span.set_attribute("request_size_bytes", request_size)
                    span.set_attribute("response_size_bytes", response_size)

                    # Performance categorization
                    if duration > 1.0:
                        span.set_attribute("performance_category", "slow")
                    elif duration > 0.5:
                        span.set_attribute("performance_category", "moderate")
                    else:
                        span.set_attribute("performance_category", "fast")

                    # Add to metrics
                    self.metrics.add_request_metric(
                        endpoint=endpoint,
                        method=method,
                        duration=duration,
                        status_code=status_code,
                        request_size=request_size,
                        response_size=response_size
                    )

                    # Add performance header
                    response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

                    return response

                except Exception as e:
                    duration = time.time() - start_time
                    logfire.error("Request processing error",
                                 method=method,
                                 endpoint=endpoint,
                                 duration_ms=duration * 1000,
                                 error=str(e))

                    # Still record the failed request
                    self.metrics.add_request_metric(
                        endpoint=endpoint,
                        method=method,
                        duration=duration,
                        status_code=500
                    )
                    raise

        return performance_middleware

    def monitor_agent_performance(self, agent_type: str):
        """Decorator for monitoring agent execution performance."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                with logfire.span(f"Agent Execution: {agent_type}") as span:
                    span.set_attribute("agent_type", agent_type)

                    try:
                        result = await func(*args, **kwargs)

                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("status", "success")

                        # Extract metrics from result if available
                        tokens_used = getattr(result, 'token_usage', 0)
                        tools_called = getattr(result, 'tools_called', [])

                        if tokens_used:
                            span.set_attribute("tokens_used", tokens_used)
                        if tools_called:
                            span.set_attribute("tools_called", len(tools_called))
                            span.set_attribute("tool_names", tools_called)

                        # Add to metrics
                        self.metrics.add_agent_metric(
                            agent_type=agent_type,
                            duration=duration,
                            success=True,
                            tokens_used=tokens_used,
                            tools_called=tools_called
                        )

                        return result

                    except Exception as e:
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("status", "error")
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))

                        # Record failed execution
                        self.metrics.add_agent_metric(
                            agent_type=agent_type,
                            duration=duration,
                            success=False
                        )

                        logfire.error(f"Agent {agent_type} execution failed",
                                     agent_type=agent_type,
                                     duration_ms=duration * 1000,
                                     error=str(e))
                        raise

            return wrapper
        return decorator

    def monitor_tool_usage(self, tool_name: str):
        """Decorator for monitoring tool usage performance."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                with logfire.span(f"Tool Usage: {tool_name}") as span:
                    span.set_attribute("tool_name", tool_name)

                    try:
                        result = await func(*args, **kwargs)

                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("status", "success")

                        # Add to metrics
                        self.metrics.add_tool_metric(
                            tool_name=tool_name,
                            duration=duration,
                            success=True
                        )

                        return result

                    except Exception as e:
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("status", "error")
                        span.set_attribute("error_type", type(e).__name__)

                        # Record failed tool usage
                        self.metrics.add_tool_metric(
                            tool_name=tool_name,
                            duration=duration,
                            success=False
                        )

                        logfire.error(f"Tool {tool_name} execution failed",
                                     tool_name=tool_name,
                                     duration_ms=duration * 1000,
                                     error=str(e))
                        raise

            return wrapper
        return decorator

    async def check_alerts(self):
        """Check for performance alerts."""
        return await self.alert_manager.check_performance_thresholds(self.metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.metrics.get_current_metrics_summary()

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return list(self.alert_history)[-limit:]


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
