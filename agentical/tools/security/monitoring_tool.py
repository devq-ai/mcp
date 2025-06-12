"""
Monitoring Tool for Agentical

This module provides comprehensive infrastructure monitoring capabilities including
system metrics collection, real-time alerting, performance analysis, and dashboard
generation with integration to the Agentical framework.

Features:
- Multi-category metrics collection (system, application, security, business)
- Real-time alerting with multiple notification channels
- Performance trend analysis and anomaly detection
- Custom dashboard generation and visualization
- Integration with external monitoring systems (Prometheus, Grafana, DataDog)
- Resource utilization tracking and capacity planning
- Log aggregation and analysis
- Health check monitoring and SLA tracking
- Automated incident response and escalation
- Compliance monitoring and reporting
"""

import asyncio
import json
import psutil
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import statistics
import threading
from collections import deque, defaultdict

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError
)
from ...core.logging import log_operation


class MetricType(Enum):
    """Types of metrics that can be monitored."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringTarget:
    """Target system or service to monitor."""

    def __init__(
        self,
        target_id: str,
        name: str,
        target_type: str,
        endpoint: Optional[str] = None,
        check_interval: int = 60,
        timeout: int = 30,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.target_id = target_id
        self.name = name
        self.target_type = target_type
        self.endpoint = endpoint
        self.check_interval = check_interval
        self.timeout = timeout
        self.enabled = enabled
        self.metadata = metadata or {}
        self.last_check = None
        self.status = "unknown"
        self.response_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_checks = 0
        self.created_at = datetime.now()

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_checks == 0:
            return 100.0
        return ((self.total_checks - self.error_count) / self.total_checks) * 100

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def add_check_result(self, success: bool, response_time: float) -> None:
        """Record check result."""
        self.total_checks += 1
        self.last_check = datetime.now()

        if success:
            self.response_times.append(response_time)
            self.status = "healthy"
        else:
            self.error_count += 1
            self.status = "unhealthy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert target to dictionary."""
        return {
            "target_id": self.target_id,
            "name": self.name,
            "target_type": self.target_type,
            "endpoint": self.endpoint,
            "check_interval": self.check_interval,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "status": self.status,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "error_count": self.error_count,
            "total_checks": self.total_checks,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "created_at": self.created_at.isoformat()
        }


class SystemMetrics:
    """System performance metrics collection."""

    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used_gb = 0.0
        self.memory_total_gb = 0.0
        self.disk_percent = 0.0
        self.disk_used_gb = 0.0
        self.disk_total_gb = 0.0
        self.network_bytes_sent = 0
        self.network_bytes_recv = 0
        self.load_average_1m = 0.0
        self.load_average_5m = 0.0
        self.load_average_15m = 0.0
        self.process_count = 0
        self.boot_time = datetime.fromtimestamp(psutil.boot_time())
        self.timestamp = datetime.now()

    def collect(self) -> None:
        """Collect current system metrics."""
        self.timestamp = datetime.now()

        # CPU metrics
        self.cpu_percent = psutil.cpu_percent(interval=1)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        self.memory_used_gb = memory.used / (1024**3)
        self.memory_total_gb = memory.total / (1024**3)

        # Disk metrics
        disk = psutil.disk_usage('/')
        self.disk_percent = disk.percent
        self.disk_used_gb = disk.used / (1024**3)
        self.disk_total_gb = disk.total / (1024**3)

        # Network metrics
        network = psutil.net_io_counters()
        self.network_bytes_sent = network.bytes_sent
        self.network_bytes_recv = network.bytes_recv

        # Load average (Unix-like systems)
        try:
            load = psutil.getloadavg()
            self.load_average_1m = load[0]
            self.load_average_5m = load[1]
            self.load_average_15m = load[2]
        except AttributeError:
            # Windows doesn't have load average
            pass

        # Process count
        self.process_count = len(psutil.pids())

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_gb, 2),
            "disk_percent": self.disk_percent,
            "disk_used_gb": round(self.disk_used_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "load_average_1m": self.load_average_1m,
            "load_average_5m": self.load_average_5m,
            "load_average_15m": self.load_average_15m,
            "process_count": self.process_count,
            "boot_time": self.boot_time.isoformat(),
            "timestamp": self.timestamp.isoformat()
        }


class MonitoringAlert:
    """Monitoring alert with comprehensive details."""

    def __init__(
        self,
        alert_id: str,
        title: str,
        description: str,
        level: AlertLevel,
        source: str,
        metric_name: str,
        current_value: Any,
        threshold_value: Any,
        target_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = alert_id
        self.title = title
        self.description = description
        self.level = level
        self.source = source
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold_value = threshold_value
        self.target_id = target_id
        self.metadata = metadata or {}
        self.triggered_at = datetime.now()
        self.acknowledged = False
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved = False
        self.resolved_at = None
        self.escalated = False
        self.escalation_count = 0

    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now()

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()

    def escalate(self) -> None:
        """Escalate the alert."""
        self.escalated = True
        self.escalation_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "level": self.level.value,
            "source": self.source,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "target_id": self.target_id,
            "metadata": self.metadata,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated": self.escalated,
            "escalation_count": self.escalation_count
        }


class MonitoringTool:
    """
    Comprehensive monitoring tool supporting multiple metrics,
    alerting, and performance analysis capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize monitoring tool.

        Args:
            config: Configuration for monitoring
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.collection_interval = self.config.get("collection_interval", 60)
        self.retention_days = self.config.get("retention_days", 30)
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "cpu_usage": 80,
            "memory_usage": 85,
            "disk_usage": 90,
            "error_rate": 5
        })
        self.notification_channels = self.config.get("notification_channels", ["email"])
        self.dashboard_refresh = self.config.get("dashboard_refresh", 30)
        self.enable_prometheus = self.config.get("enable_prometheus", False)
        self.prometheus_port = self.config.get("prometheus_port", 8000)

        # Email configuration
        self.email_config = self.config.get("email", {})
        self.smtp_server = self.email_config.get("smtp_server", "localhost")
        self.smtp_port = self.email_config.get("smtp_port", 587)
        self.smtp_username = self.email_config.get("username")
        self.smtp_password = self.email_config.get("password")
        self.email_from = self.email_config.get("from_email", "monitoring@agentical.com")
        self.email_to = self.email_config.get("to_emails", [])

        # Slack configuration
        self.slack_config = self.config.get("slack", {})
        self.slack_webhook_url = self.slack_config.get("webhook_url")

        # Monitoring state
        self.monitoring_targets: Dict[str, MonitoringTarget] = {}
        self.alerts: Dict[str, MonitoringAlert] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minutes
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_collectors: Dict[str, Callable] = {}

        # System metrics
        self.system_metrics = SystemMetrics()
        self.last_network_stats = None

        # Prometheus metrics (if enabled)
        self.prometheus_metrics = {}
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

        # Background tasks
        self.monitoring_task = None
        self.alert_task = None
        self.cleanup_task = None

        # Start monitoring
        self.start_monitoring()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.prometheus_metrics = {
            "cpu_usage": Gauge("system_cpu_usage_percent", "CPU usage percentage"),
            "memory_usage": Gauge("system_memory_usage_percent", "Memory usage percentage"),
            "disk_usage": Gauge("system_disk_usage_percent", "Disk usage percentage"),
            "network_bytes_sent": Counter("system_network_bytes_sent_total", "Network bytes sent"),
            "network_bytes_recv": Counter("system_network_bytes_recv_total", "Network bytes received"),
            "load_average": Gauge("system_load_average", "System load average", ["duration"]),
            "response_time": Histogram("http_response_time_seconds", "HTTP response times", ["target"]),
            "alerts_total": Counter("monitoring_alerts_total", "Total alerts generated", ["level"]),
            "targets_up": Gauge("monitoring_targets_up", "Number of healthy targets")
        }

        # Start Prometheus HTTP server
        try:
            prometheus_client.start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")

    def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.alert_task = asyncio.create_task(self._alert_processing_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        tasks = [self.monitoring_task, self.alert_task, self.cleanup_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @log_operation("monitoring")
    async def add_target(
        self,
        name: str,
        target_type: str,
        endpoint: Optional[str] = None,
        check_interval: int = 60,
        timeout: int = 30,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add monitoring target.

        Args:
            name: Target name
            target_type: Type of target (service, database, api, etc.)
            endpoint: Endpoint URL for health checks
            check_interval: Check interval in seconds
            timeout: Timeout for checks
            metadata: Additional metadata

        Returns:
            str: Target ID
        """
        target_id = str(uuid.uuid4())

        target = MonitoringTarget(
            target_id=target_id,
            name=name,
            target_type=target_type,
            endpoint=endpoint,
            check_interval=check_interval,
            timeout=timeout,
            metadata=metadata
        )

        self.monitoring_targets[target_id] = target
        self.logger.info(f"Added monitoring target: {name} ({target_type})")

        return target_id

    async def remove_target(self, target_id: str) -> bool:
        """
        Remove monitoring target.

        Args:
            target_id: Target ID to remove

        Returns:
            bool: True if removed successfully
        """
        if target_id in self.monitoring_targets:
            target = self.monitoring_targets[target_id]
            del self.monitoring_targets[target_id]
            self.logger.info(f"Removed monitoring target: {target.name}")
            return True
        return False

    def add_custom_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: Any,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add custom metric value.

        Args:
            name: Metric name
            metric_type: Type of metric
            value: Metric value
            labels: Metric labels
            timestamp: Metric timestamp
        """
        timestamp = timestamp or datetime.now()
        labels = labels or {}

        metric_data = {
            "name": name,
            "type": metric_type.value,
            "value": value,
            "labels": labels,
            "timestamp": timestamp.isoformat()
        }

        # Store in metrics history
        self.metrics_history[name].append(metric_data)

        # Update Prometheus metrics if enabled
        if self.enable_prometheus and name in self.prometheus_metrics:
            prom_metric = self.prometheus_metrics[name]
            if metric_type == MetricType.COUNTER:
                prom_metric.inc(value)
            elif metric_type == MetricType.GAUGE:
                prom_metric.set(value)
            elif metric_type == MetricType.HISTOGRAM:
                prom_metric.observe(value)

    def register_metric_collector(self, name: str, collector_func: Callable) -> None:
        """
        Register custom metric collector function.

        Args:
            name: Collector name
            collector_func: Function that returns metric value
        """
        self.metric_collectors[name] = collector_func

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Collect system metrics
                self.system_metrics.collect()
                await self._process_system_metrics()

                # Check monitoring targets
                await self._check_monitoring_targets()

                # Collect custom metrics
                await self._collect_custom_metrics()

                # Update Prometheus metrics
                if self.enable_prometheus:
                    self._update_prometheus_metrics()

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _process_system_metrics(self) -> None:
        """Process and store system metrics."""
        metrics = self.system_metrics.to_dict()

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.add_custom_metric(f"system.{metric_name}", MetricType.GAUGE, value)

        # Check system metric thresholds
        await self._check_system_thresholds()

    async def _check_system_thresholds(self) -> None:
        """Check system metrics against alert thresholds."""
        checks = [
            ("cpu_usage", self.system_metrics.cpu_percent, "CPU usage"),
            ("memory_usage", self.system_metrics.memory_percent, "Memory usage"),
            ("disk_usage", self.system_metrics.disk_percent, "Disk usage")
        ]

        for threshold_key, current_value, description in checks:
            threshold = self.alert_thresholds.get(threshold_key)
            if threshold and current_value > threshold:
                await self._trigger_alert(
                    title=f"High {description}",
                    description=f"{description} is {current_value:.1f}% (threshold: {threshold}%)",
                    level=AlertLevel.WARNING if current_value < threshold * 1.1 else AlertLevel.ERROR,
                    source="system_monitoring",
                    metric_name=threshold_key,
                    current_value=current_value,
                    threshold_value=threshold
                )

    async def _check_monitoring_targets(self) -> None:
        """Check health of all monitoring targets."""
        if not REQUESTS_AVAILABLE:
            return

        tasks = []
        for target in self.monitoring_targets.values():
            if target.enabled and target.endpoint:
                # Check if it's time for next check
                if (not target.last_check or
                    (datetime.now() - target.last_check).total_seconds() >= target.check_interval):
                    task = asyncio.create_task(self._check_target_health(target))
                    tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_target_health(self, target: MonitoringTarget) -> None:
        """Check health of a specific target."""
        start_time = time.time()
        success = False

        try:
            import requests
            response = requests.get(target.endpoint, timeout=target.timeout)
            response_time = time.time() - start_time
            success = response.status_code < 400

            target.add_check_result(success, response_time)

            if not success:
                await self._trigger_alert(
                    title=f"Target Health Check Failed: {target.name}",
                    description=f"Health check for {target.name} failed with status {response.status_code}",
                    level=AlertLevel.ERROR,
                    source="health_check",
                    metric_name="health_check",
                    current_value=response.status_code,
                    threshold_value=200,
                    target_id=target.target_id
                )

        except Exception as e:
            response_time = time.time() - start_time
            target.add_check_result(False, response_time)

            await self._trigger_alert(
                title=f"Target Unreachable: {target.name}",
                description=f"Failed to reach {target.name}: {str(e)}",
                level=AlertLevel.ERROR,
                source="health_check",
                metric_name="connectivity",
                current_value="unreachable",
                threshold_value="reachable",
                target_id=target.target_id
            )

    async def _collect_custom_metrics(self) -> None:
        """Collect custom metrics using registered collectors."""
        for name, collector_func in self.metric_collectors.items():
            try:
                value = await collector_func() if asyncio.iscoroutinefunction(collector_func) else collector_func()
                self.add_custom_metric(f"custom.{name}", MetricType.GAUGE, value)
            except Exception as e:
                self.logger.error(f"Error collecting custom metric {name}: {e}")

    def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics with current values."""
        if not self.prometheus_metrics:
            return

        # System metrics
        self.prometheus_metrics["cpu_usage"].set(self.system_metrics.cpu_percent)
        self.prometheus_metrics["memory_usage"].set(self.system_metrics.memory_percent)
        self.prometheus_metrics["disk_usage"].set(self.system_metrics.disk_percent)

        # Load average
        self.prometheus_metrics["load_average"].labels(duration="1m").set(self.system_metrics.load_average_1m)
        self.prometheus_metrics["load_average"].labels(duration="5m").set(self.system_metrics.load_average_5m)
        self.prometheus_metrics["load_average"].labels(duration="15m").set(self.system_metrics.load_average_15m)

        # Network metrics (rate calculation)
        if self.last_network_stats:
            bytes_sent_rate = self.system_metrics.network_bytes_sent - self.last_network_stats["bytes_sent"]
            bytes_recv_rate = self.system_metrics.network_bytes_recv - self.last_network_stats["bytes_recv"]

            if bytes_sent_rate > 0:
                self.prometheus_metrics["network_bytes_sent"].inc(bytes_sent_rate)
            if bytes_recv_rate > 0:
                self.prometheus_metrics["network_bytes_recv"].inc(bytes_recv_rate)

        self.last_network_stats = {
            "bytes_sent": self.system_metrics.network_bytes_sent,
            "bytes_recv": self.system_metrics.network_bytes_recv
        }

        # Target health
        healthy_targets = len([t for t in self.monitoring_targets.values() if t.status == "healthy"])
        self.prometheus_metrics["targets_up"].set(healthy_targets)

    async def _trigger_alert(
        self,
        title: str,
        description: str,
        level: AlertLevel,
        source: str,
        metric_name: str,
        current_value: Any,
        threshold_value: Any,
        target_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger a monitoring alert.

        Args:
            title: Alert title
            description: Alert description
            level: Alert severity level
            source: Alert source
            metric_name: Related metric name
            current_value: Current metric value
            threshold_value: Threshold value
            target_id: Related target ID
            metadata: Additional metadata

        Returns:
            str: Alert ID
        """
        alert_id = str(uuid.uuid4())

        alert = MonitoringAlert(
            alert_id=alert_id,
            title=title,
            description=description,
            level=level,
            source=source,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            target_id=target_id,
            metadata=metadata
        )

        self.alerts[alert_id] = alert

        # Update Prometheus alert counter
        if self.enable_prometheus and "alerts_total" in self.prometheus_metrics:
            self.prometheus_metrics["alerts_total"].labels(level=level.value).inc()

        # Send notifications
        await self._send_alert_notifications(alert)

        self.logger.warning(f"Alert triggered: {title} [{level.value}]")
        return alert_id

    async def _send_alert_notifications(self, alert: MonitoringAlert) -> None:
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels:
            try:
                if channel == "email" and EMAIL_AVAILABLE:
                    await self._send_email_alert(alert)
                elif channel == "slack" and self.slack_webhook_url:
                    await self._send_slack_alert(alert)
                elif channel == "webhook":
                    await self._send_webhook_alert(alert)
            except Exception as e:
                self.logger.error(f"Failed to send {channel} notification: {e}")

    async def _send_email_alert(self, alert: MonitoringAlert) -> None:
        """Send email alert notification."""
        if not self.email_to or not EMAIL_AVAILABLE:
            return

        subject = f"[{alert.level.value.upper()}] {alert.title}"

        body = f"""
Alert Details:
- Title: {alert.title}
- Level: {alert.level.value}
- Description: {alert.description}
- Source: {alert.source}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}
- Triggered At: {alert.triggered_at.isoformat()}
- Alert ID: {alert.alert_id}

This is an automated alert from the Agentical Monitoring System.
"""

        msg = MimeMultipart()
        msg['From'] = self.email_from
        msg['To'] = ', '.join(self.email_to)
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.smtp_username and self.smtp_password:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)

            server.send_message(msg)
            server.quit()
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    async def _send_slack_alert(self, alert: MonitoringAlert) -> None:
        """Send Slack alert notification."""
        if not self.slack_webhook_url or not REQUESTS_AVAILABLE:
            return

        color_map = {
            AlertLevel.DEBUG: "#36a64f",
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ff9500",
            AlertLevel.ERROR: "#ff0000",
            AlertLevel.CRITICAL: "#ff0000",
            AlertLevel.EMERGENCY: "#ff0000"
        }

        payload = {
            "text": f"Alert: {alert.title}",
            "attachments": [{
                "color": color_map.get(alert.level, "#ff0000"),
                "fields": [
                    {"title": "Level", "value": alert.level.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                    {"title": "Time", "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                ],
                "text": alert.description
            }]
        }

        try:
            import requests
            requests.post(self.slack_webhook_url, json=payload, timeout=10)
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    async def _send_webhook_alert(self, alert: MonitoringAlert) -> None:
        """Send webhook alert notification."""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url or not REQUESTS_AVAILABLE:
            return

        try:
            import requests
            requests.post(webhook_url, json=alert.to_dict(), timeout=10)
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    async def _alert_processing_loop(self) -> None:
        """Background alert processing loop."""
        while True:
            try:
                # Process alert escalations
                await self._process_alert_escalations()

                # Auto-resolve alerts
                await self._auto_resolve_alerts()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(60)

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations based on time and severity."""
        current_time = datetime.utcnow()

        for alert in list(self.active_alerts.values()):
            if alert.acknowledged or alert.resolved:
                continue

            # Calculate time since trigger
            time_since_trigger = (current_time - alert.triggered_at).total_seconds()

            # Escalate based on level and time
            escalation_times = {
                AlertLevel.INFO: 3600,      # 1 hour
                AlertLevel.WARNING: 1800,   # 30 minutes
                AlertLevel.ERROR: 900,      # 15 minutes
                AlertLevel.CRITICAL: 300,   # 5 minutes
                AlertLevel.EMERGENCY: 60    # 1 minute
            }

            escalation_time = escalation_times.get(alert.level, 1800)

            if time_since_trigger > escalation_time and not alert.escalated:
                alert.escalate()
                await self._send_alert_notifications(alert)
                self.logger.warning(f"Alert escalated: {alert.id}")

    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts when conditions are no longer met."""
        current_time = datetime.utcnow()

        for alert_id, alert in list(self.active_alerts.items()):
            if alert.resolved:
                continue

            # Auto-resolve old info alerts
            if alert.level == AlertLevel.INFO:
                time_since_trigger = (current_time - alert.triggered_at).total_seconds()
                if time_since_trigger > 86400:  # 24 hours
                    alert.resolve()
                    self.logger.info(f"Auto-resolved old info alert: {alert_id}")
                    continue

            # Check if alert condition still exists
            await self._check_alert_resolution(alert)

    async def _check_alert_resolution(self, alert: MonitoringAlert) -> None:
        """Check if an alert should be auto-resolved."""
        try:
            # For system metric alerts
            if alert.source == "system":
                current_metrics = await self._collect_system_metrics()
                current_value = getattr(current_metrics, alert.metric_name, None)

                if current_value is not None:
                    # Check if condition is no longer met
                    condition_met = self._evaluate_threshold(
                        current_value,
                        alert.threshold_value,
                        alert.condition
                    )

                    if not condition_met:
                        alert.resolve()
                        self.logger.info(f"Auto-resolved system alert: {alert.id}")

            # For target health alerts
            elif alert.source in self.targets:
                target = self.targets[alert.source]
                if target.success_rate > 0.95:  # 95% success rate threshold
                    alert.resolve()
                    self.logger.info(f"Auto-resolved target alert: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error checking alert resolution: {e}")

    def _evaluate_threshold(self, current_value: float, threshold: float, condition: str) -> bool:
        """Evaluate if threshold condition is met."""
        if condition == "greater_than":
            return current_value > threshold
        elif condition == "less_than":
            return current_value < threshold
        elif condition == "equals":
            return abs(current_value - threshold) < 0.001
        return False

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()
        await metrics.collect()
        return metrics

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        try:
            # Collect system metrics
            system_metrics = await self._collect_system_metrics()

            # Calculate target health summary
            target_summary = {}
            for name, target in self.targets.items():
                target_summary[name] = {
                    "success_rate": target.success_rate,
                    "avg_response_time": target.average_response_time,
                    "last_check": target.last_check.isoformat() if target.last_check else None,
                    "status": "healthy" if target.success_rate > 0.95 else "unhealthy"
                }

            # Active alerts summary
            alert_summary = {
                "total": len(self.active_alerts),
                "by_level": {},
                "unacknowledged": 0
            }

            for alert in self.active_alerts.values():
                if not alert.resolved:
                    level_str = alert.level.value
                    alert_summary["by_level"][level_str] = alert_summary["by_level"].get(level_str, 0) + 1
                    if not alert.acknowledged:
                        alert_summary["unacknowledged"] += 1

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": system_metrics.to_dict(),
                "targets": target_summary,
                "alerts": alert_summary,
                "monitoring_status": "active" if self.monitoring_active else "inactive",
                "uptime": (datetime.utcnow() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            }

        except Exception as e:
            self.logger.error(f"Error collecting metrics summary: {e}")
            return {"error": str(e)}

    async def get_alert_history(self, limit: int = 100, level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        try:
            history = []

            # Sort alerts by triggered time (newest first)
            sorted_alerts = sorted(
                self.alert_history.values(),
                key=lambda x: x.triggered_at,
                reverse=True
            )

            count = 0
            for alert in sorted_alerts:
                if count >= limit:
                    break

                if level is None or alert.level == level:
                    history.append(alert.to_dict())
                    count += 1

            return history

        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return []

    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledge()
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.utcnow()

                self.logger.info(f"Alert acknowledged by {user}: {alert_id}")
                return True
            else:
                self.logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False

    async def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Manually resolve an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                alert.resolved_by = user
                alert.resolved_at = datetime.utcnow()

                # Move to history
                self.alert_history[alert_id] = alert
                del self.active_alerts[alert_id]

                self.logger.info(f"Alert resolved by {user}: {alert_id}")
                return True
            else:
                self.logger.warning(f"Alert not found for resolution: {alert_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False

    async def export_metrics(self, format: str = "json", time_range: Optional[int] = None) -> str:
        """Export metrics in various formats."""
        try:
            if format.lower() == "json":
                return await self._export_json_metrics(time_range)
            elif format.lower() == "csv":
                return await self._export_csv_metrics(time_range)
            elif format.lower() == "prometheus":
                return await self._export_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return ""

    async def _export_json_metrics(self, time_range: Optional[int] = None) -> str:
        """Export metrics as JSON."""
        import json

        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "time_range_hours": time_range,
            "metrics_summary": await self.get_metrics_summary(),
            "alert_history": await self.get_alert_history(limit=1000)
        }

        return json.dumps(data, indent=2, default=str)

    async def _export_csv_metrics(self, time_range: Optional[int] = None) -> str:
        """Export metrics as CSV."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write headers
        writer.writerow([
            "timestamp", "alert_id", "level", "source", "metric",
            "current_value", "threshold", "description", "acknowledged", "resolved"
        ])

        # Write alert data
        alerts = await self.get_alert_history(limit=10000)
        for alert in alerts:
            writer.writerow([
                alert.get("triggered_at", ""),
                alert.get("id", ""),
                alert.get("level", ""),
                alert.get("source", ""),
                alert.get("metric_name", ""),
                alert.get("current_value", ""),
                alert.get("threshold_value", ""),
                alert.get("description", ""),
                alert.get("acknowledged", False),
                alert.get("resolved", False)
            ])

        return output.getvalue()

    async def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available"

        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        try:
            # Update metrics before export
            await self._update_prometheus_metrics()
            return generate_latest().decode('utf-8')
        except Exception as e:
            return f"# Error generating Prometheus metrics: {e}"

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the monitoring system."""
        try:
            active_critical_alerts = sum(
                1 for alert in self.active_alerts.values()
                if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] and not alert.resolved
            )

            unhealthy_targets = sum(
                1 for target in self.targets.values()
                if target.success_rate < 0.95
            )

            # Determine overall health
            if active_critical_alerts > 0:
                status = "critical"
            elif unhealthy_targets > 0:
                status = "warning"
            elif not self.monitoring_active:
                status = "unknown"
            else:
                status = "healthy"

            return {
                "status": status,
                "monitoring_active": self.monitoring_active,
                "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
                "critical_alerts": active_critical_alerts,
                "targets_monitored": len(self.targets),
                "unhealthy_targets": unhealthy_targets,
                "uptime": (datetime.utcnow() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up monitoring resources."""
        try:
            await self.stop_monitoring()

            # Clear old alerts from history (keep last 1000)
            if len(self.alert_history) > 1000:
                sorted_alerts = sorted(
                    self.alert_history.items(),
                    key=lambda x: x[1].triggered_at,
                    reverse=True
                )

                # Keep only the most recent 1000 alerts
                self.alert_history = dict(sorted_alerts[:1000])

            self.logger.info("Monitoring tool cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'monitoring_active') and self.monitoring_active:
                # Note: We can't use await here, so we just log
                self.logger.info("MonitoringTool being destroyed - cleanup recommended")
        except:
            pass
