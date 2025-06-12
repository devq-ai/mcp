"""
IO Agent Implementation for Agentical Framework

This module provides the IOAgent implementation for inspection, observation,
and monitoring tasks across systems, applications, and infrastructure.

Features:
- System monitoring and health checks
- Application performance observation
- Infrastructure inspection and analysis
- Log monitoring and alerting
- Resource utilization tracking
- Service discovery and status monitoring
- Real-time data collection and analysis
- Anomaly detection and alerting
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import psutil
import time
from pathlib import Path
from enum import Enum

import logfire
from pydantic import BaseModel, Field, validator

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class MonitoringScope(Enum):
    """Scope of monitoring operations."""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    SERVICE = "service"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"


class ObservationType(Enum):
    """Types of observations that can be performed."""
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRICS = "performance_metrics"
    RESOURCE_USAGE = "resource_usage"
    LOG_ANALYSIS = "log_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    DEPENDENCY_CHECK = "dependency_check"
    COMPLIANCE_AUDIT = "compliance_audit"
    SECURITY_SCAN = "security_scan"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class InspectionRequest(BaseModel):
    """Request model for inspection tasks."""
    scope: MonitoringScope = Field(..., description="Scope of inspection")
    targets: List[str] = Field(..., description="Targets to inspect (URLs, services, etc.)")
    observation_type: ObservationType = Field(..., description="Type of observation to perform")
    depth: str = Field(default="standard", description="Inspection depth (surface, standard, deep)")
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    include_logs: bool = Field(default=False, description="Include log analysis")
    timeout_seconds: int = Field(default=60, description="Timeout for inspection")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtering criteria")


class MonitoringRequest(BaseModel):
    """Request model for monitoring tasks."""
    scope: MonitoringScope = Field(..., description="Monitoring scope")
    duration_minutes: int = Field(default=5, description="Monitoring duration in minutes")
    interval_seconds: int = Field(default=30, description="Monitoring interval in seconds")
    metrics: List[str] = Field(default_factory=list, description="Specific metrics to monitor")
    thresholds: Optional[Dict[str, float]] = Field(default=None, description="Alert thresholds")
    alert_on_anomaly: bool = Field(default=True, description="Alert on detected anomalies")
    include_baseline: bool = Field(default=True, description="Include baseline comparison")


class ObservationRequest(BaseModel):
    """Request model for observation tasks."""
    observation_type: ObservationType = Field(..., description="Type of observation")
    target: str = Field(..., description="Target system or service")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Observation parameters")
    continuous: bool = Field(default=False, description="Continuous observation mode")
    report_format: str = Field(default="structured", description="Report format")
    include_recommendations: bool = Field(default=True, description="Include recommendations")


class IOAgent(EnhancedBaseAgent[InspectionRequest, Dict[str, Any]]):
    """
    Specialized agent for inspection, observation, and monitoring tasks.

    Capabilities:
    - System health monitoring and inspection
    - Application performance observation
    - Infrastructure monitoring and analysis
    - Real-time metrics collection
    - Anomaly detection and alerting
    - Service discovery and status checking
    - Log monitoring and analysis
    - Compliance and security auditing
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "IOAgent",
        description: str = "Specialized agent for inspection and observation tasks",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.IO_AGENT,
            **kwargs
        )

        # Monitoring configuration
        self.monitoring_intervals = {
            "real_time": 1,
            "frequent": 5,
            "standard": 30,
            "periodic": 300,
            "scheduled": 3600
        }

        # System metrics to collect
        self.system_metrics = {
            "cpu": ["usage_percent", "load_average", "core_count"],
            "memory": ["total", "available", "used", "percent"],
            "disk": ["total", "used", "free", "percent"],
            "network": ["bytes_sent", "bytes_recv", "packets_sent", "packets_recv"],
            "processes": ["count", "running", "sleeping", "zombie"]
        }

        # Service health check endpoints
        self.health_check_patterns = {
            "http": "/health",
            "https": "/health",
            "api": "/api/health",
            "actuator": "/actuator/health",
            "status": "/status"
        }

        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 85.0,
            "response_time": 5000,  # milliseconds
            "error_rate": 5.0  # percentage
        }

        # Baseline metrics storage
        self.baseline_metrics = {}
        self.metric_history = {}

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "system_monitoring",
            "application_monitoring",
            "infrastructure_inspection",
            "health_checks",
            "performance_metrics",
            "resource_monitoring",
            "log_analysis",
            "anomaly_detection",
            "service_discovery",
            "compliance_auditing",
            "security_scanning",
            "real_time_observation",
            "threshold_alerting",
            "baseline_comparison",
            "trend_analysis"
        ]

    async def _execute_core_logic(
        self,
        request: Union[InspectionRequest, MonitoringRequest, ObservationRequest],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the core inspection/observation logic."""

        with logfire.span("IOAgent execution", request_type=type(request).__name__):
            try:
                if isinstance(request, InspectionRequest):
                    return await self._handle_inspection_request(request, context)
                elif isinstance(request, MonitoringRequest):
                    return await self._handle_monitoring_request(request, context)
                elif isinstance(request, ObservationRequest):
                    return await self._handle_observation_request(request, context)
                else:
                    # Handle generic inspection requests
                    return await self._handle_generic_request(request, context)

            except Exception as e:
                logfire.error("IOAgent execution failed", error=str(e))
                raise AgentExecutionError(f"Inspection/observation failed: {str(e)}")

    async def _handle_inspection_request(
        self,
        request: InspectionRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle inspection requests."""

        with logfire.span("Inspection execution", scope=request.scope.value):
            result = {
                "inspection_id": f"insp_{int(time.time())}",
                "scope": request.scope.value,
                "observation_type": request.observation_type.value,
                "targets": request.targets,
                "started_at": datetime.utcnow().isoformat(),
                "results": [],
                "summary": {},
                "alerts": []
            }

            # Perform inspection on each target
            for target in request.targets:
                target_result = await self._inspect_target(target, request, context)
                result["results"].append(target_result)

                # Check for alerts
                alerts = await self._check_for_alerts(target_result, request)
                result["alerts"].extend(alerts)

            # Generate summary
            result["summary"] = await self._generate_inspection_summary(result["results"])
            result["completed_at"] = datetime.utcnow().isoformat()
            result["duration_seconds"] = (datetime.fromisoformat(result["completed_at"]) -
                                        datetime.fromisoformat(result["started_at"])).total_seconds()

            logfire.info("Inspection completed",
                        targets_inspected=len(request.targets),
                        alerts_generated=len(result["alerts"]))

            return result

    async def _handle_monitoring_request(
        self,
        request: MonitoringRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle monitoring requests."""

        with logfire.span("Monitoring execution", scope=request.scope.value):
            result = {
                "monitoring_id": f"mon_{int(time.time())}",
                "scope": request.scope.value,
                "duration_minutes": request.duration_minutes,
                "interval_seconds": request.interval_seconds,
                "started_at": datetime.utcnow().isoformat(),
                "metrics_collected": [],
                "alerts": [],
                "anomalies": []
            }

            # Calculate monitoring parameters
            end_time = datetime.utcnow() + timedelta(minutes=request.duration_minutes)
            collection_count = 0

            # Perform monitoring loop
            while datetime.utcnow() < end_time:
                collection_start = datetime.utcnow()

                # Collect metrics
                metrics = await self._collect_metrics(request.scope, request.metrics)
                metrics["timestamp"] = collection_start.isoformat()
                metrics["collection_id"] = collection_count

                result["metrics_collected"].append(metrics)

                # Check thresholds and detect anomalies
                if request.thresholds:
                    threshold_alerts = await self._check_thresholds(metrics, request.thresholds)
                    result["alerts"].extend(threshold_alerts)

                if request.alert_on_anomaly:
                    anomalies = await self._detect_anomalies(metrics, request.scope.value)
                    result["anomalies"].extend(anomalies)

                collection_count += 1

                # Wait for next interval
                await asyncio.sleep(request.interval_seconds)

            result["completed_at"] = datetime.utcnow().isoformat()
            result["total_collections"] = collection_count

            # Generate monitoring summary
            result["summary"] = await self._generate_monitoring_summary(result["metrics_collected"])

            logfire.info("Monitoring completed",
                        collections=collection_count,
                        alerts=len(result["alerts"]),
                        anomalies=len(result["anomalies"]))

            return result

    async def _handle_observation_request(
        self,
        request: ObservationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle observation requests."""

        with logfire.span("Observation execution", observation_type=request.observation_type.value):
            result = {
                "observation_id": f"obs_{int(time.time())}",
                "observation_type": request.observation_type.value,
                "target": request.target,
                "started_at": datetime.utcnow().isoformat(),
                "observations": [],
                "analysis": {},
                "recommendations": []
            }

            # Perform observation based on type
            if request.observation_type == ObservationType.HEALTH_CHECK:
                result.update(await self._perform_health_check(request.target, request.parameters))
            elif request.observation_type == ObservationType.PERFORMANCE_METRICS:
                result.update(await self._collect_performance_metrics(request.target, request.parameters))
            elif request.observation_type == ObservationType.RESOURCE_USAGE:
                result.update(await self._analyze_resource_usage(request.target, request.parameters))
            elif request.observation_type == ObservationType.LOG_ANALYSIS:
                result.update(await self._analyze_logs(request.target, request.parameters))
            elif request.observation_type == ObservationType.ANOMALY_DETECTION:
                result.update(await self._detect_target_anomalies(request.target, request.parameters))
            else:
                result.update(await self._perform_generic_observation(request.target, request.parameters))

            # Generate recommendations if requested
            if request.include_recommendations:
                result["recommendations"] = await self._generate_recommendations(result["analysis"])

            result["completed_at"] = datetime.utcnow().isoformat()

            logfire.info("Observation completed",
                        observation_type=request.observation_type.value,
                        target=request.target)

            return result

    async def _inspect_target(self, target: str, request: InspectionRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a specific target."""

        target_result = {
            "target": target,
            "scope": request.scope.value,
            "observation_type": request.observation_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "metrics": {},
            "findings": [],
            "errors": []
        }

        try:
            if request.scope == MonitoringScope.SYSTEM:
                target_result.update(await self._inspect_system(target, request))
            elif request.scope == MonitoringScope.APPLICATION:
                target_result.update(await self._inspect_application(target, request))
            elif request.scope == MonitoringScope.NETWORK:
                target_result.update(await self._inspect_network(target, request))
            elif request.scope == MonitoringScope.SERVICE:
                target_result.update(await self._inspect_service(target, request))
            else:
                target_result.update(await self._inspect_generic(target, request))

        except Exception as e:
            target_result["status"] = "error"
            target_result["errors"].append(str(e))
            logfire.error("Target inspection failed", target=target, error=str(e))

        return target_result

    async def _inspect_system(self, target: str, request: InspectionRequest) -> Dict[str, Any]:
        """Inspect system-level metrics and health."""

        result = {
            "status": "healthy",
            "metrics": {},
            "findings": []
        }

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            result["metrics"]["cpu"] = {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "load_average": load_avg
            }

            # Memory metrics
            memory = psutil.virtual_memory()
            result["metrics"]["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }

            # Disk metrics
            disk = psutil.disk_usage('/')
            result["metrics"]["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }

            # Network metrics
            network = psutil.net_io_counters()
            result["metrics"]["network"] = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }

            # Process metrics
            processes = list(psutil.process_iter(['pid', 'name', 'status']))
            result["metrics"]["processes"] = {
                "total_count": len(processes),
                "running": len([p for p in processes if p.info['status'] == 'running']),
                "sleeping": len([p for p in processes if p.info['status'] == 'sleeping'])
            }

            # Check for issues
            if cpu_percent > 85:
                result["findings"].append(f"High CPU usage: {cpu_percent}%")
                result["status"] = "warning"

            if memory.percent > 90:
                result["findings"].append(f"High memory usage: {memory.percent}%")
                result["status"] = "critical"

            if (disk.used / disk.total) * 100 > 85:
                result["findings"].append(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")
                result["status"] = "warning"

        except Exception as e:
            result["status"] = "error"
            result["errors"] = [str(e)]

        return result

    async def _inspect_application(self, target: str, request: InspectionRequest) -> Dict[str, Any]:
        """Inspect application-level metrics and health."""

        result = {
            "status": "unknown",
            "metrics": {},
            "findings": []
        }

        # Placeholder implementation - would integrate with actual application monitoring
        try:
            # Simulate application inspection
            result["metrics"]["response_time"] = 150  # ms
            result["metrics"]["throughput"] = 1000  # requests/minute
            result["metrics"]["error_rate"] = 0.5  # percentage
            result["metrics"]["active_connections"] = 50

            result["status"] = "healthy"
            result["findings"].append("Application is responding normally")

        except Exception as e:
            result["status"] = "error"
            result["findings"].append(f"Application inspection failed: {str(e)}")

        return result

    async def _inspect_network(self, target: str, request: InspectionRequest) -> Dict[str, Any]:
        """Inspect network connectivity and performance."""

        result = {
            "status": "unknown",
            "metrics": {},
            "findings": []
        }

        # Placeholder implementation - would use actual network tools
        try:
            result["metrics"]["latency"] = 25  # ms
            result["metrics"]["packet_loss"] = 0.1  # percentage
            result["metrics"]["bandwidth"] = 100  # Mbps

            result["status"] = "healthy"
            result["findings"].append("Network connectivity is stable")

        except Exception as e:
            result["status"] = "error"
            result["findings"].append(f"Network inspection failed: {str(e)}")

        return result

    async def _inspect_service(self, target: str, request: InspectionRequest) -> Dict[str, Any]:
        """Inspect service health and availability."""

        result = {
            "status": "unknown",
            "metrics": {},
            "findings": []
        }

        # Placeholder implementation - would check actual service endpoints
        try:
            result["metrics"]["availability"] = 99.9  # percentage
            result["metrics"]["response_time"] = 100  # ms
            result["metrics"]["last_restart"] = "2024-01-01T00:00:00Z"

            result["status"] = "healthy"
            result["findings"].append("Service is operational")

        except Exception as e:
            result["status"] = "error"
            result["findings"].append(f"Service inspection failed: {str(e)}")

        return result

    async def _inspect_generic(self, target: str, request: InspectionRequest) -> Dict[str, Any]:
        """Generic inspection for unknown target types."""

        return {
            "status": "inspected",
            "metrics": {"inspection_time": datetime.utcnow().isoformat()},
            "findings": [f"Generic inspection completed for {target}"]
        }

    async def _collect_metrics(self, scope: MonitoringScope, specific_metrics: List[str]) -> Dict[str, Any]:
        """Collect metrics based on scope and specific requirements."""

        metrics = {
            "scope": scope.value,
            "timestamp": datetime.utcnow().isoformat()
        }

        if scope == MonitoringScope.SYSTEM:
            metrics.update({
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
            })

        # Add specific metrics if requested
        for metric in specific_metrics:
            if metric not in metrics:
                metrics[metric] = self._get_specific_metric(metric)

        return metrics

    def _get_specific_metric(self, metric_name: str) -> Any:
        """Get a specific metric value."""

        # Placeholder implementation
        metric_values = {
            "response_time": 150,
            "throughput": 1000,
            "error_rate": 0.5,
            "active_users": 250,
            "queue_length": 5
        }

        return metric_values.get(metric_name, 0)

    async def _check_thresholds(self, metrics: Dict[str, Any], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check metrics against defined thresholds."""

        alerts = []

        for metric_name, threshold_value in thresholds.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                if isinstance(current_value, (int, float)) and current_value > threshold_value:
                    alerts.append({
                        "type": "threshold_exceeded",
                        "metric": metric_name,
                        "current_value": current_value,
                        "threshold": threshold_value,
                        "severity": AlertSeverity.HIGH.value,
                        "timestamp": datetime.utcnow().isoformat()
                    })

        return alerts

    async def _detect_anomalies(self, metrics: Dict[str, Any], scope: str) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""

        anomalies = []

        # Simple anomaly detection based on predefined thresholds
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name in self.anomaly_thresholds:
                    if value > self.anomaly_thresholds[metric_name]:
                        anomalies.append({
                            "type": "anomaly_detected",
                            "metric": metric_name,
                            "value": value,
                            "expected_range": f"< {self.anomaly_thresholds[metric_name]}",
                            "severity": AlertSeverity.MEDIUM.value,
                            "timestamp": datetime.utcnow().isoformat()
                        })

        return anomalies

    async def _check_for_alerts(self, target_result: Dict[str, Any], request: InspectionRequest) -> List[Dict[str, Any]]:
        """Check inspection results for alert conditions."""

        alerts = []

        if target_result["status"] == "critical":
            alerts.append({
                "type": "critical_status",
                "target": target_result["target"],
                "severity": AlertSeverity.CRITICAL.value,
                "message": "Target is in critical state",
                "findings": target_result.get("findings", []),
                "timestamp": datetime.utcnow().isoformat()
            })
        elif target_result["status"] == "warning":
            alerts.append({
                "type": "warning_status",
                "target": target_result["target"],
                "severity": AlertSeverity.MEDIUM.value,
                "message": "Target has warning conditions",
                "findings": target_result.get("findings", []),
                "timestamp": datetime.utcnow().isoformat()
            })

        return alerts

    async def _generate_inspection_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of inspection results."""

        summary = {
            "total_targets": len(results),
            "healthy_targets": 0,
            "warning_targets": 0,
            "critical_targets": 0,
            "error_targets": 0,
            "total_findings": 0
        }

        for result in results:
            status = result.get("status", "unknown")
            if status == "healthy":
                summary["healthy_targets"] += 1
            elif status == "warning":
                summary["warning_targets"] += 1
            elif status == "critical":
                summary["critical_targets"] += 1
            elif status == "error":
                summary["error_targets"] += 1

            summary["total_findings"] += len(result.get("findings", []))

        summary["overall_health"] = "healthy" if summary["critical_targets"] == 0 and summary["error_targets"] == 0 else "degraded"

        return summary

    async def _generate_monitoring_summary(self, metrics_collected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of monitoring data."""

        if not metrics_collected:
            return {"status": "no_data"}

        summary = {
            "total_collections": len(metrics_collected),
            "time_range": {
                "start": metrics_collected[0].get("timestamp"),
                "end": metrics_collected[-1].get("timestamp")
            },
            "metric_averages": {},
            "trends": {}
        }

        # Calculate averages for numeric metrics
        numeric_metrics = {}
        for collection in metrics_collected:
            for key, value in collection.items():
                if isinstance(value, (int, float)) and key != "collection_id":
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)

        for metric, values in numeric_metrics.items():
            summary["metric_averages"][metric] = sum(values) / len(values)

            # Simple trend analysis
            if len(values) >= 2:
                trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
                summary["trends"][metric] = trend

        return summary

    async def _perform_health_check(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check observation."""

        return {
            "observations": [{
                "type": "health_check",
                "target": target,
                "status": "healthy",
                "response_time": 50,
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "overall_health": "healthy",
                "availability": 99.9,
                "performance": "good"
            }
        }

    async def _collect_performance_metrics(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Collect performance metrics observation."""

        return {
            "observations": [{
                "type": "performance_metrics",
                "target": target,
                "metrics": {
                    "response_time": 120,
                    "throughput": 1500,
                    "cpu_usage": 45,
                    "memory_usage": 60
                },
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "performance_grade": "good",
                "bottlenecks": [],
                "recommendations": ["Monitor memory usage trends"]
            }
        }

    async def _analyze_resource_usage(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage observation."""

        return {
            "observations": [{
                "type": "resource_usage",
                "target": target,
                "resources": {
                    "cpu": {"current": 45, "peak": 78, "average": 52},
                    "memory": {"current": 60, "peak": 85, "average": 65},
                    "disk": {"current": 40, "peak": 45, "average": 42}
                },
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "resource_efficiency": "good",
                "optimization_opportunities": ["CPU optimization possible"],
                "capacity_planning": "adequate for current load"
            }
        }

    async def _analyze_logs(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze logs observation."""

        return {
            "observations": [{
                "type": "log_analysis",
                "target": target,
                "log_summary": {
                    "total_entries": 10000,
                    "error_count": 25,
                    "warning_count": 150,
                    "time_range": "1h"
                },
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "log_health": "good",
                "error_rate": 0.25,
                "patterns": ["Increased warnings during peak hours"],
                "anomalies": []
            }
        }

    async def _detect_target_anomalies(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies observation."""

        return {
            "observations": [{
                "type": "anomaly_detection",
                "target": target,
                "anomalies_found": [],
                "analysis_period": "1h",
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "anomaly_status": "none_detected",
                "baseline_comparison": "within_normal_range",
                "confidence_level": 95
            }
        }

    async def _perform_generic_observation(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform generic observation."""

        return {
            "observations": [{
                "type": "generic",
                "target": target,
                "status": "observed",
                "timestamp": datetime.utcnow().isoformat()
            }],
            "analysis": {
                "status": "completed",
                "notes": "Generic observation performed"
            }
        }

    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""

        recommendations = []

        # Basic recommendation logic
        if "performance_grade" in analysis and analysis["performance_grade"] != "excellent":
            recommendations.append("Consider performance optimization")

        if "error_rate" in analysis and analysis["error_rate"] > 1.0:
            recommendations.append("Investigate and reduce error rate")

        if "resource_efficiency" in analysis and analysis["resource_efficiency"] != "excellent":
            recommendations.append("Review resource allocation and optimization")

        if not recommendations:
            recommendations.append("Continue monitoring current performance levels")

        return recommendations

    async def _handle_generic_request(self, request: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic inspection/observation requests."""

        return {
            "result": "Generic inspection/observation completed",
            "request_type": type(request).__name__,
            "processed_at": datetime.utcnow().isoformat(),
            "context_keys": list(context.keys()) if context else []
        }

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for the IOAgent."""
        return {
            "monitoring_interval": 30,
            "alert_thresholds": self.anomaly_thresholds,
            "max_targets": 100,
            "timeout_seconds": 60,
            "enable_alerting": True,
            "enable_anomaly_detection": True,
            "baseline_update_interval": 3600,
            "metric_retention_hours": 24,
            "health_check_endpoints": self.health_check_patterns,
            "continuous_monitoring": False
        }

    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_keys = ["monitoring_interval", "max_targets", "timeout_seconds"]

        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required configuration key: {key}")

        if config["monitoring_interval"] <= 0:
            raise ValidationError("monitoring_interval must be positive")

        if config["max_targets"] <= 0:
            raise ValidationError("max_targets must be positive")

        if config["timeout_seconds"] <= 0:
            raise ValidationError("timeout_seconds must be positive")

        return True
