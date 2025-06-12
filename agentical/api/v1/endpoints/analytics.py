"""
Analytics & Monitoring API Endpoints for Agentical

SCOPE CLARIFICATION:
This module provides analytics and monitoring for the entire Agentical platform including:
- SYSTEM WORKFLOW analytics (orchestration-level workflows coordinating multiple agents)
- Agent performance metrics (individual agent execution patterns)
- Platform infrastructure monitoring (CPU, memory, disk, network)
- Cross-system integration observability

System Workflow Analytics Features:
- Multi-agent coordination performance metrics
- Business process execution analytics
- Cross-system integration monitoring
- Long-running workflow trend analysis

Agent Analytics Features:
- Individual agent performance tracking
- Agent workflow pattern analysis
- Tool usage effectiveness metrics
- Agent learning and adaptation insights

Platform Monitoring Features:
- Real-time system resource monitoring
- Logfire integration for distributed observability
- Custom analytics queries and reports
- Multi-format data export capabilities

Technical Features:
- RESTful API design with OpenAPI documentation
- Real-time metrics streaming via WebSocket
- Comprehensive error handling and validation
- Performance optimizations for large-scale analytics
- Enterprise-grade health monitoring
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from fastapi import (
    APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks,
    WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import logfire
import psutil
import platform
from collections import defaultdict

from ....core.exceptions import (
    AnalyticsError, MonitoringError, ValidationError
)
from ....core.logging import log_operation
from ....db.database import get_async_db
from ....db.models.workflow import WorkflowExecution, ExecutionStatus
from ....db.models.agent import AgentExecution
from ....db.repositories.analytics import AsyncAnalyticsRepository
from ....workflows.manager import WorkflowManager

# Initialize router
router = APIRouter(prefix="/analytics", tags=["analytics", "monitoring"])

# Pydantic Models
class MetricsTimeRange(BaseModel):
    """Time range for metrics queries."""
    start_time: datetime
    end_time: datetime

    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("end_time must be after start_time")
        return v


class MetricsQuery(BaseModel):
    """Analytics metrics query parameters."""
    metrics: List[str] = Field(..., description="List of metric names to query")
    time_range: MetricsTimeRange
    granularity: str = Field("1h", regex="^(1m|5m|15m|1h|6h|1d)$")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    group_by: Optional[List[str]] = Field(default_factory=list)
    aggregation: str = Field("avg", regex="^(avg|sum|min|max|count|p50|p90|p95|p99)$")


class SystemMetricsResponse(BaseModel):
    """System performance metrics response."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_io: Dict[str, int]
    active_connections: int
    uptime_seconds: float
    load_average: List[float]


class WorkflowMetricsResponse(BaseModel):
    """Workflow execution metrics response."""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_seconds: float
    success_rate_percent: float
    executions_per_hour: List[Dict[str, Any]]
    most_used_workflows: List[Dict[str, Any]]
    error_distribution: Dict[str, int]
    performance_trends: List[Dict[str, Any]]


class AgentMetricsResponse(BaseModel):
    """Agent execution metrics response."""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_response_time_ms: float
    success_rate_percent: float
    most_active_agents: List[Dict[str, Any]]
    error_types: Dict[str, int]
    performance_by_agent_type: List[Dict[str, Any]]


class LogfireMetricsResponse(BaseModel):
    """Logfire observability metrics response."""
    total_spans: int
    error_spans: int
    average_span_duration_ms: float
    spans_by_service: Dict[str, int]
    error_rate_percent: float
    top_errors: List[Dict[str, Any]]
    performance_insights: List[Dict[str, Any]]
    trace_analysis: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """System health check response."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]
    overall_health_score: float


class AlertRule(BaseModel):
    """Alert rule configuration."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., regex="^(gt|gte|lt|lte|eq|ne)$")
    threshold: float = Field(..., description="Alert threshold value")
    duration_minutes: int = Field(5, ge=1, le=1440, description="Duration before alerting")
    severity: str = Field("medium", regex="^(low|medium|high|critical)$")
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)


class AlertResponse(BaseModel):
    """Alert configuration response."""
    id: str
    rule: AlertRule
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime]
    trigger_count: int


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    layout: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
    widgets: List[Dict[str, Any]] = Field(..., description="Dashboard widgets")
    refresh_interval_seconds: int = Field(30, ge=10, le=3600)
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    """Dashboard response."""
    id: str
    config: DashboardConfig
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]


# WebSocket connection manager for real-time metrics
class MetricsWebSocketManager:
    """Manages WebSocket connections for real-time metrics streaming."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.subscription_configs: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str, config: Dict[str, Any]):
        """Accept a WebSocket connection for metrics streaming."""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        self.subscription_configs[client_id] = config

    def disconnect(self, websocket: WebSocket, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
                if client_id in self.subscription_configs:
                    del self.subscription_configs[client_id]

    async def broadcast_metrics(self, metrics_data: Dict[str, Any]):
        """Broadcast metrics to all connected clients."""
        if not self.active_connections:
            return

        message = json.dumps({
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics_data
        })

        disconnected = []
        for client_id, websockets in self.active_connections.items():
            for websocket in websockets:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append((websocket, client_id))

        # Clean up disconnected websockets
        for websocket, client_id in disconnected:
            self.disconnect(websocket, client_id)


metrics_ws_manager = MetricsWebSocketManager()

# Dependency injection
async def get_analytics_repository(db: AsyncSession = Depends(get_async_db)) -> AsyncAnalyticsRepository:
    """Get analytics repository instance."""
    return AsyncAnalyticsRepository(db)


# API Endpoints
@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get current system performance metrics.

    Returns real-time system metrics including CPU, memory, disk, and network usage.
    """
    with logfire.span("Get system metrics"):
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get memory information
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)

            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            disk_percent = (disk.used / disk.total) * 100

            # Get network I/O
            network_io = psutil.net_io_counters()._asdict()

            # Get active connections count
            try:
                active_connections = len(psutil.net_connections())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                active_connections = 0

            # Get system uptime
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time

            # Get load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows fallback

            metrics = SystemMetricsResponse(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_usage_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_io=network_io,
                active_connections=active_connections,
                uptime_seconds=uptime_seconds,
                load_average=load_average
            )

            logfire.info("System metrics retrieved",
                        cpu_usage=cpu_percent,
                        memory_usage=memory.percent,
                        disk_usage=disk_percent)

            return metrics

        except Exception as e:
            logfire.error("Failed to get system metrics", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system metrics"
            )


@router.get("/workflows/metrics", response_model=WorkflowMetricsResponse)
async def get_workflow_metrics(
    from_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    to_date: Optional[datetime] = Query(None, description="End date for metrics"),
    workflow_ids: Optional[str] = Query(None, description="Comma-separated workflow IDs"),
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Get workflow execution metrics and analytics.

    Returns comprehensive workflow performance metrics including execution counts,
    success rates, duration analysis, and trend data.
    """
    with logfire.span("Get workflow metrics"):
        try:
            # Set default time range if not provided
            if not to_date:
                to_date = datetime.utcnow()
            if not from_date:
                from_date = to_date - timedelta(days=7)

            # Parse workflow IDs filter
            workflow_filter = None
            if workflow_ids:
                workflow_filter = [wid.strip() for wid in workflow_ids.split(',')]

            # Get workflow metrics
            metrics = await analytics_repo.get_workflow_metrics(
                from_date=from_date,
                to_date=to_date,
                workflow_ids=workflow_filter
            )

            logfire.info("Workflow metrics retrieved",
                        from_date=from_date,
                        to_date=to_date,
                        workflow_count=len(workflow_filter) if workflow_filter else None)

            return WorkflowMetricsResponse(**metrics)

        except Exception as e:
            logfire.error("Failed to get workflow metrics", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve workflow metrics"
            )


@router.get("/agents/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    from_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    to_date: Optional[datetime] = Query(None, description="End date for metrics"),
    agent_types: Optional[str] = Query(None, description="Comma-separated agent types"),
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Get agent execution metrics and analytics.

    Returns comprehensive agent performance metrics including execution counts,
    response times, success rates, and performance by agent type.
    """
    with logfire.span("Get agent metrics"):
        try:
            # Set default time range if not provided
            if not to_date:
                to_date = datetime.utcnow()
            if not from_date:
                from_date = to_date - timedelta(days=7)

            # Parse agent types filter
            agent_type_filter = None
            if agent_types:
                agent_type_filter = [atype.strip() for atype in agent_types.split(',')]

            # Get agent metrics
            metrics = await analytics_repo.get_agent_metrics(
                from_date=from_date,
                to_date=to_date,
                agent_types=agent_type_filter
            )

            logfire.info("Agent metrics retrieved",
                        from_date=from_date,
                        to_date=to_date,
                        agent_types=agent_type_filter)

            return AgentMetricsResponse(**metrics)

        except Exception as e:
            logfire.error("Failed to get agent metrics", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve agent metrics"
            )


@router.get("/logfire/metrics", response_model=LogfireMetricsResponse)
async def get_logfire_metrics(
    from_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    to_date: Optional[datetime] = Query(None, description="End date for metrics"),
    service_names: Optional[str] = Query(None, description="Comma-separated service names"),
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Get Logfire observability metrics.

    Returns comprehensive observability metrics from Logfire including span counts,
    error rates, performance insights, and trace analysis.
    """
    with logfire.span("Get Logfire metrics"):
        try:
            # Set default time range if not provided
            if not to_date:
                to_date = datetime.utcnow()
            if not from_date:
                from_date = to_date - timedelta(hours=24)

            # Parse service names filter
            service_filter = None
            if service_names:
                service_filter = [sname.strip() for sname in service_names.split(',')]

            # Get Logfire metrics
            metrics = await analytics_repo.get_logfire_metrics(
                from_date=from_date,
                to_date=to_date,
                service_names=service_filter
            )

            logfire.info("Logfire metrics retrieved",
                        from_date=from_date,
                        to_date=to_date,
                        services=service_filter)

            return LogfireMetricsResponse(**metrics)

        except Exception as e:
            logfire.error("Failed to get Logfire metrics", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve Logfire metrics"
            )


@router.post("/query")
async def query_custom_metrics(
    query: MetricsQuery,
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Execute custom analytics queries.

    Allows flexible querying of metrics data with custom time ranges,
    aggregations, and filtering options.
    """
    with logfire.span("Query custom metrics", metrics=query.metrics):
        try:
            # Validate metrics exist
            available_metrics = await analytics_repo.get_available_metrics()
            invalid_metrics = [m for m in query.metrics if m not in available_metrics]
            if invalid_metrics:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid metrics: {', '.join(invalid_metrics)}"
                )

            # Execute query
            results = await analytics_repo.query_metrics(
                metrics=query.metrics,
                time_range=(query.time_range.start_time, query.time_range.end_time),
                granularity=query.granularity,
                filters=query.filters,
                group_by=query.group_by,
                aggregation=query.aggregation
            )

            logfire.info("Custom metrics query executed",
                        metrics_count=len(query.metrics),
                        time_range_hours=(query.time_range.end_time - query.time_range.start_time).total_seconds() / 3600,
                        result_count=len(results))

            return {
                "query": query.dict(),
                "results": results,
                "timestamp": datetime.utcnow(),
                "result_count": len(results)
            }

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to execute custom metrics query", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to execute custom metrics query"
            )


@router.get("/health", response_model=HealthCheckResponse)
async def comprehensive_health_check(
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Comprehensive system health check.

    Returns detailed health information for all system components
    including databases, external services, and system resources.
    """
    with logfire.span("Comprehensive health check"):
        try:
            health_checks = {}
            overall_score = 0.0
            total_checks = 0

            # Database health
            try:
                db_start = datetime.utcnow()
                await analytics_repo.health_check()
                db_duration = (datetime.utcnow() - db_start).total_seconds() * 1000

                health_checks["database"] = {
                    "status": "healthy",
                    "response_time_ms": db_duration,
                    "details": "Database connection successful"
                }
                overall_score += 100
            except Exception as e:
                health_checks["database"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Database connection failed"
                }
            total_checks += 1

            # System resources health
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent

            resource_score = 100
            if cpu_usage > 90:
                resource_score -= 30
            elif cpu_usage > 70:
                resource_score -= 15

            if memory_usage > 90:
                resource_score -= 30
            elif memory_usage > 80:
                resource_score -= 15

            if disk_usage > 95:
                resource_score -= 30
            elif disk_usage > 85:
                resource_score -= 15

            health_checks["system_resources"] = {
                "status": "healthy" if resource_score > 70 else "degraded" if resource_score > 40 else "unhealthy",
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage,
                "score": resource_score
            }
            overall_score += resource_score
            total_checks += 1

            # Logfire health
            try:
                logfire.info("Health check ping to Logfire")
                health_checks["logfire"] = {
                    "status": "healthy",
                    "details": "Logfire integration active"
                }
                overall_score += 100
            except Exception as e:
                health_checks["logfire"] = {
                    "status": "degraded",
                    "error": str(e),
                    "details": "Logfire integration issues"
                }
                overall_score += 50
            total_checks += 1

            # Calculate overall health score
            final_score = overall_score / total_checks if total_checks > 0 else 0

            # Determine overall status
            if final_score >= 90:
                overall_status = "healthy"
            elif final_score >= 70:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"

            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time

            response = HealthCheckResponse(
                status=overall_status,
                timestamp=datetime.utcnow(),
                version="1.0.0",  # This should come from app config
                uptime_seconds=uptime_seconds,
                checks=health_checks,
                overall_health_score=final_score
            )

            logfire.info("Health check completed",
                        overall_status=overall_status,
                        health_score=final_score,
                        checks_passed=len([c for c in health_checks.values() if c.get("status") == "healthy"]))

            return response

        except Exception as e:
            logfire.error("Health check failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Health check failed"
            )


@router.websocket("/metrics/stream")
async def stream_metrics(
    websocket: WebSocket,
    interval_seconds: int = Query(30, ge=5, le=300),
    metrics: str = Query("system,workflows,agents", description="Comma-separated metric types")
):
    """
    WebSocket endpoint for real-time metrics streaming.

    Streams live metrics data to connected clients at specified intervals.
    """
    client_id = str(uuid.uuid4())

    try:
        config = {
            "interval_seconds": interval_seconds,
            "metrics": [m.strip() for m in metrics.split(',')]
        }

        await metrics_ws_manager.connect(websocket, client_id, config)

        while True:
            try:
                # Collect requested metrics
                metrics_data = {}

                if "system" in config["metrics"]:
                    system_metrics = await get_system_metrics()
                    metrics_data["system"] = system_metrics.dict()

                # Send metrics to client
                await websocket.send_text(json.dumps({
                    "type": "metrics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": metrics_data
                }))

                # Wait for next interval
                await asyncio.sleep(config["interval_seconds"])

            except WebSocketDisconnect:
                break
            except Exception as e:
                logfire.error("Error in metrics streaming", client_id=client_id, error=str(e))
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Metrics streaming error",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                break

    except Exception as e:
        logfire.error("WebSocket connection error", error=str(e))
    finally:
        metrics_ws_manager.disconnect(websocket, client_id)


@router.get("/export/metrics")
async def export_metrics(
    format: str = Query("json", regex="^(json|csv|prometheus)$"),
    from_date: Optional[datetime] = Query(None, description="Start date for export"),
    to_date: Optional[datetime] = Query(None, description="End date for export"),
    metric_types: str = Query("all", description="Comma-separated metric types to export"),
    analytics_repo: AsyncAnalyticsRepository = Depends(get_analytics_repository)
):
    """
    Export metrics data in various formats.

    Supports exporting metrics data in JSON, CSV, or Prometheus format
    for external analysis or integration with other monitoring systems.
    """
    with logfire.span("Export metrics", format=format):
        try:
            # Set default time range if not provided
            if not to_date:
                to_date = datetime.utcnow()
            if not from_date:
                from_date = to_date - timedelta(days=1)

            # Parse metric types
            if metric_types == "all":
                types_to_export = ["system", "workflows", "agents", "logfire"]
            else:
                types_to_export = [t.strip() for t in metric_types.split(',')]

            # Collect metrics data
            export_data = await analytics_repo.export_metrics(
                from_date=from_date,
                to_date=to_date,
                metric_types=types_to_export,
                format=format
            )

            # Set appropriate content type
            if format == "json":
                media_type = "application/json"
                filename = f"metrics_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}.json"
            elif format == "csv":
                media_type = "text/csv"
                filename = f"metrics_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}.csv"
            else:  # prometheus
                media_type = "text/plain"
                filename = f"metrics_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}.txt"

            logfire.info("Metrics exported",
                        format=format,
                        from_date=from_date,
                        to_date=to_date,
                        metric_types=types_to_export)

            return StreamingResponse(
                iter([export_data]),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        except Exception as e:
            logfire.error("Failed to export metrics", format=format, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to export metrics"
            )
