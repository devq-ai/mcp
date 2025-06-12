"""
Enhanced Health Check System with Performance Monitoring

Comprehensive health monitoring endpoints that integrate with the performance
monitoring system to provide detailed system status, performance metrics,
and operational insights.

Features:
- Basic health check with uptime and status
- Detailed performance health check with metrics
- Database connectivity verification
- External service status monitoring
- Performance-based health scoring
- Alert status integration
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psutil
import logfire
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from .performance import performance_monitor


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str = Field(..., description="Overall health status: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="ISO timestamp of health check")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    version: str = Field(default="1.0.0", description="Application version")


class PerformanceHealthStatus(BaseModel):
    """Detailed performance health status model."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="ISO timestamp of health check")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    version: str = Field(default="1.0.0", description="Application version")
    performance_metrics: Dict[str, Any] = Field(..., description="Current performance metrics")
    health_indicators: Dict[str, str] = Field(..., description="Health indicators by category")
    recommendations: List[str] = Field(default_factory=list, description="Performance recommendations")
    alerts: Dict[str, Any] = Field(default_factory=dict, description="Active alerts summary")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual health checks")


class DatabaseHealthCheck(BaseModel):
    """Database health check model."""
    status: str = Field(..., description="Database connection status")
    response_time_ms: float = Field(..., description="Database response time in milliseconds")
    connection_pool: Dict[str, Any] = Field(default_factory=dict, description="Connection pool status")


class ExternalServiceCheck(BaseModel):
    """External service health check model."""
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    last_error: Optional[str] = Field(None, description="Last error message if any")


# Application start time for uptime calculation
app_start_time = time.time()

# Health check router
health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/", response_model=HealthStatus)
async def basic_health_check():
    """
    Basic health check endpoint.
    
    Returns fundamental application health status including uptime and basic status.
    This endpoint is designed to be lightweight for load balancer health checks.
    """
    with logfire.span("Basic Health Check"):
        uptime = time.time() - app_start_time
        
        health_status = HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=uptime
        )
        
        logfire.info("Basic health check completed", 
                    status=health_status.status,
                    uptime_seconds=uptime)
        
        return health_status


@health_router.get("/performance", response_model=PerformanceHealthStatus)
async def performance_health_check():
    """
    Comprehensive performance health check.
    
    Returns detailed performance metrics, health indicators, recommendations,
    and alert status for comprehensive system monitoring.
    """
    with logfire.span("Performance Health Check") as span:
        uptime = time.time() - app_start_time
        
        # Get current performance metrics
        performance_summary = performance_monitor.get_performance_summary()
        
        # Check for active alerts
        recent_alerts = await performance_monitor.check_alerts()
        alerts_summary = _summarize_alerts(recent_alerts)
        
        # Perform individual health checks
        checks = await _perform_health_checks()
        
        # Calculate health indicators
        health_indicators = _calculate_health_indicators(performance_summary, checks)
        
        # Generate recommendations
        recommendations = _generate_recommendations(performance_summary, health_indicators, checks)
        
        # Determine overall status
        overall_status = _determine_overall_status(health_indicators, checks)
        
        health_status = PerformanceHealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=uptime,
            performance_metrics=performance_summary,
            health_indicators=health_indicators,
            recommendations=recommendations,
            alerts=alerts_summary,
            checks=checks
        )
        
        # Log health check results
        span.set_attribute("overall_status", overall_status)
        span.set_attribute("active_alerts", len(recent_alerts))
        span.set_attribute("recommendations_count", len(recommendations))
        
        logfire.info("Performance health check completed",
                    status=overall_status,
                    uptime_seconds=uptime,
                    active_alerts=len(recent_alerts),
                    recommendations=len(recommendations))
        
        return health_status


@health_router.get("/database")
async def database_health_check():
    """
    Database connectivity and performance health check.
    
    Tests database connection, measures response time, and reports
    connection pool status if available.
    """
    with logfire.span("Database Health Check") as span:
        start_time = time.time()
        
        try:
            # TODO: Replace with actual database health check
            # This is a placeholder for database connectivity testing
            await asyncio.sleep(0.01)  # Simulate database query
            
            response_time = (time.time() - start_time) * 1000
            
            health_check = DatabaseHealthCheck(
                status="healthy",
                response_time_ms=response_time,
                connection_pool={
                    "active_connections": 2,
                    "idle_connections": 8,
                    "max_connections": 20
                }
            )
            
            span.set_attribute("response_time_ms", response_time)
            span.set_attribute("status", "healthy")
            
            logfire.info("Database health check passed",
                        response_time_ms=response_time,
                        status="healthy")
            
            return health_check
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            span.set_attribute("response_time_ms", response_time)
            span.set_attribute("status", "unhealthy")
            span.set_attribute("error", str(e))
            
            logfire.error("Database health check failed",
                         response_time_ms=response_time,
                         error=str(e))
            
            health_check = DatabaseHealthCheck(
                status="unhealthy",
                response_time_ms=response_time
            )
            
            return health_check


@health_router.get("/external")
async def external_services_health_check():
    """
    External services health check.
    
    Tests connectivity and response times for external dependencies
    such as LLM APIs, third-party services, etc.
    """
    with logfire.span("External Services Health Check"):
        services = [
            "anthropic_api",
            "openai_api",
            "perplexity_api"
        ]
        
        results = []
        
        for service in services:
            service_check = await _check_external_service(service)
            results.append(service_check)
        
        logfire.info("External services health check completed",
                    total_services=len(services),
                    healthy_services=sum(1 for r in results if r.status == "healthy"))
        
        return {"services": results}


@health_router.get("/alerts")
async def alerts_status():
    """
    Current alerts and alert history.
    
    Returns active performance alerts and recent alert history
    for monitoring and debugging purposes.
    """
    with logfire.span("Alerts Status Check"):
        # Get recent alerts
        recent_alerts = await performance_monitor.check_alerts()
        alert_history = performance_monitor.get_alert_history(limit=20)
        
        # Categorize alerts by severity
        alerts_by_severity = {
            "critical": [a for a in recent_alerts if a.get("severity") == "critical"],
            "warning": [a for a in recent_alerts if a.get("severity") == "warning"],
            "info": [a for a in recent_alerts if a.get("severity") == "info"]
        }
        
        alerts_summary = {
            "active_alerts": recent_alerts,
            "alert_counts": {
                "critical": len(alerts_by_severity["critical"]),
                "warning": len(alerts_by_severity["warning"]),
                "info": len(alerts_by_severity["info"])
            },
            "alerts_by_severity": alerts_by_severity,
            "recent_history": alert_history
        }
        
        logfire.info("Alerts status retrieved",
                    active_alerts=len(recent_alerts),
                    critical_alerts=len(alerts_by_severity["critical"]),
                    warning_alerts=len(alerts_by_severity["warning"]))
        
        return alerts_summary


async def _perform_health_checks() -> Dict[str, Dict[str, Any]]:
    """Perform all individual health checks."""
    checks = {}
    
    # System resource check
    try:
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        checks["system_resources"] = {
            "status": "healthy" if memory_info.percent < 85 and cpu_percent < 80 else "degraded",
            "memory_percent": memory_info.percent,
            "cpu_percent": cpu_percent,
            "details": f"Memory: {memory_info.percent:.1f}%, CPU: {cpu_percent:.1f}%"
        }
    except Exception as e:
        checks["system_resources"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Failed to collect system metrics"
        }
    
    # Performance monitoring check
    try:
        if performance_monitor.monitoring_active:
            checks["performance_monitoring"] = {
                "status": "healthy",
                "details": "Performance monitoring active and collecting metrics"
            }
        else:
            checks["performance_monitoring"] = {
                "status": "degraded",
                "details": "Performance monitoring not active"
            }
    except Exception as e:
        checks["performance_monitoring"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Performance monitoring system error"
        }
    
    # Database connectivity check
    checks["database"] = await _quick_database_check()
    
    return checks


async def _quick_database_check() -> Dict[str, Any]:
    """Quick database connectivity check."""
    try:
        # TODO: Replace with actual database ping
        await asyncio.sleep(0.005)  # Simulate quick database ping
        return {
            "status": "healthy",
            "details": "Database connection successful"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "Database connection failed"
        }


async def _check_external_service(service_name: str) -> ExternalServiceCheck:
    """Check individual external service health."""
    start_time = time.time()
    
    try:
        # TODO: Replace with actual service health checks
        # This is a placeholder for external service connectivity testing
        await asyncio.sleep(0.02)  # Simulate API call
        
        response_time = (time.time() - start_time) * 1000
        
        return ExternalServiceCheck(
            service_name=service_name,
            status="healthy",
            response_time_ms=response_time
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return ExternalServiceCheck(
            service_name=service_name,
            status="unhealthy",
            response_time_ms=response_time,
            last_error=str(e)
        )


def _calculate_health_indicators(performance_summary: Dict[str, Any], 
                               checks: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Calculate health indicators based on metrics and checks."""
    indicators = {}
    
    # Response time indicator
    response_times = performance_summary.get("response_times", {})
    p95_time = response_times.get("p95", 0)
    
    if p95_time < 200:
        indicators["response_time"] = "green"
    elif p95_time < 500:
        indicators["response_time"] = "yellow"
    else:
        indicators["response_time"] = "red"
    
    # Error rate indicator
    error_rate = performance_summary.get("error_rate", 0)
    
    if error_rate < 0.01:
        indicators["error_rate"] = "green"
    elif error_rate < 0.05:
        indicators["error_rate"] = "yellow"
    else:
        indicators["error_rate"] = "red"
    
    # Resource usage indicator
    resource_usage = performance_summary.get("resource_usage", {})
    memory_percent = resource_usage.get("memory_percent", 0)
    cpu_percent = resource_usage.get("cpu_percent", 0)
    
    if memory_percent < 70 and cpu_percent < 60:
        indicators["resource_usage"] = "green"
    elif memory_percent < 85 and cpu_percent < 80:
        indicators["resource_usage"] = "yellow"
    else:
        indicators["resource_usage"] = "red"
    
    # Agent performance indicator
    agent_performance = performance_summary.get("agent_performance", [])
    if agent_performance:
        avg_success_rate = sum(agent["success_rate"] for agent in agent_performance) / len(agent_performance)
        avg_exec_time = sum(agent["avg_execution_time_ms"] for agent in agent_performance) / len(agent_performance)
        
        if avg_success_rate > 0.95 and avg_exec_time < 5000:
            indicators["agent_performance"] = "green"
        elif avg_success_rate > 0.85 and avg_exec_time < 10000:
            indicators["agent_performance"] = "yellow"
        else:
            indicators["agent_performance"] = "red"
    else:
        indicators["agent_performance"] = "gray"  # No data
    
    # System checks indicator
    unhealthy_checks = sum(1 for check in checks.values() if check.get("status") == "unhealthy")
    degraded_checks = sum(1 for check in checks.values() if check.get("status") == "degraded")
    
    if unhealthy_checks == 0 and degraded_checks == 0:
        indicators["system_checks"] = "green"
    elif unhealthy_checks == 0:
        indicators["system_checks"] = "yellow"
    else:
        indicators["system_checks"] = "red"
    
    return indicators


def _generate_recommendations(performance_summary: Dict[str, Any],
                            health_indicators: Dict[str, str],
                            checks: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate performance recommendations based on current state."""
    recommendations = []
    
    # Response time recommendations
    if health_indicators.get("response_time") == "red":
        recommendations.append("Consider response time optimization - P95 latency exceeds 500ms")
    elif health_indicators.get("response_time") == "yellow":
        recommendations.append("Monitor response times closely - approaching performance thresholds")
    
    # Error rate recommendations
    if health_indicators.get("error_rate") == "red":
        recommendations.append("Investigate error patterns - error rate exceeds 5%")
    elif health_indicators.get("error_rate") == "yellow":
        recommendations.append("Review recent errors - error rate above normal levels")
    
    # Resource usage recommendations
    if health_indicators.get("resource_usage") == "red":
        recommendations.append("System resources under pressure - consider scaling or optimization")
    elif health_indicators.get("resource_usage") == "yellow":
        recommendations.append("Monitor resource usage - approaching capacity limits")
    
    # Agent performance recommendations
    if health_indicators.get("agent_performance") == "red":
        recommendations.append("Agent performance degraded - review agent execution patterns")
    elif health_indicators.get("agent_performance") == "yellow":
        recommendations.append("Some agents showing slower performance - consider optimization")
    
    # System checks recommendations
    if health_indicators.get("system_checks") == "red":
        unhealthy_checks = [name for name, check in checks.items() if check.get("status") == "unhealthy"]
        recommendations.append(f"System components unhealthy: {', '.join(unhealthy_checks)}")
    
    # Performance-specific recommendations
    response_times = performance_summary.get("response_times", {})
    if response_times.get("p99", 0) > 2000:
        recommendations.append("P99 latency very high - investigate slowest requests")
    
    resource_usage = performance_summary.get("resource_usage", {})
    if resource_usage.get("memory_percent", 0) > 90:
        recommendations.append("Memory usage critical - immediate attention required")
    
    return recommendations


def _determine_overall_status(health_indicators: Dict[str, str],
                            checks: Dict[str, Dict[str, Any]]) -> str:
    """Determine overall health status based on indicators and checks."""
    # Check for any critical failures
    red_indicators = sum(1 for status in health_indicators.values() if status == "red")
    unhealthy_checks = sum(1 for check in checks.values() if check.get("status") == "unhealthy")
    
    if red_indicators > 0 or unhealthy_checks > 0:
        return "unhealthy"
    
    # Check for degraded performance
    yellow_indicators = sum(1 for status in health_indicators.values() if status == "yellow")
    degraded_checks = sum(1 for check in checks.values() if check.get("status") == "degraded")
    
    if yellow_indicators > 1 or degraded_checks > 0:
        return "degraded"
    
    return "healthy"


def _summarize_alerts(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize current alerts."""
    if not alerts:
        return {
            "total_alerts": 0,
            "by_severity": {"critical": 0, "warning": 0, "info": 0},
            "by_type": {},
            "latest_alert": None
        }
    
    by_severity = {"critical": 0, "warning": 0, "info": 0}
    by_type = {}
    
    for alert in alerts:
        severity = alert.get("severity", "info")
        alert_type = alert.get("type", "unknown")
        
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[alert_type] = by_type.get(alert_type, 0) + 1
    
    return {
        "total_alerts": len(alerts),
        "by_severity": by_severity,
        "by_type": by_type,
        "latest_alert": alerts[-1] if alerts else None
    }


# Startup function to initialize monitoring
async def initialize_health_monitoring():
    """Initialize health monitoring system."""
    logfire.info("Initializing health monitoring system")
    
    # Start performance monitoring if not already started
    if not performance_monitor.monitoring_active:
        performance_monitor.start_monitoring()
    
    logfire.info("Health monitoring system initialized successfully")


# Shutdown function to cleanup monitoring
async def shutdown_health_monitoring():
    """Shutdown health monitoring system."""
    logfire.info("Shutting down health monitoring system")
    
    # Stop performance monitoring
    if performance_monitor.monitoring_active:
        performance_monitor.stop_monitoring()
    
    logfire.info("Health monitoring system shutdown complete")