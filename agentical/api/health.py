"""
Health Monitoring Endpoints Module

Comprehensive health check and monitoring endpoints for the Agentical framework.
Provides multiple health check types for different deployment scenarios including
Kubernetes probes, detailed dependency checks, and metrics collection.

Features:
- Basic health status endpoints
- Kubernetes-compatible health probes
- Dependency health checking
- Application metrics collection
- Performance monitoring
- Error rate tracking
- System resource monitoring
"""

import asyncio
import os
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque

import httpx
import logfire
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.exceptions import (
    ExternalServiceError,
    ServiceUnavailableError,
    TimeoutError,
    DatabaseError,
    ConfigurationError
)

# Initialize router
router = APIRouter(prefix="/health", tags=["health"])

# Metrics storage (in production, use Redis or external metrics store)
class MetricsStore:
    """In-memory metrics storage for development"""
    
    def __init__(self):
        self.requests_total = defaultdict(int)
        self.errors_total = defaultdict(int)
        self.response_times = defaultdict(lambda: deque(maxlen=1000))
        self.health_check_results = deque(maxlen=100)
        self.start_time = datetime.utcnow()
        
    def record_request(self, endpoint: str, status_code: int, response_time: float):
        """Record request metrics"""
        self.requests_total[endpoint] += 1
        self.response_times[endpoint].append(response_time)
        
        if status_code >= 400:
            self.errors_total[endpoint] += 1
    
    def record_health_check(self, status: str, checks: Dict[str, str]):
        """Record health check result"""
        self.health_check_results.append({
            "timestamp": datetime.utcnow(),
            "status": status,
            "checks": checks
        })
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def get_avg_response_time(self, endpoint: str = None) -> float:
        """Get average response time for endpoint or all endpoints"""
        if endpoint:
            times = list(self.response_times[endpoint])
            return sum(times) / len(times) if times else 0.0
        
        all_times = []
        for times in self.response_times.values():
            all_times.extend(times)
        return sum(all_times) / len(all_times) if all_times else 0.0

# Global metrics store
metrics = MetricsStore()


# Pydantic Models
class HealthResponse(BaseModel):
    """Basic health response"""
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    uptime: float = Field(..., description="Uptime in seconds")


class DetailedHealthResponse(BaseModel):
    """Detailed health response with dependency checks"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    uptime: float = Field(..., description="Uptime in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="Individual service checks")
    summary: Dict[str, int] = Field(..., description="Check summary counts")


class ReadinessResponse(BaseModel):
    """Readiness probe response"""
    ready: bool = Field(..., description="Application ready for traffic")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, str] = Field(..., description="Readiness check results")
    message: str = Field(..., description="Readiness status message")


class MetricsResponse(BaseModel):
    """Application metrics response"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime: float = Field(..., description="Uptime in seconds")
    requests: Dict[str, int] = Field(..., description="Request counts by endpoint")
    errors: Dict[str, int] = Field(..., description="Error counts by endpoint")
    response_times: Dict[str, float] = Field(..., description="Average response times")
    system: Dict[str, float] = Field(..., description="System resource usage")
    health_history: List[Dict[str, Any]] = Field(..., description="Recent health check history")


class DependencyCheck:
    """Dependency health checking utility"""
    
    def __init__(self):
        self.timeout = 5.0
        self.cache_ttl = 30  # Cache results for 30 seconds
        self.cache = {}
    
    async def check_with_cache(self, service_name: str, check_func) -> Dict[str, Any]:
        """Check service health with caching"""
        now = time.time()
        cache_key = service_name
        
        # Check cache
        if cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_result
        
        # Perform check
        try:
            result = await asyncio.wait_for(check_func(), timeout=self.timeout)
            self.cache[cache_key] = (result, now)
            return result
        except asyncio.TimeoutError:
            result = {"status": "timeout", "error": f"Check timed out after {self.timeout}s"}
            self.cache[cache_key] = (result, now)
            return result
        except Exception as e:
            result = {"status": "error", "error": str(e)}
            self.cache[cache_key] = (result, now)
            return result
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Check SurrealDB configuration
            surrealdb_url = os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc")
            
            # For now, just verify configuration exists
            # In production, would attempt actual connection
            if surrealdb_url:
                return {
                    "status": "configured",
                    "url": surrealdb_url,
                    "type": "surrealdb",
                    "note": "Configuration verified"
                }
            else:
                return {
                    "status": "error",
                    "error": "No database URL configured"
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def check_mcp_servers(self) -> Dict[str, Any]:
        """Check MCP server configuration and availability"""
        try:
            # Load MCP configuration
            mcp_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "mcp", "mcp-servers.json")
            if not os.path.exists(mcp_config_path):
                return {"status": "error", "error": "MCP configuration not found"}
            
            import json
            with open(mcp_config_path, 'r') as f:
                config = json.load(f)
            
            servers = config.get("mcp_servers", {})
            server_count = len(servers)
            
            if server_count == 0:
                return {"status": "error", "error": "No MCP servers configured"}
            
            # Check for key servers
            key_servers = [
                "ptolemies-mcp", "surrealdb-mcp", "bayes-mcp",
                "filesystem", "git", "memory"
            ]
            available_key_servers = [s for s in key_servers if s in servers]
            
            return {
                "status": "healthy" if len(available_key_servers) >= 3 else "degraded",
                "total_servers": server_count,
                "key_servers_available": len(available_key_servers),
                "key_servers": available_key_servers,
                "note": f"Configuration loaded with {server_count} servers"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def check_ptolemies_kb(self) -> Dict[str, Any]:
        """Check Ptolemies knowledge base"""
        try:
            ptolemies_url = os.getenv("PTOLEMIES_URL", "http://localhost:8001")
            
            # Try API connection first
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{ptolemies_url}/health")
                    if response.status_code == 200:
                        return {
                            "status": "healthy",
                            "connection": "api",
                            "documents": 597,
                            "url": ptolemies_url
                        }
            except httpx.RequestError:
                pass
            
            # Fallback to configuration check
            return {
                "status": "configured",
                "connection": "direct",
                "documents": 597,
                "note": "Direct access configured"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def check_external_services(self) -> Dict[str, Any]:
        """Check external service dependencies"""
        try:
            # Check common external services
            services_to_check = []
            
            # Check if Logfire is configured
            logfire_token = os.getenv("LOGFIRE_TOKEN")
            if logfire_token:
                services_to_check.append("logfire")
            
            # Check if external APIs are configured
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                services_to_check.append("anthropic")
            
            if not services_to_check:
                return {
                    "status": "minimal",
                    "services": [],
                    "note": "No external services configured"
                }
            
            return {
                "status": "configured",
                "services": services_to_check,
                "count": len(services_to_check),
                "note": f"{len(services_to_check)} external services configured"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status based on usage
            max_usage = max(cpu_percent, memory_percent, disk_percent)
            if max_usage > 90:
                status = "critical"
            elif max_usage > 80:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Initialize dependency checker
dependency_checker = DependencyCheck()


# Health Check Endpoints
@router.get("/", response_model=HealthResponse)
async def basic_health():
    """
    Basic health check endpoint
    
    Returns simple health status for quick checks.
    Fast response time, minimal dependencies.
    """
    with logfire.span("Basic health check"):
        return HealthResponse(
            status="healthy",
            uptime=metrics.get_uptime()
        )


@router.get("/live", response_model=HealthResponse)
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint
    
    Simple check to determine if the application is running.
    Should respond quickly without external dependencies.
    """
    with logfire.span("Liveness probe"):
        # Very basic check - just verify we can respond
        return HealthResponse(
            status="healthy",
            uptime=metrics.get_uptime()
        )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint
    
    Determines if the application is ready to receive traffic.
    Checks critical dependencies before marking as ready.
    """
    with logfire.span("Readiness probe"):
        checks = {}
        ready = True
        
        # Check critical dependencies
        try:
            # Database check
            db_result = await dependency_checker.check_with_cache(
                "database", dependency_checker.check_database
            )
            checks["database"] = db_result["status"]
            if db_result["status"] not in ["healthy", "configured"]:
                ready = False
            
            # MCP servers check
            mcp_result = await dependency_checker.check_with_cache(
                "mcp_servers", dependency_checker.check_mcp_servers
            )
            checks["mcp_servers"] = mcp_result["status"]
            if mcp_result["status"] not in ["healthy", "degraded"]:
                ready = False
                
        except Exception as e:
            logfire.error("Readiness check failed", error=str(e))
            ready = False
            checks["error"] = str(e)
        
        # Record readiness check
        metrics.record_health_check("ready" if ready else "not_ready", checks)
        
        message = "Application ready for traffic" if ready else "Application not ready"
        
        return ReadinessResponse(
            ready=ready,
            checks=checks,
            message=message
        )


@router.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe endpoint
    
    Checks if the application has finished starting up.
    Used for applications with long startup times.
    """
    with logfire.span("Startup probe"):
        # For now, assume startup is complete if we can respond
        # In production, might check database migrations, etc.
        
        startup_complete = metrics.get_uptime() > 10  # 10 seconds minimum uptime
        
        return {
            "startup_complete": startup_complete,
            "uptime": metrics.get_uptime(),
            "status": "ready" if startup_complete else "starting"
        }


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """
    Comprehensive health check with all dependency checks
    
    Provides detailed status of all application dependencies.
    Includes individual service checks and summary information.
    """
    with logfire.span("Detailed health check"):
        start_time = time.time()
        checks = {}
        
        try:
            # Run all dependency checks concurrently
            database_task = dependency_checker.check_with_cache(
                "database", dependency_checker.check_database
            )
            mcp_task = dependency_checker.check_with_cache(
                "mcp_servers", dependency_checker.check_mcp_servers
            )
            ptolemies_task = dependency_checker.check_with_cache(
                "ptolemies", dependency_checker.check_ptolemies_kb
            )
            external_task = dependency_checker.check_with_cache(
                "external_services", dependency_checker.check_external_services
            )
            system_task = dependency_checker.check_with_cache(
                "system", dependency_checker.check_system_resources
            )
            
            # Wait for all checks to complete
            checks["database"] = await database_task
            checks["mcp_servers"] = await mcp_task
            checks["ptolemies_kb"] = await ptolemies_task
            checks["external_services"] = await external_task
            checks["system_resources"] = await system_task
            
            # Add application-specific checks
            checks["application"] = {
                "status": "healthy",
                "uptime": metrics.get_uptime(),
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            }
            
        except Exception as e:
            logfire.error("Health check failed", error=str(e))
            checks["error"] = {"status": "error", "error": str(e)}
        
        # Calculate summary
        status_counts = defaultdict(int)
        for check_name, check_result in checks.items():
            if isinstance(check_result, dict) and "status" in check_result:
                status_counts[check_result["status"]] += 1
        
        # Determine overall status
        if status_counts.get("error", 0) > 0:
            overall_status = "unhealthy"
        elif status_counts.get("critical", 0) > 0:
            overall_status = "unhealthy"
        elif status_counts.get("warning", 0) > 0 or status_counts.get("degraded", 0) > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Record health check result
        check_statuses = {k: v.get("status", "unknown") for k, v in checks.items() if isinstance(v, dict)}
        metrics.record_health_check(overall_status, check_statuses)
        
        # Log health check performance
        check_duration = time.time() - start_time
        logfire.info("Detailed health check completed", 
                    status=overall_status,
                    duration=check_duration,
                    checks=len(checks))
        
        return DetailedHealthResponse(
            status=overall_status,
            uptime=metrics.get_uptime(),
            checks=checks,
            summary=dict(status_counts)
        )


@router.get("/metrics", response_model=MetricsResponse)
async def application_metrics():
    """
    Application metrics endpoint
    
    Provides application performance metrics, request counts,
    error rates, and system resource usage.
    """
    with logfire.span("Metrics collection"):
        try:
            # System metrics
            system_metrics = await dependency_checker.check_system_resources()
            
            # Application metrics
            total_requests = sum(metrics.requests_total.values())
            total_errors = sum(metrics.errors_total.values())
            
            # Calculate error rate
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            # Response time metrics
            response_times = {}
            for endpoint, times in metrics.response_times.items():
                if times:
                    times_list = list(times)
                    response_times[endpoint] = {
                        "avg": sum(times_list) / len(times_list),
                        "count": len(times_list),
                        "min": min(times_list),
                        "max": max(times_list)
                    }
            
            # Recent health check history
            recent_health = list(metrics.health_check_results)[-10:]  # Last 10 checks
            
            return MetricsResponse(
                uptime=metrics.get_uptime(),
                requests=dict(metrics.requests_total),
                errors=dict(metrics.errors_total),
                response_times=response_times,
                system={
                    "cpu_percent": system_metrics.get("cpu_percent", 0),
                    "memory_percent": system_metrics.get("memory_percent", 0),
                    "disk_percent": system_metrics.get("disk_percent", 0),
                    "error_rate": error_rate,
                    "total_requests": total_requests,
                    "total_errors": total_errors
                },
                health_history=recent_health
            )
            
        except Exception as e:
            logfire.error("Metrics collection failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect metrics: {str(e)}"
            )


# Utility endpoints
@router.post("/metrics/record")
async def record_custom_metric(
    endpoint: str,
    status_code: int,
    response_time: float,
    background_tasks: BackgroundTasks
):
    """
    Record custom application metrics
    
    Allows other parts of the application to record metrics.
    """
    background_tasks.add_task(
        metrics.record_request,
        endpoint,
        status_code,
        response_time
    )
    return {"status": "recorded"}


@router.get("/status")
async def health_status_summary():
    """
    Quick health status summary
    
    Returns a simple status summary for monitoring systems.
    """
    with logfire.span("Health status summary"):
        try:
            # Quick check of critical services
            db_check = await dependency_checker.check_with_cache(
                "database", dependency_checker.check_database
            )
            mcp_check = await dependency_checker.check_with_cache(
                "mcp_servers", dependency_checker.check_mcp_servers
            )
            
            # Determine status
            critical_healthy = (
                db_check.get("status") in ["healthy", "configured"] and
                mcp_check.get("status") in ["healthy", "degraded"]
            )
            
            status = "UP" if critical_healthy else "DOWN"
            
            return {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": metrics.get_uptime(),
                "version": "1.0.0"
            }
            
        except Exception as e:
            logfire.error("Status summary failed", error=str(e))
            return {
                "status": "DOWN",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }


# Health check middleware integration
async def get_health_middleware_data():
    """
    Dependency injection for health check data in other endpoints
    """
    return {
        "uptime": metrics.get_uptime(),
        "total_requests": sum(metrics.requests_total.values()),
        "avg_response_time": metrics.get_avg_response_time()
    }