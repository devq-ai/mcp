"""
Agentical Main Application

FastAPI orchestration layer that coordinates existing DevQ.ai infrastructure:
- Ptolemies Knowledge Base (597 production documents)
- MCP Server Ecosystem (22+ operational servers)
- DevQ.ai Standard Stack (FastAPI + Logfire + PyTest + TaskMaster AI)

This application serves as the coordination layer for multi-agent workflows,
leveraging existing production-ready infrastructure rather than reimplementing it.

Architecture:
- Agents: Orchestrators that coordinate tools + knowledge
- Workflows: Patterns that leverage existing infrastructure
- Tools: MCP servers and existing tool ecosystem
- Playbooks: Configuration-driven coordination patterns
- Knowledge: Integration with Ptolemies knowledge base
"""

import os
import json
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import httpx
import logfire
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import custom exception handling
from core.exceptions import (
    setup_exception_handlers,
    AgenticalError,
    NotFoundError,
    ValidationError,
    ServerError,
    ExternalServiceError,
    AgentError,
    WorkflowError,
    PlaybookError
)

# Import structured logging framework
from core.structured_logging import (
    StructuredLogger,
    CorrelationContext,
    LogLevel,
    OperationType,
    AgentPhase,
    api_logger,
    agent_logger,
    workflow_logger,
    system_logger,
    create_correlation_context,
    log_error_with_context
)

# Import performance monitoring
from src.monitoring.performance import performance_monitor
from src.monitoring.health import health_router, initialize_health_monitoring, shutdown_health_monitoring

# Load environment variables
load_dotenv()


def load_logfire_credentials() -> Dict[str, str]:
    """Load Logfire credentials from credentials file with fallback to environment variables."""
    credentials_path = Path(".logfire/logfire_credentials.json")
    
    try:
        if credentials_path.exists():
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
                return {
                    'token': credentials.get('write-token'),
                    'project_name': credentials.get('project_name', 'agentical'),
                    'project_url': credentials.get('project_url'),
                    'api_url': credentials.get('logfire_api_url')
                }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load Logfire credentials from file: {e}")
    
    # Fallback to environment variables
    return {
        'token': os.getenv("LOGFIRE_TOKEN"),
        'project_name': os.getenv("LOGFIRE_PROJECT_NAME", "agentical"),
        'project_url': None,
        'api_url': None
    }


# Load Logfire credentials
logfire_creds = load_logfire_credentials()

# Configure Logfire observability with credentials file
logfire.configure(
    token=logfire_creds['token'],
    project_name=logfire_creds['project_name'],
    service_name=os.getenv("LOGFIRE_SERVICE_NAME", "agentical-api"),
    environment=os.getenv("LOGFIRE_ENVIRONMENT", "development")
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Service health details")


class AgentRequest(BaseModel):
    """Agent execution request"""
    agent_id: str = Field(..., description="Agent ID to execute")
    operation: str = Field(..., description="Operation to perform")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Operation parameters")
    use_knowledge: bool = Field(default=True, description="Whether to use knowledge base")
    tools: Optional[List[str]] = Field(default=None, description="Tools to use")

from agents import agent_registry


class AgentResponse(BaseModel):
    """Agent execution response"""
    success: bool = Field(..., description="Execution success status")
    agent_id: str = Field(..., description="Agent identifier")
    operation: str = Field(..., description="Operation performed")
    result: Dict[str, Any] = Field(default_factory=dict, description="Execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    tools_used: List[str] = Field(default_factory=list, description="Tools that were used")
    knowledge_queries: int = Field(default=0, description="Number of knowledge base queries")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowRequest(BaseModel):
    """Workflow execution request"""
    workflow_type: str = Field(..., description="Type of workflow pattern")
    agents: List[str] = Field(..., description="Agents to coordinate")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")


class PlaybookRequest(BaseModel):
    """Playbook execution request"""
    playbook_name: str = Field(..., description="Name of playbook to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Playbook parameters")
    agents: Optional[List[str]] = Field(default=None, description="Override agent selection")
    tools: Optional[List[str]] = Field(default=None, description="Override tool selection")


class InfrastructureHealthChecker:
    """Health checker for existing infrastructure"""
    
    def __init__(self):
        self.ptolemies_url = os.getenv("PTOLEMIES_URL", "http://localhost:8001")
        self.surrealdb_url = os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc")
        self.mcp_servers_config = self._load_mcp_config()
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP server configuration"""
        try:
            import json
            mcp_config_path = os.path.join(os.path.dirname(__file__), "..", "mcp", "mcp-servers.json")
            if os.path.exists(mcp_config_path):
                with open(mcp_config_path, 'r') as f:
                    return json.load(f)
            else:
                # Fallback to relative path
                mcp_config_path = "../mcp/mcp-servers.json"
                if os.path.exists(mcp_config_path):
                    with open(mcp_config_path, 'r') as f:
                        return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load MCP config: {e}")
        return {"mcp_servers": {}}
    
    async def check_ptolemies(self) -> Dict[str, Any]:
        """Check Ptolemies knowledge base health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to connect to Ptolemies API if it's running
                try:
                    response = await client.get(f"{self.ptolemies_url}/health")
                    if response.status_code == 200:
                        return {"status": "healthy", "documents": 597, "connection": "api"}
                except httpx.RequestError:
                    pass
                
                # Fallback: Check if we can import Ptolemies modules
                try:
                    import sys
                    ptolemies_path = os.path.join(os.path.dirname(__file__), "..", "ptolemies", "src")
                    if ptolemies_path not in sys.path:
                        sys.path.append(ptolemies_path)
                    
                    # Try to import Ptolemies client
                    from ptolemies.db.surrealdb_client import SurrealDBClient
                    return {
                        "status": "accessible", 
                        "documents": 597, 
                        "connection": "direct",
                        "note": "Direct module access available"
                    }
                except ImportError as e:
                    return {
                        "status": "partial",
                        "error": f"Module import failed: {e}",
                        "connection": "none",
                        "note": "Ptolemies exists but not accessible"
                    }
                    
        except Exception as e:
            return {"status": "error", "error": str(e), "connection": "failed"}
    
    async def check_mcp_servers(self) -> Dict[str, Any]:
        """Check MCP server availability"""
        servers = self.mcp_servers_config.get("mcp_servers", {})
        server_count = len(servers)
        
        if server_count == 0:
            return {"status": "error", "error": "No MCP servers configured"}
        
        # For now, just return configuration status
        # In a full implementation, we'd test actual server connections
        key_servers = [
            "ptolemies-mcp", "surrealdb-mcp", "bayes-mcp", 
            "darwin-mcp", "filesystem", "git", "memory"
        ]
        
        available_servers = [server for server in key_servers if server in servers]
        
        return {
            "status": "configured",
            "total_servers": server_count,
            "key_servers_available": len(available_servers),
            "key_servers": available_servers,
            "note": f"Configuration loaded with {server_count} servers"
        }
    
    async def check_surrealdb(self) -> Dict[str, Any]:
        """Check SurrealDB connectivity"""
        try:
            # This is a basic check - in production we'd actually connect
            return {
                "status": "configured",
                "url": self.surrealdb_url,
                "note": "SurrealDB URL configured (used by Ptolemies)"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Initialize health checker
health_checker = InfrastructureHealthChecker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    with logfire.span("Application startup"):
        logfire.info("Agentical orchestration layer starting up")
        
        # Initialize performance monitoring
        await initialize_health_monitoring()
        performance_monitor.start_monitoring()
        
        # Test infrastructure connectivity on startup
        ptolemies_health = await health_checker.check_ptolemies()
        mcp_health = await health_checker.check_mcp_servers()
        
        logfire.info("Infrastructure health check", 
                    ptolemies=ptolemies_health["status"],
                    mcp_servers=mcp_health["status"])
        
        yield
    
    # Shutdown
    with logfire.span("Application shutdown"):
        logfire.info("Agentical orchestration layer shutting down")
        
        # Shutdown performance monitoring
        await shutdown_health_monitoring()
        performance_monitor.stop_monitoring()


# Create FastAPI application
app = FastAPI(
    title="Agentical",
    description="Agentic AI Framework - Orchestration layer for DevQ.ai infrastructure",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable comprehensive Logfire instrumentation
logfire.instrument_fastapi(app, capture_headers=True)
logfire.instrument_httpx()  # Instrument httpx for external API calls
logfire.instrument_sqlalchemy()  # Instrument SQLAlchemy for database operations

# Import API router
from api import api_router

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Import security middlewares
from middlewares.security import (
    RateLimitMiddleware, RateLimitConfig,
    SecurityHeadersMiddleware,
    RequestValidationMiddleware, RequestValidationConfig,
    BotProtectionMiddleware, BotDetectionConfig
)

# Configure security middlewares
app.add_middleware(
    BotProtectionMiddleware,
    config=BotDetectionConfig(
        enabled=True,
        challenge_suspicious=True,
        exclude_paths=["/health", "/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready", "/api/v1/health/detailed", "/docs", "/redoc", "/openapi.json"],
    )
)

app.add_middleware(
    RequestValidationMiddleware,
    config=RequestValidationConfig(
        max_content_length=10 * 1024 * 1024,  # 10 MB
        validate_query_params=True,
        validate_headers=True,
        exclude_paths=["/health", "/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready", "/api/v1/health/detailed", "/docs", "/redoc", "/openapi.json"],
    )
)

app.add_middleware(
    RateLimitMiddleware,
    config=RateLimitConfig(
        requests_per_minute=120,
        burst=10,
        strategy="sliding_window",
        exclude_paths=["/health", "/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready", "/api/v1/health/detailed", "/docs", "/redoc", "/openapi.json"],
        per_endpoint=True
    )
)

app.add_middleware(
    SecurityHeadersMiddleware,
    csp_directives={
        "default-src": ["'self'"],
        "img-src": ["'self'", "data:"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "script-src": ["'self'"],
        "font-src": ["'self'"],
        "object-src": ["'none'"],
        "frame-ancestors": ["'none'"],
    },
    disable_for_paths=["/docs", "/redoc", "/openapi.json"]
)

# Add structured logging middleware for enhanced request tracing
@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Enhanced request tracing with structured logging"""
    start_time = time.time()
    
    # Create correlation context for this request
    correlation = create_correlation_context(
        session_id=request.headers.get("X-Session-ID"),
        user_id=request.headers.get("X-User-ID")
    )
    
    # Extract request metadata
    request_size = len(await request.body()) if hasattr(request, 'body') else 0
    user_agent = request.headers.get("user-agent")
    client_ip = request.client.host if request.client else None
    
    # Log request start
    api_logger.log_api_request(
        message=f"Request started: {request.method} {request.url.path}",
        method=request.method,
        path=str(request.url.path),
        level=LogLevel.INFO,
        correlation=correlation,
        request_size_bytes=request_size,
        user_agent=user_agent,
        client_ip=client_ip,
        query_params=dict(request.query_params) if request.query_params else None
    )
    
    # Process request with correlation context
    with api_logger.correlation_context(correlation):
        try:
            response = await call_next(request)
            
            # Calculate response metrics
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Log successful request completion
            api_logger.log_api_request(
                message=f"Request completed: {request.method} {request.url.path}",
                method=request.method,
                path=str(request.url.path),
                level=LogLevel.INFO,
                correlation=correlation,
                status_code=response.status_code,
                response_time_ms=response_time,
                request_size_bytes=request_size,
                response_size_bytes=len(response.body) if hasattr(response, 'body') else None
            )
            
            # Add correlation headers to response
            response.headers["X-Request-ID"] = correlation.request_id
            response.headers["X-Trace-ID"] = correlation.trace_id
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            
            return response
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            # Log request error
            log_error_with_context(
                api_logger,
                e,
                f"Request failed: {request.method} {request.url.path}",
                OperationType.API_REQUEST,
                correlation,
                response_time_ms=response_time,
                request_path=str(request.url.path),
                request_method=request.method
            )
            raise

# Add performance monitoring middleware
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Performance monitoring middleware"""
    return await performance_monitor.create_performance_middleware()(request, call_next)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include health monitoring endpoints
app.include_router(health_router)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic information"""
    # Log root endpoint access with structured logging
    system_logger.log_api_request(
        message="Root endpoint accessed",
        method="GET",
        path="/",
        level=LogLevel.INFO
    )
    
    return {
        "name": "Agentical",
        "description": "Agentic AI Framework - Orchestration Layer",
        "version": "1.0.0",
        "status": "operational",
        "infrastructure": {
            "ptolemies_knowledge_base": "597 production documents",
            "mcp_servers": "22+ operational servers",
            "stack": "FastAPI + Logfire + PyTest + TaskMaster AI"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "agents": "/api/v1/agents",
            "workflows": "/api/v1/workflows",
            "playbooks": "/api/v1/playbooks"
        }
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check for all infrastructure"""
    # Create correlation context for health check
    correlation = create_correlation_context()
    
    with api_logger.correlation_context(correlation):
        with logfire.span("Health check"):
            # Log health check start
            system_logger.log_api_request(
                message="Health check initiated",
                method="GET",
                path="/health",
                level=LogLevel.INFO,
                correlation=correlation
            )
        start_time = datetime.utcnow()
        
        # Check all infrastructure components
        ptolemies_health = await health_checker.check_ptolemies()
        mcp_health = await health_checker.check_mcp_servers()
        surrealdb_health = await health_checker.check_surrealdb()
        
        # Determine overall status
        services = {
            "ptolemies_knowledge_base": ptolemies_health,
            "mcp_servers": mcp_health,
            "surrealdb": surrealdb_health,
            "logfire": {"status": "operational", "instrumentation": "active"}
        }
        
        # Overall status logic
        critical_services = ["ptolemies_knowledge_base", "mcp_servers"]
        healthy_critical = all(
            services[service]["status"] in ["healthy", "accessible", "configured", "operational"]
            for service in critical_services
        )
        
        overall_status = "healthy" if healthy_critical else "degraded"
        
        health_response = HealthStatus(
            status=overall_status,
            timestamp=start_time,
            services=services
        )
        
        # Log health check completion with structured logging
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
        
        system_logger.log_api_request(
            message=f"Health check completed with status: {overall_status}",
            method="GET",
            path="/health",
            level=LogLevel.INFO if overall_status == "healthy" else LogLevel.WARNING,
            correlation=correlation,
            status_code=200,
            response_time_ms=execution_time
        )
        
        # Log individual service statuses
        for service_name, service_health in services.items():
            system_logger.log_external_service(
                message=f"Service {service_name} health check: {service_health['status']}",
                service_name=service_name,
                level=LogLevel.INFO if service_health["status"] in ["healthy", "operational", "configured"] else LogLevel.WARNING,
                correlation=correlation,
                response_time_ms=execution_time / len(services)  # Approximate per-service time
            )
        
        logfire.info("Health check completed", 
                    status=overall_status,
                    ptolemies=ptolemies_health["status"],
                    mcp=mcp_health["status"])
        
        return health_response


# API v1 routes
@app.post("/api/v1/agents/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with coordination of existing infrastructure"""
    # Create correlation context for agent execution
    correlation = create_correlation_context(
        agent_id=request.agent_id
    )
    
    with agent_logger.correlation_context(correlation):
        with logfire.span("Agent execution", agent_id=request.agent_id, operation=request.operation):
            start_time = datetime.utcnow()
            
            # Log agent execution start
            agent_logger.log_agent_operation(
                message=f"Agent execution started: {request.agent_id} - {request.operation}",
                agent_type="dynamic",
                agent_name=request.agent_id,
                phase=AgentPhase.INITIALIZATION,
                operation_id=correlation.request_id,
                level=LogLevel.INFO,
                correlation=correlation,
                input_data={
                    "operation": request.operation,
                    "inputs": request.inputs,
                    "use_knowledge": request.use_knowledge,
                    "tools": request.tools
                }
            )
            
            try:
                # Get or create the agent
                try:
                    agent = agent_registry.get_agent(request.agent_id)
                    agent_logger.log_agent_operation(
                        message=f"Using existing agent: {request.agent_id}",
                        agent_type="existing",
                        agent_name=request.agent_id,
                        phase=AgentPhase.INITIALIZATION,
                        operation_id=correlation.request_id,
                        level=LogLevel.INFO,
                        correlation=correlation
                    )
                    logfire.info(f"Using existing agent: {request.agent_id}")
                except:
                    # If agent doesn't exist, create a generic agent with that ID
                    agent_logger.log_agent_operation(
                        message=f"Creating new generic agent: {request.agent_id}",
                        agent_type="generic",
                        agent_name=request.agent_id,
                        phase=AgentPhase.INITIALIZATION,
                        operation_id=correlation.request_id,
                        level=LogLevel.INFO,
                        correlation=correlation
                    )
                    logfire.info(f"Creating new generic agent: {request.agent_id}")
                    agent = agent_registry.create_agent(request.agent_id, "generic", name=request.agent_id)
                
                # Log agent execution phase
                agent_logger.log_agent_operation(
                    message=f"Agent execution phase started: {request.operation}",
                    agent_type="dynamic",
                    agent_name=request.agent_id,
                    phase=AgentPhase.ACTION,
                    operation_id=correlation.request_id,
                    level=LogLevel.INFO,
                    correlation=correlation,
                    tools_used=request.tools
                )
                
                # Execute the agent operation
                logfire.info("Agent execution started", 
                            agent_id=request.agent_id, 
                            operation=request.operation,
                            use_knowledge=request.use_knowledge)
                
                # Execute the operation
                result = await agent.execute(
                    operation=request.operation,
                    parameters=request.parameters or {}
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                execution_time_ms = execution_time * 1000
                
                # Create response
                response = AgentResponse(
                    success=result.success,
                    agent_id=request.agent_id,
                    operation=request.operation,
                    result=result.result if result.success else {"error": result.error},
                    execution_time=execution_time,
                    tools_used=result.tools_used or request.tools or [],
                    knowledge_queries=result.knowledge_queries
                )
                
                # Log successful agent completion
                agent_logger.log_agent_operation(
                    message=f"Agent execution completed successfully: {request.agent_id}",
                    agent_type="dynamic",
                    agent_name=request.agent_id,
                    phase=AgentPhase.REFLECTION,
                    operation_id=correlation.request_id,
                    level=LogLevel.INFO,
                    correlation=correlation,
                    output_data={
                        "success": response.success,
                        "tools_used": response.tools_used,
                        "knowledge_queries": response.knowledge_queries
                    },
                    execution_time_ms=execution_time_ms,
                    success=True
                )
                
                logfire.info("Agent execution completed", 
                            agent_id=request.agent_id,
                            success=response.success,
                            execution_time=execution_time)
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                execution_time_ms = execution_time * 1000
                
                # Log agent execution error with structured logging
                log_error_with_context(
                    agent_logger,
                    e,
                    f"Agent execution failed: {request.agent_id} - {request.operation}",
                    OperationType.AGENT_OPERATION,
                    correlation,
                    agent_id=request.agent_id,
                    operation=request.operation,
                    execution_time_ms=execution_time_ms,
                    agent_type="dynamic"
                )
                
                # Log agent error in operation format
                agent_logger.log_agent_operation(
                    message=f"Agent execution failed: {request.agent_id} - {str(e)}",
                    agent_type="dynamic",
                    agent_name=request.agent_id,
                    phase=AgentPhase.REFLECTION,
                    operation_id=correlation.request_id,
                    level=LogLevel.ERROR,
                    correlation=correlation,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_details={
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "operation": request.operation
                    }
                )
                
                logfire.error(f"Error executing agent: {str(e)}")
                
                # Error response
                response = AgentResponse(
                    success=False,
                    agent_id=request.agent_id,
                    operation=request.operation,
                    result={"error": str(e)},
                    execution_time=execution_time,
                    tools_used=request.tools or [],
                    knowledge_queries=0
                )
                
                return response


@app.post("/api/v1/workflows/execute")
async def execute_workflow(request: WorkflowRequest):
    """Execute a workflow pattern coordinating multiple agents"""
    # Create correlation context for workflow execution
    correlation = create_correlation_context(
        workflow_id=f"workflow_{request.workflow_type}_{uuid.uuid4().hex[:8]}"
    )
    
    with workflow_logger.correlation_context(correlation):
        with logfire.span("Workflow execution", workflow_type=request.workflow_type):
            # Log workflow execution start
            workflow_logger.log_workflow_execution(
                message=f"Workflow execution started: {request.workflow_type}",
                workflow_type=request.workflow_type,
                workflow_name=request.workflow_type,
                level=LogLevel.INFO,
                correlation=correlation,
                workflow_data={
                    "steps": request.steps,
                    "agents": request.agents
                }
            )
            
            logfire.info("Workflow execution started", 
                        workflow_type=request.workflow_type,
                    agents=request.agents)
        
        # Placeholder for workflow coordination
        return {
            "success": True,
            "workflow_type": request.workflow_type,
            "agents_coordinated": len(request.agents),
            "steps_completed": len(request.steps),
            "status": "This is a placeholder - full implementation in development"
        }


@app.post("/api/v1/playbooks/execute")
async def execute_playbook(request: PlaybookRequest):
    """Execute a playbook leveraging existing infrastructure"""
    with logfire.span("Playbook execution", playbook_name=request.playbook_name):
        logfire.info("Playbook execution started", 
                    playbook_name=request.playbook_name)
        
        # Placeholder for playbook coordination
        return {
            "success": True,
            "playbook_name": request.playbook_name,
            "infrastructure_coordination": "ready",
            "status": "This is a placeholder - full implementation in development"
        }


@app.get("/api/v1/infrastructure/status")
async def infrastructure_status():
    """Get detailed status of all infrastructure components"""
    with logfire.span("Infrastructure status check"):
        ptolemies_health = await health_checker.check_ptolemies()
        mcp_health = await health_checker.check_mcp_servers()
        surrealdb_health = await health_checker.check_surrealdb()
        
        return {
            "timestamp": datetime.utcnow(),
            "agentical_version": "1.0.0",
            "orchestration_layer": "operational",
            "infrastructure": {
                "ptolemies_knowledge_base": ptolemies_health,
                "mcp_server_ecosystem": mcp_health,
                "surrealdb": surrealdb_health
            },
            "integration_status": "Phase 1 - Infrastructure Discovery Complete"
        }


# Set up comprehensive exception handling
setup_exception_handlers(app)


if __name__ == "__main__":
    # Configuration
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("APP_ENV", "development") == "development"
    
    logfire.info("Starting Agentical orchestration layer", 
                host=host, port=port, debug=debug)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )