"""
Agentical Application - Simplified Version with Structured Logging

This module provides a FastAPI application that serves as the orchestration layer
for the Agentical framework, incorporating structured logging and proper error handling.

This version:
- Implements comprehensive structured logging
- Provides proper error handling and response formatting
- Includes health check and monitoring endpoints
- Configures appropriate middleware
"""

import os
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
import logfire
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import agentical modules
from agentical.core import configure_logging, log_operation
from agentical.middlewares import (
    RequestLoggingMiddleware,
    HealthCheckLoggingFilter,
    ErrorLoggingMiddleware
)

# Load environment variables
load_dotenv()

# Define models
class HealthStatus(BaseModel):
    """Health status response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")
    checks: Dict[str, bool] = Field(..., description="Individual component health checks")

class AgentRequest(BaseModel):
    """Request model for agent execution."""
    agent_id: str = Field(..., description="ID of the agent to execute")
    prompt: str = Field(..., description="User prompt for the agent")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the agent")
    tools: Optional[List[str]] = Field(None, description="Tools to make available to the agent")
    
class AgentResponse(BaseModel):
    """Response model for agent execution."""
    agent_id: str = Field(..., description="ID of the executed agent")
    response: str = Field(..., description="Agent's response")
    thinking: Optional[str] = Field(None, description="Agent's reasoning process")
    tools_used: Optional[List[str]] = Field(None, description="Tools used by the agent")
    elapsed_time: float = Field(..., description="Time taken to process in seconds")
    timestamp: str = Field(..., description="Execution timestamp")

class InfrastructureStatus(BaseModel):
    """Infrastructure status response model."""
    status: str = Field(..., description="Overall infrastructure status")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Status of individual components")

# Infrastructure health checker
class InfrastructureHealthChecker:
    """Checks health of infrastructure components."""
    
    def __init__(self):
        """Initialize the health checker."""
        self.mcp_config = self._load_mcp_config()
        
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP server configuration."""
        with log_operation("Load MCP Configuration"):
            try:
                config_path = os.path.join(os.getcwd(), "mcp-servers.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    logfire.info("MCP configuration loaded", server_count=len(config))
                    return config
                else:
                    # Try alternate location
                    alt_path = os.path.join(os.getcwd(), "mcp", "mcp-servers.json")
                    if os.path.exists(alt_path):
                        with open(alt_path, "r") as f:
                            config = json.load(f)
                        logfire.info("MCP configuration loaded from alternate path", server_count=len(config))
                        return config
                    else:
                        logfire.warning("MCP configuration not found", checked_paths=[config_path, alt_path])
                        return {}
            except Exception as e:
                logfire.error("Error loading MCP configuration", error=str(e))
                return {}
    
    async def check_ptolemies(self) -> Dict[str, Any]:
        """Check Ptolemies knowledge base connectivity."""
        with log_operation("Check Ptolemies Health"):
            result = {
                "status": "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {}
            }
            
            try:
                # Path to Ptolemies
                ptolemies_path = os.path.join("..", "ptolemies")
                if not os.path.exists(ptolemies_path):
                    ptolemies_path = os.path.join("..", "..", "ptolemies")
                
                if os.path.exists(ptolemies_path):
                    # Count documents
                    doc_count = 0
                    docs_path = os.path.join(ptolemies_path, "docs")
                    if os.path.exists(docs_path):
                        for root, dirs, files in os.walk(docs_path):
                            doc_count += len([f for f in files if f.endswith(".md")])
                    
                    result["status"] = "healthy"
                    result["details"] = {
                        "path_exists": True,
                        "document_count": doc_count
                    }
                else:
                    result["status"] = "unhealthy"
                    result["details"] = {
                        "path_exists": False,
                        "error": "Ptolemies directory not found"
                    }
            except Exception as e:
                result["status"] = "error"
                result["details"] = {
                    "error": str(e)
                }
                logfire.error("Error checking Ptolemies health", error=str(e))
            
            return result
    
    async def check_mcp_servers(self) -> Dict[str, Any]:
        """Check MCP server connectivity."""
        with log_operation("Check MCP Servers Health"):
            result = {
                "status": "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "server_count": len(self.mcp_config),
                    "servers": {}
                }
            }
            
            if not self.mcp_config:
                result["status"] = "unhealthy"
                result["details"]["error"] = "No MCP servers configured"
                return result
            
            try:
                # For now, just check if configuration exists
                if self.mcp_config and len(self.mcp_config) > 0:
                    result["status"] = "healthy"
                    result["details"]["servers"] = {
                        server_id: {"configured": True} 
                        for server_id in self.mcp_config.keys()
                    }
                else:
                    result["status"] = "unhealthy"
                    result["details"]["error"] = "Empty MCP server configuration"
            except Exception as e:
                result["status"] = "error"
                result["details"]["error"] = str(e)
                logfire.error("Error checking MCP servers health", error=str(e))
            
            return result

# Initialize health checker
health_checker = InfrastructureHealthChecker()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    with log_operation("Application Startup"):
        logfire.info("Agentical orchestration layer starting up")
        
        # Test infrastructure connectivity on startup
        ptolemies_health = await health_checker.check_ptolemies()
        mcp_health = await health_checker.check_mcp_servers()
        
        logfire.info("Infrastructure health check", 
                    ptolemies=ptolemies_health["status"],
                    mcp_servers=mcp_health["status"])
        
        yield
    
    # Shutdown
    with log_operation("Application Shutdown"):
        logfire.info("Agentical orchestration layer shutting down")

# Create FastAPI application
app = FastAPI(
    title="Agentical",
    description="Agentic AI Framework - Orchestration layer for DevQ.ai infrastructure",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure logging
configure_logging(app)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(HealthCheckLoggingFilter)
app.add_middleware(ErrorLoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logfire.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging."""
    logfire.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# API routes
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    with log_operation("Root Endpoint"):
        return {
            "name": "Agentical API",
            "version": "0.1.0",
            "description": "Orchestration layer for DevQ.ai infrastructure",
            "documentation": "/docs",
            "health": "/health",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    with log_operation("Health Check"):
        # Check Ptolemies
        ptolemies_health = await health_checker.check_ptolemies()
        
        # Check MCP servers
        mcp_health = await health_checker.check_mcp_servers()
        
        # Determine overall status
        checks = {
            "ptolemies": ptolemies_health["status"] == "healthy",
            "mcp_servers": mcp_health["status"] == "healthy",
            "api": True
        }
        
        overall_status = "healthy" if all(checks.values()) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "checks": checks
        }

@app.post("/agent", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with the specified parameters."""
    with log_operation("Execute Agent", agent_id=request.agent_id):
        # Mock agent execution for now
        start_time = datetime.utcnow()
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Calculate elapsed time
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "agent_id": request.agent_id,
            "response": f"Agent {request.agent_id} processed: {request.prompt}",
            "thinking": "Simulated agent thinking process",
            "tools_used": request.tools or [],
            "elapsed_time": elapsed_time,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/infrastructure", response_model=InfrastructureStatus)
async def infrastructure_status():
    """Get infrastructure status information."""
    with log_operation("Infrastructure Status"):
        # Check Ptolemies
        ptolemies_health = await health_checker.check_ptolemies()
        
        # Check MCP servers
        mcp_health = await health_checker.check_mcp_servers()
        
        # Combine results
        components = {
            "ptolemies": ptolemies_health,
            "mcp_servers": mcp_health
        }
        
        # Determine overall status
        status = "operational"
        if any(c["status"] == "error" for c in components.values()):
            status = "error"
        elif any(c["status"] == "unhealthy" for c in components.values()):
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components
        }

# Run server directly when script is executed
if __name__ == "__main__":
    uvicorn.run(
        "agentical_app:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )