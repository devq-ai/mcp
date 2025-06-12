"""
Agent Management API Endpoints

This module provides comprehensive REST API endpoints for managing all agent types
in the Agentical framework, including real-time monitoring, configuration,
execution control, and analytics.

Features:
- CRUD operations for all 18 specialized agent types
- Real-time status monitoring and health checks
- Agent configuration and parameter management
- Execution history and performance analytics
- WebSocket support for real-time updates
- Comprehensive error handling and validation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

import logfire
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from agentical.agents import (
    AgentRegistry, EnhancedBaseAgent, AgentType, AgentStatus, AgentPhase,
    CodeAgent, DataScienceAgent, DBAAgent, DevOpsAgent, GCPAgent, GitHubAgent,
    InfoSecAgent, LegalAgent, PulumiAgent, ResearchAgent, TesterAgent, TokenAgent,
    UATAgent, UXAgent, CodifierAgent, IOAgent, PlaybookAgent, SuperAgent
)
from agentical.core.exceptions import (
    AgentError, AgentNotFoundError, AgentExecutionError, ValidationError
)
from agentical.core.structured_logging import StructuredLogger, LogLevel
from agentical.db.session import get_db
from agentical.db.repositories.agent import AgentRepository


# Initialize router and logger
router = APIRouter(prefix="/agents", tags=["agents"])
logger = StructuredLogger("agent_api")

# Agent type mapping for dynamic instantiation
AGENT_TYPE_MAPPING = {
    "code": CodeAgent,
    "data_science": DataScienceAgent,
    "dba": DBAAgent,
    "devops": DevOpsAgent,
    "gcp": GCPAgent,
    "github": GitHubAgent,
    "infosec": InfoSecAgent,
    "legal": LegalAgent,
    "pulumi": PulumiAgent,
    "research": ResearchAgent,
    "tester": TesterAgent,
    "token": TokenAgent,
    "uat": UATAgent,
    "ux": UXAgent,
    "codifier": CodifierAgent,
    "io": IOAgent,
    "playbook": PlaybookAgent,
    "super": SuperAgent
}


# Pydantic Models
class AgentStatusResponse(BaseModel):
    """Agent status information."""
    agent_id: str
    agent_type: str
    status: str
    phase: str
    created_at: datetime
    last_active: Optional[datetime]
    execution_count: int
    success_rate: float
    average_execution_time: float
    current_operation: Optional[str]
    resource_usage: Dict[str, Any]
    health_status: str


class AgentConfigRequest(BaseModel):
    """Agent configuration request."""
    agent_type: str = Field(..., description="Type of agent to create")
    name: Optional[str] = Field(None, description="Custom agent name")
    description: Optional[str] = Field(None, description="Agent description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration parameters")
    capabilities: Optional[List[str]] = Field(None, description="Specific capabilities to enable")
    resource_limits: Optional[Dict[str, Any]] = Field(None, description="Resource constraints")

    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v not in AGENT_TYPE_MAPPING:
            raise ValueError(f"Invalid agent type. Must be one of: {list(AGENT_TYPE_MAPPING.keys())}")
        return v


class AgentConfigResponse(BaseModel):
    """Agent configuration response."""
    agent_id: str
    agent_type: str
    name: str
    description: str
    config: Dict[str, Any]
    capabilities: List[str]
    resource_limits: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AgentExecutionRequest(BaseModel):
    """Agent execution request."""
    operation: str = Field(..., description="Operation to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    priority: str = Field(default="normal", description="Execution priority")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL")


class AgentExecutionResponse(BaseModel):
    """Agent execution response."""
    execution_id: str
    agent_id: str
    operation: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metrics: Dict[str, Any]


class AgentListResponse(BaseModel):
    """Agent list response."""
    agents: List[AgentStatusResponse]
    total: int
    page: int
    page_size: int
    available_types: List[str]


class AgentAnalyticsResponse(BaseModel):
    """Agent analytics response."""
    agent_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    average_execution_time: float
    peak_execution_time: float
    total_runtime: float
    executions_by_day: Dict[str, int]
    operations_frequency: Dict[str, int]
    error_patterns: Dict[str, int]
    performance_trends: List[Dict[str, Any]]


# WebSocket connection manager
class AgentWebSocketManager:
    """Manages WebSocket connections for real-time agent updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.agent_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)

        if agent_id:
            if agent_id not in self.agent_subscribers:
                self.agent_subscribers[agent_id] = []
            self.agent_subscribers[agent_id].append(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if agent_id and agent_id in self.agent_subscribers:
            if websocket in self.agent_subscribers[agent_id]:
                self.agent_subscribers[agent_id].remove(websocket)

    async def broadcast_agent_update(self, agent_id: str, update: Dict[str, Any]):
        """Broadcast agent update to subscribers."""
        if agent_id in self.agent_subscribers:
            disconnected = []
            for websocket in self.agent_subscribers[agent_id]:
                try:
                    await websocket.send_json(update)
                except:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.disconnect(ws, agent_id)


# Initialize WebSocket manager
ws_manager = AgentWebSocketManager()


# Dependency injection
async def get_agent_registry() -> AgentRegistry:
    """Get agent registry instance."""
    # This would typically be injected from the main application
    from agentical.agents import agent_registry
    return agent_registry


async def get_agent_repository(db: Session = Depends(get_db)) -> AgentRepository:
    """Get agent repository instance."""
    return AgentRepository(db)


# API Endpoints

@router.get("/", response_model=AgentListResponse)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search agents by name"),
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """List all agents with filtering and pagination."""

    with logfire.span("List agents", page=page, page_size=page_size):
        try:
            # Get agents from registry
            all_agents = registry.list_agents()

            # Apply filters
            filtered_agents = all_agents
            if agent_type:
                filtered_agents = [a for a in filtered_agents if a.agent_type.value == agent_type]
            if status:
                filtered_agents = [a for a in filtered_agents if a.status.value == status]
            if search:
                filtered_agents = [a for a in filtered_agents if search.lower() in a.name.lower()]

            # Pagination
            total = len(filtered_agents)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_agents = filtered_agents[start_idx:end_idx]

            # Convert to response format
            agent_responses = []
            for agent in paginated_agents:
                # Get additional metrics from database
                metrics = await db_repo.get_agent_metrics(agent.agent_id)

                agent_responses.append(AgentStatusResponse(
                    agent_id=agent.agent_id,
                    agent_type=agent.agent_type.value,
                    status=agent.status.value,
                    phase=agent.current_phase.value if agent.current_phase else "idle",
                    created_at=agent.created_at,
                    last_active=agent.last_active,
                    execution_count=metrics.get("execution_count", 0),
                    success_rate=metrics.get("success_rate", 0.0),
                    average_execution_time=metrics.get("avg_execution_time", 0.0),
                    current_operation=getattr(agent, 'current_operation', None),
                    resource_usage=metrics.get("resource_usage", {}),
                    health_status=agent.health_status if hasattr(agent, 'health_status') else "unknown"
                ))

            return AgentListResponse(
                agents=agent_responses,
                total=total,
                page=page,
                page_size=page_size,
                available_types=list(AGENT_TYPE_MAPPING.keys())
            )

        except Exception as e:
            logger.log(f"Error listing agents: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.post("/", response_model=AgentConfigResponse)
async def create_agent(
    config: AgentConfigRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Create a new agent instance."""

    with logfire.span("Create agent", agent_type=config.agent_type):
        try:
            # Generate agent ID
            agent_id = f"{config.agent_type}_{uuid4().hex[:8]}"

            # Get agent class
            agent_class = AGENT_TYPE_MAPPING[config.agent_type]

            # Create agent instance
            agent = agent_class(
                agent_id=agent_id,
                name=config.name or f"{config.agent_type.title()}Agent",
                description=config.description or f"Automatically created {config.agent_type} agent",
                **config.config
            )

            # Register agent
            registry.register_agent(agent)

            # Save to database
            await db_repo.create_agent(
                agent_id=agent_id,
                agent_type=config.agent_type,
                name=agent.name,
                description=agent.description,
                config=config.config,
                capabilities=config.capabilities or agent.get_capabilities(),
                resource_limits=config.resource_limits
            )

            # Broadcast creation event
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_created",
                "agent_id": agent_id,
                "agent_type": config.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.log(f"Agent created: {agent_id}", LogLevel.INFO)

            return AgentConfigResponse(
                agent_id=agent_id,
                agent_type=config.agent_type,
                name=agent.name,
                description=agent.description,
                config=config.config,
                capabilities=agent.get_capabilities(),
                resource_limits=config.resource_limits or {},
                created_at=agent.created_at,
                updated_at=agent.created_at
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.log(f"Error creating agent: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/{agent_id}", response_model=AgentStatusResponse)
async def get_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Get detailed information about a specific agent."""

    with logfire.span("Get agent", agent_id=agent_id):
        try:
            # Get agent from registry
            agent = registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Get metrics from database
            metrics = await db_repo.get_agent_metrics(agent_id)

            return AgentStatusResponse(
                agent_id=agent_id,
                agent_type=agent.agent_type.value,
                status=agent.status.value,
                phase=agent.current_phase.value if agent.current_phase else "idle",
                created_at=agent.created_at,
                last_active=agent.last_active,
                execution_count=metrics.get("execution_count", 0),
                success_rate=metrics.get("success_rate", 0.0),
                average_execution_time=metrics.get("avg_execution_time", 0.0),
                current_operation=getattr(agent, 'current_operation', None),
                resource_usage=metrics.get("resource_usage", {}),
                health_status=agent.health_status if hasattr(agent, 'health_status') else "healthy"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error getting agent {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.put("/{agent_id}/config", response_model=AgentConfigResponse)
async def update_agent_config(
    agent_id: str,
    config: Dict[str, Any],
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Update agent configuration."""

    with logfire.span("Update agent config", agent_id=agent_id):
        try:
            # Get agent from registry
            agent = registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Update agent configuration
            for key, value in config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

            # Update in database
            await db_repo.update_agent_config(agent_id, config)

            # Broadcast update event
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_config_updated",
                "agent_id": agent_id,
                "config": config,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.log(f"Agent config updated: {agent_id}", LogLevel.INFO)

            return AgentConfigResponse(
                agent_id=agent_id,
                agent_type=agent.agent_type.value,
                name=agent.name,
                description=agent.description,
                config=config,
                capabilities=agent.get_capabilities(),
                resource_limits={},
                created_at=agent.created_at,
                updated_at=datetime.utcnow()
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error updating agent config {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to update agent config: {str(e)}")


@router.post("/{agent_id}/execute", response_model=AgentExecutionResponse)
async def execute_agent_operation(
    agent_id: str,
    request: AgentExecutionRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Execute an operation on a specific agent."""

    with logfire.span("Execute agent operation", agent_id=agent_id, operation=request.operation):
        try:
            # Get agent from registry
            agent = registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Generate execution ID
            execution_id = str(uuid4())

            # Record execution start
            start_time = datetime.utcnow()
            await db_repo.create_execution_record(
                execution_id=execution_id,
                agent_id=agent_id,
                operation=request.operation,
                parameters=request.parameters,
                started_at=start_time
            )

            # Broadcast execution start
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "execution_started",
                "agent_id": agent_id,
                "execution_id": execution_id,
                "operation": request.operation,
                "timestamp": start_time.isoformat()
            })

            try:
                # Execute the operation
                result = await agent.execute(request.operation, request.parameters)

                # Record successful completion
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                await db_repo.complete_execution_record(
                    execution_id=execution_id,
                    status="success",
                    result=result.to_dict() if hasattr(result, 'to_dict') else result,
                    completed_at=end_time,
                    duration=duration
                )

                # Broadcast execution completion
                await ws_manager.broadcast_agent_update(agent_id, {
                    "type": "execution_completed",
                    "agent_id": agent_id,
                    "execution_id": execution_id,
                    "status": "success",
                    "duration": duration,
                    "timestamp": end_time.isoformat()
                })

                return AgentExecutionResponse(
                    execution_id=execution_id,
                    agent_id=agent_id,
                    operation=request.operation,
                    status="success",
                    started_at=start_time,
                    completed_at=end_time,
                    duration=duration,
                    result=result.to_dict() if hasattr(result, 'to_dict') else result,
                    error=None,
                    metrics={"execution_time": duration, "memory_usage": 0}
                )

            except Exception as exec_error:
                # Record execution failure
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                error_msg = str(exec_error)

                await db_repo.complete_execution_record(
                    execution_id=execution_id,
                    status="failed",
                    error=error_msg,
                    completed_at=end_time,
                    duration=duration
                )

                # Broadcast execution failure
                await ws_manager.broadcast_agent_update(agent_id, {
                    "type": "execution_failed",
                    "agent_id": agent_id,
                    "execution_id": execution_id,
                    "error": error_msg,
                    "duration": duration,
                    "timestamp": end_time.isoformat()
                })

                return AgentExecutionResponse(
                    execution_id=execution_id,
                    agent_id=agent_id,
                    operation=request.operation,
                    status="failed",
                    started_at=start_time,
                    completed_at=end_time,
                    duration=duration,
                    result=None,
                    error=error_msg,
                    metrics={"execution_time": duration, "memory_usage": 0}
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error executing agent operation {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to execute operation: {str(e)}")


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Get real-time status of a specific agent."""
    return await get_agent(agent_id, registry, db_repo)


@router.post("/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Stop a running agent."""

    with logfire.span("Stop agent", agent_id=agent_id):
        try:
            # Get agent from registry
            agent = registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Stop the agent
            await agent.stop()

            # Broadcast stop event
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_stopped",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.log(f"Agent stopped: {agent_id}", LogLevel.INFO)

            return {"message": f"Agent {agent_id} stopped successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error stopping agent {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to stop agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Delete an agent instance."""

    with logfire.span("Delete agent", agent_id=agent_id):
        try:
            # Get agent from registry
            agent = registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Stop agent if running
            if agent.status != AgentStatus.STOPPED:
                await agent.stop()

            # Unregister from registry
            registry.unregister_agent(agent_id)

            # Delete from database
            await db_repo.delete_agent(agent_id)

            # Broadcast deletion event
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_deleted",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.log(f"Agent deleted: {agent_id}", LogLevel.INFO)

            return {"message": f"Agent {agent_id} deleted successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error deleting agent {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


@router.get("/{agent_id}/executions")
async def get_agent_executions(
    agent_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Get execution history for a specific agent."""

    with logfire.span("Get agent executions", agent_id=agent_id):
        try:
            executions = await db_repo.get_agent_executions(
                agent_id=agent_id,
                page=page,
                page_size=page_size,
                status_filter=status
            )

            return executions

        except Exception as e:
            logger.log(f"Error getting agent executions {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get executions: {str(e)}")


@router.get("/{agent_id}/analytics", response_model=AgentAnalyticsResponse)
async def get_agent_analytics(
    agent_id: str,
    days: int = Query(30, ge=1, le=365),
    db_repo: AgentRepository = Depends(get_agent_repository)
):
    """Get analytics and performance metrics for a specific agent."""

    with logfire.span("Get agent analytics", agent_id=agent_id):
        try:
            analytics = await db_repo.get_agent_analytics(agent_id, days)

            return AgentAnalyticsResponse(**analytics)

        except Exception as e:
            logger.log(f"Error getting agent analytics {agent_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/types/available")
async def get_available_agent_types():
    """Get list of available agent types."""

    return {
        "agent_types": [
            {
                "type": agent_type,
                "class_name": agent_class.__name__,
                "description": agent_class.__doc__ or f"{agent_type.title()} agent",
                "capabilities": getattr(agent_class, '_default_capabilities', [])
            }
            for agent_type, agent_class in AGENT_TYPE_MAPPING.items()
        ],
        "total": len(AGENT_TYPE_MAPPING)
    }


@router.websocket("/{agent_id}/ws")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for real-time agent updates."""

    await ws_manager.connect(websocket, agent_id)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Echo back any messages (ping/pong)
            await websocket.send_text(f"Agent {agent_id} WebSocket active")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, agent_id)


@router.get("/{agent_id}/logs/stream")
async def stream_agent_logs(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Stream real-time logs for a specific agent."""

    async def generate_logs():
        """Generate log stream for agent."""
        try:
            agent = registry.get_agent(agent_id)
            if not agent:
                yield f"data: {json.dumps({'error': f'Agent {agent_id} not found'})}\n\n"
                return

            # Simulate log streaming (in real implementation, this would connect to actual logs)
            while True:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": agent_id,
                    "level": "INFO",
                    "message": f"Agent {agent_id} is active",
                    "phase": "monitoring"
                }
                yield f"data: {json.dumps(log_entry)}\n\n"
                await asyncio.sleep(5)  # Send update every 5 seconds

        except Exception as e:
            error_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "level": "ERROR",
                "message": f"Log stream error: {str(e)}"
            }
            yield f"data: {json.dumps(error_entry)}\n\n"

    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
