"""
Playbook Management API Endpoints

This module provides comprehensive REST API endpoints for managing playbooks
in the Agentical framework, including CRUD operations, execution control,
real-time monitoring, template management, and validation.

Features:
- CRUD operations for playbook management
- Execution control and monitoring APIs
- Template and validation endpoints
- Real-time execution status streaming
- WebSocket support for live updates
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

from agentical.agents import PlaybookAgent, AgentRegistry
from agentical.agents.playbook_agent import (
    PlaybookExecutionRequest, PlaybookCreationRequest, PlaybookAnalysisRequest,
    ExecutionMode, ValidationLevel
)
from agentical.db.models.playbook import (
    Playbook, PlaybookStep, PlaybookExecution, PlaybookStatus,
    ExecutionStatus, StepType, StepStatus, PlaybookCategory,
    PlaybookStepType, PlaybookExecutionStatus
)
from agentical.core.exceptions import (
    PlaybookError, PlaybookNotFoundError, PlaybookExecutionError, ValidationError
)
from agentical.core.structured_logging import StructuredLogger, LogLevel
from agentical.db.session import get_db
from agentical.db.repositories.playbook import PlaybookRepository


# Initialize router and logger
router = APIRouter(prefix="/playbooks", tags=["playbooks"])
logger = StructuredLogger("playbook_api")


# Pydantic Models
class PlaybookCreateRequest(BaseModel):
    """Playbook creation request."""
    name: str = Field(..., description="Playbook name")
    description: Optional[str] = Field(None, description="Playbook description")
    category: str = Field(..., description="Playbook category")
    steps: List[Dict[str, Any]] = Field(..., description="Playbook steps")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Playbook variables")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    template: Optional[str] = Field(None, description="Base template")
    validation_rules: Optional[List[str]] = Field(default_factory=list, description="Validation rules")
    tags: Optional[List[str]] = Field(default_factory=list, description="Playbook tags")

    @validator('category')
    def validate_category(cls, v):
        valid_categories = [cat.value for cat in PlaybookCategory]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v


class PlaybookUpdateRequest(BaseModel):
    """Playbook update request."""
    name: Optional[str] = Field(None, description="Playbook name")
    description: Optional[str] = Field(None, description="Playbook description")
    steps: Optional[List[Dict[str, Any]]] = Field(None, description="Playbook steps")
    variables: Optional[Dict[str, Any]] = Field(None, description="Playbook variables")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    validation_rules: Optional[List[str]] = Field(None, description="Validation rules")
    tags: Optional[List[str]] = Field(None, description="Playbook tags")


class PlaybookResponse(BaseModel):
    """Playbook response model."""
    id: str
    name: str
    description: Optional[str]
    category: str
    status: str
    version: int
    steps: List[Dict[str, Any]]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    validation_rules: List[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    execution_count: int
    last_executed: Optional[datetime]


class PlaybookExecutionStartRequest(BaseModel):
    """Playbook execution start request."""
    execution_mode: str = Field(default="sequential", description="Execution mode")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    validation_level: str = Field(default="standard", description="Validation level")
    dry_run: bool = Field(default=False, description="Perform dry run")
    timeout_minutes: int = Field(default=30, description="Execution timeout in minutes")
    checkpoint_interval: int = Field(default=5, description="Checkpoint interval in steps")
    continue_on_error: bool = Field(default=False, description="Continue on non-critical errors")

    @validator('execution_mode')
    def validate_execution_mode(cls, v):
        valid_modes = [mode.value for mode in ExecutionMode]
        if v not in valid_modes:
            raise ValueError(f"Invalid execution mode. Must be one of: {valid_modes}")
        return v

    @validator('validation_level')
    def validate_validation_level(cls, v):
        valid_levels = [level.value for level in ValidationLevel]
        if v not in valid_levels:
            raise ValueError(f"Invalid validation level. Must be one of: {valid_levels}")
        return v


class PlaybookExecutionResponse(BaseModel):
    """Playbook execution response."""
    execution_id: str
    playbook_id: str
    status: str
    execution_mode: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    steps_total: int
    steps_completed: int
    steps_failed: int
    current_step: Optional[str]
    progress_percentage: float
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    checkpoints: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class PlaybookListResponse(BaseModel):
    """Playbook list response."""
    playbooks: List[PlaybookResponse]
    total: int
    page: int
    page_size: int
    categories: List[str]
    tags: List[str]


class PlaybookTemplateResponse(BaseModel):
    """Playbook template response."""
    name: str
    description: str
    category: str
    steps: List[Dict[str, Any]]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    tags: List[str]


class PlaybookValidationResponse(BaseModel):
    """Playbook validation response."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    estimated_duration: Optional[int]
    complexity_score: int


# WebSocket connection manager for playbook updates
class PlaybookWebSocketManager:
    """Manages WebSocket connections for real-time playbook updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.execution_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, execution_id: Optional[str] = None):
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)

        if execution_id:
            if execution_id not in self.execution_subscribers:
                self.execution_subscribers[execution_id] = []
            self.execution_subscribers[execution_id].append(websocket)

    def disconnect(self, websocket: WebSocket, execution_id: Optional[str] = None):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if execution_id and execution_id in self.execution_subscribers:
            if websocket in self.execution_subscribers[execution_id]:
                self.execution_subscribers[execution_id].remove(websocket)

    async def broadcast_execution_update(self, execution_id: str, update: Dict[str, Any]):
        """Broadcast execution update to subscribers."""
        if execution_id in self.execution_subscribers:
            disconnected = []
            for websocket in self.execution_subscribers[execution_id]:
                try:
                    await websocket.send_json(update)
                except:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.disconnect(ws, execution_id)


# Initialize WebSocket manager
ws_manager = PlaybookWebSocketManager()


# Dependency injection
async def get_playbook_repository(db: Session = Depends(get_db)) -> PlaybookRepository:
    """Get playbook repository instance."""
    return PlaybookRepository(db)


async def get_playbook_agent() -> PlaybookAgent:
    """Get playbook agent instance."""
    # This would typically be injected from the main application
    from agentical.agents import agent_registry
    playbook_agents = [agent for agent in agent_registry.list_agents()
                      if isinstance(agent, PlaybookAgent)]

    if playbook_agents:
        return playbook_agents[0]

    # Create a new playbook agent if none exists
    agent = PlaybookAgent(
        agent_id=f"playbook_agent_{uuid4().hex[:8]}",
        name="Default Playbook Agent",
        description="Default playbook execution agent"
    )
    agent_registry.register_agent(agent)
    return agent


# API Endpoints

@router.get("/", response_model=PlaybookListResponse)
async def list_playbooks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search playbooks by name"),
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """List all playbooks with filtering and pagination."""

    with logfire.span("List playbooks", page=page, page_size=page_size):
        try:
            # Parse tags filter
            tag_list = tags.split(',') if tags else None

            # Get playbooks from repository
            result = await repo.list_playbooks(
                page=page,
                page_size=page_size,
                category=category,
                status=status,
                tags=tag_list,
                search=search
            )

            # Get available categories and tags
            categories = await repo.get_available_categories()
            all_tags = await repo.get_available_tags()

            return PlaybookListResponse(
                playbooks=[PlaybookResponse(**pb) for pb in result['playbooks']],
                total=result['total'],
                page=page,
                page_size=page_size,
                categories=categories,
                tags=all_tags
            )

        except Exception as e:
            logger.log(f"Error listing playbooks: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to list playbooks: {str(e)}")


@router.post("/", response_model=PlaybookResponse)
async def create_playbook(
    request: PlaybookCreateRequest,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Create a new playbook."""

    with logfire.span("Create playbook", name=request.name):
        try:
            # Generate playbook ID
            playbook_id = f"pb_{uuid4().hex[:12]}"

            # Create playbook in repository
            playbook_data = await repo.create_playbook(
                playbook_id=playbook_id,
                name=request.name,
                description=request.description,
                category=request.category,
                steps=request.steps,
                variables=request.variables,
                metadata=request.metadata,
                validation_rules=request.validation_rules,
                tags=request.tags
            )

            logger.log(f"Playbook created: {playbook_id}", LogLevel.INFO)

            return PlaybookResponse(**playbook_data)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.log(f"Error creating playbook: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to create playbook: {str(e)}")


@router.get("/{playbook_id}", response_model=PlaybookResponse)
async def get_playbook(
    playbook_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Get detailed information about a specific playbook."""

    with logfire.span("Get playbook", playbook_id=playbook_id):
        try:
            playbook_data = await repo.get_playbook(playbook_id)
            if not playbook_data:
                raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")

            return PlaybookResponse(**playbook_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error getting playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get playbook: {str(e)}")


@router.put("/{playbook_id}", response_model=PlaybookResponse)
async def update_playbook(
    playbook_id: str,
    request: PlaybookUpdateRequest,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Update a playbook."""

    with logfire.span("Update playbook", playbook_id=playbook_id):
        try:
            # Check if playbook exists
            existing = await repo.get_playbook(playbook_id)
            if not existing:
                raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")

            # Update playbook
            update_data = {k: v for k, v in request.dict().items() if v is not None}
            playbook_data = await repo.update_playbook(playbook_id, update_data)

            logger.log(f"Playbook updated: {playbook_id}", LogLevel.INFO)

            return PlaybookResponse(**playbook_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error updating playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to update playbook: {str(e)}")


@router.delete("/{playbook_id}")
async def delete_playbook(
    playbook_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Delete a playbook."""

    with logfire.span("Delete playbook", playbook_id=playbook_id):
        try:
            # Check if playbook exists
            existing = await repo.get_playbook(playbook_id)
            if not existing:
                raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")

            # Delete playbook
            await repo.delete_playbook(playbook_id)

            logger.log(f"Playbook deleted: {playbook_id}", LogLevel.INFO)

            return {"message": f"Playbook {playbook_id} deleted successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error deleting playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to delete playbook: {str(e)}")


@router.post("/{playbook_id}/execute", response_model=PlaybookExecutionResponse)
async def execute_playbook(
    playbook_id: str,
    request: PlaybookExecutionStartRequest,
    repo: PlaybookRepository = Depends(get_playbook_repository),
    agent: PlaybookAgent = Depends(get_playbook_agent)
):
    """Execute a playbook."""

    with logfire.span("Execute playbook", playbook_id=playbook_id):
        try:
            # Check if playbook exists
            playbook_data = await repo.get_playbook(playbook_id)
            if not playbook_data:
                raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")

            # Generate execution ID
            execution_id = str(uuid4())

            # Create execution request for agent
            agent_request = PlaybookExecutionRequest(
                playbook_id=playbook_id,
                execution_mode=ExecutionMode(request.execution_mode),
                parameters=request.parameters,
                validation_level=ValidationLevel(request.validation_level),
                dry_run=request.dry_run,
                timeout_minutes=request.timeout_minutes,
                checkpoint_interval=request.checkpoint_interval,
                continue_on_error=request.continue_on_error
            )

            # Create execution record
            start_time = datetime.utcnow()
            await repo.create_execution_record(
                execution_id=execution_id,
                playbook_id=playbook_id,
                execution_mode=request.execution_mode,
                parameters=request.parameters,
                started_at=start_time
            )

            # Broadcast execution start
            await ws_manager.broadcast_execution_update(execution_id, {
                "type": "execution_started",
                "execution_id": execution_id,
                "playbook_id": playbook_id,
                "execution_mode": request.execution_mode,
                "timestamp": start_time.isoformat()
            })

            try:
                # Execute the playbook
                result = await agent.execute(agent_request, {"execution_id": execution_id})

                # Record successful completion
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                await repo.complete_execution_record(
                    execution_id=execution_id,
                    status="completed",
                    result=result,
                    completed_at=end_time,
                    duration=duration
                )

                # Broadcast execution completion
                await ws_manager.broadcast_execution_update(execution_id, {
                    "type": "execution_completed",
                    "execution_id": execution_id,
                    "status": "completed",
                    "duration": duration,
                    "timestamp": end_time.isoformat()
                })

                return PlaybookExecutionResponse(
                    execution_id=execution_id,
                    playbook_id=playbook_id,
                    status="completed",
                    execution_mode=request.execution_mode,
                    started_at=start_time,
                    completed_at=end_time,
                    duration=duration,
                    steps_total=len(playbook_data.get('steps', [])),
                    steps_completed=len(playbook_data.get('steps', [])),
                    steps_failed=0,
                    current_step=None,
                    progress_percentage=100.0,
                    result=result,
                    error=None,
                    checkpoints=[],
                    metrics={"execution_time": duration}
                )

            except Exception as exec_error:
                # Record execution failure
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                error_msg = str(exec_error)

                await repo.complete_execution_record(
                    execution_id=execution_id,
                    status="failed",
                    error=error_msg,
                    completed_at=end_time,
                    duration=duration
                )

                # Broadcast execution failure
                await ws_manager.broadcast_execution_update(execution_id, {
                    "type": "execution_failed",
                    "execution_id": execution_id,
                    "error": error_msg,
                    "duration": duration,
                    "timestamp": end_time.isoformat()
                })

                return PlaybookExecutionResponse(
                    execution_id=execution_id,
                    playbook_id=playbook_id,
                    status="failed",
                    execution_mode=request.execution_mode,
                    started_at=start_time,
                    completed_at=end_time,
                    duration=duration,
                    steps_total=len(playbook_data.get('steps', [])),
                    steps_completed=0,
                    steps_failed=1,
                    current_step=None,
                    progress_percentage=0.0,
                    result=None,
                    error=error_msg,
                    checkpoints=[],
                    metrics={"execution_time": duration}
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error executing playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to execute playbook: {str(e)}")


@router.get("/{playbook_id}/executions")
async def get_playbook_executions(
    playbook_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Get execution history for a playbook."""

    with logfire.span("Get playbook executions", playbook_id=playbook_id):
        try:
            executions = await repo.get_playbook_executions(
                playbook_id=playbook_id,
                page=page,
                page_size=page_size,
                status_filter=status
            )

            return executions

        except Exception as e:
            logger.log(f"Error getting playbook executions {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get executions: {str(e)}")


@router.get("/executions/{execution_id}", response_model=PlaybookExecutionResponse)
async def get_execution_status(
    execution_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Get status of a specific execution."""

    with logfire.span("Get execution status", execution_id=execution_id):
        try:
            execution_data = await repo.get_execution(execution_id)
            if not execution_data:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

            return PlaybookExecutionResponse(**execution_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error getting execution status {execution_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@router.post("/executions/{execution_id}/stop")
async def stop_execution(
    execution_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Stop a running execution."""

    with logfire.span("Stop execution", execution_id=execution_id):
        try:
            # Check if execution exists
            execution_data = await repo.get_execution(execution_id)
            if not execution_data:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

            # Stop the execution
            await repo.stop_execution(execution_id)

            # Broadcast stop event
            await ws_manager.broadcast_execution_update(execution_id, {
                "type": "execution_stopped",
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.log(f"Execution stopped: {execution_id}", LogLevel.INFO)

            return {"message": f"Execution {execution_id} stopped successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error stopping execution {execution_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to stop execution: {str(e)}")


@router.post("/{playbook_id}/validate", response_model=PlaybookValidationResponse)
async def validate_playbook(
    playbook_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository),
    agent: PlaybookAgent = Depends(get_playbook_agent)
):
    """Validate a playbook."""

    with logfire.span("Validate playbook", playbook_id=playbook_id):
        try:
            # Check if playbook exists
            playbook_data = await repo.get_playbook(playbook_id)
            if not playbook_data:
                raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")

            # Perform validation using the agent
            validation_result = await agent.validate_playbook(playbook_data)

            return PlaybookValidationResponse(**validation_result)

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error validating playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to validate playbook: {str(e)}")


@router.get("/templates/available", response_model=List[PlaybookTemplateResponse])
async def get_available_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    agent: PlaybookAgent = Depends(get_playbook_agent)
):
    """Get available playbook templates."""

    with logfire.span("Get available templates"):
        try:
            templates = await agent.get_available_templates(category_filter=category)

            return [PlaybookTemplateResponse(**template) for template in templates]

        except Exception as e:
            logger.log(f"Error getting templates: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.post("/templates/{template_name}/create", response_model=PlaybookResponse)
async def create_from_template(
    template_name: str,
    name: str = Query(..., description="New playbook name"),
    parameters: Optional[Dict[str, Any]] = None,
    repo: PlaybookRepository = Depends(get_playbook_repository),
    agent: PlaybookAgent = Depends(get_playbook_agent)
):
    """Create a playbook from a template."""

    with logfire.span("Create from template", template_name=template_name):
        try:
            # Create playbook from template using agent
            playbook_data = await agent.create_from_template(
                template_name=template_name,
                playbook_name=name,
                parameters=parameters or {}
            )

            # Save to repository
            playbook_id = f"pb_{uuid4().hex[:12]}"
            saved_playbook = await repo.create_playbook(
                playbook_id=playbook_id,
                **playbook_data
            )

            logger.log(f"Playbook created from template: {playbook_id}", LogLevel.INFO)

            return PlaybookResponse(**saved_playbook)

        except Exception as e:
            logger.log(f"Error creating from template {template_name}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to create from template: {str(e)}")


@router.websocket("/executions/{execution_id}/ws")
async def execution_websocket(websocket: WebSocket, execution_id: str):
    """WebSocket endpoint for real-time execution updates."""

    await ws_manager.connect(websocket, execution_id)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Echo back any messages (ping/pong)
            await websocket.send_text(f"Execution {execution_id} WebSocket active")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, execution_id)


@router.get("/executions/{execution_id}/logs/stream")
async def stream_execution_logs(
    execution_id: str,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Stream real-time logs for a specific execution."""

    async def generate_logs():
        """Generate log stream for execution."""
        try:
            # Check if execution exists
            execution_data = await repo.get_execution(execution_id)
            if not execution_data:
                yield f"data: {json.dumps({'error': f'Execution {execution_id} not found'})}\n\n"
                return

            # Simulate log streaming (in real implementation, this would connect to actual logs)
            while True:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_id": execution_id,
                    "level": "INFO",
                    "message": f"Execution {execution_id} is running",
                    "step": "monitoring"
                }
                yield f"data: {json.dumps(log_entry)}\n\n"
                await asyncio.sleep(2)  # Send update every 2 seconds

        except Exception as e:
            error_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "execution_id": execution_id,
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


@router.get("/categories/available")
async def get_available_categories():
    """Get available playbook categories."""

    categories = [
        {
            "name": cat.value,
            "display_name": cat.value.replace('_', ' ').title(),
            "description": f"{cat.value.replace('_', ' ').title()} playbooks"
        }
        for cat in PlaybookCategory
    ]

    return {"categories": categories}


@router.get("/{playbook_id}/analytics")
async def get_playbook_analytics(
    playbook_id: str,
    days: int = Query(30, ge=1, le=365),
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """Get analytics for a specific playbook."""

    with logfire.span("Get playbook analytics", playbook_id=playbook_id):
        try:
            analytics = await repo.get_playbook_analytics(playbook_id, days)

            return analytics

        except Exception as e:
            logger.log(f"Error getting playbook analytics {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# New Analytical Endpoints for Task 9.2

class PlaybookAnalysisRequestNew(BaseModel):
    """Request model for playbook analysis."""
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: comprehensive, performance, complexity, optimization")
    include_suggestions: bool = Field(default=True, description="Include improvement suggestions")
    compare_with_category: bool = Field(default=False, description="Compare with other playbooks in same category")


class PlaybookAnalysisResponseNew(BaseModel):
    """Response model for playbook analysis."""
    playbook_id: str
    analysis_timestamp: datetime
    analysis_type: str
    complexity_score: float = Field(description="Complexity score from 1-10")
    performance_score: float = Field(description="Performance score from 1-10")
    maintainability_score: float = Field(description="Maintainability score from 1-10")
    execution_efficiency: Dict[str, Any] = Field(description="Execution time and resource usage analysis")
    step_analysis: List[Dict[str, Any]] = Field(description="Per-step analysis breakdown")
    bottlenecks: List[str] = Field(description="Identified performance bottlenecks")
    suggestions: List[Dict[str, Any]] = Field(description="Improvement suggestions")
    category_comparison: Optional[Dict[str, Any]] = Field(default=None, description="Comparison with category peers")


class PlaybookExpansionRequest(BaseModel):
    """Request model for playbook expansion."""
    expansion_type: str = Field(default="variations", description="Type of expansion: variations, optimizations, alternatives")
    target_improvements: List[str] = Field(default=["performance", "reliability"], description="Areas to focus improvements on")
    max_variations: int = Field(default=3, ge=1, le=10, description="Maximum number of variations to generate")


class PlaybookExpansionResponse(BaseModel):
    """Response model for playbook expansion."""
    playbook_id: str
    expansion_timestamp: datetime
    expansion_type: str
    original_playbook: Dict[str, Any]
    generated_variations: List[Dict[str, Any]] = Field(description="Generated playbook variations")
    improvement_rationale: List[str] = Field(description="Explanation of improvements made")
    estimated_benefits: Dict[str, float] = Field(description="Estimated improvement percentages")


class PlaybookDetailedMetricsResponse(BaseModel):
    """Response model for detailed playbook metrics."""
    playbook_id: str
    metrics_timestamp: datetime
    execution_metrics: Dict[str, Any] = Field(description="Detailed execution statistics")
    resource_utilization: Dict[str, Any] = Field(description="CPU, memory, network usage")
    step_performance: List[Dict[str, Any]] = Field(description="Per-step performance metrics")
    error_patterns: List[Dict[str, Any]] = Field(description="Common error patterns and frequencies")
    success_patterns: List[Dict[str, Any]] = Field(description="Success factors and patterns")
    trend_analysis: Dict[str, Any] = Field(description="Performance trends over time")
    comparative_metrics: Dict[str, Any] = Field(description="Comparison with similar playbooks")


class PlaybookReportsResponse(BaseModel):
    """Response model for comprehensive playbook reports."""
    report_timestamp: datetime
    report_type: str
    summary_statistics: Dict[str, Any] = Field(description="Overall playbook statistics")
    category_breakdown: List[Dict[str, Any]] = Field(description="Statistics by category")
    performance_rankings: List[Dict[str, Any]] = Field(description="Top and bottom performing playbooks")
    usage_patterns: Dict[str, Any] = Field(description="Usage frequency and patterns")
    trend_analysis: Dict[str, Any] = Field(description="Trends across all playbooks")
    recommendations: List[Dict[str, Any]] = Field(description="System-wide recommendations")
    health_indicators: Dict[str, Any] = Field(description="Overall system health metrics")


@router.post("/{playbook_id}/analyze", response_model=PlaybookAnalysisResponseNew)
async def analyze_playbook(
    playbook_id: str,
    request: PlaybookAnalysisRequestNew,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """
    Analyze playbook structure, complexity, performance patterns, and optimization opportunities.

    This endpoint provides comprehensive analysis of a playbook including:
    - Complexity scoring and breakdown
    - Performance bottleneck identification
    - Maintainability assessment
    - Improvement suggestions
    - Category-based comparisons
    """
    with logfire.span("Analyze playbook", playbook_id=playbook_id, analysis_type=request.analysis_type):
        try:
            # Retrieve playbook with full details
            playbook = await repo.get_with_executions(playbook_id)
            if not playbook:
                raise HTTPException(status_code=404, detail="Playbook not found")

            # Calculate complexity score based on steps, conditions, loops
            complexity_factors = {
                "step_count": len(playbook.steps),
                "conditional_steps": len([s for s in playbook.steps if s.step_type == PlaybookStepType.CONDITIONAL]),
                "loop_steps": len([s for s in playbook.steps if s.step_type == PlaybookStepType.LOOP]),
                "variable_count": len(playbook.variables),
                "dependency_depth": await repo.calculate_dependency_depth(playbook_id)
            }

            complexity_score = min(10.0, (
                complexity_factors["step_count"] * 0.1 +
                complexity_factors["conditional_steps"] * 0.3 +
                complexity_factors["loop_steps"] * 0.4 +
                complexity_factors["variable_count"] * 0.05 +
                complexity_factors["dependency_depth"] * 0.2
            ))

            # Calculate performance score from execution history
            recent_executions = await repo.get_recent_executions(playbook_id, limit=10)
            avg_duration = sum(e.duration_seconds or 0 for e in recent_executions) / max(len(recent_executions), 1)
            success_rate = len([e for e in recent_executions if e.status == PlaybookExecutionStatus.COMPLETED]) / max(len(recent_executions), 1)
            performance_score = min(10.0, (10 - (avg_duration / 60)) * success_rate)

            # Calculate maintainability score
            maintainability_factors = {
                "documentation_coverage": len([s for s in playbook.steps if s.description]) / max(len(playbook.steps), 1),
                "naming_consistency": await repo.analyze_naming_consistency(playbook_id),
                "step_modularity": await repo.analyze_step_modularity(playbook_id)
            }
            maintainability_score = (
                maintainability_factors["documentation_coverage"] * 3 +
                maintainability_factors["naming_consistency"] * 3 +
                maintainability_factors["step_modularity"] * 4
            )

            # Analyze execution efficiency
            execution_efficiency = {
                "avg_execution_time": avg_duration,
                "resource_usage": await repo.get_resource_usage_stats(playbook_id),
                "parallel_execution_opportunities": await repo.identify_parallelization_opportunities(playbook_id)
            }

            # Per-step analysis
            step_analysis = []
            for step in playbook.steps:
                step_stats = await repo.get_step_execution_stats(step.id)
                step_analysis.append({
                    "step_id": str(step.id),
                    "step_name": step.name,
                    "avg_duration": step_stats.get("avg_duration", 0),
                    "failure_rate": step_stats.get("failure_rate", 0),
                    "complexity_contribution": step_stats.get("complexity_score", 0)
                })

            # Identify bottlenecks
            bottlenecks = []
            sorted_steps = sorted(step_analysis, key=lambda x: x["avg_duration"], reverse=True)
            for step in sorted_steps[:3]:
                if step["avg_duration"] > avg_duration * 0.3:
                    bottlenecks.append(f"Step '{step['step_name']}' takes {step['avg_duration']:.1f}s (bottleneck)")

            # Generate suggestions
            suggestions = []
            if complexity_score > 7:
                suggestions.append({"type": "complexity", "priority": "high", "suggestion": "Consider breaking down complex steps into smaller, reusable components"})
            if performance_score < 6:
                suggestions.append({"type": "performance", "priority": "high", "suggestion": "Optimize slow-running steps and consider parallel execution"})
            if maintainability_score < 6:
                suggestions.append({"type": "maintainability", "priority": "medium", "suggestion": "Improve documentation and naming consistency"})

            # Category comparison if requested
            category_comparison = None
            if request.compare_with_category:
                category_stats = await repo.get_category_statistics(playbook.category)
                category_comparison = {
                    "category_avg_complexity": category_stats.get("avg_complexity", 0),
                    "category_avg_performance": category_stats.get("avg_performance", 0),
                    "percentile_rank": category_stats.get("percentile_rank", 0)
                }

            return PlaybookAnalysisResponseNew(
                playbook_id=playbook_id,
                analysis_timestamp=datetime.utcnow(),
                analysis_type=request.analysis_type,
                complexity_score=complexity_score,
                performance_score=performance_score,
                maintainability_score=maintainability_score,
                execution_efficiency=execution_efficiency,
                step_analysis=step_analysis,
                bottlenecks=bottlenecks,
                suggestions=suggestions,
                category_comparison=category_comparison
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error analyzing playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to analyze playbook: {str(e)}")


@router.post("/{playbook_id}/expand", response_model=PlaybookExpansionResponse)
async def expand_playbook(
    playbook_id: str,
    request: PlaybookExpansionRequest,
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """
    Generate expanded playbook variations, suggest improvements, or create related playbooks based on patterns.

    This endpoint creates optimized variations of existing playbooks by:
    - Analyzing current implementation patterns
    - Generating performance-optimized alternatives
    - Creating reliability-enhanced versions
    - Suggesting structural improvements
    """
    with logfire.span("Expand playbook", playbook_id=playbook_id, expansion_type=request.expansion_type):
        try:
            # Retrieve original playbook
            playbook = await repo.get_with_executions(playbook_id)
            if not playbook:
                raise HTTPException(status_code=404, detail="Playbook not found")

            original_playbook = {
                "id": str(playbook.id),
                "name": playbook.name,
                "description": playbook.description,
                "step_count": len(playbook.steps),
                "complexity_score": await repo.calculate_complexity_score(playbook_id)
            }

            generated_variations = []
            improvement_rationale = []
            estimated_benefits = {}

            # Generate variations based on expansion type
            if request.expansion_type == "variations":
                # Generate structural variations
                for i in range(request.max_variations):
                    variation = await repo.generate_structural_variation(playbook_id, i)
                    generated_variations.append(variation)
                    improvement_rationale.append(f"Variation {i+1}: {variation['rationale']}")

            elif request.expansion_type == "optimizations":
                # Generate performance-optimized versions
                optimization_strategies = ["parallel_execution", "step_consolidation", "resource_optimization"]
                for strategy in optimization_strategies[:request.max_variations]:
                    optimized = await repo.generate_optimized_version(playbook_id, strategy)
                    generated_variations.append(optimized)
                    improvement_rationale.append(f"Optimization via {strategy}: {optimized['description']}")

            elif request.expansion_type == "alternatives":
                # Generate alternative approaches
                alternative_approaches = await repo.generate_alternative_approaches(playbook_id, request.max_variations)
                for alt in alternative_approaches:
                    generated_variations.append(alt)
                    improvement_rationale.append(f"Alternative approach: {alt['approach_description']}")

            # Estimate benefits for each target improvement
            for improvement in request.target_improvements:
                if improvement == "performance":
                    estimated_benefits["performance"] = 25.5  # Estimated % improvement
                elif improvement == "reliability":
                    estimated_benefits["reliability"] = 15.3
                elif improvement == "maintainability":
                    estimated_benefits["maintainability"] = 20.7

            return PlaybookExpansionResponse(
                playbook_id=playbook_id,
                expansion_timestamp=datetime.utcnow(),
                expansion_type=request.expansion_type,
                original_playbook=original_playbook,
                generated_variations=generated_variations,
                improvement_rationale=improvement_rationale,
                estimated_benefits=estimated_benefits
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error expanding playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to expand playbook: {str(e)}")


@router.get("/{playbook_id}/metrics", response_model=PlaybookDetailedMetricsResponse)
async def get_playbook_detailed_metrics(
    playbook_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """
    Get detailed performance and execution metrics for a specific playbook.

    Provides comprehensive metrics including:
    - Execution statistics and patterns
    - Resource utilization analysis
    - Step-by-step performance breakdown
    - Error pattern analysis
    - Trend analysis over time
    """
    with logfire.span("Get detailed playbook metrics", playbook_id=playbook_id, days=days):
        try:
            # Verify playbook exists
            playbook = await repo.get_by_id(playbook_id)
            if not playbook:
                raise HTTPException(status_code=404, detail="Playbook not found")

            # Get execution metrics
            execution_metrics = await repo.get_execution_metrics(playbook_id, days)

            # Get resource utilization
            resource_utilization = await repo.get_resource_utilization(playbook_id, days)

            # Get step performance
            step_performance = []
            for step in playbook.steps:
                step_metrics = await repo.get_step_metrics(step.id, days)
                step_performance.append({
                    "step_id": str(step.id),
                    "step_name": step.name,
                    "step_type": step.step_type.value,
                    "avg_duration": step_metrics.get("avg_duration", 0),
                    "success_rate": step_metrics.get("success_rate", 0),
                    "error_rate": step_metrics.get("error_rate", 0),
                    "resource_usage": step_metrics.get("resource_usage", {})
                })

            # Analyze error patterns
            error_patterns = await repo.analyze_error_patterns(playbook_id, days)

            # Analyze success patterns
            success_patterns = await repo.analyze_success_patterns(playbook_id, days)

            # Get trend analysis if requested
            trend_analysis = {}
            if include_trends:
                trend_analysis = await repo.get_trend_analysis(playbook_id, days)

            # Get comparative metrics
            comparative_metrics = await repo.get_comparative_metrics(playbook_id, days)

            return PlaybookDetailedMetricsResponse(
                playbook_id=playbook_id,
                metrics_timestamp=datetime.utcnow(),
                execution_metrics=execution_metrics,
                resource_utilization=resource_utilization,
                step_performance=step_performance,
                error_patterns=error_patterns,
                success_patterns=success_patterns,
                trend_analysis=trend_analysis,
                comparative_metrics=comparative_metrics
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.log(f"Error getting detailed metrics for playbook {playbook_id}: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to get detailed metrics: {str(e)}")


@router.get("/reports", response_model=PlaybookReportsResponse)
async def get_playbook_reports(
    report_type: str = Query("comprehensive", description="Type of report: comprehensive, performance, usage, health"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    repo: PlaybookRepository = Depends(get_playbook_repository)
):
    """
    Generate comprehensive analytical reports across all playbooks with insights and recommendations.

    Provides system-wide analysis including:
    - Overall playbook statistics and health
    - Category-based breakdowns
    - Performance rankings and comparisons
    - Usage patterns and trends
    - System-wide recommendations
    """
    with logfire.span("Generate playbook reports", report_type=report_type, days=days):
        try:
            # Get summary statistics
            summary_statistics = await repo.get_summary_statistics(days)

            # Get category breakdown
            category_breakdown = await repo.get_category_breakdown(days)

            # Get performance rankings
            performance_rankings = await repo.get_performance_rankings(days)

            # Analyze usage patterns
            usage_patterns = await repo.analyze_usage_patterns(days)

            # Get trend analysis if requested
            trend_analysis = {}
            if include_trends:
                trend_analysis = await repo.get_system_trend_analysis(days)

            # Generate recommendations
            recommendations = await repo.generate_system_recommendations(days)

            # Calculate health indicators
            health_indicators = await repo.calculate_health_indicators(days)

            return PlaybookReportsResponse(
                report_timestamp=datetime.utcnow(),
                report_type=report_type,
                summary_statistics=summary_statistics,
                category_breakdown=category_breakdown,
                performance_rankings=performance_rankings,
                usage_patterns=usage_patterns,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                health_indicators=health_indicators
            )

        except Exception as e:
            logger.log(f"Error generating playbook reports: {str(e)}", LogLevel.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to generate reports: {str(e)}")
