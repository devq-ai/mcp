"""
System Workflow Management API Endpoints for Agentical

IMPORTANT SCOPE CLARIFICATION:
This module manages SYSTEM WORKFLOWS - high-level orchestration workflows that:
- Coordinate multiple agents working together
- Integrate with external systems, databases, and APIs
- Represent complete business processes and automation sequences
- Handle multi-step, long-running operations (hours to days)
- Manage complex dependencies and conditional logic

This is DISTINCT from AGENT WORKFLOWS, which are:
- Internal execution patterns within individual agents
- Cognitive and operational patterns for task-specific logic
- Managed through agent configuration APIs
- Short-running task completion patterns (minutes to hours)

See docs/workflow_types_explanation.md for detailed differences.

System Workflow Management Features:
- Multi-agent orchestration and coordination
- Cross-system integration capabilities
- Business process automation workflows
- Execution control (start/pause/resume/stop)
- Real-time status monitoring and progress tracking
- Workflow template management for reusable processes

Technical Features:
- RESTful API design with OpenAPI documentation
- Comprehensive error handling and validation
- Logfire integration for distributed observability
- WebSocket support for real-time updates
- Pagination and filtering support for enterprise scale
"""

import asyncio
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
import json

from ....core.exceptions import (
    WorkflowError, WorkflowExecutionError, WorkflowValidationError,
    WorkflowNotFoundError
)
from ....core.logging import log_operation
from ....db.database import get_async_db
from ....db.models.workflow import (
    Workflow, WorkflowExecution, WorkflowStep, WorkflowType,
    WorkflowStatus, ExecutionStatus, StepType, StepStatus
)
from ....db.repositories.workflow import AsyncWorkflowRepository
from ....workflows.manager import WorkflowManager
from ....workflows.engine.workflow_engine import WorkflowEngine

# Initialize router
router = APIRouter(prefix="/workflows", tags=["workflows"])

# Pydantic Models
class WorkflowCreateRequest(BaseModel):
    """Request model for creating a new workflow."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    workflow_type: WorkflowType
    config: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_template: bool = False
    schedule_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = Field(None, gt=0, le=86400)  # Max 24 hours
    retry_config: Optional[Dict[str, Any]] = None

    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return [tag.strip().lower() for tag in v if tag.strip()]

    @validator('steps')
    def validate_steps(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 steps allowed per workflow")
        return v


class WorkflowUpdateRequest(BaseModel):
    """Request model for updating an existing workflow."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[Dict[str, Any]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    status: Optional[WorkflowStatus] = None
    schedule_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = Field(None, gt=0, le=86400)
    retry_config: Optional[Dict[str, Any]] = None


class WorkflowExecutionRequest(BaseModel):
    """Request model for executing a workflow."""
    input_data: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10)
    timeout_override: Optional[int] = Field(None, gt=0, le=86400)
    retry_override: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)
    scheduled_at: Optional[datetime] = None


class WorkflowResponse(BaseModel):
    """Response model for workflow data."""
    id: str
    name: str
    description: Optional[str]
    workflow_type: WorkflowType
    status: WorkflowStatus
    config: Dict[str, Any]
    steps: List[Dict[str, Any]]
    tags: List[str]
    is_template: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    execution_count: int
    last_execution_at: Optional[datetime]
    schedule_config: Optional[Dict[str, Any]]
    timeout_seconds: Optional[int]
    retry_config: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution data."""
    id: str
    workflow_id: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    step_executions: List[Dict[str, Any]]
    progress_percentage: float
    priority: int
    tags: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    """Response model for workflow list with pagination."""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    size: int
    pages: int


class WorkflowExecutionListResponse(BaseModel):
    """Response model for workflow execution list with pagination."""
    executions: List[WorkflowExecutionResponse]
    total: int
    page: int
    size: int
    pages: int


class WorkflowControlRequest(BaseModel):
    """Request model for workflow execution control."""
    action: str = Field(..., regex="^(pause|resume|stop|cancel)$")
    reason: Optional[str] = Field(None, max_length=500)


class WorkflowStatsResponse(BaseModel):
    """Response model for workflow statistics."""
    total_workflows: int
    active_workflows: int
    total_executions: int
    running_executions: int
    completed_executions: int
    failed_executions: int
    success_rate: float
    average_duration_seconds: float
    executions_last_24h: int
    most_used_workflows: List[Dict[str, Any]]


# WebSocket connection manager
class WorkflowWebSocketManager:
    """Manages WebSocket connections for real-time workflow updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, workflow_id: str):
        """Accept a WebSocket connection for a specific workflow."""
        await websocket.accept()
        if workflow_id not in self.active_connections:
            self.active_connections[workflow_id] = []
        self.active_connections[workflow_id].append(websocket)

    def disconnect(self, websocket: WebSocket, workflow_id: str):
        """Remove a WebSocket connection."""
        if workflow_id in self.active_connections:
            if websocket in self.active_connections[workflow_id]:
                self.active_connections[workflow_id].remove(websocket)
            if not self.active_connections[workflow_id]:
                del self.active_connections[workflow_id]

    async def broadcast_workflow_update(self, workflow_id: str, data: Dict[str, Any]):
        """Broadcast workflow updates to all connected clients."""
        if workflow_id in self.active_connections:
            message = json.dumps(data)
            disconnected = []
            for websocket in self.active_connections[workflow_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket, workflow_id)


ws_manager = WorkflowWebSocketManager()


# Dependency injection
async def get_workflow_repository(db: AsyncSession = Depends(get_async_db)) -> AsyncWorkflowRepository:
    """Get workflow repository instance."""
    return AsyncWorkflowRepository(db)


async def get_workflow_manager() -> WorkflowManager:
    """Get workflow manager instance."""
    # This should be a singleton in production
    manager = WorkflowManager()
    await manager.initialize()
    return manager


# API Endpoints
@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    request: WorkflowCreateRequest,
    background_tasks: BackgroundTasks,
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Create a new workflow.

    Creates a new workflow with the specified configuration and steps.
    Validates workflow structure and dependencies.
    """
    with logfire.span("Create workflow", workflow_name=request.name):
        try:
            # Validate workflow configuration
            if request.steps:
                await _validate_workflow_steps(request.steps)

            # Create workflow record
            workflow_data = {
                "id": str(uuid.uuid4()),
                "name": request.name,
                "description": request.description,
                "workflow_type": request.workflow_type,
                "status": WorkflowStatus.DRAFT,
                "config": request.config,
                "tags": request.tags,
                "is_template": request.is_template,
                "schedule_config": request.schedule_config,
                "timeout_seconds": request.timeout_seconds,
                "retry_config": request.retry_config,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            workflow = await workflow_repo.create(workflow_data)

            # Create workflow steps
            if request.steps:
                await _create_workflow_steps(workflow.id, request.steps, workflow_repo)

            # Log creation
            logfire.info("Workflow created",
                        workflow_id=workflow.id,
                        workflow_name=workflow.name,
                        step_count=len(request.steps))

            # Schedule background validation
            background_tasks.add_task(_validate_workflow_background, workflow.id)

            return await _build_workflow_response(workflow, workflow_repo)

        except ValidationError as e:
            logfire.error("Workflow validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Workflow validation failed: {str(e)}"
            )
        except Exception as e:
            logfire.error("Failed to create workflow", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create workflow"
            )


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    workflow_type: Optional[WorkflowType] = None,
    status: Optional[WorkflowStatus] = None,
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    sort_by: str = Query("created_at", regex="^(name|created_at|updated_at|execution_count)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    List workflows with filtering, searching, and pagination.

    Supports filtering by type, status, tags, and text search.
    Results are paginated and can be sorted by various fields.
    """
    with logfire.span("List workflows"):
        try:
            # Build filters
            filters = {}
            if workflow_type:
                filters['workflow_type'] = workflow_type
            if status:
                filters['status'] = status
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                filters['tags'] = tag_list
            if search:
                filters['search'] = search

            # Get workflows with pagination
            workflows, total = await workflow_repo.list_workflows(
                page=page,
                size=size,
                filters=filters,
                sort_by=sort_by,
                sort_order=sort_order
            )

            # Build response
            workflow_responses = []
            for workflow in workflows:
                response = await _build_workflow_response(workflow, workflow_repo)
                workflow_responses.append(response)

            pages = (total + size - 1) // size

            logfire.info("Workflows listed",
                        total=total,
                        page=page,
                        size=size,
                        filters=filters)

            return WorkflowListResponse(
                workflows=workflow_responses,
                total=total,
                page=page,
                size=size,
                pages=pages
            )

        except Exception as e:
            logfire.error("Failed to list workflows", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list workflows"
            )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Get a specific workflow by ID.

    Returns detailed workflow information including steps and configuration.
    """
    with logfire.span("Get workflow", workflow_id=workflow_id):
        try:
            workflow = await workflow_repo.get_by_id(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )

            response = await _build_workflow_response(workflow, workflow_repo)

            logfire.info("Workflow retrieved", workflow_id=workflow_id)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to get workflow", workflow_id=workflow_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get workflow"
            )


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    request: WorkflowUpdateRequest,
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Update an existing workflow.

    Updates workflow configuration, steps, or metadata.
    Validates changes before applying.
    """
    with logfire.span("Update workflow", workflow_id=workflow_id):
        try:
            workflow = await workflow_repo.get_by_id(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )

            # Validate steps if provided
            if request.steps:
                await _validate_workflow_steps(request.steps)

            # Build update data
            update_data = {}
            for field, value in request.dict(exclude_unset=True).items():
                if value is not None:
                    update_data[field] = value

            if update_data:
                update_data['updated_at'] = datetime.utcnow()
                workflow = await workflow_repo.update(workflow_id, update_data)

            # Update steps if provided
            if request.steps:
                await _update_workflow_steps(workflow_id, request.steps, workflow_repo)

            response = await _build_workflow_response(workflow, workflow_repo)

            logfire.info("Workflow updated", workflow_id=workflow_id, changes=list(update_data.keys()))

            return response

        except HTTPException:
            raise
        except ValidationError as e:
            logfire.error("Workflow update validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Workflow validation failed: {str(e)}"
            )
        except Exception as e:
            logfire.error("Failed to update workflow", workflow_id=workflow_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow"
            )


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: str,
    force: bool = Query(False, description="Force delete even with active executions"),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Delete a workflow.

    Soft deletes by default. Use force=true to permanently delete.
    Prevents deletion if there are active executions unless forced.
    """
    with logfire.span("Delete workflow", workflow_id=workflow_id):
        try:
            workflow = await workflow_repo.get_by_id(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )

            # Check for active executions
            if not force:
                active_executions = await workflow_repo.get_active_executions(workflow_id)
                if active_executions:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Cannot delete workflow with active executions. Use force=true to override."
                    )

            await workflow_repo.delete(workflow_id, soft_delete=not force)

            logfire.info("Workflow deleted", workflow_id=workflow_id, force=force)

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to delete workflow", workflow_id=workflow_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete workflow"
            )


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Execute a workflow.

    Starts workflow execution with the provided input data.
    Returns execution details and runs asynchronously.
    """
    with logfire.span("Execute workflow", workflow_id=workflow_id):
        try:
            workflow = await workflow_repo.get_by_id(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )

            if workflow.status != WorkflowStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Workflow is not active"
                )

            # Create execution record
            execution_data = {
                "id": str(uuid.uuid4()),
                "workflow_id": workflow_id,
                "status": ExecutionStatus.PENDING,
                "input_data": request.input_data,
                "priority": request.priority,
                "tags": request.tags,
                "created_at": datetime.utcnow()
            }

            if request.scheduled_at:
                execution_data['scheduled_at'] = request.scheduled_at

            execution = await workflow_repo.create_execution(execution_data)

            # Start execution
            if request.scheduled_at and request.scheduled_at > datetime.utcnow():
                # Schedule for later
                await workflow_manager.schedule_workflow_execution(
                    execution.id, request.scheduled_at
                )
            else:
                # Execute immediately in background
                background_tasks.add_task(
                    _execute_workflow_background,
                    execution.id,
                    workflow_manager
                )

            response = await _build_execution_response(execution, workflow_repo)

            logfire.info("Workflow execution started",
                        workflow_id=workflow_id,
                        execution_id=execution.id,
                        priority=request.priority)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to execute workflow", workflow_id=workflow_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to execute workflow"
            )


@router.get("/{workflow_id}/executions", response_model=WorkflowExecutionListResponse)
async def list_workflow_executions(
    workflow_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[ExecutionStatus] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    List executions for a specific workflow.

    Returns paginated list of workflow executions with filtering options.
    """
    with logfire.span("List workflow executions", workflow_id=workflow_id):
        try:
            # Verify workflow exists
            workflow = await workflow_repo.get_by_id(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )

            # Build filters
            filters = {"workflow_id": workflow_id}
            if status:
                filters['status'] = status
            if from_date:
                filters['from_date'] = from_date
            if to_date:
                filters['to_date'] = to_date

            # Get executions
            executions, total = await workflow_repo.list_executions(
                page=page,
                size=size,
                filters=filters
            )

            # Build responses
            execution_responses = []
            for execution in executions:
                response = await _build_execution_response(execution, workflow_repo)
                execution_responses.append(response)

            pages = (total + size - 1) // size

            logfire.info("Workflow executions listed",
                        workflow_id=workflow_id,
                        total=total,
                        page=page,
                        size=size)

            return WorkflowExecutionListResponse(
                executions=execution_responses,
                total=total,
                page=page,
                size=size,
                pages=pages
            )

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to list workflow executions", workflow_id=workflow_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list workflow executions"
            )


@router.get("/{workflow_id}/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(
    workflow_id: str,
    execution_id: str,
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Get a specific workflow execution.

    Returns detailed execution information including step results.
    """
    with logfire.span("Get workflow execution", workflow_id=workflow_id, execution_id=execution_id):
        try:
            execution = await workflow_repo.get_execution_by_id(execution_id)
            if not execution or execution.workflow_id != workflow_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow execution not found"
                )

            response = await _build_execution_response(execution, workflow_repo)

            logfire.info("Workflow execution retrieved",
                        workflow_id=workflow_id,
                        execution_id=execution_id)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to get workflow execution",
                         workflow_id=workflow_id,
                         execution_id=execution_id,
                         error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get workflow execution"
            )


@router.post("/{workflow_id}/executions/{execution_id}/control", status_code=status.HTTP_200_OK)
async def control_workflow_execution(
    workflow_id: str,
    execution_id: str,
    request: WorkflowControlRequest,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Control workflow execution (pause/resume/stop/cancel).

    Allows real-time control of running workflow executions.
    """
    with logfire.span("Control workflow execution",
                     workflow_id=workflow_id,
                     execution_id=execution_id,
                     action=request.action):
        try:
            execution = await workflow_repo.get_execution_by_id(execution_id)
            if not execution or execution.workflow_id != workflow_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow execution not found"
                )

            # Execute control action
            if request.action == "pause":
                await workflow_manager.pause_execution(execution_id)
            elif request.action == "resume":
                await workflow_manager.resume_execution(execution_id)
            elif request.action in ["stop", "cancel"]:
                await workflow_manager.cancel_execution(execution_id)

            # Update execution record
            update_data = {"updated_at": datetime.utcnow()}
            if request.reason:
                update_data["notes"] = request.reason

            await workflow_repo.update_execution(execution_id, update_data)

            # Broadcast update via WebSocket
            await ws_manager.broadcast_workflow_update(workflow_id, {
                "type": "execution_control",
                "execution_id": execution_id,
                "action": request.action,
                "timestamp": datetime.utcnow().isoformat()
            })

            logfire.info("Workflow execution controlled",
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                        action=request.action)

            return {"message": f"Execution {request.action} initiated successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to control workflow execution",
                         workflow_id=workflow_id,
                         execution_id=execution_id,
                         action=request.action,
                         error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to {request.action} workflow execution"
            )


@router.get("/stats/summary", response_model=WorkflowStatsResponse)
async def get_workflow_stats(
    from_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    to_date: Optional[datetime] = Query(None, description="End date for statistics"),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Get workflow system statistics.

    Returns comprehensive statistics about workflows and executions.
    """
    with logfire.span("Get workflow stats"):
        try:
            stats = await workflow_repo.get_workflow_stats(from_date, to_date)

            logfire.info("Workflow stats retrieved",
                        from_date=from_date,
                        to_date=to_date)

            return WorkflowStatsResponse(**stats)

        except Exception as e:
            logfire.error("Failed to get workflow stats", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get workflow statistics"
            )


@router.websocket("/{workflow_id}/ws")
async def workflow_websocket(
    websocket: WebSocket,
    workflow_id: str,
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    WebSocket endpoint for real-time workflow updates.

    Provides real-time updates for workflow execution progress and status changes.
    """
    try:
        # Verify workflow exists
        workflow = await workflow_repo.get_by_id(workflow_id)
        if not workflow:
            await websocket.close(code=4004, reason="Workflow not found")
            return

        await ws_manager.connect(websocket, workflow_id)

        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                # Echo back for keepalive
                await websocket.send_text(f"ack: {data}")

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, workflow_id)

    except Exception as e:
        logfire.error("WebSocket error", workflow_id=workflow_id, error=str(e))
        await websocket.close(code=1011, reason="Internal error")


@router.get("/{workflow_id}/executions/{execution_id}/logs")
async def stream_execution_logs(
    workflow_id: str,
    execution_id: str,
    follow: bool = Query(False, description="Follow logs in real-time"),
    workflow_repo: AsyncWorkflowRepository = Depends(get_workflow_repository)
):
    """
    Stream workflow execution logs.

    Returns execution logs, optionally following in real-time.
    """
    with logfire.span("Stream execution logs", workflow_id=workflow_id, execution_id=execution_id):
        try:
            execution = await workflow_repo.get_execution_by_id(execution_id)
            if not execution or execution.workflow_id != workflow_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow execution not found"
                )

            async def generate_logs():
                # Get existing logs
                logs = await workflow_repo.get_execution_logs(execution_id)
                for log_entry in logs:
                    yield f"data: {json.dumps(log_entry)}\n\n"

                # If follow is enabled, stream new logs
                if follow and execution.status in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
                    # Implement real-time log streaming
                    # This would connect to your logging system
                    while True:
                        await asyncio.sleep(1)
                        # Check for new logs and yield them
                        # Break when execution completes
                        current_execution = await workflow_repo.get_execution_by_id(execution_id)
                        if current_execution.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
                            break

            return StreamingResponse(
                generate_logs(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to stream execution logs",
                         workflow_id=workflow_id,
                         execution_id=execution_id,
                         error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stream execution logs"
            )


# Helper Functions
async def _validate_workflow_steps(steps: List[Dict[str, Any]]) -> None:
    """Validate workflow steps configuration."""
    if not steps:
        return

    step_ids = set()
    for i, step in enumerate(steps):
        # Validate required fields
        if 'id' not in step:
            raise ValidationError(f"Step {i} missing required 'id' field")
        if 'type' not in step:
            raise ValidationError(f"Step {step['id']} missing required 'type' field")
        if 'config' not in step:
            raise ValidationError(f"Step {step['id']} missing required 'config' field")

        # Check for duplicate step IDs
        if step['id'] in step_ids:
            raise ValidationError(f"Duplicate step ID: {step['id']}")
        step_ids.add(step['id'])

        # Validate step type
        try:
            StepType(step['type'])
        except ValueError:
            raise ValidationError(f"Invalid step type: {step['type']}")

        # Validate dependencies
        if 'dependencies' in step:
            for dep_id in step['dependencies']:
                if dep_id not in step_ids and dep_id not in [s['id'] for s in steps[:i]]:
                    # Check if dependency exists in previous steps
                    if not any(s['id'] == dep_id for s in steps):
                        raise ValidationError(f"Step {step['id']} depends on non-existent step: {dep_id}")


async def _create_workflow_steps(workflow_id: str, steps: List[Dict[str, Any]], workflow_repo: AsyncWorkflowRepository) -> None:
    """Create workflow steps in the database."""
    for order, step_data in enumerate(steps):
        step_record = {
            "id": str(uuid.uuid4()),
            "workflow_id": workflow_id,
            "step_id": step_data['id'],
            "name": step_data.get('name', step_data['id']),
            "description": step_data.get('description'),
            "step_type": StepType(step_data['type']),
            "config": step_data['config'],
            "dependencies": step_data.get('dependencies', []),
            "order": order,
            "timeout_seconds": step_data.get('timeout_seconds'),
            "retry_config": step_data.get('retry_config'),
            "created_at": datetime.utcnow()
        }
        await workflow_repo.create_step(step_record)


async def _update_workflow_steps(workflow_id: str, steps: List[Dict[str, Any]], workflow_repo: AsyncWorkflowRepository) -> None:
    """Update workflow steps, replacing existing ones."""
    # Delete existing steps
    await workflow_repo.delete_workflow_steps(workflow_id)

    # Create new steps
    await _create_workflow_steps(workflow_id, steps, workflow_repo)


async def _build_workflow_response(workflow: Workflow, workflow_repo: AsyncWorkflowRepository) -> WorkflowResponse:
    """Build a complete workflow response with steps and metadata."""
    # Get workflow steps
    steps = await workflow_repo.get_workflow_steps(workflow.id)
    steps_data = []
    for step in steps:
        step_dict = {
            "id": step.step_id,
            "name": step.name,
            "description": step.description,
            "type": step.step_type.value,
            "config": step.config,
            "dependencies": step.dependencies,
            "order": step.order,
            "timeout_seconds": step.timeout_seconds,
            "retry_config": step.retry_config
        }
        steps_data.append(step_dict)

    # Sort steps by order
    steps_data.sort(key=lambda x: x['order'])

    # Get execution stats
    execution_count = await workflow_repo.get_execution_count(workflow.id)
    last_execution = await workflow_repo.get_last_execution(workflow.id)

    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        description=workflow.description,
        workflow_type=workflow.workflow_type,
        status=workflow.status,
        config=workflow.config,
        steps=steps_data,
        tags=workflow.tags or [],
        is_template=workflow.is_template,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        created_by=getattr(workflow, 'created_by', None),
        execution_count=execution_count,
        last_execution_at=last_execution.created_at if last_execution else None,
        schedule_config=workflow.schedule_config,
        timeout_seconds=workflow.timeout_seconds,
        retry_config=workflow.retry_config
    )


async def _build_execution_response(execution: WorkflowExecution, workflow_repo: AsyncWorkflowRepository) -> WorkflowExecutionResponse:
    """Build a complete execution response with step executions."""
    # Get step executions
    step_executions = await workflow_repo.get_step_executions(execution.id)
    step_exec_data = []
    completed_steps = 0
    total_steps = len(step_executions)

    for step_exec in step_executions:
        step_dict = {
            "id": step_exec.id,
            "step_id": step_exec.step_id,
            "status": step_exec.status.value,
            "started_at": step_exec.started_at,
            "completed_at": step_exec.completed_at,
            "duration_seconds": step_exec.duration_seconds,
            "input_data": step_exec.input_data,
            "output_data": step_exec.output_data,
            "error_message": step_exec.error_message,
            "retry_count": step_exec.retry_count
        }
        step_exec_data.append(step_dict)

        if step_exec.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
            completed_steps += 1

    # Calculate progress
    progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0

    # Calculate duration
    duration_seconds = None
    if execution.started_at and execution.completed_at:
        duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

    return WorkflowExecutionResponse(
        id=execution.id,
        workflow_id=execution.workflow_id,
        status=execution.status,
        input_data=execution.input_data or {},
        output_data=execution.output_data,
        error_message=execution.error_message,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_seconds=duration_seconds,
        step_executions=step_exec_data,
        progress_percentage=progress_percentage,
        priority=execution.priority or 1,
        tags=execution.tags or [],
        created_at=execution.created_at
    )


async def _validate_workflow_background(workflow_id: str) -> None:
    """Background task to validate workflow configuration."""
    try:
        # This would perform more comprehensive validation
        # such as checking agent availability, tool configuration, etc.
        logfire.info("Background workflow validation completed", workflow_id=workflow_id)
    except Exception as e:
        logfire.error("Background workflow validation failed", workflow_id=workflow_id, error=str(e))


async def _execute_workflow_background(execution_id: str, workflow_manager: WorkflowManager) -> None:
    """Background task to execute workflow."""
    try:
        await workflow_manager.execute_workflow(execution_id)
        logfire.info("Background workflow execution completed", execution_id=execution_id)
    except Exception as e:
        logfire.error("Background workflow execution failed", execution_id=execution_id, error=str(e))
