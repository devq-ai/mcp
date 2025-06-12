"""
Workflow Manager for Agentical

This module provides the WorkflowManager class that handles workflow lifecycle
management, coordination between components, and high-level workflow operations.

Features:
- Workflow lifecycle management (create, start, stop, delete)
- Coordination between engine, registry, and database
- Workflow scheduling and queue management
- Performance monitoring and health checks
- Integration with FastAPI and dependency injection
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set
from enum import Enum
from contextlib import asynccontextmanager
import uuid

import logfire
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import (
    WorkflowError,
    WorkflowExecutionError,
    WorkflowValidationError,
    WorkflowNotFoundError,
    ConfigurationError
)
from ..core.logging import log_operation
from ..db.models.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    WorkflowType,
    WorkflowStatus,
    ExecutionStatus
)
from ..db.repositories.workflow import AsyncWorkflowRepository
from .engine.workflow_engine import WorkflowEngine, WorkflowEngineFactory
from .registry import WorkflowRegistry


class WorkflowManagerState(Enum):
    """Workflow manager operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class WorkflowScheduleType(Enum):
    """Workflow scheduling types."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    CRON = "cron"
    RECURRING = "recurring"
    EVENT_DRIVEN = "event_driven"


class WorkflowManager:
    """
    High-level workflow management and coordination.

    Provides a unified interface for workflow operations, managing the
    interaction between the workflow engine, registry, and database.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        engine_config: Optional[Dict[str, Any]] = None,
        registry_config: Optional[Dict[str, Any]] = None,
        enable_scheduling: bool = True,
        max_queued_workflows: int = 1000
    ):
        """Initialize the workflow manager."""
        self.db_session = db_session
        self.enable_scheduling = enable_scheduling
        self.max_queued_workflows = max_queued_workflows

        # Manager state
        self.state = WorkflowManagerState.INITIALIZING
        self.start_time = None
        self.shutdown_requested = False

        # Core components
        self.engine: Optional[WorkflowEngine] = None
        self.registry: Optional[WorkflowRegistry] = None
        self.workflow_repo = AsyncWorkflowRepository(db_session)

        # Configuration
        self.engine_config = engine_config or {}
        self.registry_config = registry_config or {}

        # Workflow queue and scheduling
        self.workflow_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queued_workflows)
        self.scheduled_workflows: Dict[str, Dict[str, Any]] = {}
        self.recurring_workflows: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.metrics = {
            "workflows_created": 0,
            "workflows_executed": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "workflows_cancelled": 0,
            "average_execution_time": 0.0,
            "queue_size": 0,
            "scheduled_count": 0
        }

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        logfire.info(
            "Workflow manager initialized",
            scheduling_enabled=enable_scheduling,
            max_queued=max_queued_workflows
        )

    async def initialize(self) -> None:
        """Initialize the workflow manager and its components."""
        with logfire.span("Initialize workflow manager"):
            try:
                # Initialize workflow registry
                self.registry = WorkflowRegistry(
                    db_session=self.db_session,
                    **self.registry_config
                )
                await self.registry.initialize()

                # Initialize workflow engine
                self.engine = WorkflowEngineFactory.create_engine(
                    db_session=self.db_session,
                    config=self.engine_config
                )

                # Register workflow handlers with engine
                await self._register_workflow_handlers()

                # Start background tasks
                if self.enable_scheduling:
                    await self._start_background_tasks()

                # Set state to running
                self.state = WorkflowManagerState.RUNNING
                self.start_time = datetime.utcnow()

                logfire.info("Workflow manager initialization complete")

            except Exception as e:
                logfire.error("Workflow manager initialization failed", error=str(e))
                self.state = WorkflowManagerState.STOPPED
                raise ConfigurationError(f"Failed to initialize workflow manager: {str(e)}")

    async def create_workflow(
        self,
        name: str,
        workflow_type: WorkflowType,
        description: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Workflow:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            workflow_type: Type of workflow
            description: Optional description
            steps: Optional workflow steps definition
            configuration: Optional workflow configuration
            tags: Optional tags for the workflow

        Returns:
            Workflow: The created workflow

        Raises:
            WorkflowValidationError: If validation fails
            WorkflowError: If creation fails
        """
        with logfire.span("Create workflow", name=name, workflow_type=workflow_type.value):
            # Validate workflow type is registered
            if not await self.registry.get_workflow_handler(workflow_type):
                raise WorkflowValidationError(f"Workflow type {workflow_type.value} is not registered")

            # Create workflow instance
            workflow = Workflow(
                name=name,
                workflow_type=workflow_type,
                description=description or f"{workflow_type.value} workflow",
                status=WorkflowStatus.DRAFT,
                configuration=configuration or {},
                created_at=datetime.utcnow()
            )

            # Add tags if provided
            if tags:
                workflow.add_tag(",".join(tags))

            # Save to database
            self.db_session.add(workflow)
            await self.db_session.commit()
            await self.db_session.refresh(workflow)

            # Create workflow steps if provided
            if steps:
                await self._create_workflow_steps(workflow, steps)

            self.metrics["workflows_created"] += 1

            logfire.info(
                "Workflow created",
                workflow_id=workflow.id,
                name=name,
                workflow_type=workflow_type.value
            )

            return workflow

    async def create_workflow_from_template(
        self,
        template_name: str,
        workflow_name: str,
        parameters: Dict[str, Any]
    ) -> Workflow:
        """Create a workflow from a template."""
        return await self.registry.create_workflow_from_template(
            template_name=template_name,
            workflow_name=workflow_name,
            parameters=parameters
        )

    async def execute_workflow(
        self,
        workflow_id: int,
        input_data: Optional[Dict[str, Any]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        schedule_type: WorkflowScheduleType = WorkflowScheduleType.IMMEDIATE,
        schedule_time: Optional[datetime] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            execution_config: Configuration overrides for this execution
            schedule_type: When to execute the workflow
            schedule_time: Specific time for scheduled execution

        Returns:
            WorkflowExecution: The execution instance

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist
            WorkflowExecutionError: If execution fails
        """
        with logfire.span("Execute workflow", workflow_id=workflow_id, schedule_type=schedule_type.value):
            # Validate workflow exists and is executable
            workflow = await self.workflow_repo.get(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(f"Workflow {workflow_id} not found")

            if workflow.status != WorkflowStatus.ACTIVE:
                raise WorkflowValidationError(
                    f"Workflow {workflow_id} is not active (status: {workflow.status.value})"
                )

            # Handle scheduling
            if schedule_type == WorkflowScheduleType.IMMEDIATE:
                return await self._execute_workflow_immediately(
                    workflow_id, input_data, execution_config
                )
            else:
                return await self._schedule_workflow_execution(
                    workflow_id, input_data, execution_config, schedule_type, schedule_time
                )

    async def _execute_workflow_immediately(
        self,
        workflow_id: int,
        input_data: Optional[Dict[str, Any]],
        execution_config: Optional[Dict[str, Any]]
    ) -> WorkflowExecution:
        """Execute a workflow immediately."""
        if self.state != WorkflowManagerState.RUNNING:
            raise WorkflowExecutionError("Workflow manager is not running")

        execution = await self.engine.execute_workflow(
            workflow_id=workflow_id,
            input_data=input_data,
            execution_config=execution_config
        )

        self.metrics["workflows_executed"] += 1
        return execution

    async def _schedule_workflow_execution(
        self,
        workflow_id: int,
        input_data: Optional[Dict[str, Any]],
        execution_config: Optional[Dict[str, Any]],
        schedule_type: WorkflowScheduleType,
        schedule_time: Optional[datetime]
    ) -> WorkflowExecution:
        """Schedule a workflow for future execution."""
        if not self.enable_scheduling:
            raise WorkflowExecutionError("Workflow scheduling is disabled")

        # Create a placeholder execution for tracking
        execution_id = str(uuid.uuid4())
        schedule_data = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "input_data": input_data,
            "execution_config": execution_config,
            "schedule_type": schedule_type.value,
            "schedule_time": schedule_time.isoformat() if schedule_time else None,
            "created_at": datetime.utcnow().isoformat()
        }

        # Store in scheduled workflows
        self.scheduled_workflows[execution_id] = schedule_data
        self.metrics["scheduled_count"] += 1

        logfire.info(
            "Workflow scheduled",
            execution_id=execution_id,
            workflow_id=workflow_id,
            schedule_type=schedule_type.value,
            schedule_time=schedule_time.isoformat() if schedule_time else None
        )

        # For now, return a mock execution
        # In production, this would create a proper scheduled execution record
        from ..db.models.workflow import WorkflowExecution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=ExecutionStatus.PENDING,
            input_data=input_data or {},
            configuration=execution_config or {},
            created_at=datetime.utcnow()
        )

        return execution

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running or scheduled workflow execution."""
        with logfire.span("Cancel workflow execution", execution_id=execution_id):
            # Try to cancel running execution
            if await self.engine.cancel_execution(execution_id):
                self.metrics["workflows_cancelled"] += 1
                return True

            # Try to cancel scheduled execution
            if execution_id in self.scheduled_workflows:
                del self.scheduled_workflows[execution_id]
                self.metrics["scheduled_count"] -= 1
                logfire.info("Scheduled workflow cancelled", execution_id=execution_id)
                return True

            return False

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running workflow execution."""
        return await self.engine.pause_execution(execution_id)

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution."""
        return await self.engine.resume_execution(execution_id)

    async def get_workflow(self, workflow_id: int) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return await self.registry.get_workflow_cached(workflow_id)

    async def get_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        workflow_type: Optional[WorkflowType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Workflow]:
        """Get workflows with optional filtering."""
        return await self.workflow_repo.get_all(
            filters={"status": status, "workflow_type": workflow_type},
            limit=limit,
            offset=offset
        )

    async def search_workflows(
        self,
        query: str,
        workflow_type: Optional[WorkflowType] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50
    ) -> List[Workflow]:
        """Search for workflows."""
        return await self.registry.search_workflows(
            query=query,
            workflow_type=workflow_type,
            status=status,
            limit=limit
        )

    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """Get the status of a workflow execution."""
        return await self.engine.get_execution_status(execution_id)

    async def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active workflow executions."""
        return await self.engine.get_active_executions()

    async def publish_workflow(self, workflow_id: int) -> bool:
        """Publish a workflow to make it available for execution."""
        workflow = await self.workflow_repo.get(workflow_id)
        if not workflow:
            return False

        workflow.publish()
        await self.workflow_repo.update(workflow_id, {"status": WorkflowStatus.ACTIVE})

        logfire.info("Workflow published", workflow_id=workflow_id)
        return True

    async def archive_workflow(self, workflow_id: int) -> bool:
        """Archive a workflow."""
        workflow = await self.workflow_repo.get(workflow_id)
        if not workflow:
            return False

        workflow.archive()
        await self.workflow_repo.update(workflow_id, {"status": WorkflowStatus.ARCHIVED})

        logfire.info("Workflow archived", workflow_id=workflow_id)
        return True

    async def delete_workflow(self, workflow_id: int) -> bool:
        """Delete a workflow."""
        # Check if workflow has active executions
        active_executions = await self.workflow_repo.get_execution_history(
            workflow_id, status=ExecutionStatus.RUNNING
        )

        if active_executions:
            raise WorkflowExecutionError(
                f"Cannot delete workflow {workflow_id} with active executions"
            )

        success = await self.workflow_repo.delete(workflow_id)
        if success:
            await self.registry.invalidate_workflow_cache(workflow_id)
            logfire.info("Workflow deleted", workflow_id=workflow_id)

        return success

    async def get_manager_status(self) -> Dict[str, Any]:
        """Get workflow manager status and metrics."""
        engine_metrics = await self.engine.get_execution_metrics() if self.engine else {}
        registry_metrics = await self.registry.get_registry_metrics() if self.registry else {}

        return {
            "state": self.state.value,
            "uptime_seconds": (
                (datetime.utcnow() - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            "metrics": self.metrics,
            "engine_metrics": engine_metrics,
            "registry_metrics": registry_metrics,
            "queue_size": self.workflow_queue.qsize(),
            "scheduled_workflows": len(self.scheduled_workflows),
            "background_tasks": len(self.background_tasks)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the workflow manager."""
        health = {
            "status": "healthy",
            "checks": {
                "manager_state": self.state == WorkflowManagerState.RUNNING,
                "engine_available": self.engine is not None,
                "registry_available": self.registry is not None,
                "database_connected": True,  # Simplified check
                "queue_not_full": self.workflow_queue.qsize() < self.max_queued_workflows
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Determine overall health
        if not all(health["checks"].values()):
            health["status"] = "unhealthy"

        return health

    async def _create_workflow_steps(
        self,
        workflow: Workflow,
        steps_data: List[Dict[str, Any]]
    ) -> None:
        """Create workflow steps from step definitions."""
        for i, step_data in enumerate(steps_data):
            step = WorkflowStep(
                workflow_id=workflow.id,
                name=step_data.get("name", f"Step {i+1}"),
                step_type=step_data.get("type", "agent_task"),
                step_order=i + 1,
                configuration=step_data.get("configuration", {}),
                depends_on=step_data.get("depends_on"),
                is_conditional=step_data.get("is_conditional", False)
            )
            self.db_session.add(step)

        await self.db_session.commit()

    async def _register_workflow_handlers(self) -> None:
        """Register workflow handlers with the engine."""
        registered_types = await self.registry.get_registered_types()

        for workflow_type in registered_types:
            handler = await self.registry.get_workflow_handler(workflow_type)
            if handler:
                self.engine.register_workflow_handler(workflow_type, handler)

    async def _start_background_tasks(self) -> None:
        """Start background tasks for scheduling and maintenance."""
        # Scheduler task
        scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.background_tasks.add(scheduler_task)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.add(cleanup_task)

        logfire.info("Background tasks started", task_count=len(self.background_tasks))

    async def _scheduler_loop(self) -> None:
        """Background scheduler loop."""
        while not self.shutdown_requested:
            try:
                await self._process_scheduled_workflows()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logfire.error("Scheduler loop error", error=str(e))
                await asyncio.sleep(30)  # Back off on error

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self.shutdown_requested:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except Exception as e:
                logfire.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(600)  # Back off on error

    async def _process_scheduled_workflows(self) -> None:
        """Process scheduled workflows that are ready to execute."""
        now = datetime.utcnow()
        ready_executions = []

        for execution_id, schedule_data in list(self.scheduled_workflows.items()):
            schedule_time_str = schedule_data.get("schedule_time")
            if schedule_time_str:
                schedule_time = datetime.fromisoformat(schedule_time_str)
                if now >= schedule_time:
                    ready_executions.append(execution_id)

        # Execute ready workflows
        for execution_id in ready_executions:
            schedule_data = self.scheduled_workflows.pop(execution_id)
            asyncio.create_task(self._execute_scheduled_workflow(schedule_data))

    async def _execute_scheduled_workflow(self, schedule_data: Dict[str, Any]) -> None:
        """Execute a scheduled workflow."""
        try:
            await self._execute_workflow_immediately(
                workflow_id=schedule_data["workflow_id"],
                input_data=schedule_data["input_data"],
                execution_config=schedule_data["execution_config"]
            )
        except Exception as e:
            logfire.error(
                "Scheduled workflow execution failed",
                execution_id=schedule_data["execution_id"],
                error=str(e)
            )

    async def _cleanup_expired_data(self) -> None:
        """Clean up expired data and caches."""
        if self.registry:
            await self.registry.cleanup_cache()

        # Clean up old scheduled workflows (older than 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        expired_schedules = [
            execution_id for execution_id, schedule_data in self.scheduled_workflows.items()
            if datetime.fromisoformat(schedule_data["created_at"]) < cutoff
        ]

        for execution_id in expired_schedules:
            del self.scheduled_workflows[execution_id]

        if expired_schedules:
            logfire.info("Expired schedules cleaned up", count=len(expired_schedules))

    async def pause(self) -> None:
        """Pause the workflow manager."""
        self.state = WorkflowManagerState.PAUSED
        logfire.info("Workflow manager paused")

    async def resume(self) -> None:
        """Resume the workflow manager."""
        self.state = WorkflowManagerState.RUNNING
        logfire.info("Workflow manager resumed")

    async def shutdown(self) -> None:
        """Shutdown the workflow manager gracefully."""
        logfire.info("Workflow manager shutting down")
        self.state = WorkflowManagerState.SHUTTING_DOWN
        self.shutdown_requested = True

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Shutdown components
        if self.engine:
            await self.engine.shutdown()

        if self.registry:
            await self.registry.shutdown()

        self.state = WorkflowManagerState.STOPPED
        logfire.info("Workflow manager shutdown complete")

    def __repr__(self) -> str:
        """String representation of the workflow manager."""
        return (
            f"WorkflowManager(state={self.state.value}, "
            f"workflows_created={self.metrics['workflows_created']}, "
            f"active_executions={len(self.engine.get_active_executions() if self.engine else [])})"
        )
