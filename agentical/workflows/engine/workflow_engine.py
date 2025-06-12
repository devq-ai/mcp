"""
Workflow Engine Core for Agentical

This module provides the core workflow orchestration engine that manages
workflow execution, state transitions, and integration with agents and tools.

Features:
- Workflow orchestration and execution management
- State persistence and recovery
- Agent and tool integration
- Error handling and retry logic
- Performance monitoring with Logfire
- Concurrent execution support
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Set
from enum import Enum
from contextlib import asynccontextmanager
import json

import logfire
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.exceptions import (
    WorkflowError,
    WorkflowExecutionError,
    WorkflowValidationError,
    WorkflowNotFoundError,
    AgentError,
    ValidationError
)
from ...core.logging import log_operation
from ...db.models.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStepExecution,
    WorkflowType,
    WorkflowStatus,
    ExecutionStatus,
    StepType,
    StepStatus
)
from ...db.repositories.workflow import AsyncWorkflowRepository
from .execution_context import ExecutionContext
from .step_executor import StepExecutor


class WorkflowEngine:
    """
    Core workflow orchestration engine.

    Manages workflow execution lifecycle, state persistence, and coordination
    between different workflow types and execution strategies.
    """

    def __init__(
        self,
        db_session: Union[Session, AsyncSession],
        max_concurrent_workflows: int = 10,
        default_timeout_minutes: int = 60,
        enable_monitoring: bool = True
    ):
        """Initialize the workflow engine."""
        self.db_session = db_session
        self.max_concurrent_workflows = max_concurrent_workflows
        self.default_timeout_minutes = default_timeout_minutes
        self.enable_monitoring = enable_monitoring

        # Repository for database operations
        if isinstance(db_session, AsyncSession):
            self.workflow_repo = AsyncWorkflowRepository(db_session)
        else:
            # Handle sync session if needed
            raise ValueError("Workflow engine requires AsyncSession")

        # Active executions tracking
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_contexts: Dict[str, ExecutionContext] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}

        # Step executor for individual step processing
        self.step_executor = StepExecutor(db_session)

        # Workflow type handlers
        self._workflow_handlers: Dict[WorkflowType, Callable] = {}
        self._register_default_handlers()

        logfire.info(
            "Workflow engine initialized",
            max_concurrent=max_concurrent_workflows,
            timeout_minutes=default_timeout_minutes
        )

    def _register_default_handlers(self) -> None:
        """Register default workflow type handlers."""
        # Will be populated when we implement standard workflow types
        pass

    def register_workflow_handler(
        self,
        workflow_type: WorkflowType,
        handler: Callable
    ) -> None:
        """Register a handler for a specific workflow type."""
        self._workflow_handlers[workflow_type] = handler
        logfire.info(
            "Workflow handler registered",
            workflow_type=workflow_type.value,
            handler=handler.__name__
        )

    async def execute_workflow(
        self,
        workflow_id: int,
        input_data: Optional[Dict[str, Any]] = None,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow with the given input data.

        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            execution_config: Configuration overrides for this execution

        Returns:
            WorkflowExecution: The execution instance

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist
            WorkflowExecutionError: If execution fails
            ValidationError: If input validation fails
        """
        with logfire.span(
            "Execute workflow",
            workflow_id=workflow_id,
            input_size=len(input_data or {})
        ):
            # Load workflow
            workflow = await self.workflow_repo.get(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(f"Workflow {workflow_id} not found")

            # Validate workflow is executable
            if workflow.status != WorkflowStatus.ACTIVE:
                raise WorkflowValidationError(
                    f"Workflow {workflow_id} is not active (status: {workflow.status.value})"
                )

            # Check concurrent execution limit
            if len(self._active_executions) >= self.max_concurrent_workflows:
                raise WorkflowExecutionError(
                    "Maximum concurrent workflow executions reached"
                )

            # Create execution record
            execution = await self._create_execution(
                workflow, input_data, execution_config
            )

            # Create execution context
            context = ExecutionContext(
                execution=execution,
                workflow=workflow,
                input_data=input_data or {},
                config=execution_config or {}
            )

            # Store in active executions
            execution_id = execution.execution_id
            self._active_executions[execution_id] = execution
            self._execution_contexts[execution_id] = context

            # Start execution task
            task = asyncio.create_task(
                self._execute_workflow_async(context)
            )
            self._execution_tasks[execution_id] = task

            logfire.info(
                "Workflow execution started",
                workflow_id=workflow_id,
                execution_id=execution_id,
                workflow_type=workflow.workflow_type.value
            )

            return execution

    async def _create_execution(
        self,
        workflow: Workflow,
        input_data: Optional[Dict[str, Any]],
        execution_config: Optional[Dict[str, Any]]
    ) -> WorkflowExecution:
        """Create a new workflow execution record."""
        execution_id = str(uuid.uuid4())

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            execution_id=execution_id,
            status=ExecutionStatus.PENDING,
            input_data=input_data or {},
            configuration=execution_config or {},
            created_at=datetime.utcnow()
        )

        # Save to database
        await self.workflow_repo.db_session.add(execution)
        await self.workflow_repo.db_session.commit()
        await self.workflow_repo.db_session.refresh(execution)

        return execution

    async def _execute_workflow_async(self, context: ExecutionContext) -> None:
        """Execute workflow asynchronously with proper error handling."""
        execution = context.execution
        workflow = context.workflow
        execution_id = execution.execution_id

        try:
            with logfire.span(
                "Workflow execution",
                workflow_id=workflow.id,
                execution_id=execution_id,
                workflow_type=workflow.workflow_type.value
            ):
                # Start execution
                await self._start_execution(execution)

                # Get workflow handler
                handler = self._workflow_handlers.get(workflow.workflow_type)
                if not handler:
                    raise WorkflowExecutionError(
                        f"No handler registered for workflow type: {workflow.workflow_type.value}"
                    )

                # Execute workflow using appropriate handler
                await handler(context)

                # Complete execution
                await self._complete_execution(execution, context.output_data)

        except Exception as e:
            logfire.error(
                "Workflow execution failed",
                workflow_id=workflow.id,
                execution_id=execution_id,
                error=str(e),
                error_type=type(e).__name__
            )
            await self._fail_execution(execution, str(e))

        finally:
            # Clean up
            self._cleanup_execution(execution_id)

    async def _start_execution(self, execution: WorkflowExecution) -> None:
        """Start workflow execution and update status."""
        execution.start_execution()
        await self.workflow_repo.update_execution_state(
            execution.id,
            ExecutionStatus.RUNNING,
            {}
        )

        logfire.info(
            "Workflow execution started",
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id
        )

    async def _complete_execution(
        self,
        execution: WorkflowExecution,
        output_data: Dict[str, Any]
    ) -> None:
        """Complete workflow execution successfully."""
        execution.complete_execution()
        execution.output_data = output_data

        await self.workflow_repo.update_execution_state(
            execution.id,
            ExecutionStatus.COMPLETED,
            {"output_data": output_data}
        )

        logfire.info(
            "Workflow execution completed",
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            duration_seconds=execution.duration.total_seconds() if execution.duration else 0
        )

    async def _fail_execution(self, execution: WorkflowExecution, error: str) -> None:
        """Fail workflow execution with error details."""
        execution.status = ExecutionStatus.FAILED
        execution.ended_at = datetime.utcnow()
        execution.error_message = error

        await self.workflow_repo.update_execution_state(
            execution.id,
            ExecutionStatus.FAILED,
            {"error_message": error}
        )

        logfire.error(
            "Workflow execution failed",
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            error=error
        )

    def _cleanup_execution(self, execution_id: str) -> None:
        """Clean up execution tracking data."""
        self._active_executions.pop(execution_id, None)
        self._execution_contexts.pop(execution_id, None)

        # Clean up task
        task = self._execution_tasks.pop(execution_id, None)
        if task and not task.done():
            task.cancel()

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running workflow execution.

        Args:
            execution_id: ID of the execution to cancel

        Returns:
            bool: True if cancellation was successful
        """
        with logfire.span("Cancel workflow execution", execution_id=execution_id):
            execution = self._active_executions.get(execution_id)
            if not execution:
                logfire.warning("Execution not found for cancellation", execution_id=execution_id)
                return False

            # Cancel the execution task
            task = self._execution_tasks.get(execution_id)
            if task:
                task.cancel()

            # Update execution status
            execution.cancel_execution()
            await self.workflow_repo.update_execution_state(
                execution.id,
                ExecutionStatus.CANCELLED,
                {}
            )

            # Clean up
            self._cleanup_execution(execution_id)

            logfire.info("Workflow execution cancelled", execution_id=execution_id)
            return True

    async def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running workflow execution.

        Args:
            execution_id: ID of the execution to pause

        Returns:
            bool: True if pause was successful
        """
        with logfire.span("Pause workflow execution", execution_id=execution_id):
            execution = self._active_executions.get(execution_id)
            if not execution:
                return False

            context = self._execution_contexts.get(execution_id)
            if context:
                context.pause()

            execution.pause_execution()
            await self.workflow_repo.update_execution_state(
                execution.id,
                ExecutionStatus.PAUSED,
                {}
            )

            logfire.info("Workflow execution paused", execution_id=execution_id)
            return True

    async def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused workflow execution.

        Args:
            execution_id: ID of the execution to resume

        Returns:
            bool: True if resume was successful
        """
        with logfire.span("Resume workflow execution", execution_id=execution_id):
            execution = self._active_executions.get(execution_id)
            if not execution:
                return False

            context = self._execution_contexts.get(execution_id)
            if context:
                context.resume()

            execution.resume_execution()
            await self.workflow_repo.update_execution_state(
                execution.id,
                ExecutionStatus.RUNNING,
                {}
            )

            logfire.info("Workflow execution resumed", execution_id=execution_id)
            return True

    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """Get the current status of a workflow execution."""
        execution = self._active_executions.get(execution_id)
        if execution:
            return execution.status

        # Check database for completed executions
        execution = await self.workflow_repo.find_one(
            {"execution_id": execution_id}
        )
        return execution.status if execution else None

    async def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all currently active workflow executions."""
        return list(self._active_executions.values())

    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get workflow engine performance metrics."""
        active_count = len(self._active_executions)

        # Get database metrics
        db_metrics = await self.workflow_repo.get_workflow_metrics()

        return {
            "active_executions": active_count,
            "max_concurrent": self.max_concurrent_workflows,
            "capacity_utilization": active_count / self.max_concurrent_workflows,
            "database_metrics": db_metrics,
            "engine_status": "healthy" if active_count < self.max_concurrent_workflows else "at_capacity"
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the workflow engine."""
        logfire.info("Workflow engine shutting down", active_executions=len(self._active_executions))

        # Cancel all active executions
        for execution_id in list(self._active_executions.keys()):
            await self.cancel_execution(execution_id)

        # Wait for tasks to complete
        if self._execution_tasks:
            await asyncio.gather(
                *self._execution_tasks.values(),
                return_exceptions=True
            )

        logfire.info("Workflow engine shutdown complete")


class WorkflowEngineFactory:
    """Factory for creating workflow engine instances."""

    @staticmethod
    def create_engine(
        db_session: AsyncSession,
        config: Optional[Dict[str, Any]] = None
    ) -> WorkflowEngine:
        """Create a workflow engine with the given configuration."""
        config = config or {}

        engine = WorkflowEngine(
            db_session=db_session,
            max_concurrent_workflows=config.get("max_concurrent_workflows", 10),
            default_timeout_minutes=config.get("default_timeout_minutes", 60),
            enable_monitoring=config.get("enable_monitoring", True)
        )

        logfire.info("Workflow engine created", config=config)
        return engine
