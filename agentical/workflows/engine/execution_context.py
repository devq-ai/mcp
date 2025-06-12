"""
Execution Context for Workflow Engine

This module provides the ExecutionContext class that manages workflow execution
state, variables, and coordination between workflow steps and components.

Features:
- Execution state management and persistence
- Variable storage and templating
- Step coordination and dependency tracking
- Error handling and recovery
- Performance monitoring
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set
from enum import Enum
from contextlib import asynccontextmanager
import uuid

import logfire

from ...core.exceptions import (
    WorkflowExecutionError,
    WorkflowValidationError,
    ValidationError
)
from ...db.models.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStepExecution,
    ExecutionStatus,
    StepStatus
)


class ExecutionPhase(Enum):
    """Execution phase tracking."""
    INITIALIZATION = "initialization"
    EXECUTION = "execution"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"
    CLEANUP = "cleanup"


class ExecutionContext:
    """
    Manages workflow execution state and coordination.

    Provides a centralized context for workflow execution that tracks
    state, variables, step progress, and handles coordination between
    different workflow components.
    """

    def __init__(
        self,
        execution: WorkflowExecution,
        workflow: Workflow,
        input_data: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """Initialize execution context."""
        self.execution = execution
        self.workflow = workflow
        self.input_data = input_data
        self.config = config

        # Execution state
        self.phase = ExecutionPhase.INITIALIZATION
        self.is_paused = False
        self.is_cancelled = False
        self.start_time = datetime.utcnow()
        self.last_checkpoint = None

        # Variable storage
        self.variables: Dict[str, Any] = {}
        self.step_results: Dict[int, Any] = {}
        self.global_context: Dict[str, Any] = {}

        # Step tracking
        self.current_step: Optional[WorkflowStep] = None
        self.completed_steps: Set[int] = set()
        self.failed_steps: Set[int] = set()
        self.skipped_steps: Set[int] = set()

        # Output data
        self.output_data: Dict[str, Any] = {}
        self.error_details: Optional[Dict[str, Any]] = None

        # Performance tracking
        self.step_durations: Dict[int, timedelta] = {}
        self.checkpoint_times: List[datetime] = []

        # Events and hooks
        self.event_handlers: Dict[str, List[callable]] = {}
        self.step_hooks: Dict[str, List[callable]] = {}

        # Initialize variables with input data
        self.variables.update(input_data)
        self.global_context.update(config)

        logfire.info(
            "Execution context initialized",
            execution_id=execution.execution_id,
            workflow_id=workflow.id,
            input_keys=list(input_data.keys())
        )

    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in the execution context."""
        self.variables[key] = value
        logfire.debug(
            "Variable set",
            execution_id=self.execution.execution_id,
            key=key,
            value_type=type(value).__name__
        )

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the execution context."""
        return self.variables.get(key, default)

    def has_variable(self, key: str) -> bool:
        """Check if a variable exists in the context."""
        return key in self.variables

    def remove_variable(self, key: str) -> bool:
        """Remove a variable from the context."""
        if key in self.variables:
            del self.variables[key]
            logfire.debug(
                "Variable removed",
                execution_id=self.execution.execution_id,
                key=key
            )
            return True
        return False

    def set_step_result(self, step_id: int, result: Any) -> None:
        """Store the result of a completed step."""
        self.step_results[step_id] = result
        logfire.debug(
            "Step result stored",
            execution_id=self.execution.execution_id,
            step_id=step_id,
            result_type=type(result).__name__
        )

    def get_step_result(self, step_id: int, default: Any = None) -> Any:
        """Get the result of a completed step."""
        return self.step_results.get(step_id, default)

    def mark_step_completed(self, step_id: int, duration: Optional[timedelta] = None) -> None:
        """Mark a step as completed."""
        self.completed_steps.add(step_id)
        if duration:
            self.step_durations[step_id] = duration

        logfire.info(
            "Step completed",
            execution_id=self.execution.execution_id,
            step_id=step_id,
            duration_seconds=duration.total_seconds() if duration else None
        )

    def mark_step_failed(self, step_id: int, error: str) -> None:
        """Mark a step as failed."""
        self.failed_steps.add(step_id)
        logfire.error(
            "Step failed",
            execution_id=self.execution.execution_id,
            step_id=step_id,
            error=error
        )

    def mark_step_skipped(self, step_id: int, reason: str) -> None:
        """Mark a step as skipped."""
        self.skipped_steps.add(step_id)
        logfire.info(
            "Step skipped",
            execution_id=self.execution.execution_id,
            step_id=step_id,
            reason=reason
        )

    def is_step_completed(self, step_id: int) -> bool:
        """Check if a step has been completed."""
        return step_id in self.completed_steps

    def is_step_failed(self, step_id: int) -> bool:
        """Check if a step has failed."""
        return step_id in self.failed_steps

    def can_execute_step(self, step: WorkflowStep) -> bool:
        """Check if a step can be executed based on dependencies."""
        if not step.depends_on:
            return True

        # Parse dependency step IDs
        dependency_ids = []
        if isinstance(step.depends_on, str):
            try:
                dependency_ids = json.loads(step.depends_on)
            except json.JSONDecodeError:
                # Assume comma-separated string
                dependency_ids = [int(x.strip()) for x in step.depends_on.split(',') if x.strip()]
        elif isinstance(step.depends_on, list):
            dependency_ids = step.depends_on

        # Check if all dependencies are completed
        for dep_id in dependency_ids:
            if dep_id not in self.completed_steps:
                return False

        return True

    def get_progress_percentage(self) -> float:
        """Calculate execution progress percentage."""
        total_steps = len(self.workflow.steps) if self.workflow.steps else 1
        completed_count = len(self.completed_steps)
        return (completed_count / total_steps) * 100.0

    def set_phase(self, phase: ExecutionPhase) -> None:
        """Set the current execution phase."""
        self.phase = phase
        logfire.debug(
            "Execution phase changed",
            execution_id=self.execution.execution_id,
            phase=phase.value
        )

    def checkpoint(self, name: Optional[str] = None) -> None:
        """Create a checkpoint in the execution."""
        self.last_checkpoint = datetime.utcnow()
        self.checkpoint_times.append(self.last_checkpoint)

        logfire.info(
            "Execution checkpoint",
            execution_id=self.execution.execution_id,
            checkpoint_name=name or f"checkpoint_{len(self.checkpoint_times)}",
            progress=self.get_progress_percentage()
        )

    def pause(self) -> None:
        """Pause the execution."""
        self.is_paused = True
        self.checkpoint("pause")
        logfire.info(
            "Execution paused",
            execution_id=self.execution.execution_id
        )

    def resume(self) -> None:
        """Resume the execution."""
        self.is_paused = False
        self.checkpoint("resume")
        logfire.info(
            "Execution resumed",
            execution_id=self.execution.execution_id
        )

    def cancel(self) -> None:
        """Cancel the execution."""
        self.is_cancelled = True
        self.checkpoint("cancel")
        logfire.info(
            "Execution cancelled",
            execution_id=self.execution.execution_id
        )

    def set_error(self, error: Exception, step_id: Optional[int] = None) -> None:
        """Set error details for the execution."""
        self.error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "step_id": step_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": self.phase.value
        }

        if step_id:
            self.mark_step_failed(step_id, str(error))

        logfire.error(
            "Execution error set",
            execution_id=self.execution.execution_id,
            error_type=type(error).__name__,
            error_message=str(error),
            step_id=step_id
        )

    def add_event_handler(self, event_type: str, handler: callable) -> None:
        """Add an event handler for the execution."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to registered handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, event_type, data)
                else:
                    handler(self, event_type, data)
            except Exception as e:
                logfire.error(
                    "Event handler failed",
                    execution_id=self.execution.execution_id,
                    event_type=event_type,
                    handler=handler.__name__,
                    error=str(e)
                )

    def add_step_hook(self, hook_type: str, handler: callable) -> None:
        """Add a step hook handler."""
        if hook_type not in self.step_hooks:
            self.step_hooks[hook_type] = []
        self.step_hooks[hook_type].append(handler)

    async def execute_step_hooks(self, hook_type: str, step: WorkflowStep, data: Dict[str, Any]) -> None:
        """Execute step hooks of the specified type."""
        handlers = self.step_hooks.get(hook_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, step, data)
                else:
                    handler(self, step, data)
            except Exception as e:
                logfire.error(
                    "Step hook failed",
                    execution_id=self.execution.execution_id,
                    hook_type=hook_type,
                    step_id=step.id,
                    handler=handler.__name__,
                    error=str(e)
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to dictionary."""
        return {
            "execution_id": self.execution.execution_id,
            "workflow_id": self.workflow.id,
            "phase": self.phase.value,
            "is_paused": self.is_paused,
            "is_cancelled": self.is_cancelled,
            "progress_percentage": self.get_progress_percentage(),
            "variables": self.variables,
            "completed_steps": list(self.completed_steps),
            "failed_steps": list(self.failed_steps),
            "skipped_steps": list(self.skipped_steps),
            "output_data": self.output_data,
            "error_details": self.error_details,
            "start_time": self.start_time.isoformat(),
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "step_count": len(self.step_results),
            "checkpoint_count": len(self.checkpoint_times)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        execution_time = datetime.utcnow() - self.start_time

        return {
            "execution_time_seconds": execution_time.total_seconds(),
            "progress_percentage": self.get_progress_percentage(),
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "skipped_steps": len(self.skipped_steps),
            "total_steps": len(self.workflow.steps) if self.workflow.steps else 0,
            "variables_count": len(self.variables),
            "checkpoints_count": len(self.checkpoint_times),
            "average_step_duration": self._calculate_average_step_duration(),
            "phase": self.phase.value,
            "is_paused": self.is_paused,
            "is_cancelled": self.is_cancelled
        }

    def _calculate_average_step_duration(self) -> float:
        """Calculate average step duration in seconds."""
        if not self.step_durations:
            return 0.0

        total_seconds = sum(duration.total_seconds() for duration in self.step_durations.values())
        return total_seconds / len(self.step_durations)

    def __repr__(self) -> str:
        """String representation of execution context."""
        return (
            f"ExecutionContext(execution_id={self.execution.execution_id}, "
            f"workflow_id={self.workflow.id}, phase={self.phase.value}, "
            f"progress={self.get_progress_percentage():.1f}%)"
        )
