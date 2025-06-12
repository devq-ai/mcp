"""
Standalone Execution Context for Validation

This module provides a standalone version of the ExecutionContext class
without external dependencies like logfire to allow for validation
of the core workflow engine functionality.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from contextlib import asynccontextmanager
import uuid


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
        execution,
        workflow,
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
        self.current_step = None
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

    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in the execution context."""
        self.variables[key] = value

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
            return True
        return False

    def set_step_result(self, step_id: int, result: Any) -> None:
        """Store the result of a completed step."""
        self.step_results[step_id] = result

    def get_step_result(self, step_id: int, default: Any = None) -> Any:
        """Get the result of a completed step."""
        return self.step_results.get(step_id, default)

    def mark_step_completed(self, step_id: int, duration: Optional[timedelta] = None) -> None:
        """Mark a step as completed."""
        self.completed_steps.add(step_id)
        if duration:
            self.step_durations[step_id] = duration

    def mark_step_failed(self, step_id: int, error: str) -> None:
        """Mark a step as failed."""
        self.failed_steps.add(step_id)

    def mark_step_skipped(self, step_id: int, reason: str) -> None:
        """Mark a step as skipped."""
        self.skipped_steps.add(step_id)

    def is_step_completed(self, step_id: int) -> bool:
        """Check if a step has been completed."""
        return step_id in self.completed_steps

    def is_step_failed(self, step_id: int) -> bool:
        """Check if a step has failed."""
        return step_id in self.failed_steps

    def can_execute_step(self, step) -> bool:
        """Check if a step can be executed based on dependencies."""
        if not hasattr(step, 'depends_on') or not step.depends_on:
            return True

        # Parse dependency step IDs
        dependency_ids = []
        if isinstance(step.depends_on, str):
            try:
                dependency_ids = json.loads(step.depends_on)
            except json.JSONDecodeError:
                # Assume comma-separated string
                try:
                    dependency_ids = [int(x.strip()) for x in step.depends_on.split(',') if x.strip()]
                except (ValueError, AttributeError):
                    return False
        elif isinstance(step.depends_on, list):
            dependency_ids = step.depends_on

        # Check if all dependencies are completed
        for dep_id in dependency_ids:
            if dep_id not in self.completed_steps:
                return False

        return True

    def get_progress_percentage(self) -> float:
        """Calculate execution progress percentage."""
        if hasattr(self.workflow, 'steps') and self.workflow.steps:
            total_steps = len(self.workflow.steps)
        else:
            total_steps = 1  # Avoid division by zero

        completed_count = len(self.completed_steps)
        return (completed_count / total_steps) * 100.0

    def set_phase(self, phase: ExecutionPhase) -> None:
        """Set the current execution phase."""
        self.phase = phase

    def checkpoint(self, name: Optional[str] = None) -> None:
        """Create a checkpoint in the execution."""
        self.last_checkpoint = datetime.utcnow()
        self.checkpoint_times.append(self.last_checkpoint)

    def pause(self) -> None:
        """Pause the execution."""
        self.is_paused = True
        self.checkpoint("pause")

    def resume(self) -> None:
        """Resume the execution."""
        self.is_paused = False
        self.checkpoint("resume")

    def cancel(self) -> None:
        """Cancel the execution."""
        self.is_cancelled = True
        self.checkpoint("cancel")

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
                # In standalone version, just continue
                pass

    def add_step_hook(self, hook_type: str, handler: callable) -> None:
        """Add a step hook handler."""
        if hook_type not in self.step_hooks:
            self.step_hooks[hook_type] = []
        self.step_hooks[hook_type].append(handler)

    async def execute_step_hooks(self, hook_type: str, step, data: Dict[str, Any]) -> None:
        """Execute step hooks of the specified type."""
        handlers = self.step_hooks.get(hook_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, step, data)
                else:
                    handler(self, step, data)
            except Exception as e:
                # In standalone version, just continue
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to dictionary."""
        return {
            "execution_id": getattr(self.execution, 'execution_id', 'unknown'),
            "workflow_id": getattr(self.workflow, 'id', 'unknown'),
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
            "total_steps": len(getattr(self.workflow, 'steps', [])),
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
        execution_id = getattr(self.execution, 'execution_id', 'unknown')
        workflow_id = getattr(self.workflow, 'id', 'unknown')
        return (
            f"ExecutionContext(execution_id={execution_id}, "
            f"workflow_id={workflow_id}, phase={self.phase.value}, "
            f"progress={self.get_progress_percentage():.1f}%)"
        )


# Standalone validation function
def test_execution_context():
    """Test the execution context functionality."""
    print("Testing ExecutionContext...")

    # Create mock objects
    class MockExecution:
        def __init__(self):
            self.execution_id = "test_123"
            self.workflow_id = 1
            self.id = 1

    class MockWorkflow:
        def __init__(self):
            self.id = 1
            self.steps = [1, 2, 3]  # Mock 3 steps

    class MockStep:
        def __init__(self, depends_on=None):
            self.depends_on = depends_on

    # Test ExecutionContext initialization
    execution = MockExecution()
    workflow = MockWorkflow()
    input_data = {"test": "data"}
    config = {"debug": True}

    context = ExecutionContext(
        execution=execution,
        workflow=workflow,
        input_data=input_data,
        config=config
    )

    # Test variable management
    context.set_variable("new_var", "value")
    assert context.get_variable("new_var") == "value"
    assert context.has_variable("new_var")
    assert context.get_variable("nonexistent", "default") == "default"
    assert context.remove_variable("new_var")
    assert not context.has_variable("new_var")
    print("✓ Variable management works")

    # Test step result management
    context.set_step_result(1, {"result": "success"})
    assert context.get_step_result(1) == {"result": "success"}
    assert context.get_step_result(999, "default") == "default"
    print("✓ Step result management works")

    # Test step completion tracking
    context.mark_step_completed(1, timedelta(seconds=2))
    assert context.is_step_completed(1)
    assert 1 in context.completed_steps

    context.mark_step_failed(2, "Test error")
    assert context.is_step_failed(2)
    assert 2 in context.failed_steps

    context.mark_step_skipped(3, "Test skip")
    assert 3 in context.skipped_steps
    print("✓ Step status tracking works")

    # Test progress calculation
    progress = context.get_progress_percentage()
    expected_progress = (1 / 3) * 100  # 1 completed out of 3 steps
    assert abs(progress - expected_progress) < 0.1
    print("✓ Progress calculation works")

    # Test dependency checking
    step_no_deps = MockStep()
    assert context.can_execute_step(step_no_deps)

    step_with_deps = MockStep(depends_on="[1]")
    assert context.can_execute_step(step_with_deps)  # Step 1 is completed

    step_with_unfulfilled_deps = MockStep(depends_on="[4]")
    assert not context.can_execute_step(step_with_unfulfilled_deps)  # Step 4 not completed
    print("✓ Dependency checking works")

    # Test phase management
    context.set_phase(ExecutionPhase.EXECUTION)
    assert context.phase == ExecutionPhase.EXECUTION
    print("✓ Phase management works")

    # Test pause/resume/cancel
    context.pause()
    assert context.is_paused
    context.resume()
    assert not context.is_paused
    context.cancel()
    assert context.is_cancelled
    print("✓ Pause/resume/cancel works")

    # Test error handling
    error = ValueError("Test error")
    context.set_error(error, 1)
    assert context.error_details is not None
    assert context.error_details["error_type"] == "ValueError"
    assert context.error_details["error_message"] == "Test error"
    print("✓ Error handling works")

    # Test serialization
    context_dict = context.to_dict()
    assert "execution_id" in context_dict
    assert "workflow_id" in context_dict
    assert "variables" in context_dict
    assert "progress_percentage" in context_dict
    print("✓ Serialization works")

    # Test metrics
    metrics = context.get_metrics()
    assert "execution_time_seconds" in metrics
    assert "progress_percentage" in metrics
    assert "completed_steps" in metrics
    print("✓ Metrics collection works")

    print("\n✅ All ExecutionContext tests passed!")
    return True


if __name__ == "__main__":
    test_execution_context()
