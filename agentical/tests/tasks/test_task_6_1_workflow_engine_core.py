"""
Test Suite for Task 6.1: Workflow Engine Core

This module contains comprehensive tests for the workflow engine core functionality,
including workflow orchestration, execution management, state persistence, and
integration with agents and tools.

Test Coverage:
- WorkflowEngine initialization and configuration
- Workflow execution lifecycle management
- ExecutionContext state management and variable handling
- StepExecutor individual step processing
- Error handling and recovery mechanisms
- Performance monitoring and metrics
- Integration with database and Logfire
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import logfire
from sqlalchemy.ext.asyncio import AsyncSession

from workflows.engine.workflow_engine import WorkflowEngine, WorkflowEngineFactory
from workflows.engine.execution_context import ExecutionContext, ExecutionPhase
from workflows.engine.step_executor import StepExecutor, StepExecutionResult
from db.models.workflow import (
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
from db.models.agent import Agent, AgentType, AgentStatus
from db.models.tool import Tool
from core.exceptions import (
    WorkflowError,
    WorkflowExecutionError,
    WorkflowValidationError,
    WorkflowNotFoundError
)


class TestWorkflowEngine:
    """Test suite for WorkflowEngine core functionality."""

    @pytest.fixture
    async def mock_db_session(self):
        """Create a mock async database session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest.fixture
    async def sample_workflow(self):
        """Create a sample workflow for testing."""
        workflow = Workflow(
            id=1,
            name="Test Sequential Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            description="Test workflow for engine testing",
            status=WorkflowStatus.ACTIVE,
            configuration={"test": True},
            created_at=datetime.utcnow()
        )

        # Add sample steps
        workflow.steps = [
            WorkflowStep(
                id=1,
                workflow_id=1,
                name="Step 1",
                step_type=StepType.AGENT_TASK,
                step_order=1,
                configuration={"agent_id": 1, "task_data": {"action": "test"}}
            ),
            WorkflowStep(
                id=2,
                workflow_id=1,
                name="Step 2",
                step_type=StepType.TOOL_EXECUTION,
                step_order=2,
                configuration={"tool_id": 1, "parameters": {"input": "test"}}
            )
        ]

        return workflow

    @pytest.fixture
    async def workflow_engine(self, mock_db_session):
        """Create a workflow engine instance for testing."""
        engine = WorkflowEngine(
            db_session=mock_db_session,
            max_concurrent_workflows=5,
            default_timeout_minutes=30,
            enable_monitoring=True
        )
        return engine

    @pytest.mark.asyncio
    async def test_workflow_engine_initialization(self, mock_db_session):
        """Test workflow engine initialization."""
        engine = WorkflowEngine(
            db_session=mock_db_session,
            max_concurrent_workflows=10,
            default_timeout_minutes=60,
            enable_monitoring=True
        )

        assert engine.db_session == mock_db_session
        assert engine.max_concurrent_workflows == 10
        assert engine.default_timeout_minutes == 60
        assert engine.enable_monitoring is True
        assert len(engine._active_executions) == 0
        assert len(engine._execution_contexts) == 0
        assert len(engine._execution_tasks) == 0

    @pytest.mark.asyncio
    async def test_workflow_engine_factory(self, mock_db_session):
        """Test workflow engine factory creation."""
        config = {
            "max_concurrent_workflows": 15,
            "default_timeout_minutes": 45,
            "enable_monitoring": False
        }

        engine = WorkflowEngineFactory.create_engine(mock_db_session, config)

        assert engine.max_concurrent_workflows == 15
        assert engine.default_timeout_minutes == 45
        assert engine.enable_monitoring is False

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow_engine, sample_workflow, mock_db_session):
        """Test successful workflow execution."""
        # Mock repository behavior
        workflow_engine.workflow_repo.get = AsyncMock(return_value=sample_workflow)

        # Mock handler registration
        async def mock_handler(context):
            context.output_data = {"result": "success"}
            return context.output_data

        workflow_engine._workflow_handlers[WorkflowType.SEQUENTIAL] = mock_handler

        # Execute workflow
        execution = await workflow_engine.execute_workflow(
            workflow_id=1,
            input_data={"test_input": "value"},
            execution_config={"debug": True}
        )

        assert execution is not None
        assert execution.workflow_id == 1
        assert execution.input_data == {"test_input": "value"}
        assert execution.configuration == {"debug": True}
        assert execution.status == ExecutionStatus.PENDING

    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, workflow_engine):
        """Test workflow execution with non-existent workflow."""
        workflow_engine.workflow_repo.get = AsyncMock(return_value=None)

        with pytest.raises(WorkflowNotFoundError):
            await workflow_engine.execute_workflow(workflow_id=999)

    @pytest.mark.asyncio
    async def test_execute_workflow_inactive_status(self, workflow_engine, sample_workflow):
        """Test workflow execution with inactive workflow."""
        sample_workflow.status = WorkflowStatus.DRAFT
        workflow_engine.workflow_repo.get = AsyncMock(return_value=sample_workflow)

        with pytest.raises(WorkflowValidationError):
            await workflow_engine.execute_workflow(workflow_id=1)

    @pytest.mark.asyncio
    async def test_concurrent_execution_limit(self, workflow_engine, sample_workflow):
        """Test concurrent execution limit enforcement."""
        workflow_engine.workflow_repo.get = AsyncMock(return_value=sample_workflow)

        # Fill up the concurrent execution slots
        for i in range(workflow_engine.max_concurrent_workflows):
            execution_id = f"exec_{i}"
            mock_execution = MagicMock()
            mock_execution.execution_id = execution_id
            workflow_engine._active_executions[execution_id] = mock_execution

        with pytest.raises(WorkflowExecutionError, match="Maximum concurrent workflow executions reached"):
            await workflow_engine.execute_workflow(workflow_id=1)

    @pytest.mark.asyncio
    async def test_cancel_execution(self, workflow_engine):
        """Test workflow execution cancellation."""
        execution_id = "test_execution_123"
        mock_execution = MagicMock()
        mock_execution.id = 1
        mock_execution.execution_id = execution_id
        mock_execution.cancel_execution = MagicMock()

        workflow_engine._active_executions[execution_id] = mock_execution
        workflow_engine.workflow_repo.update_execution_state = AsyncMock()

        result = await workflow_engine.cancel_execution(execution_id)

        assert result is True
        mock_execution.cancel_execution.assert_called_once()
        workflow_engine.workflow_repo.update_execution_state.assert_called_once()
        assert execution_id not in workflow_engine._active_executions

    @pytest.mark.asyncio
    async def test_pause_resume_execution(self, workflow_engine):
        """Test workflow execution pause and resume."""
        execution_id = "test_execution_456"
        mock_execution = MagicMock()
        mock_execution.id = 1
        mock_execution.execution_id = execution_id
        mock_execution.pause_execution = MagicMock()
        mock_execution.resume_execution = MagicMock()

        mock_context = MagicMock()
        mock_context.pause = MagicMock()
        mock_context.resume = MagicMock()

        workflow_engine._active_executions[execution_id] = mock_execution
        workflow_engine._execution_contexts[execution_id] = mock_context
        workflow_engine.workflow_repo.update_execution_state = AsyncMock()

        # Test pause
        result = await workflow_engine.pause_execution(execution_id)
        assert result is True
        mock_execution.pause_execution.assert_called_once()
        mock_context.pause.assert_called_once()

        # Test resume
        result = await workflow_engine.resume_execution(execution_id)
        assert result is True
        mock_execution.resume_execution.assert_called_once()
        mock_context.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_execution_metrics(self, workflow_engine):
        """Test execution metrics collection."""
        # Add some mock active executions
        for i in range(3):
            execution_id = f"exec_{i}"
            workflow_engine._active_executions[execution_id] = MagicMock()

        # Mock repository metrics
        workflow_engine.workflow_repo.get_workflow_metrics = AsyncMock(return_value={
            "total_workflows": 10,
            "active_workflows": 5
        })

        metrics = await workflow_engine.get_execution_metrics()

        assert metrics["active_executions"] == 3
        assert metrics["max_concurrent"] == workflow_engine.max_concurrent_workflows
        assert metrics["capacity_utilization"] == 3 / workflow_engine.max_concurrent_workflows
        assert "database_metrics" in metrics
        assert metrics["engine_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_shutdown(self, workflow_engine):
        """Test workflow engine shutdown."""
        # Add mock executions and tasks
        execution_id = "test_execution"
        mock_execution = MagicMock()
        mock_task = AsyncMock()

        workflow_engine._active_executions[execution_id] = mock_execution
        workflow_engine._execution_tasks[execution_id] = mock_task
        workflow_engine.cancel_execution = AsyncMock(return_value=True)

        await workflow_engine.shutdown()

        workflow_engine.cancel_execution.assert_called_once_with(execution_id)
        assert len(workflow_engine._active_executions) == 0


class TestExecutionContext:
    """Test suite for ExecutionContext functionality."""

    @pytest.fixture
    def sample_execution(self):
        """Create a sample workflow execution."""
        return WorkflowExecution(
            id=1,
            workflow_id=1,
            execution_id="test_exec_123",
            status=ExecutionStatus.RUNNING,
            input_data={"input": "test"},
            configuration={"config": "value"},
            created_at=datetime.utcnow()
        )

    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow."""
        return Workflow(
            id=1,
            name="Test Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            status=WorkflowStatus.ACTIVE
        )

    @pytest.fixture
    def execution_context(self, sample_execution, sample_workflow):
        """Create an execution context for testing."""
        return ExecutionContext(
            execution=sample_execution,
            workflow=sample_workflow,
            input_data={"test": "data"},
            config={"debug": True}
        )

    def test_execution_context_initialization(self, execution_context, sample_execution, sample_workflow):
        """Test execution context initialization."""
        assert execution_context.execution == sample_execution
        assert execution_context.workflow == sample_workflow
        assert execution_context.input_data == {"test": "data"}
        assert execution_context.config == {"debug": True}
        assert execution_context.phase == ExecutionPhase.INITIALIZATION
        assert not execution_context.is_paused
        assert not execution_context.is_cancelled
        assert len(execution_context.variables) == 1  # Initialized with input_data
        assert execution_context.variables["test"] == "data"

    def test_variable_management(self, execution_context):
        """Test variable management functionality."""
        # Test set and get variable
        execution_context.set_variable("new_var", "new_value")
        assert execution_context.get_variable("new_var") == "new_value"
        assert execution_context.has_variable("new_var")

        # Test get with default
        assert execution_context.get_variable("nonexistent", "default") == "default"

        # Test remove variable
        assert execution_context.remove_variable("new_var")
        assert not execution_context.has_variable("new_var")
        assert not execution_context.remove_variable("nonexistent")

    def test_step_result_management(self, execution_context):
        """Test step result management."""
        step_id = 1
        result_data = {"output": "step_result"}

        # Set step result
        execution_context.set_step_result(step_id, result_data)
        assert execution_context.get_step_result(step_id) == result_data

        # Test get with default
        assert execution_context.get_step_result(999, "default") == "default"

    def test_step_status_tracking(self, execution_context):
        """Test step status tracking."""
        step_id = 1
        duration = timedelta(seconds=5)

        # Mark step completed
        execution_context.mark_step_completed(step_id, duration)
        assert execution_context.is_step_completed(step_id)
        assert step_id in execution_context.completed_steps
        assert execution_context.step_durations[step_id] == duration

        # Mark step failed
        step_id_2 = 2
        execution_context.mark_step_failed(step_id_2, "Test error")
        assert execution_context.is_step_failed(step_id_2)
        assert step_id_2 in execution_context.failed_steps

        # Mark step skipped
        step_id_3 = 3
        execution_context.mark_step_skipped(step_id_3, "Test skip")
        assert step_id_3 in execution_context.skipped_steps

    def test_dependency_checking(self, execution_context):
        """Test step dependency checking."""
        # Create mock step with dependencies
        step = MagicMock()
        step.depends_on = "[1, 2]"  # JSON string format

        # No dependencies completed - should not be executable
        assert not execution_context.can_execute_step(step)

        # Complete one dependency
        execution_context.mark_step_completed(1)
        assert not execution_context.can_execute_step(step)

        # Complete all dependencies
        execution_context.mark_step_completed(2)
        assert execution_context.can_execute_step(step)

        # Test step with no dependencies
        step_no_deps = MagicMock()
        step_no_deps.depends_on = None
        assert execution_context.can_execute_step(step_no_deps)

    def test_progress_calculation(self, execution_context):
        """Test progress percentage calculation."""
        # Mock workflow with 4 steps
        execution_context.workflow.steps = [MagicMock() for _ in range(4)]

        # No steps completed
        assert execution_context.get_progress_percentage() == 0.0

        # Complete 2 steps
        execution_context.mark_step_completed(1)
        execution_context.mark_step_completed(2)
        assert execution_context.get_progress_percentage() == 50.0

        # Complete all steps
        execution_context.mark_step_completed(3)
        execution_context.mark_step_completed(4)
        assert execution_context.get_progress_percentage() == 100.0

    def test_phase_management(self, execution_context):
        """Test execution phase management."""
        # Test phase setting
        execution_context.set_phase(ExecutionPhase.EXECUTION)
        assert execution_context.phase == ExecutionPhase.EXECUTION

    def test_checkpoint_functionality(self, execution_context):
        """Test checkpoint functionality."""
        initial_checkpoint_count = len(execution_context.checkpoint_times)

        execution_context.checkpoint("test_checkpoint")

        assert len(execution_context.checkpoint_times) == initial_checkpoint_count + 1
        assert execution_context.last_checkpoint is not None

    def test_pause_resume_cancel(self, execution_context):
        """Test pause, resume, and cancel functionality."""
        # Test pause
        execution_context.pause()
        assert execution_context.is_paused

        # Test resume
        execution_context.resume()
        assert not execution_context.is_paused

        # Test cancel
        execution_context.cancel()
        assert execution_context.is_cancelled

    def test_error_handling(self, execution_context):
        """Test error handling in execution context."""
        error = ValueError("Test error")
        step_id = 1

        execution_context.set_error(error, step_id)

        assert execution_context.error_details is not None
        assert execution_context.error_details["error_type"] == "ValueError"
        assert execution_context.error_details["error_message"] == "Test error"
        assert execution_context.error_details["step_id"] == step_id
        assert step_id in execution_context.failed_steps

    @pytest.mark.asyncio
    async def test_event_handling(self, execution_context):
        """Test event handling functionality."""
        event_data = []

        def event_handler(context, event_type, data):
            event_data.append((event_type, data))

        execution_context.add_event_handler("test_event", event_handler)
        await execution_context.emit_event("test_event", {"test": "data"})

        assert len(event_data) == 1
        assert event_data[0] == ("test_event", {"test": "data"})

    def test_context_serialization(self, execution_context):
        """Test execution context serialization."""
        context_dict = execution_context.to_dict()

        assert "execution_id" in context_dict
        assert "workflow_id" in context_dict
        assert "phase" in context_dict
        assert "progress_percentage" in context_dict
        assert "variables" in context_dict
        assert "completed_steps" in context_dict

    def test_metrics_collection(self, execution_context):
        """Test metrics collection."""
        # Add some data to calculate metrics
        execution_context.mark_step_completed(1, timedelta(seconds=2))
        execution_context.mark_step_completed(2, timedelta(seconds=4))
        execution_context.mark_step_failed(3, "Test error")

        metrics = execution_context.get_metrics()

        assert "execution_time_seconds" in metrics
        assert "progress_percentage" in metrics
        assert metrics["completed_steps"] == 2
        assert metrics["failed_steps"] == 1
        assert metrics["average_step_duration"] == 3.0  # (2+4)/2


class TestStepExecutor:
    """Test suite for StepExecutor functionality."""

    @pytest.fixture
    async def mock_db_session(self):
        """Create a mock async database session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.get = AsyncMock()
        return session

    @pytest.fixture
    def step_executor(self, mock_db_session):
        """Create a step executor for testing."""
        return StepExecutor(mock_db_session)

    @pytest.fixture
    def sample_execution_context(self):
        """Create a sample execution context."""
        execution = MagicMock()
        execution.id = 1
        execution.execution_id = "test_exec"

        workflow = MagicMock()
        workflow.id = 1

        context = ExecutionContext(
            execution=execution,
            workflow=workflow,
            input_data={"test": "data"},
            config={"debug": True}
        )
        return context

    @pytest.mark.asyncio
    async def test_agent_task_execution(self, step_executor, sample_execution_context, mock_db_session):
        """Test agent task step execution."""
        # Create agent task step
        step = WorkflowStep(
            id=1,
            name="Agent Task",
            step_type=StepType.AGENT_TASK,
            configuration={
                "agent_id": 1,
                "task_data": {"action": "process"}
            }
        )

        # Mock agent
        mock_agent = Agent(
            id=1,
            name="Test Agent",
            agent_type=AgentType.CODE_AGENT,
            status=AgentStatus.ACTIVE
        )
        mock_db_session.get.return_value = mock_agent

        input_data = {"input": "test_data"}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert "agent_id" in result.output_data
        assert "task_result" in result.output_data
        assert result.metadata["agent_type"] == "code_agent"

    @pytest.mark.asyncio
    async def test_tool_execution(self, step_executor, sample_execution_context, mock_db_session):
        """Test tool execution step."""
        # Create tool execution step
        step = WorkflowStep(
            id=2,
            name="Tool Execution",
            step_type=StepType.TOOL_EXECUTION,
            configuration={
                "tool_id": 1,
                "parameters": {"param1": "value1"}
            }
        )

        # Mock tool
        mock_tool = Tool(
            id=1,
            name="Test Tool",
            description="Test tool for execution"
        )
        mock_db_session.get.return_value = mock_tool

        input_data = {"input": "test_data"}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert "tool_id" in result.output_data
        assert "tool_name" in result.output_data
        assert result.metadata["tool_name"] == "Test Tool"

    @pytest.mark.asyncio
    async def test_condition_execution(self, step_executor, sample_execution_context):
        """Test condition step execution."""
        # Create condition step
        step = WorkflowStep(
            id=3,
            name="Condition Check",
            step_type=StepType.CONDITION,
            configuration={
                "condition": "{value} > 5",
                "true_path": "step_4",
                "false_path": "step_5"
            }
        )

        # Test with true condition
        input_data = {"value": 10}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert result.output_data["condition_result"] is True
        assert result.output_data["next_path"] == "step_4"

        # Test with false condition
        input_data = {"value": 3}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert result.output_data["condition_result"] is False
        assert result.output_data["next_path"] == "step_5"

    @pytest.mark.asyncio
    async def test_loop_execution(self, step_executor, sample_execution_context):
        """Test loop step execution."""
        # Create loop step
        step = WorkflowStep(
            id=4,
            name="Loop Processing",
            step_type=StepType.LOOP,
            configuration={
                "loop_type": "foreach",
                "loop_data": ["item1", "item2", "item3"],
                "max_iterations": 10
            }
        )

        input_data = {}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert result.output_data["loop_type"] == "foreach"
        assert result.output_data["iterations"] == 3
        assert len(result.output_data["results"]) == 3

    @pytest.mark.asyncio
    async def test_parallel_execution(self, step_executor, sample_execution_context):
        """Test parallel step execution."""
        # Create parallel step
        step = WorkflowStep(
            id=5,
            name="Parallel Processing",
            step_type=StepType.PARALLEL,
            configuration={
                "tasks": [
                    {"task_id": "task1", "config": {"param": "value1"}},
                    {"task_id": "task2", "config": {"param": "value2"}},
                    {"task_id": "task3", "config": {"param": "value3"}}
                ]
            }
        )

        input_data = {"shared_data": "test"}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert result.output_data["total_tasks"] == 3
        assert result.output_data["successful_tasks"] == 3
        assert result.output_data["failed_tasks"] == 0

    @pytest.mark.asyncio
    async def test_wait_step(self, step_executor, sample_execution_context):
        """Test wait step execution."""
        step = WorkflowStep(
            id=6,
            name="Wait Step",
            step_type=StepType.WAIT,
            configuration={
                "seconds": 0.1,  # Short wait for testing
                "reason": "Test wait"
            }
        )

        start_time = datetime.utcnow()
        result = await step_executor.execute_step(step, sample_execution_context, {})
        end_time = datetime.utcnow()

        assert result.success
        assert result.output_data["wait_seconds"] == 0.1
        assert result.output_data["wait_reason"] == "Test wait"
        assert (end_time - start_time).total_seconds() >= 0.1

    @pytest.mark.asyncio
    async def test_data_transform_step(self, step_executor, sample_execution_context):
        """Test data transformation step."""
        step = WorkflowStep(
            id=7,
            name="Data Transform",
            step_type=StepType.DATA_TRANSFORM,
            configuration={
                "type": "json",
                "transformation": {
                    "output_field": "Processed: {input_field}",
                    "status": "completed"
                }
            }
        )

        input_data = {"input_field": "test_data"}
        result = await step_executor.execute_step(step, sample_execution_context, input_data)

        assert result.success
        assert "transformed_data" in result.output_data
        assert result.output_data["transformed_data"]["output_field"] == "Processed: test_data"
        assert result.output_data["transformed_data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_step_execution_error_handling(self, step_executor, sample_execution_context, mock_db_session):
        """Test step execution error handling."""
        # Create agent task step with invalid agent
        step = WorkflowStep(
            id=8,
            name="Invalid Agent Task",
            step_type=StepType.AGENT_TASK,
            configuration={
                "agent_id": 999,  # Non-existent agent
                "task_data": {"action": "test"}
            }
        )

        mock_db_session.get.return_value = None  # Agent not found

        input_data = {"input": "test"}

        with pytest.raises(WorkflowExecutionError):
            await step_executor.execute_step(step, sample_execution_context, input_data)

    @pytest.mark.asyncio
    async def test_step_execution_result_serialization(self, step_executor, sample_execution_context):
        """Test step execution result serialization."""
        step = WorkflowStep(
            id=9,
            name="Test Step",
            step_type=StepType.WAIT,
            configuration={"seconds": 0.001}
        )

        result = await step_executor.execute_step(step, sample_execution_context, {})
        result_dict = result.to_dict()

        assert "success" in result_dict
        assert "output_data" in result_dict
        assert "error" in result_dict
        assert "metadata" in result_dict
        assert "execution_time_seconds" in result_dict
        assert isinstance(result_dict["execution_time_seconds"], float)

    @pytest.mark.asyncio
    async def test_missing_configuration_error(self, step_executor, sample_execution_context):
        """Test step execution with missing required configuration."""
        # Agent task without agent_id
        step = WorkflowStep(
            id=10,
            name="Invalid Config",
            step_type=StepType.AGENT_TASK,
            configuration={}  # Missing agent_id
        )

        with pytest.raises(WorkflowExecutionError):
            await step_executor.execute_step(step, sample_execution_context, {})


class TestWorkflowEngineIntegration:
    """Integration tests for the complete workflow engine system."""

    @pytest.fixture
    async def integrated_setup(self, mock_db_session):
        """Set up integrated workflow engine system for testing."""
        engine = WorkflowEngine(mock_db_session)

        # Create sample workflow with steps
        workflow = Workflow(
            id=1,
            name="Integration Test Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            status=WorkflowStatus.ACTIVE
        )

        steps = [
            WorkflowStep(
                id=1,
                workflow_id=1,
                name="Data Transform",
                step_type=StepType.DATA_TRANSFORM,
                step_order=1,
                configuration={
                    "type": "json",
                    "transformation": {"processed": "true"}
                }
            ),
            WorkflowStep(
                id=2,
                workflow_id=1,
                name="Wait Step",
                step_type=StepType.WAIT,
                step_order=2,
                configuration={"seconds": 0.001}
            )
        ]

        workflow.steps = steps
        return engine, workflow

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, integrated_setup):
        """Test complete end-to-end workflow execution."""
        engine, workflow = integrated_setup

        # Mock repository
        engine.workflow_repo.get = AsyncMock(return_value=workflow)
        engine.workflow_repo.update_execution_state = AsyncMock()

        # Register a simple sequential handler
        async def sequential_handler(context):
            for step in context.workflow.steps:
                if context.can_execute_step(step):
                    result = await engine.step_executor.execute_step(
                        step, context, context.variables
                    )
                    if not result.success:
                        raise WorkflowExecutionError(f"Step {step.id} failed: {result.error}")

                    context.set_step_result(step.id, result.output_data)
                    context.mark_step_completed(step.id, result.execution_time)

            context.output_data = {"workflow_completed": True}
            return context.output_data

        engine.register_workflow_handler(WorkflowType.SEQUENTIAL, sequential_handler)

        # Execute workflow
        execution = await engine.execute_workflow(
            workflow_id=1,
            input_data={"test_input": "integration_test"},
            execution_config={"integration": True}
        )

        # Wait a moment for async execution
        await asyncio.sleep(0.1)

        assert execution is not None
        assert execution.workflow_id == 1
        assert execution.input_data == {"test_input": "integration_test"}

    @pytest.mark.asyncio
    async def test_workflow_execution_with_dependencies(self, integrated_setup):
        """Test workflow execution with step dependencies."""
        engine, workflow = integrated_setup

        # Modify workflow to have dependencies
        workflow.steps[1].depends_on = "[1]"  # Step 2 depends on Step 1

        engine.workflow_repo.get = AsyncMock(return_value=workflow)
        engine.workflow_repo.update_execution_state = AsyncMock()

        # Register handler that respects dependencies
        async def dependency_aware_handler(context):
            executed_steps = []

            # Execute steps in dependency order
            while len(executed_steps) < len(context.workflow.steps):
                for step in context.workflow.steps:
                    if step.id not in executed_steps and context.can_execute_step(step):
                        result = await engine.step_executor.execute_step(
                            step, context, context.variables
                        )
                        context.mark_step_completed(step.id, result.execution_time)
                        executed_steps.append(step.id)
                        break
                else:
                    # No executable steps found - check for deadlock
                    break

            context.output_data = {"executed_steps": executed_steps}
            return context.output_data

        engine.register_workflow_handler(WorkflowType.SEQUENTIAL, dependency_aware_handler)

        execution = await engine.execute_workflow(workflow_id=1)
        await asyncio.sleep(0.1)

        assert execution is not None

    @pytest.mark.asyncio
    async def test_concurrent_workflow_executions(self, integrated_setup):
        """Test multiple concurrent workflow executions."""
        engine, workflow = integrated_setup

        engine.workflow_repo.get = AsyncMock(return_value=workflow)
        engine.workflow_repo.update_execution_state = AsyncMock()

        # Simple handler for concurrent testing
        async def simple_handler(context):
            await asyncio.sleep(0.05)  # Simulate work
            context.output_data = {"completed": True}
            return context.output_data

        engine.register_workflow_handler(WorkflowType.SEQUENTIAL, simple_handler)

        # Start multiple executions
        executions = []
        for i in range(3):
            execution = await engine.execute_workflow(
                workflow_id=1,
                input_data={"execution_number": i}
            )
            executions.append(execution)

        # Verify all executions were created
        assert len(executions) == 3
        assert len(engine._active_executions) == 3

        # Wait for executions to complete
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_workflow_execution_error_recovery(self, integrated_setup):
        """Test workflow execution error handling and recovery."""
        engine, workflow = integrated_setup

        engine.workflow_repo.get = AsyncMock(return_value=workflow)
        engine.workflow_repo.update_execution_state = AsyncMock()

        # Handler that simulates an error
        async def error_handler(context):
            raise ValueError("Simulated workflow error")

        engine.register_workflow_handler(WorkflowType.SEQUENTIAL, error_handler)

        execution = await engine.execute_workflow(workflow_id=1)

        # Wait for execution to fail
        await asyncio.sleep(0.1)

        # Verify error handling
        assert execution.execution_id not in engine._active_executions


class TestStepExecutionResult:
    """Test suite for StepExecutionResult."""

    def test_step_execution_result_creation(self):
        """Test creation of step execution result."""
        result = StepExecutionResult(
            success=True,
            output_data={"result": "success"},
            error=None,
            metadata={"step_type": "test"}
        )

        assert result.success is True
        assert result.output_data == {"result": "success"}
        assert result.error is None
        assert result.metadata == {"step_type": "test"}
        assert result.execution_time is None

    def test_step_execution_result_with_error(self):
        """Test creation of step execution result with error."""
        result = StepExecutionResult(
            success=False,
            output_data={},
            error="Test error message",
            metadata={"error_type": "ValueError"}
        )

        assert result.success is False
        assert result.output_data == {}
        assert result.error == "Test error message"
        assert result.metadata["error_type"] == "ValueError"

    def test_step_execution_result_serialization(self):
        """Test step execution result serialization."""
        result = StepExecutionResult(
            success=True,
            output_data={"key": "value"},
            metadata={"test": True}
        )
        result.execution_time = timedelta(seconds=2.5)

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["output_data"] == {"key": "value"}
        assert result_dict["error"] is None
        assert result_dict["metadata"] == {"test": True}
        assert result_dict["execution_time_seconds"] == 2.5


class TestWorkflowEnginePerformance:
    """Performance and stress tests for the workflow engine."""

    @pytest.mark.asyncio
    async def test_high_volume_step_execution(self, mock_db_session):
        """Test step execution under high volume."""
        executor = StepExecutor(mock_db_session)

        # Create execution context
        execution = MagicMock()
        execution.id = 1
        execution.execution_id = "perf_test"

        workflow = MagicMock()
        workflow.id = 1

        context = ExecutionContext(
            execution=execution,
            workflow=workflow,
            input_data={},
            config={}
        )

        # Create multiple wait steps for performance testing
        steps = []
        for i in range(50):
            step = WorkflowStep(
                id=i + 1,
                name=f"Wait Step {i + 1}",
                step_type=StepType.WAIT,
                configuration={"seconds": 0.001}  # Very short wait
            )
            steps.append(step)

        # Execute all steps
        start_time = datetime.utcnow()
        results = []

        for step in steps:
            result = await executor.execute_step(step, context, {})
            results.append(result)

        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()

        # Verify all steps completed successfully
        assert len(results) == 50
        assert all(result.success for result in results)

        # Performance assertion (should complete in reasonable time)
        assert total_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_context(self, mock_db_session):
        """Test memory usage with large execution context."""
        execution = MagicMock()
        execution.id = 1
        execution.execution_id = "memory_test"

        workflow = MagicMock()
        workflow.id = 1
        workflow.steps = [MagicMock() for _ in range(1000)]  # Large number of steps

        context = ExecutionContext(
            execution=execution,
            workflow=workflow,
            input_data={},
            config={}
        )

        # Add large amount of data to context
        for i in range(1000):
            context.set_variable(f"var_{i}", f"value_{i}" * 100)  # Large strings
            context.set_step_result(i, {"large_data": "x" * 1000})

        # Test context operations still work
        assert context.get_variable("var_0") == "value_0" * 100
        assert len(context.variables) == 1000
        assert len(context.step_results) == 1000

        # Test progress calculation with many steps
        for i in range(500):  # Complete half the steps
            context.mark_step_completed(i)

        progress = context.get_progress_percentage()
        assert progress == 50.0

    @pytest.mark.asyncio
    async def test_concurrent_context_operations(self, mock_db_session):
        """Test concurrent operations on execution context."""
        execution = MagicMock()
        execution.id = 1
        execution.execution_id = "concurrent_test"

        workflow = MagicMock()
        workflow.id = 1

        context = ExecutionContext(
            execution=execution,
            workflow=workflow,
            input_data={},
            config={}
        )

        async def concurrent_operations():
            """Perform concurrent operations on context."""
            tasks = []

            # Add variables concurrently
            for i in range(100):
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(context.set_variable, f"concurrent_var_{i}", i)
                ))

            # Wait for all operations to complete
            await asyncio.gather(*tasks)

        await concurrent_operations()

        # Verify all variables were set
        assert len(context.variables) == 100
        for i in range(100):
            assert context.get_variable(f"concurrent_var_{i}") == i


# Additional utility tests
class TestWorkflowEngineUtilities:
    """Test utility functions and edge cases."""

    @pytest.mark.asyncio
    async def test_workflow_engine_factory_default_config(self, mock_db_session):
        """Test workflow engine factory with default configuration."""
        engine = WorkflowEngineFactory.create_engine(mock_db_session)

        assert engine.max_concurrent_workflows == 10  # Default value
        assert engine.default_timeout_minutes == 60   # Default value
        assert engine.enable_monitoring is True       # Default value

    @pytest.mark.asyncio
    async def test_execution_context_edge_cases(self):
        """Test execution context edge cases."""
        execution = MagicMock()
        execution.execution_id = "edge_case_test"

        workflow = MagicMock()
        workflow.steps = None  # Edge case: no steps

        context = ExecutionContext(
            execution=execution,
            workflow=workflow,
            input_data={},
            config={}
        )

        # Test progress calculation with no steps
        progress = context.get_progress_percentage()
        assert progress == 0.0

        # Test dependency checking with malformed depends_on
        step = MagicMock()
        step.depends_on = "invalid_json"

        # Should handle gracefully and return False
        can_execute = context.can_execute_step(step)
        assert can_execute is False

    def test_step_execution_result_edge_cases(self):
        """Test step execution result edge cases."""
        # Test with None values
        result = StepExecutionResult(
            success=True,
            output_data=None,
            error=None,
            metadata=None
        )

        # Should handle None gracefully
        result_dict = result.to_dict()
        assert result_dict["output_data"] is None
        assert result_dict["metadata"] == {}  # Default empty dict

    @pytest.mark.asyncio
    async def test_step_executor_unsupported_step_type(self, mock_db_session):
        """Test step executor with unsupported step type."""
        executor = StepExecutor(mock_db_session)

        # Create a step with an unsupported type (simulate by removing handler)
        step = WorkflowStep(
            id=1,
            name="Unsupported Step",
            step_type=StepType.AGENT_TASK  # Valid type but remove handler
        )

        # Remove the handler to simulate unsupported type
        del executor._step_handlers[StepType.AGENT_TASK]

        execution = MagicMock()
        workflow = MagicMock()
        context = ExecutionContext(execution, workflow, {}, {})

        with pytest.raises(WorkflowExecutionError, match="No handler registered"):
            await executor.execute_step(step, context, {})


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
