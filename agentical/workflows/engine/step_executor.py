"""
Step Executor for Workflow Engine

This module provides the StepExecutor class that handles individual workflow
step execution, including agent task execution, tool invocation, condition
evaluation, and other step types.

Features:
- Individual step execution and coordination
- Agent task integration
- Tool execution and management
- Condition evaluation and branching
- Loop and parallel step handling
- Error handling and retry logic
- Performance monitoring with Logfire
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import uuid

import logfire
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.exceptions import (
    WorkflowExecutionError,
    WorkflowValidationError,
    AgentError,
    ValidationError
)
from ...core.logging import log_operation
from ...db.models.workflow import (
    WorkflowStep,
    WorkflowStepExecution,
    StepType,
    StepStatus
)
from ...db.models.agent import Agent, AgentStatus
from ...db.models.tool import Tool


class StepExecutionResult:
    """Result of a step execution."""

    def __init__(
        self,
        success: bool,
        output_data: Dict[str, Any],
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.output_data = output_data
        self.error = error
        self.metadata = metadata or {}
        self.execution_time: Optional[timedelta] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output_data": self.output_data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_seconds": self.execution_time.total_seconds() if self.execution_time else None
        }


class StepExecutor:
    """
    Executes individual workflow steps.

    Handles different step types including agent tasks, tool execution,
    conditions, loops, and other workflow step types with proper error
    handling and monitoring.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize the step executor."""
        self.db_session = db_session

        # Step type handlers
        self._step_handlers: Dict[StepType, Callable] = {
            StepType.AGENT_TASK: self._execute_agent_task,
            StepType.TOOL_EXECUTION: self._execute_tool,
            StepType.CONDITION: self._execute_condition,
            StepType.LOOP: self._execute_loop,
            StepType.PARALLEL: self._execute_parallel,
            StepType.WAIT: self._execute_wait,
            StepType.WEBHOOK: self._execute_webhook,
            StepType.SCRIPT: self._execute_script,
            StepType.HUMAN_INPUT: self._execute_human_input,
            StepType.DATA_TRANSFORM: self._execute_data_transform
        }

        logfire.info("Step executor initialized")

    async def execute_step(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """
        Execute a workflow step.

        Args:
            step: The workflow step to execute
            context: The execution context
            input_data: Input data for the step

        Returns:
            StepExecutionResult: The execution result

        Raises:
            WorkflowExecutionError: If step execution fails
        """
        with logfire.span(
            "Execute workflow step",
            step_id=step.id,
            step_type=step.step_type.value,
            execution_id=context.execution.execution_id
        ):
            start_time = datetime.utcnow()

            # Create step execution record
            step_execution = await self._create_step_execution(step, context, input_data)

            try:
                # Set current step in context
                context.current_step = step

                # Execute step hooks (before)
                await context.execute_step_hooks("before_step", step, input_data)

                # Get step handler
                handler = self._step_handlers.get(step.step_type)
                if not handler:
                    raise WorkflowExecutionError(
                        f"No handler registered for step type: {step.step_type.value}"
                    )

                # Execute the step
                result = await handler(step, context, input_data)

                # Calculate execution time
                end_time = datetime.utcnow()
                result.execution_time = end_time - start_time

                # Update step execution record
                await self._complete_step_execution(step_execution, result)

                # Execute step hooks (after)
                await context.execute_step_hooks("after_step", step, result.output_data)

                # Update context
                context.set_step_result(step.id, result.output_data)
                context.mark_step_completed(step.id, result.execution_time)

                logfire.info(
                    "Step executed successfully",
                    step_id=step.id,
                    execution_time_seconds=result.execution_time.total_seconds()
                )

                return result

            except Exception as e:
                # Calculate execution time for failed step
                end_time = datetime.utcnow()
                execution_time = end_time - start_time

                # Create error result
                error_result = StepExecutionResult(
                    success=False,
                    output_data={},
                    error=str(e),
                    metadata={
                        "error_type": type(e).__name__,
                        "step_type": step.step_type.value
                    }
                )
                error_result.execution_time = execution_time

                # Update step execution record
                await self._fail_step_execution(step_execution, error_result)

                # Execute error hooks
                await context.execute_step_hooks("step_error", step, {"error": str(e)})

                # Update context
                context.mark_step_failed(step.id, str(e))
                context.set_error(e, step.id)

                logfire.error(
                    "Step execution failed",
                    step_id=step.id,
                    step_type=step.step_type.value,
                    error=str(e),
                    execution_time_seconds=execution_time.total_seconds()
                )

                # Re-raise if not handling retries
                if not await self._should_retry_step(step, context):
                    raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}")

                # Implement retry logic
                return await self._retry_step_execution(step, context, input_data, e)

    async def _create_step_execution(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> WorkflowStepExecution:
        """Create a step execution record."""
        step_execution = WorkflowStepExecution(
            workflow_execution_id=context.execution.id,
            step_id=step.id,
            status=StepStatus.RUNNING,
            input_data=input_data,
            started_at=datetime.utcnow()
        )

        self.db_session.add(step_execution)
        await self.db_session.commit()
        await self.db_session.refresh(step_execution)

        return step_execution

    async def _complete_step_execution(
        self,
        step_execution: WorkflowStepExecution,
        result: StepExecutionResult
    ) -> None:
        """Complete a step execution record."""
        step_execution.complete_execution()
        step_execution.output_data = result.output_data
        step_execution.metadata = result.metadata

        await self.db_session.commit()

    async def _fail_step_execution(
        self,
        step_execution: WorkflowStepExecution,
        result: StepExecutionResult
    ) -> None:
        """Fail a step execution record."""
        step_execution.status = StepStatus.FAILED
        step_execution.ended_at = datetime.utcnow()
        step_execution.error_message = result.error
        step_execution.metadata = result.metadata

        await self.db_session.commit()

    async def _execute_agent_task(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute an agent task step."""
        config = step.configuration or {}
        agent_id = config.get("agent_id")
        task_data = config.get("task_data", {})

        if not agent_id:
            raise WorkflowValidationError("Agent task step requires agent_id in configuration")

        # Get agent from database
        agent = await self.db_session.get(Agent, agent_id)
        if not agent:
            raise WorkflowExecutionError(f"Agent {agent_id} not found")

        if agent.status != AgentStatus.ACTIVE:
            raise WorkflowExecutionError(f"Agent {agent_id} is not active")

        # Merge input data with task data
        task_input = {**task_data, **input_data}

        with logfire.span("Execute agent task", agent_id=agent_id, agent_type=agent.agent_type.value):
            # TODO: Implement actual agent task execution
            # This would integrate with the agent system to execute the task

            # For now, simulate agent execution
            await asyncio.sleep(0.1)  # Simulate processing time

            result_data = {
                "agent_id": agent_id,
                "agent_type": agent.agent_type.value,
                "task_result": "Task completed successfully",
                "processed_data": task_input
            }

            return StepExecutionResult(
                success=True,
                output_data=result_data,
                metadata={
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type.value,
                    "execution_mode": "simulated"
                }
            )

    async def _execute_tool(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a tool execution step."""
        config = step.configuration or {}
        tool_id = config.get("tool_id")
        tool_params = config.get("parameters", {})

        if not tool_id:
            raise WorkflowValidationError("Tool execution step requires tool_id in configuration")

        # Get tool from database
        tool = await self.db_session.get(Tool, tool_id)
        if not tool:
            raise WorkflowExecutionError(f"Tool {tool_id} not found")

        # Merge input data with tool parameters
        tool_input = {**tool_params, **input_data}

        with logfire.span("Execute tool", tool_id=tool_id, tool_name=tool.name):
            # TODO: Implement actual tool execution
            # This would integrate with the tool system to execute the tool

            # For now, simulate tool execution
            await asyncio.sleep(0.1)  # Simulate processing time

            result_data = {
                "tool_id": tool_id,
                "tool_name": tool.name,
                "tool_result": "Tool executed successfully",
                "output": tool_input
            }

            return StepExecutionResult(
                success=True,
                output_data=result_data,
                metadata={
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "execution_mode": "simulated"
                }
            )

    async def _execute_condition(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a condition step."""
        config = step.configuration or {}
        condition_expr = config.get("condition")
        true_path = config.get("true_path")
        false_path = config.get("false_path")

        if not condition_expr:
            raise WorkflowValidationError("Condition step requires condition expression")

        # Evaluate condition
        # For now, use simple string-based evaluation
        # In production, this should use a safe expression evaluator
        try:
            # Simple variable substitution and evaluation
            variables = {**context.variables, **input_data}

            # Replace variables in condition expression
            evaluated_expr = condition_expr
            for key, value in variables.items():
                evaluated_expr = evaluated_expr.replace(f"{{{key}}}", str(value))

            # Simple condition evaluation (extend as needed)
            condition_result = eval(evaluated_expr)  # Note: Use safe evaluator in production

            result_data = {
                "condition_result": condition_result,
                "condition_expression": condition_expr,
                "evaluated_expression": evaluated_expr,
                "next_path": true_path if condition_result else false_path
            }

            return StepExecutionResult(
                success=True,
                output_data=result_data,
                metadata={
                    "condition_type": "expression",
                    "condition_result": condition_result
                }
            )

        except Exception as e:
            raise WorkflowExecutionError(f"Condition evaluation failed: {str(e)}")

    async def _execute_loop(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a loop step."""
        config = step.configuration or {}
        loop_type = config.get("loop_type", "for")  # for, while, foreach
        loop_data = config.get("loop_data", [])
        max_iterations = config.get("max_iterations", 100)

        results = []
        iteration_count = 0

        if loop_type == "foreach":
            items = loop_data if isinstance(loop_data, list) else input_data.get("items", [])

            for item in items:
                if iteration_count >= max_iterations:
                    break

                # Process each item
                item_result = {
                    "iteration": iteration_count,
                    "item": item,
                    "processed": True
                }
                results.append(item_result)
                iteration_count += 1

        result_data = {
            "loop_type": loop_type,
            "iterations": iteration_count,
            "results": results
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "loop_type": loop_type,
                "iteration_count": iteration_count
            }
        )

    async def _execute_parallel(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a parallel step."""
        config = step.configuration or {}
        parallel_tasks = config.get("tasks", [])

        if not parallel_tasks:
            raise WorkflowValidationError("Parallel step requires tasks configuration")

        # Execute tasks in parallel
        tasks = []
        for task_config in parallel_tasks:
            task = asyncio.create_task(
                self._execute_parallel_task(task_config, context, input_data)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "task_index": i,
                    "error": str(result)
                })
            else:
                successful_results.append({
                    "task_index": i,
                    "result": result
                })

        result_data = {
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "total_tasks": len(parallel_tasks),
            "results": successful_results,
            "errors": failed_results
        }

        return StepExecutionResult(
            success=len(failed_results) == 0,
            output_data=result_data,
            metadata={
                "execution_type": "parallel",
                "task_count": len(parallel_tasks)
            }
        )

    async def _execute_parallel_task(
        self,
        task_config: Dict[str, Any],
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task in a parallel step."""
        # Simulate parallel task execution
        await asyncio.sleep(0.1)

        return {
            "task_config": task_config,
            "input_data": input_data,
            "result": "Parallel task completed"
        }

    async def _execute_wait(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a wait step."""
        config = step.configuration or {}
        wait_seconds = config.get("seconds", 1)
        wait_reason = config.get("reason", "Waiting as configured")

        await asyncio.sleep(wait_seconds)

        result_data = {
            "wait_seconds": wait_seconds,
            "wait_reason": wait_reason,
            "waited_at": datetime.utcnow().isoformat()
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "step_type": "wait",
                "duration_seconds": wait_seconds
            }
        )

    async def _execute_webhook(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a webhook step."""
        config = step.configuration or {}
        webhook_url = config.get("url")
        method = config.get("method", "POST")
        headers = config.get("headers", {})

        if not webhook_url:
            raise WorkflowValidationError("Webhook step requires URL configuration")

        # TODO: Implement actual webhook execution
        # This would make HTTP requests to the specified webhook URL

        result_data = {
            "webhook_url": webhook_url,
            "method": method,
            "status": "simulated",
            "response": "Webhook execution simulated"
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "webhook_url": webhook_url,
                "method": method,
                "execution_mode": "simulated"
            }
        )

    async def _execute_script(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a script step."""
        config = step.configuration or {}
        script_type = config.get("type", "python")
        script_content = config.get("content")

        if not script_content:
            raise WorkflowValidationError("Script step requires script content")

        # TODO: Implement actual script execution
        # This would execute the script in a secure environment

        result_data = {
            "script_type": script_type,
            "execution_status": "simulated",
            "output": "Script execution simulated"
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "script_type": script_type,
                "execution_mode": "simulated"
            }
        )

    async def _execute_human_input(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a human input step."""
        config = step.configuration or {}
        prompt_message = config.get("prompt", "Please provide input")
        input_type = config.get("input_type", "text")

        # TODO: Implement actual human input handling
        # This would pause execution and wait for human input

        result_data = {
            "prompt_message": prompt_message,
            "input_type": input_type,
            "status": "simulated",
            "user_input": "Simulated user input"
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "input_type": input_type,
                "execution_mode": "simulated"
            }
        )

    async def _execute_data_transform(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a data transformation step."""
        config = step.configuration or {}
        transform_type = config.get("type", "json")
        transformation = config.get("transformation", {})

        # Simple data transformation
        transformed_data = input_data.copy()

        # Apply transformations based on configuration
        for key, transform_rule in transformation.items():
            if isinstance(transform_rule, str):
                # Simple string replacement or formatting
                transformed_data[key] = transform_rule.format(**input_data)
            elif isinstance(transform_rule, dict):
                # More complex transformations
                transformed_data[key] = transform_rule

        result_data = {
            "original_data": input_data,
            "transformed_data": transformed_data,
            "transform_type": transform_type
        }

        return StepExecutionResult(
            success=True,
            output_data=result_data,
            metadata={
                "transform_type": transform_type,
                "keys_transformed": len(transformation)
            }
        )

    async def _should_retry_step(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext'
    ) -> bool:
        """Determine if a failed step should be retried."""
        retry_config = step.configuration.get("retry", {}) if step.configuration else {}
        max_retries = retry_config.get("max_retries", 0)

        # Check if step has retry attempts remaining
        # This would be tracked in the step execution record
        return max_retries > 0  # Simplified logic

    async def _retry_step_execution(
        self,
        step: WorkflowStep,
        context: 'ExecutionContext',
        input_data: Dict[str, Any],
        original_error: Exception
    ) -> StepExecutionResult:
        """Retry step execution with backoff."""
        retry_config = step.configuration.get("retry", {}) if step.configuration else {}
        retry_delay = retry_config.get("delay_seconds", 1)

        # Wait before retry
        await asyncio.sleep(retry_delay)

        # For now, just return the original error
        # In production, this would implement proper retry logic
        raise WorkflowExecutionError(f"Step {step.id} failed after retry: {str(original_error)}")
