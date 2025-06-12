"""
Tool Execution Framework for Agentical

This module provides the ToolExecutor class that handles tool execution
with comprehensive error handling, validation, monitoring, and integration
with MCP servers and workflow systems.

Features:
- Unified tool execution interface
- Parameter validation and schema enforcement
- Async execution with timeout and cancellation
- Error handling and retry logic
- Performance monitoring and observability
- Integration with MCP client and tool registry
- Execution context and state management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from contextlib import asynccontextmanager
import logging

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    ToolTimeoutError
)
from ...core.logging import log_operation
from ...db.models.tool import Tool, ToolExecution, ExecutionStatus
from ..core.tool_registry import ToolRegistry, ToolRegistryEntry
from ..mcp.mcp_client import MCPClient


class ExecutionMode(Enum):
    """Tool execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"


class ExecutionPriority(Enum):
    """Execution priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ToolExecutionResult:
    """Result of a tool execution."""

    def __init__(
        self,
        execution_id: str,
        tool_name: str,
        success: bool,
        result_data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.execution_id = execution_id
        self.tool_name = tool_name
        self.success = success
        self.result_data = result_data
        self.error = error
        self.metadata = metadata or {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_time: Optional[timedelta] = None

    def set_timing(self, start_time: datetime, end_time: datetime) -> None:
        """Set execution timing information."""
        self.start_time = start_time
        self.end_time = end_time
        self.execution_time = end_time - start_time

    def get_execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time.total_seconds() if self.execution_time else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "result_data": self.result_data,
            "error": self.error,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_seconds": self.get_execution_time_seconds()
        }


class ExecutionContext:
    """Context for tool execution with state management."""

    def __init__(
        self,
        execution_id: str,
        tool_entry: ToolRegistryEntry,
        parameters: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.ASYNC,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout_seconds: int = 300,
        retry_attempts: int = 3,
        user_context: Optional[Dict[str, Any]] = None
    ):
        self.execution_id = execution_id
        self.tool_entry = tool_entry
        self.parameters = parameters
        self.mode = mode
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.user_context = user_context or {}

        # Execution state
        self.status = ExecutionStatus.PENDING
        self.current_attempt = 0
        self.is_cancelled = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Results and errors
        self.intermediate_results: List[Any] = []
        self.error_history: List[str] = []
        self.execution_log: List[Dict[str, Any]] = []

        # Performance tracking
        self.metrics = {
            "validation_time": 0.0,
            "execution_time": 0.0,
            "retry_count": 0,
            "memory_usage": 0,
            "network_calls": 0
        }

    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None) -> None:
        """Log an execution event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.execution_log.append(event)

    def add_error(self, error: str) -> None:
        """Add an error to the error history."""
        self.error_history.append(error)
        self.log_event("error", error)

    def can_retry(self) -> bool:
        """Check if execution can be retried."""
        return (
            not self.is_cancelled and
            self.current_attempt < self.retry_attempts and
            self.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_entry.tool.name,
            "parameters": self.parameters,
            "mode": self.mode.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "current_attempt": self.current_attempt,
            "retry_attempts": self.retry_attempts,
            "is_cancelled": self.is_cancelled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "error_count": len(self.error_history),
            "log_entries": len(self.execution_log)
        }


class ToolExecutor:
    """
    Comprehensive tool execution framework.

    Provides unified interface for executing tools with validation,
    error handling, monitoring, and integration with MCP servers.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        mcp_client: MCPClient,
        max_concurrent_executions: int = 50,
        default_timeout_seconds: int = 300,
        enable_monitoring: bool = True
    ):
        """Initialize the tool executor."""
        self.tool_registry = tool_registry
        self.mcp_client = mcp_client
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout_seconds = default_timeout_seconds
        self.enable_monitoring = enable_monitoring

        # Active executions tracking
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)

        # Performance tracking
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "cancelled_executions": 0,
            "average_execution_time": 0.0,
            "concurrent_peak": 0
        }

        # Execution hooks
        self._pre_execution_hooks: List[Callable] = []
        self._post_execution_hooks: List[Callable] = []
        self._error_hooks: List[Callable] = []

        logging.info(f"Tool executor initialized with max concurrent: {max_concurrent_executions}")

    async def execute_tool(
        self,
        tool_identifier: Union[str, int],
        parameters: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.ASYNC,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout_seconds: Optional[int] = None,
        retry_attempts: int = 3,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """
        Execute a tool with the given parameters.

        Args:
            tool_identifier: Tool ID or name
            parameters: Tool execution parameters
            mode: Execution mode (sync, async, batch, stream)
            priority: Execution priority
            timeout_seconds: Custom timeout (uses default if None)
            retry_attempts: Number of retry attempts
            user_context: Additional user context

        Returns:
            ToolExecutionResult: Execution result

        Raises:
            ToolNotFoundError: If tool is not found
            ToolValidationError: If parameters are invalid
            ToolExecutionError: If execution fails
        """
        # Get tool from registry
        tool_entry = await self.tool_registry.get_tool(tool_identifier)
        if not tool_entry:
            raise ToolNotFoundError(f"Tool '{tool_identifier}' not found")

        if not tool_entry.is_enabled:
            raise ToolExecutionError(f"Tool '{tool_entry.tool.name}' is disabled")

        # Create execution context
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            execution_id=execution_id,
            tool_entry=tool_entry,
            parameters=parameters,
            mode=mode,
            priority=priority,
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
            retry_attempts=retry_attempts,
            user_context=user_context
        )

        # Track active execution
        self._active_executions[execution_id] = context

        try:
            # Execute with concurrency control
            async with self._execution_semaphore:
                result = await self._execute_with_retry(context)

                # Update registry usage statistics
                await self.tool_registry.update_tool_usage(
                    tool_identifier,
                    result.get_execution_time_seconds(),
                    result.success
                )

                return result

        finally:
            # Clean up
            self._active_executions.pop(execution_id, None)
            self._execution_tasks.pop(execution_id, None)

    async def _execute_with_retry(self, context: ExecutionContext) -> ToolExecutionResult:
        """Execute tool with retry logic."""
        start_time = datetime.utcnow()
        context.start_time = start_time
        context.status = ExecutionStatus.RUNNING

        last_error = None

        while context.current_attempt <= context.retry_attempts:
            context.current_attempt += 1
            context.log_event("attempt", f"Execution attempt {context.current_attempt}")

            try:
                # Validate parameters
                await self._validate_parameters(context)

                # Execute pre-execution hooks
                await self._execute_hooks(self._pre_execution_hooks, context, "pre_execution")

                # Execute the tool
                result_data = await self._execute_tool_core(context)

                # Execute post-execution hooks
                await self._execute_hooks(self._post_execution_hooks, context, "post_execution")

                # Success - create result
                end_time = datetime.utcnow()
                context.end_time = end_time
                context.status = ExecutionStatus.COMPLETED

                result = ToolExecutionResult(
                    execution_id=context.execution_id,
                    tool_name=context.tool_entry.tool.name,
                    success=True,
                    result_data=result_data,
                    metadata={
                        "attempt": context.current_attempt,
                        "execution_mode": context.mode.value,
                        "priority": context.priority.value
                    }
                )
                result.set_timing(start_time, end_time)

                self._metrics["successful_executions"] += 1
                context.log_event("success", "Tool execution completed successfully")

                return result

            except asyncio.TimeoutError as e:
                error_msg = f"Tool execution timeout after {context.timeout_seconds} seconds"
                context.add_error(error_msg)
                context.status = ExecutionStatus.TIMEOUT
                last_error = ToolTimeoutError(error_msg)
                self._metrics["timeout_executions"] += 1

            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                context.add_error(error_msg)
                context.status = ExecutionStatus.FAILED
                last_error = e

                # Execute error hooks
                await self._execute_hooks(self._error_hooks, context, "error", {"error": str(e)})

            # Check if we can retry
            if not context.can_retry():
                break

            # Wait before retry (exponential backoff)
            retry_delay = min(2 ** (context.current_attempt - 1), 30)
            context.log_event("retry", f"Waiting {retry_delay}s before retry")
            await asyncio.sleep(retry_delay)

        # All attempts failed
        end_time = datetime.utcnow()
        context.end_time = end_time
        self._metrics["failed_executions"] += 1

        result = ToolExecutionResult(
            execution_id=context.execution_id,
            tool_name=context.tool_entry.tool.name,
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            metadata={
                "attempts": context.current_attempt,
                "errors": context.error_history,
                "execution_mode": context.mode.value
            }
        )
        result.set_timing(start_time, end_time)

        context.log_event("failed", f"Tool execution failed after {context.current_attempt} attempts")
        return result

    async def _validate_parameters(self, context: ExecutionContext) -> None:
        """Validate tool parameters."""
        validation_start = datetime.utcnow()

        try:
            # Use schema validation if available (MCP tools)
            if context.tool_entry.schema:
                context.tool_entry.schema.validate_parameters(context.parameters)

            # Additional validation using tool parameters
            for param in context.tool_entry.parameters:
                if param.is_required and param.name not in context.parameters:
                    raise ToolValidationError(f"Required parameter '{param.name}' is missing")

                if param.name in context.parameters:
                    value = context.parameters[param.name]
                    if not param.validate_value(value):
                        raise ToolValidationError(f"Invalid value for parameter '{param.name}'")

            validation_time = (datetime.utcnow() - validation_start).total_seconds()
            context.metrics["validation_time"] = validation_time
            context.log_event("validation", "Parameter validation completed")

        except Exception as e:
            context.log_event("validation_error", f"Parameter validation failed: {str(e)}")
            raise ToolValidationError(f"Parameter validation failed: {str(e)}")

    async def _execute_tool_core(self, context: ExecutionContext) -> Any:
        """Core tool execution logic."""
        execution_start = datetime.utcnow()

        try:
            # Check if this is an MCP tool
            if context.tool_entry.mcp_server:
                result = await self._execute_mcp_tool(context)
            else:
                result = await self._execute_native_tool(context)

            execution_time = (datetime.utcnow() - execution_start).total_seconds()
            context.metrics["execution_time"] = execution_time

            return result

        except asyncio.TimeoutError:
            context.log_event("timeout", f"Tool execution timed out after {context.timeout_seconds}s")
            raise

        except Exception as e:
            context.log_event("execution_error", f"Tool execution error: {str(e)}")
            raise ToolExecutionError(f"Tool execution failed: {str(e)}")

    async def _execute_mcp_tool(self, context: ExecutionContext) -> Any:
        """Execute tool via MCP server."""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.mcp_client.execute_tool(
                    context.tool_entry.mcp_server,
                    context.tool_entry.schema.name,
                    context.parameters
                ),
                timeout=context.timeout_seconds
            )

            context.log_event("mcp_execution", f"MCP tool executed on server {context.tool_entry.mcp_server}")
            return result.get("result", result)

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"MCP tool execution failed: {str(e)}")

    async def _execute_native_tool(self, context: ExecutionContext) -> Any:
        """Execute native (non-MCP) tool."""
        # This would implement native tool execution
        # For now, simulate execution
        context.log_event("native_execution", "Native tool execution simulated")
        await asyncio.sleep(0.1)  # Simulate processing

        return {
            "status": "success",
            "message": f"Native tool {context.tool_entry.tool.name} executed",
            "parameters": context.parameters
        }

    async def _execute_hooks(
        self,
        hooks: List[Callable],
        context: ExecutionContext,
        hook_type: str,
        data: Dict[str, Any] = None
    ) -> None:
        """Execute hooks with error handling."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context, hook_type, data or {})
                else:
                    hook(context, hook_type, data or {})
            except Exception as e:
                logging.error(f"Hook execution failed ({hook_type}): {str(e)}")
                # Don't let hook failures break tool execution

    async def batch_execute(
        self,
        tool_requests: List[Dict[str, Any]],
        max_parallel: int = 10,
        fail_fast: bool = False
    ) -> List[ToolExecutionResult]:
        """Execute multiple tools in batch."""
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_single(request: Dict[str, Any]) -> ToolExecutionResult:
            async with semaphore:
                try:
                    return await self.execute_tool(**request)
                except Exception as e:
                    # Create error result
                    return ToolExecutionResult(
                        execution_id=str(uuid.uuid4()),
                        tool_name=request.get("tool_identifier", "unknown"),
                        success=False,
                        error=str(e)
                    )

        # Create tasks
        tasks = [execute_single(request) for request in tool_requests]

        # Execute with optional fail-fast behavior
        if fail_fast:
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                if not result.success:
                    # Cancel remaining tasks
                    for remaining_task in tasks:
                        if not remaining_task.done():
                            remaining_task.cancel()
                    break
            return results
        else:
            return await asyncio.gather(*tasks, return_exceptions=False)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running tool execution."""
        if execution_id in self._active_executions:
            context = self._active_executions[execution_id]
            context.is_cancelled = True
            context.status = ExecutionStatus.CANCELLED
            context.log_event("cancelled", "Execution cancelled by user")

            # Cancel the task if it exists
            task = self._execution_tasks.get(execution_id)
            if task and not task.done():
                task.cancel()

            self._metrics["cancelled_executions"] += 1
            return True

        return False

    def add_pre_execution_hook(self, hook: Callable) -> None:
        """Add a pre-execution hook."""
        self._pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: Callable) -> None:
        """Add a post-execution hook."""
        self._post_execution_hooks.append(hook)

    def add_error_hook(self, hook: Callable) -> None:
        """Add an error hook."""
        self._error_hooks.append(hook)

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a running execution."""
        if execution_id in self._active_executions:
            return self._active_executions[execution_id].to_dict()
        return None

    async def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get all active executions."""
        return [context.to_dict() for context in self._active_executions.values()]

    async def get_executor_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics."""
        # Update current metrics
        self._metrics["total_executions"] = (
            self._metrics["successful_executions"] +
            self._metrics["failed_executions"] +
            self._metrics["timeout_executions"] +
            self._metrics["cancelled_executions"]
        )

        active_count = len(self._active_executions)
        if active_count > self._metrics.get("concurrent_peak", 0):
            self._metrics["concurrent_peak"] = active_count

        success_rate = 0.0
        if self._metrics["total_executions"] > 0:
            success_rate = (self._metrics["successful_executions"] / self._metrics["total_executions"]) * 100

        return {
            **self._metrics,
            "active_executions": active_count,
            "max_concurrent": self.max_concurrent_executions,
            "success_rate": success_rate,
            "capacity_utilization": (active_count / self.max_concurrent_executions) * 100
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the executor."""
        mcp_health = await self.mcp_client.health_check_all()
        registry_metrics = await self.tool_registry.get_registry_metrics()

        healthy_servers = sum(1 for healthy in mcp_health.values() if healthy)
        total_servers = len(mcp_health)

        return {
            "status": "healthy" if healthy_servers > 0 else "degraded",
            "active_executions": len(self._active_executions),
            "mcp_servers": {
                "healthy": healthy_servers,
                "total": total_servers,
                "health_details": mcp_health
            },
            "registry_status": {
                "total_tools": registry_metrics.get("total_tools", 0),
                "enabled_tools": registry_metrics.get("enabled_tools", 0)
            },
            "executor_metrics": await self.get_executor_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def shutdown(self) -> None:
        """Shutdown the executor gracefully."""
        logging.info("Tool executor shutting down")

        # Cancel all active executions
        for execution_id in list(self._active_executions.keys()):
            await self.cancel_execution(execution_id)

        # Wait for tasks to complete
        if self._execution_tasks:
            await asyncio.gather(
                *self._execution_tasks.values(),
                return_exceptions=True
            )

        # Clear tracking data
        self._active_executions.clear()
        self._execution_tasks.clear()

        logging.info("Tool executor shutdown complete")

    def __repr__(self) -> str:
        """String representation of the executor."""
        return (
            f"ToolExecutor(active={len(self._active_executions)}, "
            f"max_concurrent={self.max_concurrent_executions}, "
            f"total_executions={self._metrics['total_executions']})"
        )
