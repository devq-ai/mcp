"""
Structured Logging Framework for Agentical

This module provides comprehensive structured logging capabilities that enhance
the Logfire SDK foundation with context-aware logging for agent operations,
workflows, tools, and system events.

Features:
- Standardized log schemas for different operation types
- Request correlation and distributed tracing
- Agent-specific logging with decision context
- Performance monitoring integration
- Efficient log aggregation patterns
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps

import logfire
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Standard log levels for structured logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OperationType(str, Enum):
    """Types of operations for categorization"""
    API_REQUEST = "api_request"
    AGENT_OPERATION = "agent_operation"
    WORKFLOW_EXECUTION = "workflow_execution"
    TOOL_USAGE = "tool_usage"
    DATABASE_OPERATION = "database_operation"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"
    PERFORMANCE_METRIC = "performance_metric"


class AgentPhase(str, Enum):
    """Agent operation phases for decision tracking"""
    INITIALIZATION = "initialization"
    PERCEPTION = "perception"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"
    COORDINATION = "coordination"
    TERMINATION = "termination"


@dataclass
class CorrelationContext:
    """Context information for request correlation"""
    request_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    parent_operation_id: Optional[str] = None
    trace_id: Optional[str] = None

    @classmethod
    def generate(cls, **kwargs) -> 'CorrelationContext':
        """Generate new correlation context with unique IDs"""
        return cls(
            request_id=kwargs.get('request_id', str(uuid.uuid4())),
            session_id=kwargs.get('session_id'),
            user_id=kwargs.get('user_id'),
            agent_id=kwargs.get('agent_id'),
            workflow_id=kwargs.get('workflow_id'),
            parent_operation_id=kwargs.get('parent_operation_id'),
            trace_id=kwargs.get('trace_id', str(uuid.uuid4()))
        )


class BaseLogSchema(BaseModel):
    """Base schema for all structured log entries"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: LogLevel
    operation_type: OperationType
    message: str
    correlation: CorrelationContext
    component: str
    version: str = "1.0.0"
    environment: str = "development"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            CorrelationContext: lambda v: asdict(v)
        }


class APIRequestSchema(BaseLogSchema):
    """Schema for API request logging"""
    operation_type: OperationType = OperationType.API_REQUEST
    method: str
    path: str
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, Any]] = None


class AgentOperationSchema(BaseLogSchema):
    """Schema for agent operation logging"""
    operation_type: OperationType = OperationType.AGENT_OPERATION
    agent_type: str
    agent_name: str
    phase: AgentPhase
    operation_id: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    decision_rationale: Optional[str] = None
    tools_used: Optional[List[str]] = None
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None


class WorkflowExecutionSchema(BaseLogSchema):
    """Schema for workflow execution logging"""
    operation_type: OperationType = OperationType.WORKFLOW_EXECUTION
    workflow_type: str
    workflow_name: str
    step_name: Optional[str] = None
    step_index: Optional[int] = None
    total_steps: Optional[int] = None
    agents_involved: Optional[List[str]] = None
    workflow_data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None


class ToolUsageSchema(BaseLogSchema):
    """Schema for tool usage logging"""
    operation_type: OperationType = OperationType.TOOL_USAGE
    tool_name: str
    tool_category: str
    operation: str
    input_parameters: Optional[Dict[str, Any]] = None
    output_result: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, Any]] = None


class DatabaseOperationSchema(BaseLogSchema):
    """Schema for database operation logging"""
    operation_type: OperationType = OperationType.DATABASE_OPERATION
    database_type: str  # "sqlalchemy", "surrealdb", etc.
    operation: str  # "select", "insert", "update", "delete", etc.
    table_name: Optional[str] = None
    query_hash: Optional[str] = None  # Hash of query for privacy
    rows_affected: Optional[int] = None
    execution_time_ms: Optional[float] = None
    connection_pool_info: Optional[Dict[str, Any]] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None


class ExternalServiceSchema(BaseLogSchema):
    """Schema for external service interaction logging"""
    operation_type: OperationType = OperationType.EXTERNAL_SERVICE
    service_name: str
    endpoint: Optional[str] = None
    method: Optional[str] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None
    retry_attempt: Optional[int] = None


class PerformanceMetricSchema(BaseLogSchema):
    """Schema for performance metrics logging"""
    operation_type: OperationType = OperationType.PERFORMANCE_METRIC
    metric_name: str
    metric_value: float
    metric_unit: str
    tags: Optional[Dict[str, str]] = None
    threshold_exceeded: bool = False
    benchmark_comparison: Optional[Dict[str, Any]] = None


class StructuredLogger:
    """Enhanced structured logger that builds on Logfire foundation"""

    def __init__(self, component: str, environment: str = "development"):
        self.component = component
        self.environment = environment
        self._context_stack: List[CorrelationContext] = []

    def _get_current_context(self) -> Optional[CorrelationContext]:
        """Get the current correlation context from the stack"""
        return self._context_stack[-1] if self._context_stack else None

    def _log_structured(self, schema: BaseLogSchema, extra_attributes: Optional[Dict[str, Any]] = None):
        """Log structured data using Logfire with proper context"""
        # Set component and environment
        schema.component = self.component
        schema.environment = self.environment

        # Convert to dict for logging
        log_data = schema.dict()

        # Extract main attributes for Logfire span
        span_name = f"{schema.operation_type.value}_{schema.correlation.request_id[:8]}"

        # Create span attributes
        attributes = {
            "operation_type": schema.operation_type.value,
            "component": schema.component,
            "level": schema.level.value,
            "request_id": schema.correlation.request_id,
        }

        # Add correlation IDs if available
        if schema.correlation.agent_id:
            attributes["agent_id"] = schema.correlation.agent_id
        if schema.correlation.workflow_id:
            attributes["workflow_id"] = schema.correlation.workflow_id
        if schema.correlation.trace_id:
            attributes["trace_id"] = schema.correlation.trace_id

        # Add extra attributes if provided
        if extra_attributes:
            attributes.update(extra_attributes)

        # Log using Logfire with appropriate level
        with logfire.span(span_name, **attributes):
            if schema.level == LogLevel.DEBUG:
                logfire.debug(schema.message, **log_data)
            elif schema.level == LogLevel.INFO:
                logfire.info(schema.message, **log_data)
            elif schema.level == LogLevel.WARNING:
                logfire.warning(schema.message, **log_data)
            elif schema.level == LogLevel.ERROR:
                logfire.error(schema.message, **log_data)
            elif schema.level == LogLevel.CRITICAL:
                logfire.error(schema.message, **log_data)

    @contextmanager
    def correlation_context(self, context: CorrelationContext):
        """Context manager for maintaining correlation context"""
        self._context_stack.append(context)
        try:
            yield context
        finally:
            self._context_stack.pop()

    def log_api_request(
        self,
        message: str,
        method: str,
        path: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log API request with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = APIRequestSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            method=method,
            path=path,
            **kwargs
        )

        self._log_structured(schema)

    def log_agent_operation(
        self,
        message: str,
        agent_type: str,
        agent_name: str,
        phase: AgentPhase,
        operation_id: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log agent operation with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = AgentOperationSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            agent_type=agent_type,
            agent_name=agent_name,
            phase=phase,
            operation_id=operation_id,
            **kwargs
        )

        self._log_structured(schema)

    def log_workflow_execution(
        self,
        message: str,
        workflow_type: str,
        workflow_name: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log workflow execution with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = WorkflowExecutionSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            workflow_type=workflow_type,
            workflow_name=workflow_name,
            **kwargs
        )

        self._log_structured(schema)

    def log_tool_usage(
        self,
        message: str,
        tool_name: str,
        tool_category: str,
        operation: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log tool usage with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = ToolUsageSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            tool_name=tool_name,
            tool_category=tool_category,
            operation=operation,
            **kwargs
        )

        self._log_structured(schema)

    def log_database_operation(
        self,
        message: str,
        database_type: str,
        operation: str,
        level: LogLevel = LogLevel.DEBUG,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log database operation with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = DatabaseOperationSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            database_type=database_type,
            operation=operation,
            **kwargs
        )

        self._log_structured(schema)

    def log_external_service(
        self,
        message: str,
        service_name: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log external service interaction with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = ExternalServiceSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            service_name=service_name,
            **kwargs
        )

        self._log_structured(schema)

    def log_performance_metric(
        self,
        message: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        level: LogLevel = LogLevel.INFO,
        correlation: Optional[CorrelationContext] = None,
        **kwargs
    ):
        """Log performance metric with structured data"""
        context = correlation or self._get_current_context() or CorrelationContext.generate()

        schema = PerformanceMetricSchema(
            level=level,
            message=message,
            correlation=context,
            component=self.component,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            **kwargs
        )

        self._log_structured(schema)


def timed_operation(logger: StructuredLogger, operation_name: str, operation_type: OperationType = OperationType.SYSTEM_EVENT):
    """Decorator for timing operations and logging performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            correlation = CorrelationContext.generate()

            with logger.correlation_context(correlation):
                try:
                    result = await func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000

                    logger.log_performance_metric(
                        message=f"Operation {operation_name} completed successfully",
                        metric_name=f"{operation_name}_execution_time",
                        metric_value=execution_time,
                        metric_unit="milliseconds",
                        level=LogLevel.INFO,
                        correlation=correlation,
                        tags={"operation": operation_name, "success": "true"}
                    )

                    return result
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000

                    logger.log_performance_metric(
                        message=f"Operation {operation_name} failed after {execution_time:.2f}ms",
                        metric_name=f"{operation_name}_execution_time",
                        metric_value=execution_time,
                        metric_unit="milliseconds",
                        level=LogLevel.ERROR,
                        correlation=correlation,
                        tags={"operation": operation_name, "success": "false", "error": str(e)}
                    )
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            correlation = CorrelationContext.generate()

            with logger.correlation_context(correlation):
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000

                    logger.log_performance_metric(
                        message=f"Operation {operation_name} completed successfully",
                        metric_name=f"{operation_name}_execution_time",
                        metric_value=execution_time,
                        metric_unit="milliseconds",
                        level=LogLevel.INFO,
                        correlation=correlation,
                        tags={"operation": operation_name, "success": "true"}
                    )

                    return result
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000

                    logger.log_performance_metric(
                        message=f"Operation {operation_name} failed after {execution_time:.2f}ms",
                        metric_name=f"{operation_name}_execution_time",
                        metric_value=execution_time,
                        metric_unit="milliseconds",
                        level=LogLevel.ERROR,
                        correlation=correlation,
                        tags={"operation": operation_name, "success": "false", "error": str(e)}
                    )
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global logger instances for different components
api_logger = StructuredLogger("api")
agent_logger = StructuredLogger("agent")
workflow_logger = StructuredLogger("workflow")
tool_logger = StructuredLogger("tool")
database_logger = StructuredLogger("database")
system_logger = StructuredLogger("system")


# Convenience functions for common logging patterns
def get_logger(component: str) -> StructuredLogger:
    """Get or create a structured logger for a component"""
    return StructuredLogger(component)


def create_correlation_context(**kwargs) -> CorrelationContext:
    """Create a new correlation context with provided parameters"""
    return CorrelationContext.generate(**kwargs)


def log_error_with_context(
    logger: StructuredLogger,
    error: Exception,
    message: str,
    operation_type: OperationType = OperationType.ERROR_EVENT,
    correlation: Optional[CorrelationContext] = None,
    **extra_context
):
    """Log an error with full context and exception details"""
    context = correlation or logger._get_current_context() or CorrelationContext.generate()

    error_details = {
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": getattr(error, '__traceback__', None)
    }
    error_details.update(extra_context)

    # Use base schema for error logging
    schema = BaseLogSchema(
        level=LogLevel.ERROR,
        operation_type=operation_type,
        message=message,
        correlation=context,
        component=logger.component,
        environment=logger.environment
    )

    logger._log_structured(schema, {"error_details": error_details})


__all__ = [
    "LogLevel",
    "OperationType",
    "AgentPhase",
    "CorrelationContext",
    "BaseLogSchema",
    "APIRequestSchema",
    "AgentOperationSchema",
    "WorkflowExecutionSchema",
    "ToolUsageSchema",
    "DatabaseOperationSchema",
    "ExternalServiceSchema",
    "PerformanceMetricSchema",
    "StructuredLogger",
    "timed_operation",
    "api_logger",
    "agent_logger",
    "workflow_logger",
    "tool_logger",
    "database_logger",
    "system_logger",
    "get_logger",
    "create_correlation_context",
    "log_error_with_context"
]
