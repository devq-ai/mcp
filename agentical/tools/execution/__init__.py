"""
Execution Package for Agentical Tools

This package contains the tool execution framework including the tool executor,
execution context management, and result handling for the Agentical tool system.

Features:
- ToolExecutor: Unified tool execution interface
- ExecutionContext: Execution state and context management
- ToolExecutionResult: Comprehensive execution result handling
- Async execution with timeout and cancellation support
- Batch execution and parallel processing
- Performance monitoring and metrics collection
- Error handling and retry logic
- Integration with MCP servers and tool registry
"""

from .tool_executor import (
    ToolExecutor,
    ToolExecutionResult,
    ExecutionContext,
    ExecutionMode,
    ExecutionPriority
)

__all__ = [
    # Executor components
    "ToolExecutor",
    "ToolExecutionResult",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionPriority"
]

# Package metadata
__version__ = "1.0.0"
__description__ = "Tool execution framework for Agentical"

# Execution configuration defaults
DEFAULT_EXECUTION_CONFIG = {
    "max_concurrent_executions": 50,
    "default_timeout_seconds": 300,
    "retry_attempts": 3,
    "enable_monitoring": True,
    "batch_size_limit": 100,
    "parallel_limit": 10
}

# Execution modes and their characteristics
EXECUTION_MODE_INFO = {
    "sync": {
        "description": "Synchronous execution with blocking behavior",
        "use_case": "Simple operations requiring immediate results",
        "performance": "Lower concurrency, immediate feedback"
    },
    "async": {
        "description": "Asynchronous execution with non-blocking behavior",
        "use_case": "Most tool executions, allows high concurrency",
        "performance": "High concurrency, optimal for I/O operations"
    },
    "batch": {
        "description": "Batch execution of multiple tools",
        "use_case": "Processing multiple tools with shared context",
        "performance": "Optimized for bulk operations"
    },
    "stream": {
        "description": "Streaming execution with real-time results",
        "use_case": "Long-running operations with progressive output",
        "performance": "Real-time feedback, memory efficient"
    }
}

# Priority levels and their impact
PRIORITY_LEVELS = {
    "low": {
        "description": "Low priority execution",
        "queue_position": "Back of queue",
        "timeout_multiplier": 1.5,
        "retry_multiplier": 1.0
    },
    "normal": {
        "description": "Normal priority execution",
        "queue_position": "Standard queue position",
        "timeout_multiplier": 1.0,
        "retry_multiplier": 1.0
    },
    "high": {
        "description": "High priority execution",
        "queue_position": "Front of queue",
        "timeout_multiplier": 0.8,
        "retry_multiplier": 1.2
    },
    "critical": {
        "description": "Critical priority execution",
        "queue_position": "Immediate processing",
        "timeout_multiplier": 0.6,
        "retry_multiplier": 1.5
    }
}

# Performance thresholds for execution monitoring
EXECUTION_THRESHOLDS = {
    "fast_execution_seconds": 1.0,
    "normal_execution_seconds": 10.0,
    "slow_execution_seconds": 30.0,
    "timeout_warning_seconds": 120.0,
    "memory_warning_mb": 100,
    "memory_critical_mb": 500
}

def get_execution_info() -> dict:
    """Get comprehensive information about the execution framework."""
    return {
        "package_version": __version__,
        "default_config": DEFAULT_EXECUTION_CONFIG,
        "execution_modes": EXECUTION_MODE_INFO,
        "priority_levels": PRIORITY_LEVELS,
        "performance_thresholds": EXECUTION_THRESHOLDS,
        "supported_features": [
            "async_execution",
            "batch_processing",
            "timeout_handling",
            "retry_logic",
            "priority_queuing",
            "performance_monitoring",
            "error_tracking",
            "cancellation_support",
            "context_management",
            "result_serialization"
        ]
    }

def validate_execution_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate execution configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    # Validate max_concurrent_executions
    if "max_concurrent_executions" in config:
        value = config["max_concurrent_executions"]
        if not isinstance(value, int) or value < 1 or value > 1000:
            errors.append("max_concurrent_executions must be an integer between 1 and 1000")

    # Validate default_timeout_seconds
    if "default_timeout_seconds" in config:
        value = config["default_timeout_seconds"]
        if not isinstance(value, int) or value < 1 or value > 3600:
            errors.append("default_timeout_seconds must be an integer between 1 and 3600")

    # Validate retry_attempts
    if "retry_attempts" in config:
        value = config["retry_attempts"]
        if not isinstance(value, int) or value < 0 or value > 10:
            errors.append("retry_attempts must be an integer between 0 and 10")

    # Validate batch_size_limit
    if "batch_size_limit" in config:
        value = config["batch_size_limit"]
        if not isinstance(value, int) or value < 1 or value > 1000:
            errors.append("batch_size_limit must be an integer between 1 and 1000")

    # Validate parallel_limit
    if "parallel_limit" in config:
        value = config["parallel_limit"]
        if not isinstance(value, int) or value < 1 or value > 100:
            errors.append("parallel_limit must be an integer between 1 and 100")

    return len(errors) == 0, errors

def create_default_executor(tool_registry, mcp_client) -> 'ToolExecutor':
    """
    Create a tool executor with default configuration.

    Args:
        tool_registry: Tool registry instance
        mcp_client: MCP client instance

    Returns:
        ToolExecutor: Configured executor instance
    """
    return ToolExecutor(
        tool_registry=tool_registry,
        mcp_client=mcp_client,
        **DEFAULT_EXECUTION_CONFIG
    )
