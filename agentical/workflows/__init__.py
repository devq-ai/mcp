"""
Workflows Package for Agentical

This package provides comprehensive workflow orchestration capabilities for the
Agentical framework, including workflow engine, registry, management, and
implementation of standard and graph-based workflow types.

Features:
- Workflow engine with multi-type execution support
- Workflow registry for discovery and management
- Standard workflow types (sequential, parallel, conditional, loop, pipeline)
- Pydantic-graph workflows for advanced AI agent coordination
- Comprehensive monitoring and observability with Logfire
- Integration with FastAPI and async operations
"""

from .engine.workflow_engine import (
    WorkflowEngine,
    WorkflowEngineFactory
)

from .engine.execution_context import (
    ExecutionContext,
    ExecutionPhase
)

from .engine.step_executor import (
    StepExecutor,
    StepExecutionResult
)

from .registry import (
    WorkflowRegistry,
    WorkflowRegistryEntry,
    WorkflowDiscoveryMode
)

from .manager import (
    WorkflowManager,
    WorkflowManagerState,
    WorkflowScheduleType
)

# Core workflow system components
__all__ = [
    # Engine components
    "WorkflowEngine",
    "WorkflowEngineFactory",
    "ExecutionContext",
    "ExecutionPhase",
    "StepExecutor",
    "StepExecutionResult",

    # Registry components
    "WorkflowRegistry",
    "WorkflowRegistryEntry",
    "WorkflowDiscoveryMode",

    # Manager components
    "WorkflowManager",
    "WorkflowManagerState",
    "WorkflowScheduleType"
]

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Package metadata
WORKFLOW_TYPES_SUPPORTED = [
    "sequential",
    "parallel",
    "conditional",
    "loop",
    "pipeline",
    "agent_feedback",
    "handoff",
    "human_loop",
    "self_feedback",
    "versus"
]

STEP_TYPES_SUPPORTED = [
    "agent_task",
    "tool_execution",
    "condition",
    "loop",
    "parallel",
    "wait",
    "webhook",
    "script",
    "human_input",
    "data_transform"
]

# Default configuration
DEFAULT_ENGINE_CONFIG = {
    "max_concurrent_workflows": 10,
    "default_timeout_minutes": 60,
    "enable_monitoring": True
}

DEFAULT_REGISTRY_CONFIG = {
    "discovery_mode": "hybrid",
    "auto_discover_paths": [],
    "enable_caching": True
}

DEFAULT_MANAGER_CONFIG = {
    "enable_scheduling": True,
    "max_queued_workflows": 1000
}
