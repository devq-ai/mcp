"""
Workflow Engine Package for Agentical

This package contains the core workflow orchestration engine components
including the main workflow engine, execution context management, and
step execution logic.

Features:
- WorkflowEngine: Core orchestration and execution management
- ExecutionContext: Workflow state and variable management
- StepExecutor: Individual step processing and execution
- Performance monitoring and error handling
- Integration with Logfire observability
"""

from .workflow_engine import (
    WorkflowEngine,
    WorkflowEngineFactory
)

from .execution_context import (
    ExecutionContext,
    ExecutionPhase
)

from .step_executor import (
    StepExecutor,
    StepExecutionResult
)

__all__ = [
    # Core engine
    "WorkflowEngine",
    "WorkflowEngineFactory",

    # Execution management
    "ExecutionContext",
    "ExecutionPhase",

    # Step processing
    "StepExecutor",
    "StepExecutionResult"
]

# Engine package metadata
__version__ = "1.0.0"
__description__ = "Workflow orchestration engine for Agentical"
