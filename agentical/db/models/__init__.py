"""
Database Models Package

This package contains all SQLAlchemy ORM models for the Agentical framework,
organized into domain-specific modules. These models define the database schema
and provide object-oriented access to database entities.

Features:
- SQLAlchemy ORM integration
- Base model with common fields
- Domain-specific model implementations
- Type annotations for IDE support
- Integration with Logfire observability
"""

from agentical.db.models.base import (
    BaseModel,
    TimestampMixin,
    UUIDMixin,
    SoftDeleteMixin,
    JSONSerializableMixin,
    MetadataMixin
)

from agentical.db.models.user import (
    User,
    Role,
    user_roles
)

from agentical.db.models.agent import (
    Agent,
    AgentCapability,
    AgentConfiguration,
    AgentExecution,
    AgentStatus,
    AgentType,
    ExecutionStatus as AgentExecutionStatus
)

from agentical.db.models.tool import (
    Tool,
    ToolCapability,
    ToolParameter,
    ToolExecution,
    ToolType,
    ToolStatus,
    ExecutionStatus as ToolExecutionStatus
)

from agentical.db.models.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStepExecution,
    WorkflowType,
    WorkflowStatus,
    ExecutionStatus as WorkflowExecutionStatus,
    StepType as WorkflowStepType,
    StepStatus as WorkflowStepStatus
)

from agentical.db.models.task import (
    Task,
    TaskExecution,
    TaskResult,
    TaskPriority,
    TaskStatus,
    TaskType,
    ExecutionStatus as TaskExecutionStatus
)

from agentical.db.models.playbook import (
    Playbook,
    PlaybookStep,
    PlaybookVariable,
    PlaybookExecution,
    PlaybookStepExecution,
    PlaybookTemplate,
    PlaybookCategory,
    PlaybookStatus,
    ExecutionStatus as PlaybookExecutionStatus,
    StepType as PlaybookStepType,
    StepStatus as PlaybookStepStatus,
    VariableType
)

# Export all models for easy imports
__all__ = [
    # Base models and mixins
    "BaseModel",
    "TimestampMixin",
    "UUIDMixin",
    "SoftDeleteMixin",
    "JSONSerializableMixin",
    "MetadataMixin",

    # User models
    "User",
    "Role",
    "user_roles",

    # Agent models
    "Agent",
    "AgentCapability",
    "AgentConfiguration",
    "AgentExecution",
    "AgentStatus",
    "AgentType",
    "AgentExecutionStatus",

    # Tool models
    "Tool",
    "ToolCapability",
    "ToolParameter",
    "ToolExecution",
    "ToolType",
    "ToolStatus",
    "ToolExecutionStatus",

    # Workflow models
    "Workflow",
    "WorkflowStep",
    "WorkflowExecution",
    "WorkflowStepExecution",
    "WorkflowType",
    "WorkflowStatus",
    "WorkflowExecutionStatus",
    "WorkflowStepType",
    "WorkflowStepStatus",

    # Task models
    "Task",
    "TaskExecution",
    "TaskResult",
    "TaskPriority",
    "TaskStatus",
    "TaskType",
    "TaskExecutionStatus",

    # Playbook models
    "Playbook",
    "PlaybookStep",
    "PlaybookVariable",
    "PlaybookExecution",
    "PlaybookStepExecution",
    "PlaybookTemplate",
    "PlaybookCategory",
    "PlaybookStatus",
    "PlaybookExecutionStatus",
    "PlaybookStepType",
    "PlaybookStepStatus",
    "VariableType",
]
