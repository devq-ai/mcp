"""
Workflow Data Models for Agentical

This module defines the Workflow-related models for the Agentical framework,
including workflow entities, steps, executions, and templates.

Features:
- Workflow entity with multi-step process orchestration
- Workflow step management with execution order
- Execution history and state tracking
- Template system for reusable workflow patterns
- Integration with agent and task systems
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json

from sqlalchemy import (
    Column, String, Integer, Text, Boolean, DateTime, ForeignKey,
    Enum as SQLEnum, JSON, Float, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property

from .base import BaseModel


class WorkflowType(Enum):
    """Workflow categories and types."""
    # Core workflow types
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PIPELINE = "pipeline"

    # Business workflows
    DATA_PROCESSING = "data_processing"
    CONTENT_GENERATION = "content_generation"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"

    # Development workflows
    CI_CD = "ci_cd"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    CODE_REVIEW = "code_review"

    # Custom workflows
    CUSTOM = "custom"
    TEMPLATE = "template"


class WorkflowStatus(Enum):
    """Workflow operational status."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExecutionStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    TIMEOUT = "timeout"


class StepType(Enum):
    """Workflow step types."""
    AGENT_TASK = "agent_task"
    TOOL_EXECUTION = "tool_execution"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    WEBHOOK = "webhook"
    SCRIPT = "script"
    HUMAN_INPUT = "human_input"
    DATA_TRANSFORM = "data_transform"


class StepStatus(Enum):
    """Workflow step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class Workflow(BaseModel):
    """Workflow entity for multi-step process orchestration."""

    # Basic identification
    name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Workflow classification
    workflow_type = Column(SQLEnum(WorkflowType), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    tags = Column(JSON, nullable=True)  # List of tags for categorization

    # Status and availability
    status = Column(SQLEnum(WorkflowStatus), default=WorkflowStatus.DRAFT, nullable=False, index=True)
    is_public = Column(Boolean, default=False, nullable=False)
    is_template = Column(Boolean, default=False, nullable=False)

    # Configuration
    configuration = Column(JSON, nullable=True)
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    variables = Column(JSON, nullable=True)  # Workflow variables and defaults

    # Execution settings
    max_concurrent_executions = Column(Integer, default=1, nullable=False)
    timeout_minutes = Column(Integer, nullable=True)
    retry_attempts = Column(Integer, default=0, nullable=False)

    # Performance and usage
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    average_execution_time = Column(Float, default=0.0, nullable=False)

    # Versioning
    version = Column(String(20), default="1.0", nullable=False)
    parent_workflow_id = Column(Integer, ForeignKey('workflow.id'), nullable=True, index=True)

    # Timestamps
    last_execution_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)

    # Ownership
    created_by_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Relationships
    created_by = relationship("User", back_populates="created_workflows")
    parent_workflow = relationship("Workflow", remote_side="Workflow.id")
    child_workflows = relationship("Workflow", remote_side="Workflow.parent_workflow_id")
    steps = relationship("WorkflowStep", back_populates="workflow", cascade="all, delete-orphan", order_by="WorkflowStep.step_order")
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")
    agent_executions = relationship("AgentExecution", back_populates="workflow")
    tool_executions = relationship("ToolExecution", back_populates="workflow")

    # Indexes for performance
    __table_args__ = (
        Index('idx_workflow_type_status', 'workflow_type', 'status'),
        Index('idx_workflow_category_public', 'category', 'is_public'),
        Index('idx_workflow_template', 'is_template'),
        Index('idx_workflow_parent', 'parent_workflow_id'),
    )

    def __repr__(self) -> str:
        return f"<Workflow(name={self.name}, type={self.workflow_type.value}, status={self.status.value})>"

    @validates('name')
    def validate_name(self, key, name):
        """Validate workflow name."""
        if not name or len(name.strip()) < 2:
            raise ValueError("Workflow name must be at least 2 characters")
        return name.strip()

    @validates('max_concurrent_executions')
    def validate_max_concurrent_executions(self, key, value):
        """Validate concurrent execution limit."""
        if value < 1 or value > 1000:
            raise ValueError("Max concurrent executions must be between 1 and 1000")
        return value

    @validates('timeout_minutes')
    def validate_timeout_minutes(self, key, value):
        """Validate timeout value."""
        if value is not None and (value < 1 or value > 43200):  # Max 30 days
            raise ValueError("Timeout must be between 1 and 43200 minutes")
        return value

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate workflow success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @hybrid_property
    def is_executable(self) -> bool:
        """Check if workflow can be executed."""
        return self.status == WorkflowStatus.ACTIVE and self.is_active and len(self.steps) > 0

    @hybrid_property
    def step_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)

    def get_tags(self) -> List[str]:
        """Get list of tags."""
        if not self.tags:
            return []
        return self.tags if isinstance(self.tags, list) else []

    def add_tag(self, tag: str) -> None:
        """Add tag to workflow."""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag.lower().strip())
            self.tags = tags

    def remove_tag(self, tag: str) -> None:
        """Remove tag from workflow."""
        tags = self.get_tags()
        if tag in tags:
            tags.remove(tag)
            self.tags = tags

    def get_variables(self) -> Dict[str, Any]:
        """Get workflow variables."""
        if not self.variables:
            return {}
        return self.variables if isinstance(self.variables, dict) else {}

    def set_variable(self, key: str, value: Any) -> None:
        """Set workflow variable."""
        variables = self.get_variables()
        variables[key] = value
        self.variables = variables

    def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if not self.configuration:
            return default
        return self.configuration.get(key, default)

    def set_configuration(self, key: str, value: Any) -> None:
        """Set configuration value."""
        if not self.configuration:
            self.configuration = {}
        self.configuration[key] = value

    def update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update workflow performance metrics."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = execution_time
        else:
            # Rolling average
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1) + execution_time) /
                self.total_executions
            )

        self.last_execution_at = datetime.utcnow()

    def publish(self) -> None:
        """Publish workflow for use."""
        self.status = WorkflowStatus.ACTIVE
        self.published_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive workflow."""
        self.status = WorkflowStatus.ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "workflow_type": self.workflow_type.value,
            "status": self.status.value,
            "success_rate": self.success_rate,
            "is_executable": self.is_executable,
            "step_count": self.step_count,
            "tags": self.get_tags(),
            "variables": self.get_variables()
        })
        return result


class WorkflowStep(BaseModel):
    """Individual workflow step with execution configuration."""

    workflow_id = Column(Integer, ForeignKey('workflow.id'), nullable=False, index=True)
    step_name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Step configuration
    step_type = Column(SQLEnum(StepType), nullable=False, index=True)
    step_order = Column(Integer, nullable=False, index=True)
    is_optional = Column(Boolean, default=False, nullable=False)
    is_parallel = Column(Boolean, default=False, nullable=False)

    # Execution configuration
    configuration = Column(JSON, nullable=True)
    input_mapping = Column(JSON, nullable=True)  # How to map workflow data to step input
    output_mapping = Column(JSON, nullable=True)  # How to map step output to workflow data
    conditions = Column(JSON, nullable=True)  # Conditions for step execution

    # Tool/Agent configuration
    agent_type = Column(String(50), nullable=True)
    tool_name = Column(String(100), nullable=True)
    parameters = Column(JSON, nullable=True)

    # Timeout and retry
    timeout_minutes = Column(Integer, nullable=True)
    retry_attempts = Column(Integer, default=0, nullable=False)
    retry_delay_seconds = Column(Integer, default=30, nullable=False)

    # Dependencies
    depends_on_steps = Column(JSON, nullable=True)  # List of step names this depends on

    # Performance tracking
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    average_execution_time = Column(Float, default=0.0, nullable=False)

    # Relationships
    workflow = relationship("Workflow", back_populates="steps")
    executions = relationship("WorkflowStepExecution", back_populates="step", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint('workflow_id', 'step_name', name='uq_workflow_step_name'),
        UniqueConstraint('workflow_id', 'step_order', name='uq_workflow_step_order'),
        Index('idx_step_type_order', 'step_type', 'step_order'),
        Index('idx_step_parallel', 'is_parallel'),
    )

    def __repr__(self) -> str:
        return f"<WorkflowStep(workflow_id={self.workflow_id}, name={self.step_name}, order={self.step_order})>"

    @validates('step_order')
    def validate_step_order(self, key, order):
        """Validate step order."""
        if order < 1:
            raise ValueError("Step order must be >= 1")
        return order

    @validates('retry_attempts')
    def validate_retry_attempts(self, key, attempts):
        """Validate retry attempts."""
        if attempts < 0 or attempts > 10:
            raise ValueError("Retry attempts must be between 0 and 10")
        return attempts

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def get_depends_on_steps(self) -> List[str]:
        """Get list of step dependencies."""
        if not self.depends_on_steps:
            return []
        return self.depends_on_steps if isinstance(self.depends_on_steps, list) else []

    def add_dependency(self, step_name: str) -> None:
        """Add step dependency."""
        dependencies = self.get_depends_on_steps()
        if step_name not in dependencies:
            dependencies.append(step_name)
            self.depends_on_steps = dependencies

    def remove_dependency(self, step_name: str) -> None:
        """Remove step dependency."""
        dependencies = self.get_depends_on_steps()
        if step_name in dependencies:
            dependencies.remove(step_name)
            self.depends_on_steps = dependencies

    def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if not self.configuration:
            return default
        return self.configuration.get(key, default)

    def set_configuration(self, key: str, value: Any) -> None:
        """Set configuration value."""
        if not self.configuration:
            self.configuration = {}
        self.configuration[key] = value

    def update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update step performance metrics."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = execution_time
        else:
            # Rolling average
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1) + execution_time) /
                self.total_executions
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "step_type": self.step_type.value,
            "success_rate": self.success_rate,
            "depends_on_steps": self.get_depends_on_steps()
        })
        return result


class WorkflowExecution(BaseModel):
    """Workflow execution tracking and state management."""

    workflow_id = Column(Integer, ForeignKey('workflow.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution context
    user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)
    triggered_by = Column(String(50), nullable=True)  # 'user', 'schedule', 'webhook', 'api'

    # Execution details
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING, nullable=False, index=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_data = Column(JSON, nullable=True)
    context_data = Column(JSON, nullable=True)  # Execution context and variables

    # Performance metrics
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)

    # Progress tracking
    current_step_order = Column(Integer, nullable=True)
    completed_steps = Column(Integer, default=0, nullable=False)
    failed_steps = Column(Integer, default=0, nullable=False)
    skipped_steps = Column(Integer, default=0, nullable=False)

    # Resource usage
    total_cost = Column(Float, default=0.0, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)

    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    user = relationship("User", back_populates="workflow_executions")
    step_executions = relationship("WorkflowStepExecution", back_populates="workflow_execution", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_workflow_execution_status_time', 'status', 'started_at'),
        Index('idx_workflow_execution_workflow_time', 'workflow_id', 'started_at'),
        Index('idx_workflow_execution_user', 'user_id'),
    )

    def __repr__(self) -> str:
        return f"<WorkflowExecution(id={self.execution_id}, workflow_id={self.workflow_id}, status={self.status.value})>"

    @validates('execution_id')
    def validate_execution_id(self, key, execution_id):
        """Validate execution ID format."""
        if not execution_id or len(execution_id) < 10:
            raise ValueError("Execution ID must be at least 10 characters")
        return execution_id

    def start_execution(self, input_data: Optional[Dict] = None) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.input_data = input_data or {}
        self.current_step_order = 1

    def complete_execution(self, success: bool, output_data: Optional[Dict] = None, error_data: Optional[Dict] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.output_data = output_data
        self.error_data = error_data

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def pause_execution(self) -> None:
        """Pause execution."""
        self.status = ExecutionStatus.PAUSED

    def resume_execution(self) -> None:
        """Resume execution."""
        self.status = ExecutionStatus.RUNNING

    def cancel_execution(self, reason: str = None) -> None:
        """Cancel execution."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if reason:
            if not self.error_data:
                self.error_data = {}
            self.error_data["cancellation_reason"] = reason

    def timeout_execution(self) -> None:
        """Mark execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        if not self.error_data:
            self.error_data = {}
        self.error_data["timeout"] = True

    @hybrid_property
    def duration(self) -> Optional[timedelta]:
        """Get execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if execution is in a completed state."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]

    @hybrid_property
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status in [ExecutionStatus.RUNNING, ExecutionStatus.PAUSED]

    @hybrid_property
    def progress_percentage(self) -> float:
        """Calculate execution progress percentage."""
        total_steps = self.completed_steps + self.failed_steps + self.skipped_steps
        if total_steps == 0:
            return 0.0
        # Estimate remaining steps based on workflow
        if self.workflow and self.workflow.step_count > 0:
            return min(100.0, (total_steps / self.workflow.step_count) * 100.0)
        return 0.0

    def update_progress(self, step_order: int, step_status: StepStatus) -> None:
        """Update execution progress."""
        self.current_step_order = step_order

        if step_status == StepStatus.COMPLETED:
            self.completed_steps += 1
        elif step_status == StepStatus.FAILED:
            self.failed_steps += 1
        elif step_status == StepStatus.SKIPPED:
            self.skipped_steps += 1

    def get_context_data(self) -> Dict[str, Any]:
        """Get execution context data."""
        if not self.context_data:
            return {}
        return self.context_data if isinstance(self.context_data, dict) else {}

    def set_context_variable(self, key: str, value: Any) -> None:
        """Set context variable."""
        context = self.get_context_data()
        context[key] = value
        self.context_data = context

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "status": self.status.value,
            "duration_seconds": self.execution_time_seconds,
            "is_completed": self.is_completed,
            "is_running": self.is_running,
            "progress_percentage": self.progress_percentage
        })
        return result


class WorkflowStepExecution(BaseModel):
    """Individual workflow step execution tracking."""

    workflow_execution_id = Column(Integer, ForeignKey('workflowexecution.id'), nullable=False, index=True)
    step_id = Column(Integer, ForeignKey('workflowstep.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution details
    status = Column(SQLEnum(StepStatus), default=StepStatus.PENDING, nullable=False, index=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_data = Column(JSON, nullable=True)

    # Performance metrics
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)

    # Retry tracking
    attempt_number = Column(Integer, default=1, nullable=False)
    max_attempts = Column(Integer, default=1, nullable=False)

    # Agent/Tool execution references
    agent_execution_id = Column(String(100), nullable=True)
    tool_execution_id = Column(String(100), nullable=True)

    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="step_executions")
    step = relationship("WorkflowStep", back_populates="executions")

    # Constraints
    __table_args__ = (
        Index('idx_step_execution_status_time', 'status', 'started_at'),
        Index('idx_step_execution_workflow', 'workflow_execution_id'),
        Index('idx_step_execution_attempt', 'attempt_number'),
    )

    def __repr__(self) -> str:
        return f"<WorkflowStepExecution(id={self.execution_id}, step_id={self.step_id}, status={self.status.value})>"

    def start_execution(self, input_data: Optional[Dict] = None) -> None:
        """Mark step execution as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.input_data = input_data

    def complete_execution(self, success: bool, output_data: Optional[Dict] = None, error_data: Optional[Dict] = None) -> None:
        """Mark step execution as completed."""
        self.status = StepStatus.COMPLETED if success else StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.output_data = output_data
        self.error_data = error_data

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def skip_execution(self, reason: str = None) -> None:
        """Mark step execution as skipped."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow()
        if reason:
            self.error_data = {"skip_reason": reason}

    def cancel_execution(self) -> None:
        """Cancel step execution."""
        self.status = StepStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    @hybrid_property
    def duration(self) -> Optional[timedelta]:
        """Get step execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if step execution is in a completed state."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED]

    @hybrid_property
    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.status == StepStatus.FAILED and self.attempt_number < self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Convert step execution to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "status": self.status.value,
            "duration_seconds": self.execution_time_seconds,
            "is_completed": self.is_completed,
            "can_retry": self.can_retry
        })
        return result
