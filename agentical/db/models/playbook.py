"""
Playbook Data Models for Agentical

This module defines the Playbook-related models for the Agentical framework,
including playbook entities, steps, variables, and executions.

Features:
- Playbook entity for strategic execution frameworks
- Playbook step management with execution sequences
- Variable management and templating
- Execution history and state tracking
- Integration with agent and workflow systems
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


class PlaybookCategory(Enum):
    """Playbook categories and types."""
    # Strategic categories
    INCIDENT_RESPONSE = "incident_response"
    TROUBLESHOOTING = "troubleshooting"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    SECURITY = "security"

    # Development playbooks
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    RELEASE = "release"
    ONBOARDING = "onboarding"

    # Operations playbooks
    MONITORING = "monitoring"
    BACKUP = "backup"
    DISASTER_RECOVERY = "disaster_recovery"
    CAPACITY_PLANNING = "capacity_planning"

    # Business playbooks
    CUSTOMER_SUPPORT = "customer_support"
    SALES = "sales"
    MARKETING = "marketing"
    COMPLIANCE = "compliance"

    # Custom playbooks
    CUSTOM = "custom"
    TEMPLATE = "template"


class PlaybookStatus(Enum):
    """Playbook operational status."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExecutionStatus(Enum):
    """Playbook execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(Enum):
    """Playbook step types."""
    ACTION = "action"
    DECISION = "decision"
    VERIFICATION = "verification"
    NOTIFICATION = "notification"
    WAIT = "wait"
    LOOP = "loop"
    CONDITION = "condition"
    PARALLEL = "parallel"
    AGENT_TASK = "agent_task"
    TOOL_EXECUTION = "tool_execution"
    HUMAN_INPUT = "human_input"
    SCRIPT = "script"


class StepStatus(Enum):
    """Playbook step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class VariableType(Enum):
    """Playbook variable types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"
    REFERENCE = "reference"


class Playbook(BaseModel):
    """Playbook entity for strategic execution frameworks."""

    # Basic identification
    name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Playbook classification
    category = Column(SQLEnum(PlaybookCategory), nullable=False, index=True)
    tags = Column(JSON, nullable=True)  # List of tags for categorization

    # Strategic information
    purpose = Column(Text, nullable=False)
    strategy = Column(Text, nullable=False)
    success_criteria = Column(JSON, nullable=True)  # List of success criteria

    # Status and availability
    status = Column(SQLEnum(PlaybookStatus), default=PlaybookStatus.DRAFT, nullable=False, index=True)
    is_public = Column(Boolean, default=False, nullable=False)
    is_template = Column(Boolean, default=False, nullable=False)

    # Configuration
    configuration = Column(JSON, nullable=True)
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)

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
    parent_playbook_id = Column(Integer, ForeignKey('playbook.id'), nullable=True, index=True)

    # Timestamps
    last_execution_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)

    # Ownership
    created_by_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Relationships
    created_by = relationship("User", back_populates="created_playbooks")
    parent_playbook = relationship("Playbook", remote_side="Playbook.id")
    child_playbooks = relationship("Playbook", remote_side="Playbook.parent_playbook_id")
    steps = relationship("PlaybookStep", back_populates="playbook", cascade="all, delete-orphan", order_by="PlaybookStep.step_order")
    variables = relationship("PlaybookVariable", back_populates="playbook", cascade="all, delete-orphan")
    executions = relationship("PlaybookExecution", back_populates="playbook", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_playbook_category_status', 'category', 'status'),
        Index('idx_playbook_public', 'is_public'),
        Index('idx_playbook_template', 'is_template'),
        Index('idx_playbook_parent', 'parent_playbook_id'),
    )

    def __repr__(self) -> str:
        return f"<Playbook(name={self.name}, category={self.category.value}, status={self.status.value})>"

    @validates('name')
    def validate_name(self, key, name):
        """Validate playbook name."""
        if not name or len(name.strip()) < 2:
            raise ValueError("Playbook name must be at least 2 characters")
        return name.strip()

    @validates('max_concurrent_executions')
    def validate_max_concurrent_executions(self, key, value):
        """Validate concurrent execution limit."""
        if value < 1 or value > 100:
            raise ValueError("Max concurrent executions must be between 1 and 100")
        return value

    @validates('timeout_minutes')
    def validate_timeout_minutes(self, key, value):
        """Validate timeout value."""
        if value is not None and (value < 1 or value > 43200):  # Max 30 days
            raise ValueError("Timeout must be between 1 and 43200 minutes")
        return value

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate playbook success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @hybrid_property
    def is_executable(self) -> bool:
        """Check if playbook can be executed."""
        return self.status == PlaybookStatus.ACTIVE and self.is_active and len(self.steps) > 0

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
        """Add tag to playbook."""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag.lower().strip())
            self.tags = tags

    def remove_tag(self, tag: str) -> None:
        """Remove tag from playbook."""
        tags = self.get_tags()
        if tag in tags:
            tags.remove(tag)
            self.tags = tags

    def get_success_criteria(self) -> List[str]:
        """Get success criteria."""
        if not self.success_criteria:
            return []
        return self.success_criteria if isinstance(self.success_criteria, list) else []

    def add_success_criteria(self, criteria: str) -> None:
        """Add success criteria."""
        criteria_list = self.get_success_criteria()
        if criteria not in criteria_list:
            criteria_list.append(criteria)
            self.success_criteria = criteria_list

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
        """Update playbook performance metrics."""
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
        """Publish playbook for use."""
        self.status = PlaybookStatus.ACTIVE
        self.published_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive playbook."""
        self.status = PlaybookStatus.ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert playbook to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "category": self.category.value,
            "status": self.status.value,
            "success_rate": self.success_rate,
            "is_executable": self.is_executable,
            "step_count": self.step_count,
            "tags": self.get_tags(),
            "success_criteria": self.get_success_criteria()
        })
        return result


class PlaybookStep(BaseModel):
    """Individual playbook step with execution configuration."""

    playbook_id = Column(Integer, ForeignKey('playbook.id'), nullable=False, index=True)
    step_name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Step configuration
    step_type = Column(SQLEnum(StepType), nullable=False, index=True)
    step_order = Column(Integer, nullable=False, index=True)
    is_optional = Column(Boolean, default=False, nullable=False)
    is_manual = Column(Boolean, default=False, nullable=False)

    # Execution configuration
    configuration = Column(JSON, nullable=True)
    instructions = Column(Text, nullable=True)  # Human-readable instructions
    automation_script = Column(Text, nullable=True)  # Automation code

    # Conditions and dependencies
    conditions = Column(JSON, nullable=True)  # Conditions for step execution
    depends_on_steps = Column(JSON, nullable=True)  # List of step names this depends on

    # Tool/Agent configuration
    agent_type = Column(String(50), nullable=True)
    tool_name = Column(String(100), nullable=True)
    parameters = Column(JSON, nullable=True)

    # Validation and verification
    verification_method = Column(String(50), nullable=True)  # 'manual', 'automated', 'none'
    verification_criteria = Column(JSON, nullable=True)
    success_indicators = Column(JSON, nullable=True)

    # Timeout and retry
    timeout_minutes = Column(Integer, nullable=True)
    retry_attempts = Column(Integer, default=0, nullable=False)
    retry_delay_seconds = Column(Integer, default=30, nullable=False)

    # Performance tracking
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    average_execution_time = Column(Float, default=0.0, nullable=False)

    # Relationships
    playbook = relationship("Playbook", back_populates="steps")
    executions = relationship("PlaybookStepExecution", back_populates="step", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint('playbook_id', 'step_name', name='uq_playbook_step_name'),
        UniqueConstraint('playbook_id', 'step_order', name='uq_playbook_step_order'),
        Index('idx_step_type_order', 'step_type', 'step_order'),
        Index('idx_step_manual', 'is_manual'),
        Index('idx_step_optional', 'is_optional'),
    )

    def __repr__(self) -> str:
        return f"<PlaybookStep(playbook_id={self.playbook_id}, name={self.step_name}, order={self.step_order})>"

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

    def get_verification_criteria(self) -> List[str]:
        """Get verification criteria."""
        if not self.verification_criteria:
            return []
        return self.verification_criteria if isinstance(self.verification_criteria, list) else []

    def get_success_indicators(self) -> List[str]:
        """Get success indicators."""
        if not self.success_indicators:
            return []
        return self.success_indicators if isinstance(self.success_indicators, list) else []

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
            "depends_on_steps": self.get_depends_on_steps(),
            "verification_criteria": self.get_verification_criteria(),
            "success_indicators": self.get_success_indicators()
        })
        return result


class PlaybookVariable(BaseModel):
    """Playbook variable definitions and management."""

    playbook_id = Column(Integer, ForeignKey('playbook.id'), nullable=False, index=True)
    variable_name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Variable type and constraints
    variable_type = Column(SQLEnum(VariableType), nullable=False, index=True)
    is_required = Column(Boolean, default=False, nullable=False)
    is_sensitive = Column(Boolean, default=False, nullable=False)

    # Value and defaults
    default_value = Column(JSON, nullable=True)
    current_value = Column(JSON, nullable=True)

    # Validation constraints
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    min_length = Column(Integer, nullable=True)
    max_length = Column(Integer, nullable=True)
    pattern = Column(String(500), nullable=True)  # Regex pattern
    enum_values = Column(JSON, nullable=True)  # Allowed values

    # Scope and visibility
    scope = Column(String(50), default="playbook", nullable=False)  # 'playbook', 'step', 'global'
    is_output = Column(Boolean, default=False, nullable=False)

    # Relationships
    playbook = relationship("Playbook", back_populates="variables")

    # Constraints
    __table_args__ = (
        UniqueConstraint('playbook_id', 'variable_name', name='uq_playbook_variable'),
        Index('idx_variable_type', 'variable_type'),
        Index('idx_variable_required', 'is_required'),
        Index('idx_variable_sensitive', 'is_sensitive'),
    )

    def __repr__(self) -> str:
        return f"<PlaybookVariable(playbook_id={self.playbook_id}, name={self.variable_name}, type={self.variable_type.value})>"

    @validates('variable_name')
    def validate_variable_name(self, key, name):
        """Validate variable name."""
        if not name or len(name.strip()) < 1:
            raise ValueError("Variable name must be at least 1 character")
        # Allow only alphanumeric and underscores
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError("Variable name must be a valid identifier")
        return name.strip()

    def get_enum_values(self) -> List[Any]:
        """Get allowed enum values."""
        if not self.enum_values:
            return []
        return self.enum_values if isinstance(self.enum_values, list) else []

    def validate_value(self, value: Any) -> tuple[bool, List[str]]:
        """Validate variable value against constraints."""
        errors = []

        # Required validation
        if self.is_required and value is None:
            errors.append(f"Variable '{self.variable_name}' is required")
            return False, errors

        if value is None:
            return True, []

        # Type validation
        if self.variable_type == VariableType.STRING and not isinstance(value, str):
            errors.append(f"Variable '{self.variable_name}' must be a string")
        elif self.variable_type == VariableType.INTEGER and not isinstance(value, int):
            errors.append(f"Variable '{self.variable_name}' must be an integer")
        elif self.variable_type == VariableType.FLOAT and not isinstance(value, (int, float)):
            errors.append(f"Variable '{self.variable_name}' must be a number")
        elif self.variable_type == VariableType.BOOLEAN and not isinstance(value, bool):
            errors.append(f"Variable '{self.variable_name}' must be a boolean")
        elif self.variable_type == VariableType.LIST and not isinstance(value, list):
            errors.append(f"Variable '{self.variable_name}' must be a list")
        elif self.variable_type == VariableType.DICT and not isinstance(value, dict):
            errors.append(f"Variable '{self.variable_name}' must be a dictionary")

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Variable '{self.variable_name}' must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Variable '{self.variable_name}' must be <= {self.max_value}")

        # Length validation for strings and lists
        if isinstance(value, (str, list)):
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(f"Variable '{self.variable_name}' must be at least {self.min_length} in length")
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(f"Variable '{self.variable_name}' must be at most {self.max_length} in length")

        # Pattern validation for strings
        if isinstance(value, str) and self.pattern:
            import re
            if not re.match(self.pattern, value):
                errors.append(f"Variable '{self.variable_name}' does not match required pattern")

        # Enum validation
        enum_values = self.get_enum_values()
        if enum_values and value not in enum_values:
            errors.append(f"Variable '{self.variable_name}' must be one of: {enum_values}")

        return len(errors) == 0, errors

    def set_value(self, value: Any) -> bool:
        """Set variable value with validation."""
        is_valid, errors = self.validate_value(value)
        if is_valid:
            self.current_value = value
        return is_valid

    def get_value(self) -> Any:
        """Get current value or default."""
        if self.current_value is not None:
            return self.current_value
        return self.default_value

    def reset_to_default(self) -> None:
        """Reset variable to default value."""
        self.current_value = self.default_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert variable to dictionary."""
        result = super().to_dict()
        result.update({
            "variable_type": self.variable_type.value,
            "has_value": self.current_value is not None,
            "enum_values": self.get_enum_values(),
            "current_value": self.get_value() if not self.is_sensitive else "[REDACTED]"
        })
        return result


class PlaybookExecution(BaseModel):
    """Playbook execution tracking and state management."""

    playbook_id = Column(Integer, ForeignKey('playbook.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution context
    user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)
    triggered_by = Column(String(50), nullable=True)  # 'user', 'schedule', 'incident', 'api'
    trigger_context = Column(JSON, nullable=True)  # Context data for the trigger

    # Execution details
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING, nullable=False, index=True)
    input_variables = Column(JSON, nullable=True)
    output_variables = Column(JSON, nullable=True)
    error_data = Column(JSON, nullable=True)

    # Performance metrics
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)

    # Progress tracking
    current_step_order = Column(Integer, nullable=True)
    completed_steps = Column(Integer, default=0, nullable=False)
    failed_steps = Column(Integer, default=0, nullable=False)
    skipped_steps = Column(Integer, default=0, nullable=False)

    # Results and outcomes
    success_criteria_met = Column(JSON, nullable=True)  # Which criteria were met
    overall_success = Column(Boolean, nullable=True)
    lessons_learned = Column(Text, nullable=True)

    # Relationships
    playbook = relationship("Playbook", back_populates="executions")
    user = relationship("User", back_populates="playbook_executions")
    step_executions = relationship("PlaybookStepExecution", back_populates="playbook_execution", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_playbook_execution_status_time', 'status', 'started_at'),
        Index('idx_playbook_execution_playbook_time', 'playbook_id', 'started_at'),
        Index('idx_playbook_execution_trigger', 'triggered_by'),
    )

    def __repr__(self) -> str:
        return f"<PlaybookExecution(id={self.execution_id}, playbook_id={self.playbook_id}, status={self.status.value})>"

    def start_execution(self, input_variables: Optional[Dict] = None) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.input_variables = input_variables or {}
        self.current_step_order = 1

    def complete_execution(self, success: bool, output_variables: Optional[Dict] = None, error_data: Optional[Dict] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.output_variables = output_variables
        self.error_data = error_data
        self.overall_success = success

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

    @hybrid_property
    def duration(self) -> Optional[timedelta]:
        """Get execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if execution is in a completed state."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]

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
        # Estimate remaining steps based on playbook
        if self.playbook and self.playbook.step_count > 0:
            return min(100.0, (total_steps / self.playbook.step_count) * 100.0)
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

    def get_success_criteria_met(self) -> List[str]:
        """Get list of success criteria that were met."""
        if not self.success_criteria_met:
            return []
        return self.success_criteria_met if isinstance(self.success_criteria_met, list) else []

    def mark_success_criteria(self, criteria: str, met: bool = True) -> None:
        """Mark a success criteria as met or not met."""
        criteria_met = self.get_success_criteria_met()
        if met and criteria not in criteria_met:
            criteria_met.append(criteria)
        elif not met and criteria in criteria_met:
            criteria_met.remove(criteria)
        self.success_criteria_met = criteria_met

    def get_input_variables(self) -> Dict[str, Any]:
        """Get input variables."""
        if not self.input_variables:
            return {}
        return self.input_variables if isinstance(self.input_variables, dict) else {}

    def get_output_variables(self) -> Dict[str, Any]:
        """Get output variables."""
        if not self.output_variables:
            return {}
        return self.output_variables if isinstance(self.output_variables, dict) else {}

    def set_variable(self, key: str, value: Any, is_output: bool = False) -> None:
        """Set execution variable."""
        if is_output:
            if not self.output_variables:
                self.output_variables = {}
            self.output_variables[key] = value
        else:
            if not self.input_variables:
                self.input_variables = {}
            self.input_variables[key] = value


class PlaybookStepExecution(BaseModel):
    """Individual step execution within a playbook execution."""

    __tablename__ = "playbook_step_executions"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(__import__('uuid').uuid4()))

    # Foreign keys
    playbook_execution_id = Column(String(36), ForeignKey("playbook_executions.id"), nullable=False)
    playbook_step_id = Column(String(36), ForeignKey("playbook_steps.id"), nullable=False)

    # Execution metadata
    step_order = Column(Integer, nullable=False)
    status = Column(SQLEnum(StepStatus), nullable=False, default=StepStatus.PENDING)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Execution context
    input_data = Column(JSON, nullable=True, default=dict)
    output_data = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)

    # Performance metrics
    execution_time_seconds = Column(Float, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # Relationships
    playbook_execution = relationship("PlaybookExecution", back_populates="step_executions")
    playbook_step = relationship("PlaybookStep")

    # Constraints
    __table_args__ = (
        UniqueConstraint('playbook_execution_id', 'playbook_step_id', name='_execution_step_uc'),
        Index('idx_step_execution_status', 'status'),
        Index('idx_step_execution_timing', 'started_at', 'completed_at'),
    )

    def __repr__(self) -> str:
        return f"<PlaybookStepExecution(id={self.id}, step={self.playbook_step_id}, status={self.status.value})>"

    def start_step(self) -> None:
        """Start step execution."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete_step(self, output_data: Dict[str, Any] = None) -> None:
        """Complete step execution successfully."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if output_data:
            self.output_data = output_data

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def fail_step(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def skip_step(self, reason: str = None) -> None:
        """Skip step execution."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow()
        if reason:
            self.error_message = f"Skipped: {reason}"

    @hybrid_property
    def duration(self) -> Optional[timedelta]:
        """Get step execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'playbook_execution_id': self.playbook_execution_id,
            'playbook_step_id': self.playbook_step_id,
            'step_order': self.step_order,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'execution_time_seconds': self.execution_time_seconds,
            'retry_count': self.retry_count,
            'duration': str(self.duration) if self.duration else None,
        }


class PlaybookTemplate(BaseModel):
    """Reusable playbook templates for common scenarios."""

    __tablename__ = "playbook_templates"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(__import__('uuid').uuid4()))

    # Template metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(SQLEnum(PlaybookCategory), nullable=False)
    version = Column(String(50), nullable=False, default="1.0.0")

    # Template structure
    template_data = Column(JSON, nullable=False)  # Contains steps, variables, config
    default_variables = Column(JSON, nullable=True, default=dict)

    # Usage metadata
    usage_count = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=True)

    # Publishing
    is_public = Column(Boolean, nullable=False, default=False)
    created_by = Column(String(36), nullable=False)

    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='_template_name_version_uc'),
        Index('idx_template_category', 'category'),
        Index('idx_template_public', 'is_public'),
    )

    def __repr__(self) -> str:
        return f"<PlaybookTemplate(id={self.id}, name={self.name}, version={self.version})>"

    def create_playbook(self, name: str, created_by: str, **kwargs) -> 'Playbook':
        """Create a new playbook from this template."""
        template_data = self.template_data or {}

        # Create playbook with template data
        playbook_data = {
            'name': name,
            'description': template_data.get('description', self.description),
            'category': self.category,
            'created_by': created_by,
            'tags': template_data.get('tags', []),
            'configuration': template_data.get('configuration', {}),
            **kwargs
        }

        # Increment usage count
        self.usage_count += 1

        return Playbook(**playbook_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'version': self.version,
            'template_data': self.template_data,
            'default_variables': self.default_variables,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'is_public': self.is_public,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
