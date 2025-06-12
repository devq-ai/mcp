"""
Task Data Models for Agentical

This module defines the Task-related models for the Agentical framework,
including task entities, dependencies, executions, and results.

Features:
- Task entity with priority and status management
- Task dependency tracking with complex relationships
- Execution history and progress tracking
- Result storage and performance metrics
- Integration with agent and workflow systems
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json

from sqlalchemy import (
    Column, String, Integer, Text, Boolean, DateTime, ForeignKey,
    Enum as SQLEnum, JSON, Float, Index, UniqueConstraint, Table
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property

from .base import BaseModel


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """Task status values."""
    BACKLOG = "backlog"
    PLANNING = "planning"
    TODO = "todo"
    DOING = "doing"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class TaskType(Enum):
    """Task categories and types."""
    # Development tasks
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    ENHANCEMENT = "enhancement"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"

    # Operations tasks
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    MONITORING = "monitoring"
    BACKUP = "backup"

    # Research and analysis
    RESEARCH = "research"
    ANALYSIS = "analysis"
    INVESTIGATION = "investigation"
    PROTOTYPE = "prototype"

    # General tasks
    MEETING = "meeting"
    REVIEW = "review"
    TRAINING = "training"
    ADMIN = "admin"
    CUSTOM = "custom"


class ExecutionStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# Task dependency association table
task_dependencies = Table(
    'task_dependencies',
    BaseModel.metadata,
    Column('task_id', Integer, ForeignKey('task.id'), primary_key=True),
    Column('depends_on_task_id', Integer, ForeignKey('task.id'), primary_key=True),
    Column('dependency_type', String(50), default='blocks'),  # 'blocks', 'relates_to', 'duplicates'
    Column('created_at', DateTime, default=datetime.utcnow),
    Index('idx_task_dep_task', 'task_id'),
    Index('idx_task_dep_depends', 'depends_on_task_id'),
    Index('idx_task_dep_type', 'dependency_type')
)


class Task(BaseModel):
    """Task entity with dependency tracking and execution state."""

    # Basic identification
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)

    # Task classification
    task_type = Column(SQLEnum(TaskType), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    tags = Column(JSON, nullable=True)  # List of tags for categorization

    # Status and priority
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.BACKLOG, nullable=False, index=True)
    priority = Column(SQLEnum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False, index=True)

    # Planning and estimation
    story_points = Column(Integer, nullable=True)
    estimated_hours = Column(Float, nullable=True)
    actual_hours = Column(Float, nullable=True)
    complexity_score = Column(Integer, nullable=True)  # 1-10 scale

    # Progress tracking
    progress_percentage = Column(Float, default=0.0, nullable=False)
    subtasks_completed = Column(Integer, default=0, nullable=False)
    subtasks_total = Column(Integer, default=0, nullable=False)

    # Dates and deadlines
    due_date = Column(DateTime, nullable=True)
    start_date = Column(DateTime, nullable=True)
    completed_date = Column(DateTime, nullable=True)

    # Assignment and ownership
    assigned_to_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)
    created_by_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)
    project_id = Column(Integer, nullable=True, index=True)  # External project reference

    # Workflow integration
    workflow_id = Column(Integer, ForeignKey('workflow.id'), nullable=True, index=True)
    workflow_step_order = Column(Integer, nullable=True)

    # Parent-child relationships
    parent_task_id = Column(Integer, ForeignKey('task.id'), nullable=True, index=True)

    # Configuration and metadata
    configuration = Column(JSON, nullable=True)
    custom_fields = Column(JSON, nullable=True)
    attachments = Column(JSON, nullable=True)  # List of file references

    # Performance metrics
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)

    # Relationships
    assigned_to = relationship("User", foreign_keys=[assigned_to_user_id], back_populates="assigned_tasks")
    created_by = relationship("User", foreign_keys=[created_by_user_id], back_populates="created_tasks")
    workflow = relationship("Workflow", back_populates="tasks")
    parent_task = relationship("Task", remote_side="Task.id")
    subtasks = relationship("Task", remote_side="Task.parent_task_id")

    # Dependencies (many-to-many)
    dependencies = relationship(
        "Task",
        secondary=task_dependencies,
        primaryjoin=id == task_dependencies.c.task_id,
        secondaryjoin=id == task_dependencies.c.depends_on_task_id,
        back_populates="dependent_tasks"
    )
    dependent_tasks = relationship(
        "Task",
        secondary=task_dependencies,
        primaryjoin=id == task_dependencies.c.depends_on_task_id,
        secondaryjoin=id == task_dependencies.c.task_id,
        back_populates="dependencies"
    )

    executions = relationship("TaskExecution", back_populates="task", cascade="all, delete-orphan")
    results = relationship("TaskResult", back_populates="task", cascade="all, delete-orphan")
    agent_executions = relationship("AgentExecution", back_populates="task")

    # Indexes for performance
    __table_args__ = (
        Index('idx_task_status_priority', 'status', 'priority'),
        Index('idx_task_type_category', 'task_type', 'category'),
        Index('idx_task_assigned_status', 'assigned_to_user_id', 'status'),
        Index('idx_task_due_date', 'due_date'),
        Index('idx_task_project_status', 'project_id', 'status'),
        Index('idx_task_parent', 'parent_task_id'),
        Index('idx_task_workflow', 'workflow_id', 'workflow_step_order'),
    )

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title={self.title[:50]}, status={self.status.value})>"

    @validates('title')
    def validate_title(self, key, title):
        """Validate task title."""
        if not title or len(title.strip()) < 3:
            raise ValueError("Task title must be at least 3 characters")
        return title.strip()

    @validates('story_points')
    def validate_story_points(self, key, points):
        """Validate story points."""
        if points is not None and (points < 1 or points > 100):
            raise ValueError("Story points must be between 1 and 100")
        return points

    @validates('complexity_score')
    def validate_complexity_score(self, key, score):
        """Validate complexity score."""
        if score is not None and (score < 1 or score > 10):
            raise ValueError("Complexity score must be between 1 and 10")
        return score

    @validates('progress_percentage')
    def validate_progress_percentage(self, key, percentage):
        """Validate progress percentage."""
        if percentage < 0 or percentage > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        return percentage

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.DONE, TaskStatus.CANCELLED, TaskStatus.ARCHIVED]

    @hybrid_property
    def is_blocked(self) -> bool:
        """Check if task is blocked."""
        return self.status == TaskStatus.BLOCKED

    @hybrid_property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.due_date or self.is_completed:
            return False
        return datetime.utcnow() > self.due_date

    @hybrid_property
    def days_until_due(self) -> Optional[int]:
        """Calculate days until due date."""
        if not self.due_date or self.is_completed:
            return None
        delta = self.due_date - datetime.utcnow()
        return delta.days

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate task execution success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @hybrid_property
    def has_subtasks(self) -> bool:
        """Check if task has subtasks."""
        return self.subtasks_total > 0

    @hybrid_property
    def subtask_completion_rate(self) -> float:
        """Calculate subtask completion rate."""
        if self.subtasks_total == 0:
            return 0.0
        return self.subtasks_completed / self.subtasks_total

    def get_tags(self) -> List[str]:
        """Get list of tags."""
        if not self.tags:
            return []
        return self.tags if isinstance(self.tags, list) else []

    def add_tag(self, tag: str) -> None:
        """Add tag to task."""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag.lower().strip())
            self.tags = tags

    def remove_tag(self, tag: str) -> None:
        """Remove tag from task."""
        tags = self.get_tags()
        if tag in tags:
            tags.remove(tag)
            self.tags = tags

    def get_custom_fields(self) -> Dict[str, Any]:
        """Get custom fields."""
        if not self.custom_fields:
            return {}
        return self.custom_fields if isinstance(self.custom_fields, dict) else {}

    def set_custom_field(self, key: str, value: Any) -> None:
        """Set custom field value."""
        fields = self.get_custom_fields()
        fields[key] = value
        self.custom_fields = fields

    def get_attachments(self) -> List[Dict[str, Any]]:
        """Get task attachments."""
        if not self.attachments:
            return []
        return self.attachments if isinstance(self.attachments, list) else []

    def add_attachment(self, filename: str, url: str, size: int = None, mime_type: str = None) -> None:
        """Add attachment to task."""
        attachments = self.get_attachments()
        attachment = {
            "filename": filename,
            "url": url,
            "uploaded_at": datetime.utcnow().isoformat(),
            "size": size,
            "mime_type": mime_type
        }
        attachments.append(attachment)
        self.attachments = attachments

    def start_task(self) -> None:
        """Mark task as started."""
        if self.status == TaskStatus.TODO:
            self.status = TaskStatus.DOING
            self.start_date = datetime.utcnow()

    def complete_task(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.DONE
        self.completed_date = datetime.utcnow()
        self.progress_percentage = 100.0

    def block_task(self, reason: str = None) -> None:
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        if reason:
            self.set_custom_field("blocked_reason", reason)

    def unblock_task(self) -> None:
        """Remove block from task."""
        if self.status == TaskStatus.BLOCKED:
            self.status = TaskStatus.TODO
            custom_fields = self.get_custom_fields()
            if "blocked_reason" in custom_fields:
                del custom_fields["blocked_reason"]
                self.custom_fields = custom_fields

    def update_progress(self, percentage: float) -> None:
        """Update task progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))

        # Auto-complete if 100%
        if self.progress_percentage >= 100.0 and not self.is_completed:
            self.complete_task()

    def add_dependency(self, task: 'Task', dependency_type: str = 'blocks') -> None:
        """Add task dependency."""
        if task not in self.dependencies:
            self.dependencies.append(task)

    def remove_dependency(self, task: 'Task') -> None:
        """Remove task dependency."""
        if task in self.dependencies:
            self.dependencies.remove(task)

    def can_start(self) -> bool:
        """Check if task can be started (all dependencies completed)."""
        if self.status != TaskStatus.TODO:
            return False

        # Check if all dependencies are completed
        for dependency in self.dependencies:
            if not dependency.is_completed:
                return False

        return True

    def update_performance_metrics(self, success: bool) -> None:
        """Update task performance metrics."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

    def estimate_completion_date(self) -> Optional[datetime]:
        """Estimate completion date based on progress and remaining hours."""
        if self.is_completed or not self.estimated_hours or self.progress_percentage <= 0:
            return None

        if not self.start_date:
            return None

        # Calculate remaining hours
        remaining_progress = (100.0 - self.progress_percentage) / 100.0
        remaining_hours = self.estimated_hours * remaining_progress

        # Estimate completion (assuming 8 hours per day)
        remaining_days = remaining_hours / 8.0
        return datetime.utcnow() + timedelta(days=remaining_days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "is_completed": self.is_completed,
            "is_blocked": self.is_blocked,
            "is_overdue": self.is_overdue,
            "days_until_due": self.days_until_due,
            "success_rate": self.success_rate,
            "has_subtasks": self.has_subtasks,
            "subtask_completion_rate": self.subtask_completion_rate,
            "can_start": self.can_start(),
            "tags": self.get_tags(),
            "custom_fields": self.get_custom_fields(),
            "attachments": self.get_attachments(),
            "estimated_completion": self.estimate_completion_date().isoformat() if self.estimate_completion_date() else None
        })
        return result


class TaskExecution(BaseModel):
    """Task execution tracking and performance monitoring."""

    task_id = Column(Integer, ForeignKey('task.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution context
    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=True, index=True)
    workflow_execution_id = Column(Integer, ForeignKey('workflowexecution.id'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Execution details
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING, nullable=False, index=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_data = Column(JSON, nullable=True)
    context_data = Column(JSON, nullable=True)

    # Performance metrics
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)

    # Resource usage
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)

    # Progress tracking
    progress_percentage = Column(Float, default=0.0, nullable=False)
    steps_completed = Column(Integer, default=0, nullable=False)
    steps_total = Column(Integer, default=1, nullable=False)

    # Relationships
    task = relationship("Task", back_populates="executions")
    agent = relationship("Agent", back_populates="task_executions")
    workflow_execution = relationship("WorkflowExecution", back_populates="task_executions")
    user = relationship("User", back_populates="task_executions")

    # Indexes for performance
    __table_args__ = (
        Index('idx_task_execution_status_time', 'status', 'started_at'),
        Index('idx_task_execution_task_time', 'task_id', 'started_at'),
        Index('idx_task_execution_agent', 'agent_id'),
    )

    def __repr__(self) -> str:
        return f"<TaskExecution(id={self.execution_id}, task_id={self.task_id}, status={self.status.value})>"

    def start_execution(self, input_data: Optional[Dict] = None) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.input_data = input_data

    def complete_execution(self, success: bool, output_data: Optional[Dict] = None, error_data: Optional[Dict] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.output_data = output_data
        self.error_data = error_data
        self.progress_percentage = 100.0

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def update_progress(self, percentage: float, steps_completed: int = None) -> None:
        """Update execution progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if steps_completed is not None:
            self.steps_completed = steps_completed

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
        """Check if execution is completed."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        result = super().to_dict()
        result.update({
            "status": self.status.value,
            "duration_seconds": self.execution_time_seconds,
            "is_completed": self.is_completed
        })
        return result


class TaskResult(BaseModel):
    """Task execution results and outputs."""

    task_id = Column(Integer, ForeignKey('task.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=True, index=True)

    # Result classification
    result_type = Column(String(50), nullable=False, index=True)  # 'output', 'artifact', 'metric', 'error'
    result_name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Result data
    result_data = Column(JSON, nullable=True)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256 hash

    # Metadata
    format_type = Column(String(50), nullable=True)  # 'json', 'csv', 'image', 'text', etc.
    schema_version = Column(String(20), nullable=True)
    quality_score = Column(Float, nullable=True)  # 0.0 to 1.0

    # Validation
    is_validated = Column(Boolean, default=False, nullable=False)
    validation_errors = Column(JSON, nullable=True)
    validated_at = Column(DateTime, nullable=True)

    # Usage tracking
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime, nullable=True)

    # Relationships
    task = relationship("Task", back_populates="results")

    # Indexes for performance
    __table_args__ = (
        Index('idx_task_result_type', 'result_type'),
        Index('idx_task_result_task_type', 'task_id', 'result_type'),
        Index('idx_task_result_validation', 'is_validated'),
    )

    def __repr__(self) -> str:
        return f"<TaskResult(task_id={self.task_id}, name={self.result_name}, type={self.result_type})>"

    @validates('quality_score')
    def validate_quality_score(self, key, score):
        """Validate quality score."""
        if score is not None and (score < 0.0 or score > 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return score

    def validate_result(self) -> bool:
        """Validate result data."""
        errors = []

        # Basic validation
        if not self.result_data and not self.file_path:
            errors.append("Result must have either data or file path")

        # File validation
        if self.file_path:
            # Additional file validation could be added here
            pass

        # Data validation based on format
        if self.result_data and self.format_type:
            if self.format_type == 'json':
                try:
                    if isinstance(self.result_data, str):
                        json.loads(self.result_data)
                except json.JSONDecodeError:
                    errors.append("Invalid JSON format")

        self.validation_errors = errors if errors else None
        self.is_validated = len(errors) == 0
        self.validated_at = datetime.utcnow()

        return self.is_validated

    def record_access(self) -> None:
        """Record result access."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()

    def get_result_data(self) -> Any:
        """Get result data with access tracking."""
        self.record_access()
        return self.result_data

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = super().to_dict()
        result.update({
            "has_file": bool(self.file_path),
            "has_data": bool(self.result_data),
            "validation_status": "validated" if self.is_validated else "pending"
        })
        return result
