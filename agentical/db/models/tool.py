"""
Tool Data Models for Agentical

This module defines the Tool-related models for the Agentical framework,
including tool entities, parameters, execution logs, and capabilities.

Features:
- Tool entity with parameter schema definitions
- Tool execution history and performance tracking
- Capability mapping and discovery
- Integration with agent and workflow systems
- Schema validation for inputs and outputs
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


class ToolType(Enum):
    """Tool categories and types."""
    # Core MCP tools
    FILESYSTEM = "filesystem"
    GIT = "git"
    MEMORY = "memory"
    FETCH = "fetch"
    SEQUENTIAL_THINKING = "sequential_thinking"

    # Knowledge and analysis
    PTOLEMIES = "ptolemies"
    CONTEXT7 = "context7"
    BAYES = "bayes"
    DARWIN = "darwin"

    # External integrations
    GITHUB = "github"
    CRAWL4AI = "crawl4ai"
    CALENDAR = "calendar"
    STRIPE = "stripe"

    # Development tools
    JUPYTER = "jupyter"
    SHADCN_UI = "shadcn_ui"
    MAGIC = "magic"

    # Specialized tools
    SOLVER_Z3 = "solver_z3"
    SOLVER_PYSAT = "solver_pysat"
    SOLVER_MZN = "solver_mzn"

    # Database tools
    SURREALDB = "surrealdb"

    # Custom tools
    CUSTOM = "custom"
    API = "api"
    SCRIPT = "script"


class ToolStatus(Enum):
    """Tool operational status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    ERROR = "error"


class ExecutionStatus(Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Tool(BaseModel):
    """Tool entity with parameter schemas and capabilities."""

    # Basic identification
    name = Column(String(100), nullable=False, unique=True, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Tool classification
    tool_type = Column(SQLEnum(ToolType), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    tags = Column(JSON, nullable=True)  # List of tags for categorization

    # Status and availability
    status = Column(SQLEnum(ToolStatus), default=ToolStatus.AVAILABLE, nullable=False, index=True)
    is_public = Column(Boolean, default=True, nullable=False)
    requires_auth = Column(Boolean, default=False, nullable=False)

    # Schema definitions
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    parameters_schema = Column(JSON, nullable=True)

    # Configuration and connection
    configuration = Column(JSON, nullable=True)
    connection_string = Column(String(500), nullable=True)
    endpoint_url = Column(String(500), nullable=True)

    # Performance and usage
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    average_execution_time = Column(Float, default=0.0, nullable=False)

    # Cost and billing
    cost_per_execution = Column(Float, nullable=True)
    total_cost = Column(Float, default=0.0, nullable=False)

    # Timestamps
    last_execution_at = Column(DateTime, nullable=True)
    last_health_check_at = Column(DateTime, nullable=True)

    # Ownership
    created_by_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Relationships
    created_by = relationship("User", back_populates="created_tools")
    capabilities = relationship("ToolCapability", back_populates="tool", cascade="all, delete-orphan")
    executions = relationship("ToolExecution", back_populates="tool", cascade="all, delete-orphan")
    parameters = relationship("ToolParameter", back_populates="tool", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_tool_type_status', 'tool_type', 'status'),
        Index('idx_tool_category_public', 'category', 'is_public'),
        Index('idx_tool_last_execution', 'last_execution_at'),
    )

    def __repr__(self) -> str:
        return f"<Tool(name={self.name}, type={self.tool_type.value}, status={self.status.value})>"

    @validates('name')
    def validate_name(self, key, name):
        """Validate tool name."""
        if not name or len(name.strip()) < 2:
            raise ValueError("Tool name must be at least 2 characters")
        return name.strip().lower().replace(' ', '_')

    @validates('endpoint_url')
    def validate_endpoint_url(self, key, url):
        """Validate endpoint URL format."""
        if url and not (url.startswith('http://') or url.startswith('https://') or url.startswith('ws://')):
            raise ValueError("Endpoint URL must start with http://, https://, or ws://")
        return url

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate tool success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @hybrid_property
    def is_available(self) -> bool:
        """Check if tool is available for use."""
        return self.status == ToolStatus.AVAILABLE and self.is_active

    def get_tags(self) -> List[str]:
        """Get list of tags."""
        if not self.tags:
            return []
        return self.tags if isinstance(self.tags, list) else []

    def add_tag(self, tag: str) -> None:
        """Add tag to tool."""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag.lower().strip())
            self.tags = tags

    def remove_tag(self, tag: str) -> None:
        """Remove tag from tool."""
        tags = self.get_tags()
        if tag in tags:
            tags.remove(tag)
            self.tags = tags

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

    def update_performance_metrics(self, execution_time: float, success: bool, cost: float = 0.0) -> None:
        """Update tool performance metrics."""
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

        # Update cost tracking
        self.total_cost += cost
        self.last_execution_at = datetime.utcnow()

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate input data against schema."""
        errors = []

        if not self.input_schema:
            return True, []

        # Basic schema validation (can be extended with jsonschema)
        if isinstance(self.input_schema, dict):
            required_fields = self.input_schema.get('required', [])
            properties = self.input_schema.get('properties', {})

            for field in required_fields:
                if field not in input_data:
                    errors.append(f"Required field '{field}' is missing")

            for field, value in input_data.items():
                if field in properties:
                    field_schema = properties[field]
                    field_type = field_schema.get('type')

                    if field_type == 'string' and not isinstance(value, str):
                        errors.append(f"Field '{field}' must be a string")
                    elif field_type == 'integer' and not isinstance(value, int):
                        errors.append(f"Field '{field}' must be an integer")
                    elif field_type == 'number' and not isinstance(value, (int, float)):
                        errors.append(f"Field '{field}' must be a number")
                    elif field_type == 'boolean' and not isinstance(value, bool):
                        errors.append(f"Field '{field}' must be a boolean")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "tool_type": self.tool_type.value,
            "status": self.status.value,
            "success_rate": self.success_rate,
            "is_available": self.is_available,
            "tags": self.get_tags()
        })
        return result


class ToolCapability(BaseModel):
    """Tool capability definitions and requirements."""

    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False, index=True)
    capability_name = Column(String(100), nullable=False, index=True)
    capability_type = Column(String(50), nullable=False)  # 'input', 'output', 'processing'
    description = Column(Text, nullable=True)

    # Capability specifications
    data_types = Column(JSON, nullable=True)  # Supported data types
    operations = Column(JSON, nullable=True)  # Supported operations
    constraints = Column(JSON, nullable=True)  # Operational constraints

    # Performance characteristics
    min_response_time = Column(Float, nullable=True)
    max_response_time = Column(Float, nullable=True)
    throughput_limit = Column(Integer, nullable=True)

    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    tool = relationship("Tool", back_populates="capabilities")

    # Constraints
    __table_args__ = (
        UniqueConstraint('tool_id', 'capability_name', name='uq_tool_capability'),
        Index('idx_capability_type', 'capability_type'),
    )

    def __repr__(self) -> str:
        return f"<ToolCapability(tool_id={self.tool_id}, name={self.capability_name})>"

    def get_data_types(self) -> List[str]:
        """Get supported data types."""
        if not self.data_types:
            return []
        return self.data_types if isinstance(self.data_types, list) else []

    def get_operations(self) -> List[str]:
        """Get supported operations."""
        if not self.operations:
            return []
        return self.operations if isinstance(self.operations, list) else []

    def update_usage(self) -> None:
        """Update capability usage statistics."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()


class ToolParameter(BaseModel):
    """Tool parameter definitions and validation rules."""

    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    parameter_type = Column(String(50), nullable=False)  # 'string', 'integer', 'float', 'boolean', 'object'
    description = Column(Text, nullable=True)

    # Parameter constraints
    is_required = Column(Boolean, default=False, nullable=False)
    default_value = Column(JSON, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    min_length = Column(Integer, nullable=True)
    max_length = Column(Integer, nullable=True)
    pattern = Column(String(500), nullable=True)  # Regex pattern
    enum_values = Column(JSON, nullable=True)  # Allowed values

    # Documentation
    examples = Column(JSON, nullable=True)
    documentation_url = Column(String(500), nullable=True)

    # Relationships
    tool = relationship("Tool", back_populates="parameters")

    # Constraints
    __table_args__ = (
        UniqueConstraint('tool_id', 'parameter_name', name='uq_tool_parameter'),
        Index('idx_parameter_required', 'is_required'),
    )

    def __repr__(self) -> str:
        return f"<ToolParameter(tool_id={self.tool_id}, name={self.parameter_name}, type={self.parameter_type})>"

    def get_enum_values(self) -> List[Any]:
        """Get allowed enum values."""
        if not self.enum_values:
            return []
        return self.enum_values if isinstance(self.enum_values, list) else []

    def get_examples(self) -> List[Any]:
        """Get parameter examples."""
        if not self.examples:
            return []
        return self.examples if isinstance(self.examples, list) else []

    def validate_value(self, value: Any) -> tuple[bool, List[str]]:
        """Validate parameter value against constraints."""
        errors = []

        # Type validation
        if self.parameter_type == 'string' and not isinstance(value, str):
            errors.append(f"Parameter '{self.parameter_name}' must be a string")
        elif self.parameter_type == 'integer' and not isinstance(value, int):
            errors.append(f"Parameter '{self.parameter_name}' must be an integer")
        elif self.parameter_type == 'float' and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{self.parameter_name}' must be a number")
        elif self.parameter_type == 'boolean' and not isinstance(value, bool):
            errors.append(f"Parameter '{self.parameter_name}' must be a boolean")

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Parameter '{self.parameter_name}' must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Parameter '{self.parameter_name}' must be <= {self.max_value}")

        # Length validation for strings
        if isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(f"Parameter '{self.parameter_name}' must be at least {self.min_length} characters")
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(f"Parameter '{self.parameter_name}' must be at most {self.max_length} characters")

        # Enum validation
        enum_values = self.get_enum_values()
        if enum_values and value not in enum_values:
            errors.append(f"Parameter '{self.parameter_name}' must be one of: {enum_values}")

        return len(errors) == 0, errors


class ToolExecution(BaseModel):
    """Tool execution logs and performance tracking."""

    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution context
    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=True, index=True)
    workflow_id = Column(Integer, ForeignKey('workflow.id'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Execution details
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING, nullable=False, index=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_data = Column(JSON, nullable=True)

    # Performance metrics
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)

    # Resource usage
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    network_requests = Column(Integer, nullable=True)

    # Cost and billing
    cost = Column(Float, nullable=True)

    # Validation results
    input_validation_errors = Column(JSON, nullable=True)
    output_validation_errors = Column(JSON, nullable=True)

    # Relationships
    tool = relationship("Tool", back_populates="executions")
    agent = relationship("Agent", back_populates="tool_executions")
    workflow = relationship("Workflow", back_populates="tool_executions")
    user = relationship("User", back_populates="tool_executions")

    # Indexes for performance
    __table_args__ = (
        Index('idx_tool_execution_status_time', 'status', 'started_at'),
        Index('idx_tool_execution_tool_time', 'tool_id', 'started_at'),
        Index('idx_tool_execution_performance', 'execution_time_seconds'),
    )

    def __repr__(self) -> str:
        return f"<ToolExecution(id={self.execution_id}, tool_id={self.tool_id}, status={self.status.value})>"

    def start_execution(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete_execution(self, success: bool, output_data: Optional[Dict] = None, error_data: Optional[Dict] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.output_data = output_data
        self.error_data = error_data

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def timeout_execution(self) -> None:
        """Mark execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        if not self.error_data:
            self.error_data = {}
        self.error_data["timeout"] = True

    def cancel_execution(self, reason: str = None) -> None:
        """Cancel execution."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if not self.error_data:
            self.error_data = {}
        self.error_data["cancelled"] = True
        if reason:
            self.error_data["reason"] = reason

    @hybrid_property
    def duration(self) -> Optional[timedelta]:
        """Get execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if execution is in a completed state."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT, ExecutionStatus.CANCELLED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "status": self.status.value,
            "duration_seconds": self.execution_time_seconds,
            "is_completed": self.is_completed
        })
        return result
