"""
Agent Data Models for Agentical

This module defines the Agent-related models for the Agentical framework,
including agent entities, capabilities, executions, and configurations.

Features:
- Agent entity with polymorphic inheritance for specialized types
- Agent capability mapping and discovery
- Execution history and performance tracking
- Configuration management with validation
- Integration with tool and task systems
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


class AgentStatus(Enum):
    """Agent operational status."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentType(Enum):
    """Specialized agent types in the Agentical framework."""
    # Base agent types
    CODE_AGENT = "code_agent"
    DATA_SCIENCE_AGENT = "data_science_agent"
    DBA_AGENT = "dba_agent"
    DEVOPS_AGENT = "devops_agent"
    GCP_AGENT = "gcp_agent"
    GITHUB_AGENT = "github_agent"
    LEGAL_AGENT = "legal_agent"
    INFOSEC_AGENT = "infosec_agent"
    PULUMI_AGENT = "pulumi_agent"
    RESEARCH_AGENT = "research_agent"
    TESTER_AGENT = "tester_agent"
    TOKEN_AGENT = "token_agent"
    UAT_AGENT = "uat_agent"
    UX_AGENT = "ux_agent"

    # Custom agent types
    CODIFIER_AGENT = "codifier_agent"
    IO_AGENT = "io_agent"
    PLAYBOOK_AGENT = "playbook_agent"
    SUPER_AGENT = "super_agent"

    # Generic
    GENERIC_AGENT = "generic_agent"


class ExecutionStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Agent(BaseModel):
    """Core agent entity with configuration and state management."""

    # Basic identification
    name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)

    # Agent type and configuration
    agent_type = Column(SQLEnum(AgentType), nullable=False, index=True)
    status = Column(SQLEnum(AgentStatus), default=AgentStatus.INACTIVE, nullable=False, index=True)

    # Configuration and capabilities
    configuration = Column(JSON, nullable=True)
    available_tools = Column(JSON, nullable=True)  # List of tool names
    max_concurrent_executions = Column(Integer, default=1, nullable=False)

    # Performance and metrics
    total_executions = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    average_execution_time = Column(Float, default=0.0, nullable=False)

    # Timestamps
    last_execution_at = Column(DateTime, nullable=True)
    last_health_check_at = Column(DateTime, nullable=True)

    # Ownership and access
    created_by_user_id = Column(Integer, ForeignKey('user.id'), nullable=True, index=True)

    # Relationships
    created_by = relationship("User", back_populates="created_agents")
    capabilities = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
    executions = relationship("AgentExecution", back_populates="agent", cascade="all, delete-orphan")
    configurations = relationship("AgentConfiguration", back_populates="agent", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_agent_type_status', 'agent_type', 'status'),
        Index('idx_agent_created_by', 'created_by_user_id'),
        Index('idx_agent_last_execution', 'last_execution_at'),
    )

    def __repr__(self) -> str:
        return f"<Agent(name={self.name}, type={self.agent_type.value}, status={self.status.value})>"

    @validates('name')
    def validate_name(self, key, name):
        """Validate agent name."""
        if not name or len(name.strip()) < 2:
            raise ValueError("Agent name must be at least 2 characters")
        return name.strip()

    @validates('max_concurrent_executions')
    def validate_max_concurrent_executions(self, key, value):
        """Validate concurrent execution limit."""
        if value < 1 or value > 100:
            raise ValueError("Max concurrent executions must be between 1 and 100")
        return value

    @hybrid_property
    def success_rate(self) -> float:
        """Calculate agent success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @hybrid_property
    def is_available(self) -> bool:
        """Check if agent is available for execution."""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.INACTIVE] and self.is_active

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
        """Update agent performance metrics."""
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

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if not self.available_tools:
            return []
        return self.available_tools if isinstance(self.available_tools, list) else []

    def add_tool(self, tool_name: str) -> None:
        """Add tool to available tools."""
        tools = self.get_available_tools()
        if tool_name not in tools:
            tools.append(tool_name)
            self.available_tools = tools

    def remove_tool(self, tool_name: str) -> None:
        """Remove tool from available tools."""
        tools = self.get_available_tools()
        if tool_name in tools:
            tools.remove(tool_name)
            self.available_tools = tools

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "success_rate": self.success_rate,
            "is_available": self.is_available,
            "available_tools": self.get_available_tools(),
            "agent_type": self.agent_type.value,
            "status": self.status.value
        })
        return result


class AgentCapability(BaseModel):
    """Agent capability mapping for tool and skill management."""

    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=False, index=True)
    capability_name = Column(String(100), nullable=False, index=True)
    capability_type = Column(String(50), nullable=False)  # 'tool', 'skill', 'knowledge'
    description = Column(Text, nullable=True)

    # Capability configuration
    parameters = Column(JSON, nullable=True)
    required_tools = Column(JSON, nullable=True)  # List of required tool names

    # Performance tracking
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    success_rate = Column(Float, default=0.0, nullable=False)

    # Relationships
    agent = relationship("Agent", back_populates="capabilities")

    # Constraints
    __table_args__ = (
        UniqueConstraint('agent_id', 'capability_name', name='uq_agent_capability'),
        Index('idx_capability_type', 'capability_type'),
        Index('idx_capability_usage', 'usage_count'),
    )

    def __repr__(self) -> str:
        return f"<AgentCapability(agent_id={self.agent_id}, name={self.capability_name})>"

    @validates('capability_name')
    def validate_capability_name(self, key, name):
        """Validate capability name."""
        if not name or len(name.strip()) < 2:
            raise ValueError("Capability name must be at least 2 characters")
        return name.strip().lower().replace(' ', '_')

    def update_usage(self, success: bool) -> None:
        """Update capability usage statistics."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()

        # Update success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Rolling average
            current_successes = self.success_rate * (self.usage_count - 1)
            if success:
                current_successes += 1
            self.success_rate = current_successes / self.usage_count

    def get_required_tools(self) -> List[str]:
        """Get list of required tools."""
        if not self.required_tools:
            return []
        return self.required_tools if isinstance(self.required_tools, list) else []


class AgentConfiguration(BaseModel):
    """Agent configuration versioning and management."""

    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=False, index=True)
    config_name = Column(String(100), nullable=False, index=True)
    config_version = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)

    # Configuration data
    configuration_data = Column(JSON, nullable=False)
    schema_version = Column(String(10), default="1.0", nullable=False)

    # Status and validation
    is_active = Column(Boolean, default=False, nullable=False)
    is_validated = Column(Boolean, default=False, nullable=False)
    validation_errors = Column(JSON, nullable=True)

    # Timestamps
    applied_at = Column(DateTime, nullable=True)
    validated_at = Column(DateTime, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="configurations")

    # Constraints
    __table_args__ = (
        UniqueConstraint('agent_id', 'config_name', 'config_version', name='uq_agent_config_version'),
        Index('idx_config_active', 'is_active'),
        Index('idx_config_applied', 'applied_at'),
    )

    def __repr__(self) -> str:
        return f"<AgentConfiguration(agent_id={self.agent_id}, name={self.config_name}, version={self.config_version})>"

    def activate(self) -> None:
        """Activate this configuration."""
        self.is_active = True
        self.applied_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate this configuration."""
        self.is_active = False

    def validate_configuration(self) -> bool:
        """Validate configuration data."""
        errors = []

        if not self.configuration_data:
            errors.append("Configuration data is required")

        # Add specific validation logic here
        if isinstance(self.configuration_data, dict):
            # Validate required fields based on agent type
            required_fields = ["name", "type"]
            for field in required_fields:
                if field not in self.configuration_data:
                    errors.append(f"Required field '{field}' is missing")

        self.validation_errors = errors if errors else None
        self.is_validated = len(errors) == 0
        self.validated_at = datetime.utcnow()

        return self.is_validated


class AgentExecution(BaseModel):
    """Agent execution tracking with performance metrics."""

    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=False, index=True)
    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    # Execution context
    task_id = Column(Integer, ForeignKey('task.id'), nullable=True, index=True)
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
    tools_used = Column(JSON, nullable=True)  # List of tool names used

    # Cost and billing
    token_usage = Column(Integer, nullable=True)
    cost_estimate = Column(Float, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="executions")
    task = relationship("Task", back_populates="agent_executions")
    workflow = relationship("Workflow", back_populates="agent_executions")
    user = relationship("User", back_populates="agent_executions")

    # Indexes for performance
    __table_args__ = (
        Index('idx_execution_status_time', 'status', 'started_at'),
        Index('idx_execution_agent_time', 'agent_id', 'started_at'),
        Index('idx_execution_performance', 'execution_time_seconds'),
    )

    def __repr__(self) -> str:
        return f"<AgentExecution(id={self.execution_id}, agent_id={self.agent_id}, status={self.status.value})>"

    @validates('execution_id')
    def validate_execution_id(self, key, execution_id):
        """Validate execution ID format."""
        if not execution_id or len(execution_id) < 10:
            raise ValueError("Execution ID must be at least 10 characters")
        return execution_id

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

    def cancel_execution(self, reason: str = None) -> None:
        """Cancel execution."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if reason:
            self.error_data = {"cancellation_reason": reason}

    def timeout_execution(self) -> None:
        """Mark execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        self.error_data = {"error": "Execution timeout"}

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

    def get_tools_used(self) -> List[str]:
        """Get list of tools used during execution."""
        if not self.tools_used:
            return []
        return self.tools_used if isinstance(self.tools_used, list) else []

    def add_tool_usage(self, tool_name: str) -> None:
        """Add tool to the list of used tools."""
        tools = self.get_tools_used()
        if tool_name not in tools:
            tools.append(tool_name)
            self.tools_used = tools

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary with computed fields."""
        result = super().to_dict()
        result.update({
            "status": self.status.value,
            "duration_seconds": self.execution_time_seconds,
            "is_completed": self.is_completed,
            "tools_used": self.get_tools_used()
        })
        return result
