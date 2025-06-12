"""
Enhanced Base Agent Architecture for Agentical Framework

This module provides the enhanced base agent architecture that integrates with
the repository pattern, observability framework, and provides comprehensive
lifecycle management for all agent types.

Features:
- Generic base agent class with full lifecycle management
- Repository pattern integration for state persistence
- Comprehensive observability with Logfire integration
- Resource management and constraint enforcement
- Error recovery and fault tolerance mechanisms
- Configuration management with validation
- Inter-agent communication framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generic, TypeVar, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import asyncio
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

import logfire
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from agentical.db.repositories.agent import AsyncAgentRepository
from agentical.db.models.agent import Agent, AgentStatus, AgentType, ExecutionStatus
from agentical.core.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentConfigurationError,
    ValidationError,
    NotFoundError
)
from agentical.core.structured_logging import (
    StructuredLogger,
    CorrelationContext,
    LogLevel,
    OperationType,
    AgentPhase
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for configuration types
ConfigType = TypeVar('ConfigType', bound=BaseModel)


class AgentState(Enum):
    """Enhanced agent state enumeration."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ResourceConstraints(BaseModel):
    """Resource constraints for agent execution."""
    max_memory_mb: Optional[int] = Field(default=512, description="Maximum memory usage in MB")
    max_cpu_percent: Optional[float] = Field(default=80.0, description="Maximum CPU usage percentage")
    max_execution_time_seconds: Optional[int] = Field(default=300, description="Maximum execution time")
    max_concurrent_operations: Optional[int] = Field(default=1, description="Maximum concurrent operations")
    max_disk_usage_mb: Optional[int] = Field(default=100, description="Maximum disk usage in MB")


class AgentConfiguration(BaseModel, Generic[ConfigType]):
    """Generic agent configuration with validation."""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: AgentType = Field(..., description="Type of the agent")
    name: str = Field(..., description="Human-readable agent name")
    description: Optional[str] = Field(default=None, description="Agent description")

    # Resource and execution constraints
    resource_constraints: ResourceConstraints = Field(default_factory=ResourceConstraints)
    timeout_seconds: int = Field(default=300, description="Default operation timeout")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed operations")

    # Agent-specific configuration
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools_enabled: List[str] = Field(default_factory=list, description="Available tools")

    # Observability settings
    enable_tracing: bool = Field(default=True, description="Enable execution tracing")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")

    # Custom configuration for specific agent types
    custom_config: Optional[ConfigType] = Field(default=None, description="Agent-type specific configuration")

    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Agent ID cannot be empty")
        return v.strip()

    @validator('resource_constraints')
    def validate_resource_constraints(cls, v):
        if v.max_memory_mb and v.max_memory_mb <= 0:
            raise ValueError("Memory constraint must be positive")
        if v.max_cpu_percent and (v.max_cpu_percent <= 0 or v.max_cpu_percent > 100):
            raise ValueError("CPU constraint must be between 0 and 100")
        return v


@dataclass
class ExecutionContext:
    """Comprehensive execution context for agent operations."""
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_context: CorrelationContext = field(default_factory=CorrelationContext.generate)
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    timeout_at: Optional[datetime] = None
    retry_count: int = 0

    # Resource tracking
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Observability
    logfire_spans: List[Any] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Comprehensive execution result with metrics and observability."""
    success: bool
    execution_id: str
    agent_id: str
    operation: str

    # Result data
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Performance metrics
    execution_time_ms: float = 0.0
    memory_peak_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Execution metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0

    # Observability
    logs_captured: int = 0
    spans_created: int = 0
    metrics_collected: Dict[str, Any] = field(default_factory=dict)


class EnhancedBaseAgent(ABC, Generic[ConfigType]):
    """
    Enhanced base agent class with comprehensive lifecycle management,
    repository integration, and observability.

    This class provides:
    - Full lifecycle management (initialize, execute, cleanup)
    - State persistence through repository pattern
    - Resource management and constraint enforcement
    - Comprehensive error handling and recovery
    - Observability with Logfire integration
    - Configuration validation and management
    """

    def __init__(
        self,
        config: AgentConfiguration[ConfigType],
        db_session: AsyncSession,
        logger_instance: Optional[StructuredLogger] = None
    ):
        """
        Initialize the enhanced base agent.

        Args:
            config: Agent configuration with validation
            db_session: Async database session for repository operations
            logger_instance: Optional structured logger instance
        """
        self.config = config
        self.db_session = db_session
        self.agent_repo = AsyncAgentRepository(db_session)

        # Setup logging
        self.logger = logger_instance or StructuredLogger(
            component=f"agent:{config.agent_type.value}",
            service_name="agentical-agent",
            correlation_context=CorrelationContext.generate(
                agent_id=config.agent_id
            )
        )

        # Agent state
        self.state = AgentState.INITIALIZING
        self.current_execution: Optional[ExecutionContext] = None
        self.execution_history: List[ExecutionResult] = []

        # Resource management
        self.allocated_resources: Dict[str, Any] = {}
        self.active_operations: Dict[str, ExecutionContext] = {}

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.average_execution_time = 0.0

        # Agent lifecycle hooks
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            'before_initialize': [],
            'after_initialize': [],
            'before_execute': [],
            'after_execute': [],
            'before_cleanup': [],
            'after_cleanup': [],
            'on_error': [],
            'on_state_change': []
        }

        with logfire.span("Agent initialization", agent_id=config.agent_id, agent_type=config.agent_type.value):
            self.logger.log_agent_operation(
                message=f"Initializing agent {config.agent_id}",
                agent_type=config.agent_type.value,
                agent_name=config.name,
                phase=AgentPhase.INITIALIZATION,
                level=LogLevel.INFO
            )

    async def initialize(self) -> None:
        """
        Initialize the agent with comprehensive setup and validation.

        This method:
        1. Validates configuration and resources
        2. Sets up infrastructure connections
        3. Registers the agent in the database
        4. Performs agent-specific initialization
        5. Updates agent state to IDLE
        """
        with logfire.span("Agent initialization", agent_id=self.config.agent_id):
            try:
                self.state = AgentState.INITIALIZING
                await self._execute_lifecycle_hooks('before_initialize')

                # Validate configuration
                await self._validate_configuration()

                # Allocate initial resources
                await self._allocate_resources()

                # Register or update agent in database
                await self._register_agent()

                # Agent-specific initialization
                await self._agent_initialize()

                # Final state update
                self.state = AgentState.IDLE
                await self._update_agent_state()

                await self._execute_lifecycle_hooks('after_initialize')

                self.logger.log_agent_operation(
                    message=f"Agent {self.config.agent_id} initialized successfully",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.INFO,
                    success=True
                )

            except Exception as e:
                self.state = AgentState.ERROR
                await self._update_agent_state()
                await self._execute_lifecycle_hooks('on_error', error=e)

                self.logger.log_agent_operation(
                    message=f"Agent initialization failed: {str(e)}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )

                raise AgentError(f"Agent initialization failed: {str(e)}")

    async def execute(self, operation: str, parameters: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute an agent operation with comprehensive monitoring and error handling.

        Args:
            operation: The operation to execute
            parameters: Operation parameters

        Returns:
            Execution result with metrics and observability data
        """
        execution_context = ExecutionContext(
            operation=operation,
            parameters=parameters or {},
            correlation_context=CorrelationContext.generate(
                agent_id=self.config.agent_id,
                request_id=str(uuid4())
            )
        )

        with logfire.span(
            "Agent execution",
            agent_id=self.config.agent_id,
            operation=operation,
            execution_id=execution_context.execution_id
        ):
            try:
                # Pre-execution validation and setup
                await self._pre_execution_setup(execution_context)

                # Execute the operation
                result_data = await self._execute_operation(execution_context)

                # Post-execution cleanup and metrics
                execution_result = await self._post_execution_cleanup(execution_context, result_data)

                return execution_result

            except Exception as e:
                # Handle execution errors
                return await self._handle_execution_error(execution_context, e)

    async def cleanup(self) -> None:
        """
        Cleanup agent resources and prepare for shutdown.

        This method:
        1. Completes any running operations
        2. Releases allocated resources
        3. Persists final state to database
        4. Performs agent-specific cleanup
        5. Updates state to STOPPED
        """
        with logfire.span("Agent cleanup", agent_id=self.config.agent_id):
            try:
                self.state = AgentState.STOPPING
                await self._execute_lifecycle_hooks('before_cleanup')

                # Wait for active operations to complete or timeout
                await self._complete_active_operations()

                # Release resources
                await self._release_resources()

                # Agent-specific cleanup
                await self._agent_cleanup()

                # Final state update
                self.state = AgentState.STOPPED
                await self._update_agent_state()

                await self._execute_lifecycle_hooks('after_cleanup')

                self.logger.log_agent_operation(
                    message=f"Agent {self.config.agent_id} cleanup completed",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.INFO,
                    success=True
                )

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Agent cleanup failed: {str(e)}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )

                raise AgentError(f"Agent cleanup failed: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status including performance metrics.

        Returns:
            Dictionary containing agent status, metrics, and operational data
        """
        with logfire.span("Get agent status", agent_id=self.config.agent_id):
            # Get metrics from repository
            agent_metrics = await self.agent_repo.get_agent_metrics(
                self._get_agent_db_id()
            ) if hasattr(self, '_agent_db_id') else {}

            status = {
                "agent_id": self.config.agent_id,
                "agent_type": self.config.agent_type.value,
                "name": self.config.name,
                "state": self.state.value,
                "current_execution": self.current_execution.execution_id if self.current_execution else None,
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "failed_executions": self.failed_executions,
                "success_rate": (self.successful_executions / self.total_executions) if self.total_executions > 0 else 0,
                "average_execution_time": self.average_execution_time,
                "active_operations": len(self.active_operations),
                "allocated_resources": self.allocated_resources,
                "capabilities": self.config.capabilities,
                "last_updated": datetime.utcnow().isoformat()
            }

            # Add repository metrics if available
            if agent_metrics:
                status.update({
                    "repository_metrics": agent_metrics
                })

            return status

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for the agent.

        Returns:
            Comprehensive metrics including performance, resource usage, and operational data
        """
        with logfire.span("Get agent metrics", agent_id=self.config.agent_id):
            try:
                # Get metrics from repository
                repo_metrics = await self.agent_repo.get_agent_metrics(
                    self._get_agent_db_id()
                ) if hasattr(self, '_agent_db_id') else {}

                metrics = {
                    "agent_id": self.config.agent_id,
                    "performance": {
                        "total_executions": self.total_executions,
                        "successful_executions": self.successful_executions,
                        "failed_executions": self.failed_executions,
                        "success_rate": (self.successful_executions / self.total_executions) if self.total_executions > 0 else 0,
                        "average_execution_time": self.average_execution_time,
                    },
                    "operational": {
                        "state": self.state.value,
                        "active_operations": len(self.active_operations),
                        "execution_history_size": len(self.execution_history),
                        "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0
                    },
                    "resources": {
                        "allocated": self.allocated_resources,
                        "constraints": self.config.resource_constraints.dict(),
                        "current_usage": await self._get_current_resource_usage()
                    },
                    "configuration": {
                        "agent_type": self.config.agent_type.value,
                        "capabilities": self.config.capabilities,
                        "tools_enabled": self.config.tools_enabled,
                        "timeout_seconds": self.config.timeout_seconds,
                        "retry_attempts": self.config.retry_attempts
                    }
                }

                # Add repository metrics
                if repo_metrics:
                    metrics["repository_metrics"] = repo_metrics

                return metrics

            except Exception as e:
                self.logger.log_error_with_context(
                    error=e,
                    operation="get_metrics",
                    level=LogLevel.ERROR
                )
                raise AgentError(f"Failed to get agent metrics: {str(e)}")

    # Abstract methods that must be implemented by concrete agent classes

    @abstractmethod
    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass

    @abstractmethod
    async def _execute_operation(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute the core agent operation."""
        pass

    @abstractmethod
    async def _agent_cleanup(self) -> None:
        """Agent-specific cleanup logic."""
        pass

    # Lifecycle hook management

    def add_lifecycle_hook(self, event: str, callback: Callable) -> None:
        """Add a lifecycle hook for the specified event."""
        if event in self._lifecycle_hooks:
            self._lifecycle_hooks[event].append(callback)
        else:
            raise ValueError(f"Unknown lifecycle event: {event}")

    async def _execute_lifecycle_hooks(self, event: str, **kwargs) -> None:
        """Execute all registered hooks for the specified event."""
        for hook in self._lifecycle_hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self, **kwargs)
                else:
                    hook(self, **kwargs)
            except Exception as e:
                self.logger.log_error_with_context(
                    error=e,
                    operation=f"lifecycle_hook:{event}",
                    level=LogLevel.WARNING
                )

    # Private implementation methods

    async def _validate_configuration(self) -> None:
        """Validate agent configuration and constraints."""
        try:
            # Validate resource constraints
            constraints = self.config.resource_constraints
            if constraints.max_memory_mb and constraints.max_memory_mb <= 0:
                raise ValidationError("Memory constraint must be positive")
            if constraints.max_cpu_percent and (constraints.max_cpu_percent <= 0 or constraints.max_cpu_percent > 100):
                raise ValidationError("CPU constraint must be between 0 and 100")

            # Validate agent-specific configuration
            await self._validate_agent_configuration()

        except Exception as e:
            raise AgentConfigurationError(f"Configuration validation failed: {str(e)}")

    async def _validate_agent_configuration(self) -> None:
        """Agent-specific configuration validation. Override in subclasses."""
        pass

    async def _allocate_resources(self) -> None:
        """Allocate initial resources based on configuration."""
        self.allocated_resources = {
            "memory_mb": min(self.config.resource_constraints.max_memory_mb or 512, 512),
            "cpu_percent": min(self.config.resource_constraints.max_cpu_percent or 50, 50),
            "disk_mb": min(self.config.resource_constraints.max_disk_usage_mb or 100, 100),
            "allocated_at": datetime.utcnow().isoformat()
        }

    async def _release_resources(self) -> None:
        """Release allocated resources."""
        self.allocated_resources.clear()

    async def _register_agent(self) -> None:
        """Register or update agent in the database."""
        try:
            # Check if agent exists
            existing_agent = await self.agent_repo.find_one({"name": self.config.agent_id})

            agent_data = {
                "name": self.config.agent_id,
                "display_name": self.config.name,
                "description": self.config.description,
                "agent_type": self.config.agent_type,
                "status": AgentStatus.ACTIVE,
                "configuration": self.config.dict(),
                "available_tools": self.config.tools_enabled,
                "max_concurrent_executions": self.config.resource_constraints.max_concurrent_operations or 1
            }

            if existing_agent:
                # Update existing agent
                await self.agent_repo.update(existing_agent.id, agent_data)
                self._agent_db_id = existing_agent.id
            else:
                # Create new agent
                new_agent = await self.agent_repo.create(agent_data)
                self._agent_db_id = new_agent.id

        except Exception as e:
            raise AgentError(f"Failed to register agent in database: {str(e)}")

    async def _update_agent_state(self) -> None:
        """Update agent state in the database."""
        if hasattr(self, '_agent_db_id'):
            try:
                await self.agent_repo.update_state(self._agent_db_id, {
                    "state": self.state.value,
                    "last_activity": datetime.utcnow().isoformat(),
                    "total_executions": self.total_executions,
                    "successful_executions": self.successful_executions,
                    "failed_executions": self.failed_executions,
                    "average_execution_time": self.average_execution_time
                })
            except Exception as e:
                self.logger.log_error_with_context(
                    error=e,
                    operation="update_agent_state",
                    level=LogLevel.WARNING
                )

    def _get_agent_db_id(self) -> int:
        """Get the database ID for this agent."""
        if hasattr(self, '_agent_db_id'):
            return self._agent_db_id
        raise AgentError("Agent not registered in database")

    async def _pre_execution_setup(self, context: ExecutionContext) -> None:
        """Setup before operation execution."""
        self.state = AgentState.RUNNING
        self.current_execution = context
        self.active_operations[context.execution_id] = context

        # Set timeout
        if self.config.timeout_seconds:
            context.timeout_at = datetime.utcnow() + timedelta(seconds=self.config.timeout_seconds)

        await self._execute_lifecycle_hooks('before_execute', context=context)

    async def _post_execution_cleanup(self, context: ExecutionContext, result_data: Dict[str, Any]) -> ExecutionResult:
        """Cleanup after successful execution."""
        execution_time = (datetime.utcnow() - context.started_at).total_seconds() * 1000

        # Update metrics
        self.total_executions += 1
        self.successful_executions += 1
        self.average_execution_time = ((self.average_execution_time * (self.total_executions - 1)) + execution_time) / self.total_executions

        # Create result
        result = ExecutionResult(
            success=True,
            execution_id=context.execution_id,
            agent_id=self.config.agent_id,
            operation=context.operation,
            result=result_data,
            execution_time_ms=execution_time,
            started_at=context.started_at,
            completed_at=datetime.utcnow(),
            retry_count=context.retry_count
        )

        # Cleanup
        self.current_execution = None
        self.active_operations.pop(context.execution_id, None)
        self.state = AgentState.IDLE
        self.execution_history.append(result)

        await self._update_agent_state()
        await self._execute_lifecycle_hooks('after_execute', context=context, result=result)

        return result

    async def _handle_execution_error(self, context: ExecutionContext, error: Exception) -> ExecutionResult:
        """Handle execution errors and create error result."""
        execution_time = (datetime.utcnow() - context.started_at).total_seconds() * 1000

        # Update metrics
        self.total_executions += 1
        self.failed_executions += 1

        # Create error result
        result = ExecutionResult(
            success=False,
            execution_id=context.execution_id,
            agent_id=self.config.agent_id,
            operation=context.operation,
            error=str(error),
            error_type=type(error).__name__,
            execution_time_ms=execution_time,
            started_at=context.started_at,
            completed_at=datetime.utcnow(),
            retry_count=context.retry_count
        )

        # Cleanup
        self.current_execution = None
        self.active_operations.pop(context.execution_id, None)
        self.state = AgentState.ERROR if isinstance(error, (AgentError, AgentExecutionError)) else AgentState.IDLE
        self.execution_history.append(result)

        await self._update_agent_state()
        await self._execute_lifecycle_hooks('on_error', context=context, error=error, result=result)

        self.logger.log_agent_operation(
            message=f"Agent execution failed: {str(error)}",
            agent_type=self.config.agent_type.value,
            agent_name=self.config.name,
            phase=AgentPhase.ACTION,
            level=LogLevel.ERROR,
            success=False,
            error_details={"error": str(error), "error_type": type(error).__name__}
        )

        return result

    async def _complete_active_operations(self, timeout_seconds: int = 30) -> None:
        """Wait for active operations to complete or timeout."""
        if not self.active_operations:
            return

        timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        while self.active_operations and datetime.utcnow() < timeout_at:
            await asyncio.sleep(0.1)

        # Force cleanup remaining operations
        for execution_id in list(self.active_operations.keys()):
            self.active_operations.pop(execution_id, None)

    async def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage. Override in subclasses for actual monitoring."""
        return {
            "memory_mb": 0,
            "cpu_percent": 0,
            "disk_mb": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
