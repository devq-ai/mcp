"""
Enhanced Agent Registry and Discovery System for Agentical Framework

This module provides the centralized agent registry and discovery system that serves
as the central nervous system for the entire agent ecosystem. It enables dynamic
agent registration, discovery, health monitoring, and lifecycle management.

Features:
- Centralized agent registration and deregistration
- Type-based and capability-based agent discovery
- Real-time health monitoring and status tracking
- Load balancing and intelligent agent selection
- Registry persistence through repository pattern
- Comprehensive observability with Logfire integration
- Production-ready high availability and fault tolerance
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Callable
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

import logfire
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent, AgentState
from agentical.db.models.agent import Agent, AgentType, AgentStatus
from agentical.db.repositories.agent import AsyncAgentRepository
from agentical.core.exceptions import (
    AgentError,
    AgentRegistrationError,
    AgentDiscoveryError,
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


class RegistryStatus(Enum):
    """Registry operational status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class SelectionStrategy(Enum):
    """Agent selection strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    HIGHEST_HEALTH = "highest_health"
    CLOSEST = "closest"
    CUSTOM = "custom"


class DiscoveryRequest(BaseModel):
    """Flexible agent discovery request model."""
    agent_type: Optional[AgentType] = Field(default=None, description="Specific agent type to find")
    capabilities: Optional[List[str]] = Field(default=None, description="Required agent capabilities")
    status: Optional[AgentStatus] = Field(default=AgentStatus.ACTIVE, description="Required agent status")
    max_load: Optional[float] = Field(default=0.8, description="Maximum acceptable load percentage")
    min_health_score: Optional[float] = Field(default=0.7, description="Minimum health score required")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Agent metadata tags to match")
    region: Optional[str] = Field(default=None, description="Geographic region preference")
    exclude_agents: Optional[List[str]] = Field(default=None, description="Agent IDs to exclude")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results")

    @validator('max_load')
    def validate_max_load(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("max_load must be between 0 and 1")
        return v

    @validator('min_health_score')
    def validate_min_health_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("min_health_score must be between 0 and 1")
        return v


class SelectionCriteria(BaseModel):
    """Agent selection criteria for intelligent agent choosing."""
    discovery_request: DiscoveryRequest
    selection_strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED
    prefer_local: bool = Field(default=False, description="Prefer agents in the same region")
    sticky_session: Optional[str] = Field(default=None, description="Session ID for sticky routing")
    custom_selector: Optional[str] = Field(default=None, description="Custom selection function name")


@dataclass
class AgentInfo:
    """Comprehensive agent information for discovery and selection."""
    agent_id: str
    agent_type: AgentType
    capabilities: List[str]
    status: AgentStatus
    endpoint: str
    health_score: float
    current_load: float
    last_heartbeat: datetime
    registration_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    region: Optional[str] = None
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "health_score": self.health_score,
            "current_load": self.current_load,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "registration_time": self.registration_time.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "region": self.region,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentInfo':
        """Create agent info from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            capabilities=data["capabilities"],
            status=AgentStatus(data["status"]),
            endpoint=data["endpoint"],
            health_score=data["health_score"],
            current_load=data["current_load"],
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            registration_time=datetime.fromisoformat(data["registration_time"]),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", {}),
            region=data.get("region"),
            version=data.get("version", "1.0.0")
        )


@dataclass
class RegistryMetrics:
    """Registry performance and operational metrics."""
    total_agents: int = 0
    active_agents: int = 0
    failed_agents: int = 0
    total_discoveries: int = 0
    average_discovery_time: float = 0.0
    total_registrations: int = 0
    total_deregistrations: int = 0
    health_check_failures: int = 0
    last_cleanup: Optional[datetime] = None
    registry_uptime: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_agents": self.total_agents,
            "active_agents": self.active_agents,
            "failed_agents": self.failed_agents,
            "total_discoveries": self.total_discoveries,
            "average_discovery_time": self.average_discovery_time,
            "total_registrations": self.total_registrations,
            "total_deregistrations": self.total_deregistrations,
            "health_check_failures": self.health_check_failures,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
            "registry_uptime": self.registry_uptime
        }


class EnhancedAgentRegistry:
    """
    Enhanced agent registry with comprehensive discovery and lifecycle management.

    This class provides the central nervous system for the agent ecosystem,
    enabling dynamic agent registration, intelligent discovery, health monitoring,
    and production-ready features for high availability and performance.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        logger_instance: Optional[StructuredLogger] = None,
        health_check_interval: int = 30,
        cleanup_interval: int = 300,
        agent_timeout: int = 120
    ):
        """
        Initialize the enhanced agent registry.

        Args:
            db_session: Async database session for repository operations
            logger_instance: Optional structured logger instance
            health_check_interval: Seconds between health checks
            cleanup_interval: Seconds between cleanup operations
            agent_timeout: Seconds before considering agent failed
        """
        self.db_session = db_session
        self.agent_repo = AsyncAgentRepository(db_session)

        # Setup logging
        self.logger = logger_instance or StructuredLogger(
            component="agent-registry",
            service_name="agentical-registry",
            correlation_context=CorrelationContext.generate()
        )

        # Registry state
        self.status = RegistryStatus.INITIALIZING
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, EnhancedBaseAgent] = {}

        # Index structures for fast discovery
        self.agents_by_type: Dict[AgentType, Set[str]] = defaultdict(set)
        self.agents_by_capability: Dict[str, Set[str]] = defaultdict(set)
        self.agents_by_status: Dict[AgentStatus, Set[str]] = defaultdict(set)
        self.agents_by_region: Dict[str, Set[str]] = defaultdict(set)

        # Configuration
        self.health_check_interval = health_check_interval
        self.cleanup_interval = cleanup_interval
        self.agent_timeout = agent_timeout

        # Metrics and monitoring
        self.metrics = RegistryMetrics()
        self.start_time = datetime.utcnow()

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Selection strategies
        self._selection_strategies: Dict[SelectionStrategy, Callable] = {
            SelectionStrategy.ROUND_ROBIN: self._select_round_robin,
            SelectionStrategy.LEAST_LOADED: self._select_least_loaded,
            SelectionStrategy.RANDOM: self._select_random,
            SelectionStrategy.HIGHEST_HEALTH: self._select_highest_health,
            SelectionStrategy.CLOSEST: self._select_closest
        }

        # Round-robin counters
        self._round_robin_counters: Dict[str, int] = defaultdict(int)

        with logfire.span("Registry initialization"):
            self.logger.log_agent_operation(
                message="Initializing enhanced agent registry",
                operation_type="registry_init",
                phase=AgentPhase.INITIALIZATION,
                level=LogLevel.INFO
            )

    async def start(self) -> None:
        """Start the registry and background tasks."""
        with logfire.span("Registry startup"):
            try:
                self.status = RegistryStatus.ACTIVE
                self._running = True

                # Start background tasks
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

                # Load existing agents from repository
                await self._load_existing_agents()

                self.logger.log_agent_operation(
                    message="Agent registry started successfully",
                    operation_type="registry_start",
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.INFO,
                    success=True
                )

            except Exception as e:
                self.status = RegistryStatus.OFFLINE
                self.logger.log_agent_operation(
                    message=f"Registry startup failed: {str(e)}",
                    operation_type="registry_start",
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                raise AgentRegistrationError(f"Registry startup failed: {str(e)}")

    async def stop(self) -> None:
        """Stop the registry and cleanup background tasks."""
        with logfire.span("Registry shutdown"):
            try:
                self._running = False
                self.status = RegistryStatus.OFFLINE

                # Cancel background tasks
                if self._health_check_task:
                    self._health_check_task.cancel()
                if self._cleanup_task:
                    self._cleanup_task.cancel()

                # Deregister all agents
                agent_ids = list(self.agents.keys())
                for agent_id in agent_ids:
                    await self.deregister_agent(agent_id)

                self.logger.log_agent_operation(
                    message="Agent registry stopped successfully",
                    operation_type="registry_stop",
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.INFO,
                    success=True
                )

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Registry shutdown failed: {str(e)}",
                    operation_type="registry_stop",
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )

    async def register_agent(
        self,
        agent: EnhancedBaseAgent,
        endpoint: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        region: Optional[str] = None
    ) -> str:
        """
        Register an agent with the registry.

        Args:
            agent: Enhanced base agent instance to register
            endpoint: Agent endpoint URL for communication
            tags: Metadata tags for the agent
            region: Geographic region for the agent

        Returns:
            Registration ID for the agent

        Raises:
            AgentRegistrationError: If registration fails
        """
        start_time = time.time()

        with logfire.span("Agent registration", agent_id=agent.config.agent_id):
            try:
                agent_id = agent.config.agent_id

                # Check if agent is already registered
                if agent_id in self.agents:
                    raise AgentRegistrationError(f"Agent {agent_id} is already registered")

                # Create agent info
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_type=agent.config.agent_type,
                    capabilities=agent.config.capabilities,
                    status=AgentStatus.ACTIVE,
                    endpoint=endpoint or f"local://{agent_id}",
                    health_score=1.0,
                    current_load=0.0,
                    last_heartbeat=datetime.utcnow(),
                    registration_time=datetime.utcnow(),
                    metadata=agent.config.dict(),
                    tags=tags or {},
                    region=region
                )

                # Store in registry
                self.agents[agent_id] = agent_info
                self.agent_instances[agent_id] = agent

                # Update indexes
                self.agents_by_type[agent_info.agent_type].add(agent_id)
                self.agents_by_status[agent_info.status].add(agent_id)

                for capability in agent_info.capabilities:
                    self.agents_by_capability[capability].add(agent_id)

                if agent_info.region:
                    self.agents_by_region[agent_info.region].add(agent_id)

                # Persist to repository
                await self._persist_agent_registration(agent_info)

                # Update metrics
                self.metrics.total_agents += 1
                self.metrics.active_agents += 1
                self.metrics.total_registrations += 1

                registration_time = (time.time() - start_time) * 1000

                self.logger.log_agent_operation(
                    message=f"Agent {agent_id} registered successfully",
                    agent_type=agent_info.agent_type.value,
                    agent_name=agent_id,
                    operation_type="agent_registration",
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.INFO,
                    success=True,
                    execution_time_ms=registration_time,
                    metadata={
                        "agent_type": agent_info.agent_type.value,
                        "capabilities": agent_info.capabilities,
                        "endpoint": agent_info.endpoint,
                        "region": agent_info.region
                    }
                )

                return agent_id

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Agent registration failed: {str(e)}",
                    agent_type=agent.config.agent_type.value if agent else "unknown",
                    operation_type="agent_registration",
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                raise AgentRegistrationError(f"Agent registration failed: {str(e)}")

    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.

        Args:
            agent_id: ID of the agent to deregister

        Returns:
            True if deregistration was successful

        Raises:
            NotFoundError: If agent is not found
        """
        start_time = time.time()

        with logfire.span("Agent deregistration", agent_id=agent_id):
            try:
                if agent_id not in self.agents:
                    raise NotFoundError(f"Agent {agent_id} not found in registry")

                agent_info = self.agents[agent_id]

                # Remove from indexes
                self.agents_by_type[agent_info.agent_type].discard(agent_id)
                self.agents_by_status[agent_info.status].discard(agent_id)

                for capability in agent_info.capabilities:
                    self.agents_by_capability[capability].discard(agent_id)

                if agent_info.region:
                    self.agents_by_region[agent_info.region].discard(agent_id)

                # Remove from registry
                del self.agents[agent_id]
                if agent_id in self.agent_instances:
                    del self.agent_instances[agent_id]

                # Update repository
                await self._persist_agent_deregistration(agent_id)

                # Update metrics
                self.metrics.total_agents -= 1
                if agent_info.status == AgentStatus.ACTIVE:
                    self.metrics.active_agents -= 1
                self.metrics.total_deregistrations += 1

                deregistration_time = (time.time() - start_time) * 1000

                self.logger.log_agent_operation(
                    message=f"Agent {agent_id} deregistered successfully",
                    agent_type=agent_info.agent_type.value,
                    agent_name=agent_id,
                    operation_type="agent_deregistration",
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.INFO,
                    success=True,
                    execution_time_ms=deregistration_time
                )

                return True

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Agent deregistration failed: {str(e)}",
                    agent_name=agent_id,
                    operation_type="agent_deregistration",
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                raise

    async def discover_agents(self, request: DiscoveryRequest) -> List[AgentInfo]:
        """
        Discover agents based on criteria.

        Args:
            request: Discovery request with filtering criteria

        Returns:
            List of matching agent information

        Raises:
            AgentDiscoveryError: If discovery fails
        """
        start_time = time.time()

        with logfire.span("Agent discovery"):
            try:
                matching_agents: Set[str] = set()

                # Start with all agents if no specific criteria
                if not any([request.agent_type, request.capabilities, request.region]):
                    matching_agents = set(self.agents.keys())
                else:
                    # Filter by agent type
                    if request.agent_type:
                        matching_agents = self.agents_by_type[request.agent_type].copy()
                    else:
                        matching_agents = set(self.agents.keys())

                    # Filter by capabilities (intersection)
                    if request.capabilities:
                        capability_agents = set()
                        for capability in request.capabilities:
                            if not capability_agents:
                                capability_agents = self.agents_by_capability[capability].copy()
                            else:
                                capability_agents &= self.agents_by_capability[capability]
                        matching_agents &= capability_agents

                    # Filter by region
                    if request.region:
                        matching_agents &= self.agents_by_region[request.region]

                # Apply additional filters
                filtered_agents = []
                for agent_id in matching_agents:
                    if agent_id not in self.agents:
                        continue

                    agent_info = self.agents[agent_id]

                    # Filter by status
                    if request.status and agent_info.status != request.status:
                        continue

                    # Filter by load
                    if request.max_load and agent_info.current_load > request.max_load:
                        continue

                    # Filter by health score
                    if request.min_health_score and agent_info.health_score < request.min_health_score:
                        continue

                    # Filter by exclusions
                    if request.exclude_agents and agent_id in request.exclude_agents:
                        continue

                    # Filter by tags
                    if request.tags:
                        if not all(
                            agent_info.tags.get(key) == value
                            for key, value in request.tags.items()
                        ):
                            continue

                    filtered_agents.append(agent_info)

                # Limit results
                if request.max_results:
                    filtered_agents = filtered_agents[:request.max_results]

                # Update metrics
                discovery_time = (time.time() - start_time) * 1000
                self.metrics.total_discoveries += 1
                self.metrics.average_discovery_time = (
                    (self.metrics.average_discovery_time * (self.metrics.total_discoveries - 1) + discovery_time)
                    / self.metrics.total_discoveries
                )

                self.logger.log_agent_operation(
                    message=f"Agent discovery completed: {len(filtered_agents)} agents found",
                    operation_type="agent_discovery",
                    phase=AgentPhase.ACTION,
                    level=LogLevel.INFO,
                    success=True,
                    execution_time_ms=discovery_time,
                    metadata={
                        "request": request.dict(),
                        "results_count": len(filtered_agents),
                        "total_agents": len(self.agents)
                    }
                )

                return filtered_agents

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Agent discovery failed: {str(e)}",
                    operation_type="agent_discovery",
                    phase=AgentPhase.ACTION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                raise AgentDiscoveryError(f"Agent discovery failed: {str(e)}")

    async def select_agent(self, criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """
        Select an optimal agent based on selection criteria.

        Args:
            criteria: Selection criteria including discovery request and strategy

        Returns:
            Selected agent information or None if no suitable agent found

        Raises:
            AgentDiscoveryError: If selection fails
        """
        with logfire.span("Agent selection"):
            try:
                # Discover candidate agents
                candidates = await self.discover_agents(criteria.discovery_request)

                if not candidates:
                    return None

                # Apply selection strategy
                strategy = criteria.selection_strategy
                if strategy in self._selection_strategies:
                    selected_agent = await self._selection_strategies[strategy](candidates, criteria)
                else:
                    # Default to least loaded
                    selected_agent = await self._select_least_loaded(candidates, criteria)

                if selected_agent:
                    self.logger.log_agent_operation(
                        message=f"Agent selected: {selected_agent.agent_id}",
                        agent_type=selected_agent.agent_type.value,
                        agent_name=selected_agent.agent_id,
                        operation_type="agent_selection",
                        phase=AgentPhase.ACTION,
                        level=LogLevel.INFO,
                        success=True,
                        metadata={
                            "strategy": strategy.value,
                            "candidates_count": len(candidates),
                            "selected_agent": selected_agent.agent_id
                        }
                    )

                return selected_agent

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Agent selection failed: {str(e)}",
                    operation_type="agent_selection",
                    phase=AgentPhase.ACTION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                raise AgentDiscoveryError(f"Agent selection failed: {str(e)}")

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a specific agent.

        Args:
            agent_id: ID of the agent to query

        Returns:
            Comprehensive agent status information

        Raises:
            NotFoundError: If agent is not found
        """
        with logfire.span("Get agent status", agent_id=agent_id):
            if agent_id not in self.agents:
                raise NotFoundError(f"Agent {agent_id} not found in registry")

            agent_info = self.agents[agent_id]
            agent_instance = self.agent_instances.get(agent_id)

            status = {
                "agent_info": agent_info.to_dict(),
                "registry_metrics": {
                    "registration_time": agent_info.registration_time.isoformat(),
                    "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                    "uptime_seconds": (datetime.utcnow() - agent_info.registration_time).total_seconds(),
                    "heartbeat_age_seconds": (datetime.utcnow() - agent_info.last_heartbeat).total_seconds()
                }
            }

            # Add agent instance status if available
            if agent_instance:
                try:
                    instance_status = await agent_instance.get_status()
                    status["instance_status"] = instance_status
                except Exception as e:
                    status["instance_status_error"] = str(e)

            return status

    async def get_registry_status(self) -> Dict[str, Any]:
        """
        Get comprehensive registry status and metrics.

        Returns:
            Registry status including metrics and operational information
        """
        with logfire.span("Get registry status"):
            # Update metrics
            self.metrics.registry_uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Count agents by status
            status_counts = defaultdict(int)
            for agent_info in self.agents.values():
                status_counts[agent_info.status.value] += 1

            # Count agents by type
            type_counts = defaultdict(int)
            for agent_info in self.agents.values():
                type_counts[agent_info.agent_type.value] += 1

            registry_status = {
                "registry_info": {
                    "status": self.status.value,
                    "total_agents": len(self.agents),
                    "start_time": self.start_time.isoformat(),
                    "uptime_seconds": self.metrics.registry_uptime
                },
                "agent_statistics": {
                    "by_status": dict(status_counts),
                    "by_type": dict(type_counts),
                    "by_region": {
                        region: len(agents)
                        for region, agents in self.agents_by_region.items()
                    }
                },
                "performance_metrics": self.metrics.to_dict(),
                "configuration": {
                    "health_check_interval": self.health_check_interval,
                    "cleanup_interval": self.cleanup_interval,
                    "agent_timeout": self.agent_timeout
                }
            }

            return registry_status

    async def update_agent_heartbeat(self, agent_id: str, health_score: Optional[float] = None, current_load: Optional[float] = None) -> bool:
        """
        Update agent heartbeat and health information.

        Args:
            agent_id: ID of the agent
            health_score: Optional updated health score
            current_load: Optional updated load percentage

        Returns:
            True if update was successful

        Raises:
            NotFoundError: If agent is not found
        """
        if agent_id not in self.agents:
            raise NotFoundError(f"Agent {agent_id} not found in registry")

        agent_info = self.agents[agent_id]
        agent_info.last_heartbeat = datetime.utcnow()

        if health_score is not None:
            agent_info.health_score = max(0.0, min(1.0, health_score))

        if current_load is not None:
            agent_info.current_load = max(0.0, min(1.0, current_load))

        # Update status based on health
        if agent_info.health_score < 0.3:
            await self._update_agent_status(agent_id, AgentStatus.ERROR)
        elif agent_info.status == AgentStatus.ERROR and agent_info.health_score > 0.7:
            await self._update_agent_status(agent_id, AgentStatus.ACTIVE)

        return True

    # Selection strategy implementations
    async def _select_round_robin(self, candidates: List[AgentInfo], criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """Round-robin selection strategy."""
        if not candidates:
            return None

        # Create a key for round-robin tracking
        key = f"{criteria.discovery_request.agent_type}_{len(candidates)}"
        index = self._round_robin_counters[key] % len(candidates)
        self._round_robin_counters[key] += 1

        return candidates[index]

    async def _select_least_loaded(self, candidates: List[AgentInfo], criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """Least loaded selection strategy."""
        if not candidates:
            return None

        return min(candidates, key=lambda agent: agent.current_load)

    async def _select_random(self, candidates: List[AgentInfo], criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """Random selection strategy."""
        if not candidates:
            return None

        import random
        return random.choice(candidates)

    async def _select_highest_health(self, candidates: List[AgentInfo], criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """Highest health score selection strategy."""
        if not candidates:
            return None

        return max(candidates, key=lambda agent: agent.health_score)

    async def _select_closest(self, candidates: List[AgentInfo], criteria: SelectionCriteria) -> Optional[AgentInfo]:
        """Closest (same region) selection strategy."""
        if not candidates:
            return None

        # Prefer agents in the same region if specified
        if criteria.prefer_local and criteria.discovery_request.region:
            same_region = [
                agent for agent in candidates
                if agent.region == criteria.discovery_request.region
            ]
            if same_region:
                return await self._select_least_loaded(same_region, criteria)

        # Fall back to least loaded
        return await self._select_least_loaded(candidates, criteria)

    # Internal helper methods
    async def _update_agent_status(self, agent_id: str, new_status: AgentStatus) -> None:
        """Update agent status and indexes."""
        if agent_id not in self.agents:
            return

        agent_info = self.agents[agent_id]
        old_status = agent_info.status

        # Update indexes
        self.agents_by_status[old_status].discard(agent_id)
        self.agents_by_status[new_status].add(agent_id)

        # Update agent info
        agent_info.status = new_status

        # Update metrics
        if old_status == AgentStatus.ACTIVE and new_status != AgentStatus.ACTIVE:
            self.metrics.active_agents -= 1
        elif old_status != AgentStatus.ACTIVE and new_status == AgentStatus.ACTIVE:
            self.metrics.active_agents += 1

        if new_status == AgentStatus.ERROR:
            self.metrics.failed_agents += 1

    async def _load_existing_agents(self) -> None:
        """Load existing agents from repository on startup."""
        try:
            # This would load persisted agent data from the repository
            # For now, we start with empty registry
            pass
        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Failed to load existing agents: {str(e)}",
                operation_type="agent_load",
                level=LogLevel.WARNING,
                error_details={"error": str(e)}
            )

    async def _persist_agent_registration(self, agent_info: AgentInfo) -> None:
        """Persist agent registration to repository."""
        try:
            # Create or update agent record in database
            agent_data = {
                'name': agent_info.agent_id,
                'agent_type': agent_info.agent_type,
                'status': agent_info.status,
                'configuration': agent_info.metadata,
                'available_tools': agent_info.capabilities
            }

            await self.agent_repo.create(agent_data)

        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Failed to persist agent registration: {str(e)}",
                agent_name=agent_info.agent_id,
                operation_type="agent_persist",
                level=LogLevel.WARNING,
                error_details={"error": str(e)}
            )

    async def _persist_agent_deregistration(self, agent_id: str) -> None:
        """Persist agent deregistration to repository."""
        try:
            # Update agent status in database to inactive
            agents = await self.agent_repo.get_by_name(agent_id)
            if agents:
                await self.agent_repo.update_state(
                    agents[0].id,
                    AgentStatus.INACTIVE,
                    {"deregistered_at": datetime.utcnow().isoformat()}
                )

        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Failed to persist agent deregistration: {str(e)}",
                agent_name=agent_id,
                operation_type="agent_persist",
                level=LogLevel.WARNING,
                error_details={"error": str(e)}
            )

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)

                if not self._running:
                    break

                await self._perform_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Health check loop error: {str(e)}",
                    operation_type="health_check",
                    level=LogLevel.ERROR,
                    error_details={"error": str(e)}
                )

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered agents."""
        current_time = datetime.utcnow()
        failed_agents = []

        for agent_id, agent_info in self.agents.items():
            # Check if agent has timed out
            time_since_heartbeat = (current_time - agent_info.last_heartbeat).total_seconds()

            if time_since_heartbeat > self.agent_timeout:
                failed_agents.append(agent_id)
                self.metrics.health_check_failures += 1

                self.logger.log_agent_operation(
                    message=f"Agent {agent_id} failed health check (timeout: {time_since_heartbeat}s)",
                    agent_name=agent_id,
                    operation_type="health_check",
                    level=LogLevel.WARNING,
                    metadata={"timeout_seconds": time_since_heartbeat}
                )

        # Update status for failed agents
        for agent_id in failed_agents:
            await self._update_agent_status(agent_id, AgentStatus.ERROR)

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                if not self._running:
                    break

                await self._perform_cleanup()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Cleanup loop error: {str(e)}",
                    operation_type="cleanup",
                    level=LogLevel.ERROR,
                    error_details={"error": str(e)}
                )

    async def _perform_cleanup(self) -> None:
        """Perform cleanup operations."""
        current_time = datetime.utcnow()

        # Clean up failed agents that have been offline too long
        cleanup_threshold = timedelta(hours=1)
        agents_to_remove = []

        for agent_id, agent_info in self.agents.items():
            if (agent_info.status == AgentStatus.ERROR and
                current_time - agent_info.last_heartbeat > cleanup_threshold):
                agents_to_remove.append(agent_id)

        # Remove offline agents
        for agent_id in agents_to_remove:
            try:
                await self.deregister_agent(agent_id)
                self.logger.log_agent_operation(
                    message=f"Cleaned up offline agent: {agent_id}",
                    agent_name=agent_id,
                    operation_type="cleanup",
                    level=LogLevel.INFO
                )
            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Failed to cleanup agent {agent_id}: {str(e)}",
                    agent_name=agent_id,
                    operation_type="cleanup",
                    level=LogLevel.WARNING,
                    error_details={"error": str(e)}
                )

        self.metrics.last_cleanup = current_time

    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentInfo]:
        """Get all agents of a specific type."""
        agent_ids = self.agents_by_type.get(agent_type, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]

    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get all agents with a specific capability."""
        agent_ids = self.agents_by_capability.get(capability, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]

    def get_agents_by_status(self, status: AgentStatus) -> List[AgentInfo]:
        """Get all agents with a specific status."""
        agent_ids = self.agents_by_status.get(status, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]

    def get_active_agents_count(self) -> int:
        """Get count of active agents."""
        return len(self.agents_by_status.get(AgentStatus.ACTIVE, set()))

    def get_total_agents_count(self) -> int:
        """Get total count of registered agents."""
        return len(self.agents)


# Registry exception classes
class AgentRegistrationError(AgentError):
    """Exception raised when agent registration fails."""
    pass


class AgentDiscoveryError(AgentError):
    """Exception raised when agent discovery fails."""
    pass


# Convenience functions for common operations
async def create_registry(
    db_session: AsyncSession,
    health_check_interval: int = 30,
    cleanup_interval: int = 300,
    agent_timeout: int = 120
) -> EnhancedAgentRegistry:
    """
    Create and start an enhanced agent registry.

    Args:
        db_session: Database session for persistence
        health_check_interval: Seconds between health checks
        cleanup_interval: Seconds between cleanup operations
        agent_timeout: Seconds before considering agent failed

    Returns:
        Started registry instance
    """
    registry = EnhancedAgentRegistry(
        db_session=db_session,
        health_check_interval=health_check_interval,
        cleanup_interval=cleanup_interval,
        agent_timeout=agent_timeout
    )

    await registry.start()
    return registry


async def discover_agent_by_type(
    registry: EnhancedAgentRegistry,
    agent_type: AgentType,
    selection_strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED
) -> Optional[AgentInfo]:
    """
    Convenience function to discover and select an agent by type.

    Args:
        registry: Agent registry instance
        agent_type: Type of agent to find
        selection_strategy: Strategy for selecting among candidates

    Returns:
        Selected agent info or None if no suitable agent found
    """
    request = DiscoveryRequest(agent_type=agent_type)
    criteria = SelectionCriteria(
        discovery_request=request,
        selection_strategy=selection_strategy
    )

    return await registry.select_agent(criteria)


async def discover_agent_by_capability(
    registry: EnhancedAgentRegistry,
    capabilities: List[str],
    selection_strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED
) -> Optional[AgentInfo]:
    """
    Convenience function to discover and select an agent by capabilities.

    Args:
        registry: Agent registry instance
        capabilities: Required capabilities
        selection_strategy: Strategy for selecting among candidates

    Returns:
        Selected agent info or None if no suitable agent found
    """
    request = DiscoveryRequest(capabilities=capabilities)
    criteria = SelectionCriteria(
        discovery_request=request,
        selection_strategy=selection_strategy
    )

    return await registry.select_agent(criteria)
