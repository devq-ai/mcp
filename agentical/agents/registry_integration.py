"""
Registry Integration Mixin for Enhanced Base Agent

This module provides the integration layer between the Enhanced Base Agent
and the Agent Registry system, enabling automatic registration, heartbeat
management, and lifecycle integration.

Features:
- Automatic agent registration on initialization
- Heartbeat management for health monitoring
- Graceful deregistration on cleanup
- Registry-aware status updates
- Health score calculation and reporting
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from abc import ABC

import logfire
from agentical.agents.agent_registry_enhanced import (
    EnhancedAgentRegistry,
    AgentInfo,
    RegistryStatus
)
from agentical.core.structured_logging import (
    StructuredLogger,
    LogLevel,
    AgentPhase
)
from agentical.core.exceptions import AgentError


class RegistryIntegrationMixin(ABC):
    """
    Mixin class that provides registry integration capabilities to Enhanced Base Agent.

    This mixin enables agents to automatically register with the registry,
    maintain heartbeats, and handle lifecycle events in coordination with
    the centralized agent registry.
    """

    def __init__(self, *args, **kwargs):
        """Initialize registry integration components."""
        super().__init__(*args, **kwargs)

        # Registry integration state
        self._registry: Optional[EnhancedAgentRegistry] = None
        self._registry_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: int = 15  # seconds
        self._last_heartbeat: Optional[datetime] = None
        self._registration_endpoint: Optional[str] = None
        self._registration_tags: Dict[str, str] = {}
        self._registration_region: Optional[str] = None

        # Health tracking
        self._health_score: float = 1.0
        self._load_percentage: float = 0.0
        self._performance_history: list = []

    def set_registry(
        self,
        registry: EnhancedAgentRegistry,
        endpoint: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        region: Optional[str] = None,
        heartbeat_interval: int = 15
    ) -> None:
        """
        Configure the agent to use a specific registry.

        Args:
            registry: The agent registry instance to use
            endpoint: Agent endpoint URL for communication
            tags: Metadata tags for the agent
            region: Geographic region for the agent
            heartbeat_interval: Seconds between heartbeats
        """
        self._registry = registry
        self._registration_endpoint = endpoint or f"local://{self.config.agent_id}"
        self._registration_tags = tags or {}
        self._registration_region = region
        self._heartbeat_interval = heartbeat_interval

    async def _register_with_registry(self) -> None:
        """Register this agent with the configured registry."""
        if not self._registry:
            return

        with logfire.span("Registry registration", agent_id=self.config.agent_id):
            try:
                self._registry_id = await self._registry.register_agent(
                    agent=self,
                    endpoint=self._registration_endpoint,
                    tags=self._registration_tags,
                    region=self._registration_region
                )

                # Start heartbeat task
                if self._heartbeat_task is None:
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                self.logger.log_agent_operation(
                    message=f"Agent registered with registry: {self._registry_id}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.INFO,
                    success=True,
                    metadata={
                        "registry_id": self._registry_id,
                        "endpoint": self._registration_endpoint,
                        "region": self._registration_region
                    }
                )

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Registry registration failed: {str(e)}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.INITIALIZATION,
                    level=LogLevel.ERROR,
                    success=False,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )
                # Don't fail agent initialization if registry registration fails

    async def _deregister_from_registry(self) -> None:
        """Deregister this agent from the registry."""
        if not self._registry or not self._registry_id:
            return

        with logfire.span("Registry deregistration", agent_id=self.config.agent_id):
            try:
                # Stop heartbeat task
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    self._heartbeat_task = None

                # Deregister from registry
                await self._registry.deregister_agent(self._registry_id)

                self.logger.log_agent_operation(
                    message=f"Agent deregistered from registry: {self._registry_id}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.INFO,
                    success=True,
                    metadata={"registry_id": self._registry_id}
                )

                self._registry_id = None

            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Registry deregistration failed: {str(e)}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    phase=AgentPhase.TERMINATION,
                    level=LogLevel.WARNING,
                    error_details={"error": str(e), "error_type": type(e).__name__}
                )

    async def _heartbeat_loop(self) -> None:
        """Background task for sending periodic heartbeats to registry."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                if not self._registry or not self._registry_id:
                    break

                # Calculate current health score and load
                health_score = await self._calculate_health_score()
                load_percentage = await self._calculate_load_percentage()

                # Send heartbeat
                await self._registry.update_agent_heartbeat(
                    agent_id=self._registry_id,
                    health_score=health_score,
                    current_load=load_percentage
                )

                self._last_heartbeat = datetime.utcnow()
                self._health_score = health_score
                self._load_percentage = load_percentage

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_agent_operation(
                    message=f"Heartbeat failed: {str(e)}",
                    agent_type=self.config.agent_type.value,
                    agent_name=self.config.name,
                    operation_type="heartbeat",
                    level=LogLevel.WARNING,
                    error_details={"error": str(e)}
                )

                # Continue trying heartbeats even if some fail
                continue

    async def _calculate_health_score(self) -> float:
        """
        Calculate the current health score for this agent.

        Returns:
            Health score between 0.0 and 1.0
        """
        try:
            health_factors = []

            # Factor 1: Agent state (40% weight)
            state_score = 1.0
            if self.state.value in ['error', 'maintenance']:
                state_score = 0.0
            elif self.state.value in ['stopping', 'paused']:
                state_score = 0.3
            elif self.state.value == 'initializing':
                state_score = 0.5
            elif self.state.value in ['idle', 'running']:
                state_score = 1.0

            health_factors.append(('state', state_score, 0.4))

            # Factor 2: Success rate (30% weight)
            success_rate = 1.0
            if self.total_executions > 0:
                success_rate = self.successful_executions / self.total_executions
            health_factors.append(('success_rate', success_rate, 0.3))

            # Factor 3: Resource utilization (20% weight)
            resource_score = 1.0
            if hasattr(self, 'allocated_resources') and self.allocated_resources:
                # Simple resource utilization check
                total_used = sum(
                    usage for usage in self.allocated_resources.values()
                    if isinstance(usage, (int, float))
                )
                # Normalize to 0-1 range (assuming 100 is max)
                resource_score = max(0.0, 1.0 - (total_used / 100.0))
            health_factors.append(('resources', resource_score, 0.2))

            # Factor 4: Recent performance (10% weight)
            performance_score = 1.0
            if len(self._performance_history) > 0:
                recent_performance = self._performance_history[-10:]  # Last 10 operations
                avg_time = sum(recent_performance) / len(recent_performance)
                # Score based on execution time (lower is better)
                performance_score = max(0.0, min(1.0, 1.0 - (avg_time / 5000.0)))  # 5s max
            health_factors.append(('performance', performance_score, 0.1))

            # Calculate weighted health score
            total_score = sum(score * weight for _, score, weight in health_factors)

            return max(0.0, min(1.0, total_score))

        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Health score calculation failed: {str(e)}",
                operation_type="health_calculation",
                level=LogLevel.WARNING,
                error_details={"error": str(e)}
            )
            return 0.5  # Default moderate health score

    async def _calculate_load_percentage(self) -> float:
        """
        Calculate the current load percentage for this agent.

        Returns:
            Load percentage between 0.0 and 1.0
        """
        try:
            load_factors = []

            # Factor 1: Active operations (50% weight)
            active_ops = len(getattr(self, 'active_operations', {}))
            max_concurrent = getattr(self.config, 'max_concurrent_operations', 1)
            ops_load = min(1.0, active_ops / max_concurrent) if max_concurrent > 0 else 0.0
            load_factors.append(ops_load * 0.5)

            # Factor 2: Resource constraints (30% weight)
            resource_load = 0.0
            if hasattr(self.config, 'resource_constraints'):
                constraints = self.config.resource_constraints
                current_usage = await self._get_current_resource_usage()

                # Check memory usage
                if constraints.max_memory_mb and 'memory_mb' in current_usage:
                    memory_load = current_usage['memory_mb'] / constraints.max_memory_mb
                    resource_load = max(resource_load, memory_load)

                # Check CPU usage
                if constraints.max_cpu_percent and 'cpu_percent' in current_usage:
                    cpu_load = current_usage['cpu_percent'] / constraints.max_cpu_percent
                    resource_load = max(resource_load, cpu_load)

            load_factors.append(resource_load * 0.3)

            # Factor 3: Queue depth or pending work (20% weight)
            queue_load = 0.0
            if hasattr(self, 'pending_operations'):
                pending_count = len(self.pending_operations)
                # Assume max queue size of 10
                queue_load = min(1.0, pending_count / 10.0)
            load_factors.append(queue_load * 0.2)

            # Calculate total load
            total_load = sum(load_factors)

            return max(0.0, min(1.0, total_load))

        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Load calculation failed: {str(e)}",
                operation_type="load_calculation",
                level=LogLevel.WARNING,
                error_details={"error": str(e)}
            )
            return 0.0  # Default to no load on error

    def _record_performance(self, execution_time_ms: float) -> None:
        """Record performance data for health calculation."""
        self._performance_history.append(execution_time_ms)

        # Keep only recent history
        if len(self._performance_history) > 50:
            self._performance_history = self._performance_history[-25:]

    async def get_registry_status(self) -> Dict[str, Any]:
        """
        Get status information related to registry integration.

        Returns:
            Dictionary containing registry integration status
        """
        status = {
            "registry_integrated": self._registry is not None,
            "registry_id": self._registry_id,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "health_score": self._health_score,
            "load_percentage": self._load_percentage,
            "heartbeat_interval": self._heartbeat_interval,
            "endpoint": self._registration_endpoint,
            "tags": self._registration_tags,
            "region": self._registration_region
        }

        if self._registry:
            try:
                registry_status = await self._registry.get_registry_status()
                status["registry_info"] = {
                    "status": registry_status.get("registry_info", {}).get("status"),
                    "total_agents": registry_status.get("registry_info", {}).get("total_agents"),
                    "active_agents": registry_status.get("agent_statistics", {}).get("by_status", {}).get("active", 0)
                }
            except Exception as e:
                status["registry_info_error"] = str(e)

        return status

    def is_registered(self) -> bool:
        """Check if agent is currently registered with a registry."""
        return self._registry is not None and self._registry_id is not None

    async def _update_registry_on_state_change(self, new_state) -> None:
        """Update registry when agent state changes."""
        if not self.is_registered():
            return

        try:
            # Calculate new health score based on state change
            health_score = await self._calculate_health_score()

            # Update heartbeat with new health score
            await self._registry.update_agent_heartbeat(
                agent_id=self._registry_id,
                health_score=health_score
            )

        except Exception as e:
            self.logger.log_agent_operation(
                message=f"Failed to update registry on state change: {str(e)}",
                operation_type="registry_update",
                level=LogLevel.WARNING,
                error_details={"error": str(e), "new_state": new_state}
            )

    # Lifecycle hook integration methods
    async def _on_initialize_with_registry(self) -> None:
        """Called during agent initialization to set up registry integration."""
        await self._register_with_registry()

    async def _on_cleanup_with_registry(self) -> None:
        """Called during agent cleanup to teardown registry integration."""
        await self._deregister_from_registry()

    async def _on_execution_complete_with_registry(self, execution_time_ms: float) -> None:
        """Called after successful execution to update performance tracking."""
        self._record_performance(execution_time_ms)

        # Update load percentage after execution
        if self.is_registered():
            try:
                load_percentage = await self._calculate_load_percentage()
                await self._registry.update_agent_heartbeat(
                    agent_id=self._registry_id,
                    current_load=load_percentage
                )
            except Exception as e:
                # Don't fail execution for registry update errors
                pass

    async def _on_error_with_registry(self, error: Exception) -> None:
        """Called when agent encounters an error to update health status."""
        if self.is_registered():
            try:
                # Reduce health score on errors
                health_score = max(0.1, self._health_score - 0.2)
                await self._registry.update_agent_heartbeat(
                    agent_id=self._registry_id,
                    health_score=health_score
                )
            except Exception as e:
                # Don't compound errors
                pass
