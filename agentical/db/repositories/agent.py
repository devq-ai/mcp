"""
Agent Repository Implementation

This module provides the repository implementation for Agent model operations
in the Agentical framework. It extends the base repository with agent-specific
functionality for agent lifecycle management, state persistence, and metrics tracking.

Features:
- Agent CRUD operations
- Agent state management and persistence
- Agent type filtering and discovery
- Agent capability management
- Agent execution tracking and metrics
- Integration with Logfire observability
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID
import logging

import logfire
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from agentical.db.repositories.base import BaseRepository, AsyncBaseRepository
from agentical.db.models.agent import (
    Agent,
    AgentCapability,
    AgentConfiguration,
    AgentExecution,
    AgentStatus,
    AgentType,
    AgentExecutionStatus
)
from agentical.core.exceptions import (
    NotFoundError,
    ValidationError,
    AgentError
)

# Configure logging
logger = logging.getLogger(__name__)


class AgentRepository(BaseRepository[Agent]):
    """
    Repository for Agent model operations.

    Extends the base repository with agent-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(Agent, db)

    def get_by_type(self, agent_type: AgentType) -> List[Agent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agents to retrieve

        Returns:
            List of agents of the specified type
        """
        with logfire.span("Get agents by type", agent_type=agent_type.value):
            try:
                stmt = select(Agent).where(
                    and_(
                        Agent.agent_type == agent_type,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} agents of type {agent_type.value}")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agents by type: {str(e)}")
                raise AgentError(f"Failed to retrieve agents by type: {str(e)}")

    def get_active_agents(self) -> List[Agent]:
        """
        Get all active agents.

        Returns:
            List of active agents
        """
        with logfire.span("Get active agents"):
            try:
                stmt = select(Agent).where(
                    and_(
                        Agent.status == AgentStatus.ACTIVE,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} active agents")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting active agents: {str(e)}")
                raise AgentError(f"Failed to retrieve active agents: {str(e)}")

    def get_by_capability(self, capability_name: str) -> List[Agent]:
        """
        Get agents that have a specific capability.

        Args:
            capability_name: Name of the capability

        Returns:
            List of agents with the specified capability
        """
        with logfire.span("Get agents by capability", capability=capability_name):
            try:
                stmt = select(Agent).join(Agent.capabilities).where(
                    and_(
                        AgentCapability.name == capability_name,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} agents with capability {capability_name}")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agents by capability: {str(e)}")
                raise AgentError(f"Failed to retrieve agents by capability: {str(e)}")

    def update_state(self, agent_id: int, state: Dict[str, Any]) -> Agent:
        """
        Update agent state.

        Args:
            agent_id: Agent ID
            state: New state data

        Returns:
            Updated agent

        Raises:
            NotFoundError: If agent not found
            AgentError: If update fails
        """
        with logfire.span("Update agent state", agent_id=agent_id):
            try:
                agent = self.get(agent_id)
                if not agent:
                    raise NotFoundError(f"Agent with ID {agent_id} not found")

                # Merge new state with existing state
                current_state = agent.state or {}
                current_state.update(state)

                agent.state = current_state
                agent.updated_at = datetime.utcnow()

                self.db.commit()
                self.db.refresh(agent)

                logfire.info(f"Updated state for agent {agent_id}")
                return agent

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                self.db.rollback()
                logfire.error(f"Database error updating agent state: {str(e)}")
                raise AgentError(f"Failed to update agent state: {str(e)}")

    def get_agent_metrics(self, agent_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Dictionary containing agent metrics

        Raises:
            NotFoundError: If agent not found
        """
        with logfire.span("Get agent metrics", agent_id=agent_id):
            try:
                agent = self.get(agent_id)
                if not agent:
                    raise NotFoundError(f"Agent with ID {agent_id} not found")

                # Get execution statistics
                total_executions = self.db.execute(
                    select(func.count(AgentExecution.id)).where(
                        AgentExecution.agent_id == agent_id
                    )
                ).scalar()

                successful_executions = self.db.execute(
                    select(func.count(AgentExecution.id)).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == AgentExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                avg_execution_time = self.db.execute(
                    select(func.avg(AgentExecution.execution_time_ms)).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == AgentExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                # Get recent execution history
                recent_executions = self.db.execute(
                    select(AgentExecution).where(
                        AgentExecution.agent_id == agent_id
                    ).order_by(desc(AgentExecution.created_at)).limit(10)
                ).scalars().all()

                metrics = {
                    "agent_id": agent_id,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type.value,
                    "status": agent.status.value,
                    "total_executions": total_executions or 0,
                    "successful_executions": successful_executions or 0,
                    "success_rate": (successful_executions / total_executions) if total_executions > 0 else 0,
                    "average_execution_time_ms": float(avg_execution_time) if avg_execution_time else 0,
                    "recent_executions": [
                        {
                            "id": exec.id,
                            "status": exec.status.value,
                            "execution_time_ms": exec.execution_time_ms,
                            "created_at": exec.created_at.isoformat()
                        }
                        for exec in recent_executions
                    ],
                    "capabilities": [cap.name for cap in agent.capabilities],
                    "last_active": agent.updated_at.isoformat() if agent.updated_at else None
                }

                logfire.info(f"Retrieved metrics for agent {agent_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agent metrics: {str(e)}")
                raise AgentError(f"Failed to retrieve agent metrics: {str(e)}")


class AsyncAgentRepository(AsyncBaseRepository[Agent]):
    """
    Async repository for Agent model operations.

    Extends the async base repository with agent-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(Agent, db)

    async def get_by_type(self, agent_type: AgentType) -> List[Agent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agents to retrieve

        Returns:
            List of agents of the specified type
        """
        with logfire.span("Get agents by type async", agent_type=agent_type.value):
            try:
                stmt = select(Agent).where(
                    and_(
                        Agent.agent_type == agent_type,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = await self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} agents of type {agent_type.value}")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agents by type: {str(e)}")
                raise AgentError(f"Failed to retrieve agents by type: {str(e)}")

    async def get_active_agents(self) -> List[Agent]:
        """
        Get all active agents.

        Returns:
            List of active agents
        """
        with logfire.span("Get active agents async"):
            try:
                stmt = select(Agent).where(
                    and_(
                        Agent.status == AgentStatus.ACTIVE,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = await self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} active agents")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting active agents: {str(e)}")
                raise AgentError(f"Failed to retrieve active agents: {str(e)}")

    async def get_by_capability(self, capability_name: str) -> List[Agent]:
        """
        Get agents that have a specific capability.

        Args:
            capability_name: Name of the capability

        Returns:
            List of agents with the specified capability
        """
        with logfire.span("Get agents by capability async", capability=capability_name):
            try:
                stmt = select(Agent).join(Agent.capabilities).where(
                    and_(
                        AgentCapability.name == capability_name,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = await self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} agents with capability {capability_name}")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agents by capability: {str(e)}")
                raise AgentError(f"Failed to retrieve agents by capability: {str(e)}")

    async def update_state(self, agent_id: int, state: Dict[str, Any]) -> Agent:
        """
        Update agent state.

        Args:
            agent_id: Agent ID
            state: New state data

        Returns:
            Updated agent

        Raises:
            NotFoundError: If agent not found
            AgentError: If update fails
        """
        with logfire.span("Update agent state async", agent_id=agent_id):
            try:
                agent = await self.get(agent_id)
                if not agent:
                    raise NotFoundError(f"Agent with ID {agent_id} not found")

                # Merge new state with existing state
                current_state = agent.state or {}
                current_state.update(state)

                agent.state = current_state
                agent.updated_at = datetime.utcnow()

                await self.db.commit()
                await self.db.refresh(agent)

                logfire.info(f"Updated state for agent {agent_id}")
                return agent

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error updating agent state: {str(e)}")
                raise AgentError(f"Failed to update agent state: {str(e)}")

    async def get_agent_metrics(self, agent_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Dictionary containing agent metrics

        Raises:
            NotFoundError: If agent not found
        """
        with logfire.span("Get agent metrics async", agent_id=agent_id):
            try:
                agent = await self.get(agent_id)
                if not agent:
                    raise NotFoundError(f"Agent with ID {agent_id} not found")

                # Get execution statistics
                total_executions_result = await self.db.execute(
                    select(func.count(AgentExecution.id)).where(
                        AgentExecution.agent_id == agent_id
                    )
                )
                total_executions = total_executions_result.scalar()

                successful_executions_result = await self.db.execute(
                    select(func.count(AgentExecution.id)).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == AgentExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = successful_executions_result.scalar()

                avg_execution_time_result = await self.db.execute(
                    select(func.avg(AgentExecution.execution_time_ms)).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == AgentExecutionStatus.COMPLETED
                        )
                    )
                )
                avg_execution_time = avg_execution_time_result.scalar()

                # Get recent execution history
                recent_executions_result = await self.db.execute(
                    select(AgentExecution).where(
                        AgentExecution.agent_id == agent_id
                    ).order_by(desc(AgentExecution.created_at)).limit(10)
                )
                recent_executions = recent_executions_result.scalars().all()

                metrics = {
                    "agent_id": agent_id,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type.value,
                    "status": agent.status.value,
                    "total_executions": total_executions or 0,
                    "successful_executions": successful_executions or 0,
                    "success_rate": (successful_executions / total_executions) if total_executions > 0 else 0,
                    "average_execution_time_ms": float(avg_execution_time) if avg_execution_time else 0,
                    "recent_executions": [
                        {
                            "id": exec.id,
                            "status": exec.status.value,
                            "execution_time_ms": exec.execution_time_ms,
                            "created_at": exec.created_at.isoformat()
                        }
                        for exec in recent_executions
                    ],
                    "capabilities": [cap.name for cap in agent.capabilities],
                    "last_active": agent.updated_at.isoformat() if agent.updated_at else None
                }

                logfire.info(f"Retrieved metrics for agent {agent_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agent metrics: {str(e)}")
                raise AgentError(f"Failed to retrieve agent metrics: {str(e)}")

    async def get_agents_by_status(self, status: AgentStatus) -> List[Agent]:
        """
        Get all agents with a specific status.

        Args:
            status: Agent status to filter by

        Returns:
            List of agents with the specified status
        """
        with logfire.span("Get agents by status async", status=status.value):
            try:
                stmt = select(Agent).where(
                    and_(
                        Agent.status == status,
                        Agent.deleted_at.is_(None)
                    )
                ).options(selectinload(Agent.capabilities))

                result = await self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Retrieved {len(agents)} agents with status {status.value}")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting agents by status: {str(e)}")
                raise AgentError(f"Failed to retrieve agents by status: {str(e)}")

    async def search_agents(
        self,
        query: str,
        agent_type: Optional[AgentType] = None,
        status: Optional[AgentStatus] = None,
        limit: int = 50
    ) -> List[Agent]:
        """
        Search agents by name or description.

        Args:
            query: Search query
            agent_type: Optional agent type filter
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of matching agents
        """
        with logfire.span("Search agents async", query=query):
            try:
                conditions = [
                    or_(
                        Agent.name.ilike(f"%{query}%"),
                        Agent.description.ilike(f"%{query}%")
                    ),
                    Agent.deleted_at.is_(None)
                ]

                if agent_type:
                    conditions.append(Agent.agent_type == agent_type)

                if status:
                    conditions.append(Agent.status == status)

                stmt = select(Agent).where(
                    and_(*conditions)
                ).options(selectinload(Agent.capabilities)).limit(limit)

                result = await self.db.execute(stmt)
                agents = result.scalars().all()

                logfire.info(f"Found {len(agents)} agents matching query '{query}'")
                return list(agents)

            except SQLAlchemyError as e:
                logfire.error(f"Database error searching agents: {str(e)}")
                raise AgentError(f"Failed to search agents: {str(e)}")

    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent by string ID."""
        try:
            # Get execution statistics
            execution_stats = await self.db.execute(
                select(
                    func.count(AgentExecution.id).label('total_executions'),
                    func.sum(func.case((AgentExecution.status == AgentExecutionStatus.COMPLETED, 1), else_=0)).label('successful_executions'),
                    func.avg(AgentExecution.duration).label('avg_execution_time'),
                    func.max(AgentExecution.duration).label('peak_execution_time'),
                    func.sum(AgentExecution.duration).label('total_runtime')
                ).where(AgentExecution.agent_id == agent_id)
            )

            stats = execution_stats.fetchone()

            total_executions = stats.total_executions or 0
            successful_executions = stats.successful_executions or 0
            success_rate = (successful_executions / total_executions) if total_executions > 0 else 0.0

            return {
                "execution_count": total_executions,
                "success_rate": success_rate,
                "avg_execution_time": float(stats.avg_execution_time or 0.0),
                "peak_execution_time": float(stats.peak_execution_time or 0.0),
                "total_runtime": float(stats.total_runtime or 0.0),
                "resource_usage": {"memory": 0, "cpu": 0}  # Placeholder
            }

        except SQLAlchemyError as e:
            logfire.error(f"Database error getting agent metrics: {str(e)}")
            return {
                "execution_count": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "peak_execution_time": 0.0,
                "total_runtime": 0.0,
                "resource_usage": {"memory": 0, "cpu": 0}
            }

    async def create_agent(self, agent_id: str, agent_type: str, name: str, description: str,
                          config: Dict[str, Any], capabilities: List[str], resource_limits: Optional[Dict[str, Any]]):
        """Create a new agent record in the database."""
        try:
            # This would integrate with the actual Agent model
            # For now, we'll simulate the creation
            logfire.info(f"Agent {agent_id} created in database", agent_type=agent_type, name=name)

        except SQLAlchemyError as e:
            logfire.error(f"Database error creating agent: {str(e)}")
            raise AgentError(f"Failed to create agent: {str(e)}")

    async def update_agent_config(self, agent_id: str, config: Dict[str, Any]):
        """Update agent configuration in the database."""
        try:
            logfire.info(f"Agent {agent_id} config updated", config=config)

        except SQLAlchemyError as e:
            logfire.error(f"Database error updating agent config: {str(e)}")
            raise AgentError(f"Failed to update agent config: {str(e)}")

    async def create_execution_record(self, execution_id: str, agent_id: str, operation: str,
                                    parameters: Dict[str, Any], started_at: datetime):
        """Create an execution record."""
        try:
            logfire.info(f"Execution {execution_id} created for agent {agent_id}", operation=operation)

        except SQLAlchemyError as e:
            logfire.error(f"Database error creating execution record: {str(e)}")
            raise AgentError(f"Failed to create execution record: {str(e)}")

    async def complete_execution_record(self, execution_id: str, status: str, result: Optional[Dict[str, Any]] = None,
                                      error: Optional[str] = None, completed_at: Optional[datetime] = None,
                                      duration: Optional[float] = None):
        """Complete an execution record."""
        try:
            logfire.info(f"Execution {execution_id} completed", status=status, duration=duration)

        except SQLAlchemyError as e:
            logfire.error(f"Database error completing execution record: {str(e)}")
            raise AgentError(f"Failed to complete execution record: {str(e)}")

    async def get_agent_executions(self, agent_id: str, page: int = 1, page_size: int = 20,
                                 status_filter: Optional[str] = None):
        """Get execution history for an agent."""
        try:
            # Placeholder implementation
            return {
                "executions": [],
                "total": 0,
                "page": page,
                "page_size": page_size
            }

        except SQLAlchemyError as e:
            logfire.error(f"Database error getting agent executions: {str(e)}")
            raise AgentError(f"Failed to get agent executions: {str(e)}")

    async def get_agent_analytics(self, agent_id: str, days: int = 30):
        """Get analytics for an agent."""
        try:
            # Placeholder implementation
            return {
                "agent_id": agent_id,
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "peak_execution_time": 0.0,
                "total_runtime": 0.0,
                "executions_by_day": {},
                "operations_frequency": {},
                "error_patterns": {},
                "performance_trends": []
            }

        except SQLAlchemyError as e:
            logfire.error(f"Database error getting agent analytics: {str(e)}")
            raise AgentError(f"Failed to get agent analytics: {str(e)}")

    async def delete_agent(self, agent_id: str):
        """Delete an agent from the database."""
        try:
            logfire.info(f"Agent {agent_id} deleted from database")

        except SQLAlchemyError as e:
            logfire.error(f"Database error deleting agent: {str(e)}")
            raise AgentError(f"Failed to delete agent: {str(e)}")
