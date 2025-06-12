"""
Tool Repository Implementation

This module provides the repository implementation for Tool model operations
in the Agentical framework. It extends the base repository with tool-specific
functionality for tool discovery, capability management, and usage tracking.

Features:
- Tool CRUD operations
- Tool capability management and discovery
- Tool category filtering and search
- Tool usage tracking and metrics
- Tool availability and status management
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
from agentical.db.models.tool import (
    Tool,
    ToolCapability,
    ToolParameter,
    ToolExecution,
    ToolType,
    ToolStatus,
    ToolExecutionStatus
)
from agentical.core.exceptions import (
    NotFoundError,
    ValidationError,
    ToolError
)

# Configure logging
logger = logging.getLogger(__name__)


class ToolRepository(BaseRepository[Tool]):
    """
    Repository for Tool model operations.

    Extends the base repository with tool-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(Tool, db)

    def get_by_category(self, category: str) -> List[Tool]:
        """
        Get all tools in a specific category.

        Args:
            category: Tool category

        Returns:
            List of tools in the specified category
        """
        with logfire.span("Get tools by category", category=category):
            try:
                stmt = select(Tool).where(
                    and_(
                        Tool.category == category,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} tools in category {category}")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tools by category: {str(e)}")
                raise ToolError(f"Failed to retrieve tools by category: {str(e)}")

    def get_available_tools(self) -> List[Tool]:
        """
        Get all available tools.

        Returns:
            List of available tools
        """
        with logfire.span("Get available tools"):
            try:
                stmt = select(Tool).where(
                    and_(
                        Tool.status == ToolStatus.AVAILABLE,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} available tools")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting available tools: {str(e)}")
                raise ToolError(f"Failed to retrieve available tools: {str(e)}")

    def get_by_capability(self, capability_name: str) -> List[Tool]:
        """
        Get tools that have a specific capability.

        Args:
            capability_name: Name of the capability

        Returns:
            List of tools with the specified capability
        """
        with logfire.span("Get tools by capability", capability=capability_name):
            try:
                stmt = select(Tool).join(Tool.capabilities).where(
                    and_(
                        ToolCapability.name == capability_name,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} tools with capability {capability_name}")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tools by capability: {str(e)}")
                raise ToolError(f"Failed to retrieve tools by capability: {str(e)}")

    def track_usage(self, tool_id: int, usage_data: Dict[str, Any]) -> ToolExecution:
        """
        Track tool usage with execution data.

        Args:
            tool_id: Tool ID
            usage_data: Usage tracking data

        Returns:
            Tool execution record

        Raises:
            NotFoundError: If tool not found
            ToolError: If tracking fails
        """
        with logfire.span("Track tool usage", tool_id=tool_id):
            try:
                tool = self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                # Create execution record
                execution = ToolExecution(
                    tool_id=tool_id,
                    input_parameters=usage_data.get("input_parameters", {}),
                    output_result=usage_data.get("output_result", {}),
                    execution_time_ms=usage_data.get("execution_time_ms"),
                    status=usage_data.get("status", ToolExecutionStatus.COMPLETED),
                    error_message=usage_data.get("error_message"),
                    metadata=usage_data.get("metadata", {}),
                    created_at=datetime.utcnow()
                )

                self.db.add(execution)
                self.db.commit()
                self.db.refresh(execution)

                # Update tool usage statistics
                tool.usage_count = (tool.usage_count or 0) + 1
                tool.last_used = datetime.utcnow()
                self.db.commit()

                logfire.info(f"Tracked usage for tool {tool_id}")
                return execution

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                self.db.rollback()
                logfire.error(f"Database error tracking tool usage: {str(e)}")
                raise ToolError(f"Failed to track tool usage: {str(e)}")

    def get_tool_capabilities(self, tool_id: int) -> List[ToolCapability]:
        """
        Get all capabilities for a specific tool.

        Args:
            tool_id: Tool ID

        Returns:
            List of tool capabilities

        Raises:
            NotFoundError: If tool not found
        """
        with logfire.span("Get tool capabilities", tool_id=tool_id):
            try:
                tool = self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                stmt = select(ToolCapability).where(
                    ToolCapability.tool_id == tool_id
                )

                result = self.db.execute(stmt)
                capabilities = result.scalars().all()

                logfire.info(f"Retrieved {len(capabilities)} capabilities for tool {tool_id}")
                return list(capabilities)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tool capabilities: {str(e)}")
                raise ToolError(f"Failed to retrieve tool capabilities: {str(e)}")

    def get_tool_metrics(self, tool_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a tool.

        Args:
            tool_id: Tool ID

        Returns:
            Dictionary containing tool metrics

        Raises:
            NotFoundError: If tool not found
        """
        with logfire.span("Get tool metrics", tool_id=tool_id):
            try:
                tool = self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                # Get execution statistics
                total_executions = self.db.execute(
                    select(func.count(ToolExecution.id)).where(
                        ToolExecution.tool_id == tool_id
                    )
                ).scalar()

                successful_executions = self.db.execute(
                    select(func.count(ToolExecution.id)).where(
                        and_(
                            ToolExecution.tool_id == tool_id,
                            ToolExecution.status == ToolExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                avg_execution_time = self.db.execute(
                    select(func.avg(ToolExecution.execution_time_ms)).where(
                        and_(
                            ToolExecution.tool_id == tool_id,
                            ToolExecution.status == ToolExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                # Get recent execution history
                recent_executions = self.db.execute(
                    select(ToolExecution).where(
                        ToolExecution.tool_id == tool_id
                    ).order_by(desc(ToolExecution.created_at)).limit(10)
                ).scalars().all()

                metrics = {
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "tool_type": tool.tool_type.value,
                    "category": tool.category,
                    "status": tool.status.value,
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
                    "capabilities": [cap.name for cap in tool.capabilities],
                    "last_used": tool.last_used.isoformat() if tool.last_used else None,
                    "usage_count": tool.usage_count or 0
                }

                logfire.info(f"Retrieved metrics for tool {tool_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tool metrics: {str(e)}")
                raise ToolError(f"Failed to retrieve tool metrics: {str(e)}")


class AsyncToolRepository(AsyncBaseRepository[Tool]):
    """
    Async repository for Tool model operations.

    Extends the async base repository with tool-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(Tool, db)

    async def get_by_category(self, category: str) -> List[Tool]:
        """
        Get all tools in a specific category.

        Args:
            category: Tool category

        Returns:
            List of tools in the specified category
        """
        with logfire.span("Get tools by category async", category=category):
            try:
                stmt = select(Tool).where(
                    and_(
                        Tool.category == category,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = await self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} tools in category {category}")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tools by category: {str(e)}")
                raise ToolError(f"Failed to retrieve tools by category: {str(e)}")

    async def get_available_tools(self) -> List[Tool]:
        """
        Get all available tools.

        Returns:
            List of available tools
        """
        with logfire.span("Get available tools async"):
            try:
                stmt = select(Tool).where(
                    and_(
                        Tool.status == ToolStatus.AVAILABLE,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = await self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} available tools")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting available tools: {str(e)}")
                raise ToolError(f"Failed to retrieve available tools: {str(e)}")

    async def get_by_capability(self, capability_name: str) -> List[Tool]:
        """
        Get tools that have a specific capability.

        Args:
            capability_name: Name of the capability

        Returns:
            List of tools with the specified capability
        """
        with logfire.span("Get tools by capability async", capability=capability_name):
            try:
                stmt = select(Tool).join(Tool.capabilities).where(
                    and_(
                        ToolCapability.name == capability_name,
                        Tool.deleted_at.is_(None)
                    )
                ).options(selectinload(Tool.capabilities))

                result = await self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Retrieved {len(tools)} tools with capability {capability_name}")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tools by capability: {str(e)}")
                raise ToolError(f"Failed to retrieve tools by capability: {str(e)}")

    async def track_usage(self, tool_id: int, usage_data: Dict[str, Any]) -> ToolExecution:
        """
        Track tool usage with execution data.

        Args:
            tool_id: Tool ID
            usage_data: Usage tracking data

        Returns:
            Tool execution record

        Raises:
            NotFoundError: If tool not found
            ToolError: If tracking fails
        """
        with logfire.span("Track tool usage async", tool_id=tool_id):
            try:
                tool = await self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                # Create execution record
                execution = ToolExecution(
                    tool_id=tool_id,
                    input_parameters=usage_data.get("input_parameters", {}),
                    output_result=usage_data.get("output_result", {}),
                    execution_time_ms=usage_data.get("execution_time_ms"),
                    status=usage_data.get("status", ToolExecutionStatus.COMPLETED),
                    error_message=usage_data.get("error_message"),
                    metadata=usage_data.get("metadata", {}),
                    created_at=datetime.utcnow()
                )

                self.db.add(execution)
                await self.db.commit()
                await self.db.refresh(execution)

                # Update tool usage statistics
                tool.usage_count = (tool.usage_count or 0) + 1
                tool.last_used = datetime.utcnow()
                await self.db.commit()

                logfire.info(f"Tracked usage for tool {tool_id}")
                return execution

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error tracking tool usage: {str(e)}")
                raise ToolError(f"Failed to track tool usage: {str(e)}")

    async def get_tool_capabilities(self, tool_id: int) -> List[ToolCapability]:
        """
        Get all capabilities for a specific tool.

        Args:
            tool_id: Tool ID

        Returns:
            List of tool capabilities

        Raises:
            NotFoundError: If tool not found
        """
        with logfire.span("Get tool capabilities async", tool_id=tool_id):
            try:
                tool = await self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                stmt = select(ToolCapability).where(
                    ToolCapability.tool_id == tool_id
                )

                result = await self.db.execute(stmt)
                capabilities = result.scalars().all()

                logfire.info(f"Retrieved {len(capabilities)} capabilities for tool {tool_id}")
                return list(capabilities)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tool capabilities: {str(e)}")
                raise ToolError(f"Failed to retrieve tool capabilities: {str(e)}")

    async def get_tool_metrics(self, tool_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a tool.

        Args:
            tool_id: Tool ID

        Returns:
            Dictionary containing tool metrics

        Raises:
            NotFoundError: If tool not found
        """
        with logfire.span("Get tool metrics async", tool_id=tool_id):
            try:
                tool = await self.get(tool_id)
                if not tool:
                    raise NotFoundError(f"Tool with ID {tool_id} not found")

                # Get execution statistics
                total_executions_result = await self.db.execute(
                    select(func.count(ToolExecution.id)).where(
                        ToolExecution.tool_id == tool_id
                    )
                )
                total_executions = total_executions_result.scalar()

                successful_executions_result = await self.db.execute(
                    select(func.count(ToolExecution.id)).where(
                        and_(
                            ToolExecution.tool_id == tool_id,
                            ToolExecution.status == ToolExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = successful_executions_result.scalar()

                avg_execution_time_result = await self.db.execute(
                    select(func.avg(ToolExecution.execution_time_ms)).where(
                        and_(
                            ToolExecution.tool_id == tool_id,
                            ToolExecution.status == ToolExecutionStatus.COMPLETED
                        )
                    )
                )
                avg_execution_time = avg_execution_time_result.scalar()

                # Get recent execution history
                recent_executions_result = await self.db.execute(
                    select(ToolExecution).where(
                        ToolExecution.tool_id == tool_id
                    ).order_by(desc(ToolExecution.created_at)).limit(10)
                )
                recent_executions = recent_executions_result.scalars().all()

                metrics = {
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "tool_type": tool.tool_type.value,
                    "category": tool.category,
                    "status": tool.status.value,
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
                    "capabilities": [cap.name for cap in tool.capabilities],
                    "last_used": tool.last_used.isoformat() if tool.last_used else None,
                    "usage_count": tool.usage_count or 0
                }

                logfire.info(f"Retrieved metrics for tool {tool_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tool metrics: {str(e)}")
                raise ToolError(f"Failed to retrieve tool metrics: {str(e)}")

    async def search_tools(
        self,
        query: str,
        category: Optional[str] = None,
        tool_type: Optional[ToolType] = None,
        status: Optional[ToolStatus] = None,
        limit: int = 50
    ) -> List[Tool]:
        """
        Search tools by name or description.

        Args:
            query: Search query
            category: Optional category filter
            tool_type: Optional tool type filter
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of matching tools
        """
        with logfire.span("Search tools async", query=query):
            try:
                conditions = [
                    or_(
                        Tool.name.ilike(f"%{query}%"),
                        Tool.description.ilike(f"%{query}%")
                    ),
                    Tool.deleted_at.is_(None)
                ]

                if category:
                    conditions.append(Tool.category == category)

                if tool_type:
                    conditions.append(Tool.tool_type == tool_type)

                if status:
                    conditions.append(Tool.status == status)

                stmt = select(Tool).where(
                    and_(*conditions)
                ).options(selectinload(Tool.capabilities)).limit(limit)

                result = await self.db.execute(stmt)
                tools = result.scalars().all()

                logfire.info(f"Found {len(tools)} tools matching query '{query}'")
                return list(tools)

            except SQLAlchemyError as e:
                logfire.error(f"Database error searching tools: {str(e)}")
                raise ToolError(f"Failed to search tools: {str(e)}")

    async def get_most_used_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently used tools.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of tools with usage statistics
        """
        with logfire.span("Get most used tools async", limit=limit):
            try:
                stmt = select(Tool).where(
                    and_(
                        Tool.usage_count > 0,
                        Tool.deleted_at.is_(None)
                    )
                ).order_by(desc(Tool.usage_count)).limit(limit)

                result = await self.db.execute(stmt)
                tools = result.scalars().all()

                tools_with_stats = []
                for tool in tools:
                    tools_with_stats.append({
                        "id": tool.id,
                        "name": tool.name,
                        "category": tool.category,
                        "usage_count": tool.usage_count or 0,
                        "last_used": tool.last_used.isoformat() if tool.last_used else None
                    })

                logfire.info(f"Retrieved {len(tools_with_stats)} most used tools")
                return tools_with_stats

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting most used tools: {str(e)}")
                raise ToolError(f"Failed to retrieve most used tools: {str(e)}")
