"""
Playbook Repository Implementation

This module provides the repository implementation for Playbook model operations
in the Agentical framework. It extends the base repository with playbook-specific
functionality for playbook lifecycle management, execution tracking, and template management.

Features:
- Playbook CRUD operations
- Playbook execution state management and tracking
- Playbook template management and versioning
- Playbook step execution monitoring
- Playbook category filtering and discovery
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
from agentical.db.models.playbook import (
    Playbook,
    PlaybookStep,
    PlaybookVariable,
    PlaybookExecution,
    PlaybookStepExecution,
    PlaybookTemplate,
    PlaybookCategory,
    PlaybookStatus,
    PlaybookExecutionStatus,
    PlaybookStepType,
    PlaybookStepStatus,
    VariableType
)
from agentical.core.exceptions import (
    NotFoundError,
    ValidationError,
    PlaybookError
)

# Configure logging
logger = logging.getLogger(__name__)


class PlaybookRepository(BaseRepository[Playbook]):
    """
    Repository for Playbook model operations.

    Extends the base repository with playbook-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(Playbook, db)

    def get_by_status(self, status: PlaybookStatus) -> List[Playbook]:
        """
        Get all playbooks with a specific status.

        Args:
            status: Playbook status to filter by

        Returns:
            List of playbooks with the specified status
        """
        with logfire.span("Get playbooks by status", status=status.value):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.status == status,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps), selectinload(Playbook.variables))

                result = self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks with status {status.value}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by status: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by status: {str(e)}")

    def get_by_category(self, category: PlaybookCategory) -> List[Playbook]:
        """
        Get all playbooks in a specific category.

        Args:
            category: Playbook category to filter by

        Returns:
            List of playbooks in the specified category
        """
        with logfire.span("Get playbooks by category", category=category.value):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.category == category,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps), selectinload(Playbook.variables))

                result = self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks in category {category.value}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by category: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by category: {str(e)}")

    def get_execution_history(self, playbook_id: int) -> List[PlaybookExecution]:
        """
        Get execution history for a specific playbook.

        Args:
            playbook_id: Playbook ID

        Returns:
            List of playbook executions

        Raises:
            NotFoundError: If playbook not found
        """
        with logfire.span("Get playbook execution history", playbook_id=playbook_id):
            try:
                playbook = self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                stmt = select(PlaybookExecution).where(
                    PlaybookExecution.playbook_id == playbook_id
                ).order_by(desc(PlaybookExecution.created_at))

                result = self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for playbook {playbook_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbook execution history: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbook execution history: {str(e)}")

    def update_execution_state(self, playbook_id: int, state: Dict[str, Any]) -> Playbook:
        """
        Update playbook execution state.

        Args:
            playbook_id: Playbook ID
            state: New execution state data

        Returns:
            Updated playbook

        Raises:
            NotFoundError: If playbook not found
            PlaybookError: If update fails
        """
        with logfire.span("Update playbook execution state", playbook_id=playbook_id):
            try:
                playbook = self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                # Merge new state with existing state
                current_state = playbook.execution_state or {}
                current_state.update(state)

                playbook.execution_state = current_state
                playbook.updated_at = datetime.utcnow()

                self.db.commit()
                self.db.refresh(playbook)

                logfire.info(f"Updated execution state for playbook {playbook_id}")
                return playbook

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                self.db.rollback()
                logfire.error(f"Database error updating playbook execution state: {str(e)}")
                raise PlaybookError(f"Failed to update playbook execution state: {str(e)}")

    def get_playbook_metrics(self, playbook_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a playbook.

        Args:
            playbook_id: Playbook ID

        Returns:
            Dictionary containing playbook metrics

        Raises:
            NotFoundError: If playbook not found
        """
        with logfire.span("Get playbook metrics", playbook_id=playbook_id):
            try:
                playbook = self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                # Get execution statistics
                total_executions = self.db.execute(
                    select(func.count(PlaybookExecution.id)).where(
                        PlaybookExecution.playbook_id == playbook_id
                    )
                ).scalar()

                successful_executions = self.db.execute(
                    select(func.count(PlaybookExecution.id)).where(
                        and_(
                            PlaybookExecution.playbook_id == playbook_id,
                            PlaybookExecution.status == PlaybookExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                avg_execution_time = self.db.execute(
                    select(func.avg(PlaybookExecution.execution_time_ms)).where(
                        and_(
                            PlaybookExecution.playbook_id == playbook_id,
                            PlaybookExecution.status == PlaybookExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                # Get step statistics
                total_steps = self.db.execute(
                    select(func.count(PlaybookStep.id)).where(
                        PlaybookStep.playbook_id == playbook_id
                    )
                ).scalar()

                # Get recent execution history
                recent_executions = self.db.execute(
                    select(PlaybookExecution).where(
                        PlaybookExecution.playbook_id == playbook_id
                    ).order_by(desc(PlaybookExecution.created_at)).limit(10)
                ).scalars().all()

                metrics = {
                    "playbook_id": playbook_id,
                    "playbook_name": playbook.name,
                    "category": playbook.category.value,
                    "status": playbook.status.value,
                    "version": playbook.version,
                    "total_executions": total_executions or 0,
                    "successful_executions": successful_executions or 0,
                    "success_rate": (successful_executions / total_executions) if total_executions > 0 else 0,
                    "average_execution_time_ms": float(avg_execution_time) if avg_execution_time else 0,
                    "total_steps": total_steps or 0,
                    "recent_executions": [
                        {
                            "id": exec.id,
                            "status": exec.status.value,
                            "execution_time_ms": exec.execution_time_ms,
                            "created_at": exec.created_at.isoformat()
                        }
                        for exec in recent_executions
                    ],
                    "last_executed": playbook.last_executed.isoformat() if playbook.last_executed else None,
                    "execution_count": playbook.execution_count or 0
                }

                logfire.info(f"Retrieved metrics for playbook {playbook_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbook metrics: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbook metrics: {str(e)}")

    def get_by_template(self, template_id: int) -> List[Playbook]:
        """
        Get all playbooks created from a specific template.

        Args:
            template_id: Template ID

        Returns:
            List of playbooks created from the template
        """
        with logfire.span("Get playbooks by template", template_id=template_id):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.template_id == template_id,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps))

                result = self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks from template {template_id}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by template: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by template: {str(e)}")

    def get_active_playbooks(self) -> List[Playbook]:
        """
        Get all active playbooks.

        Returns:
            List of active playbooks
        """
        with logfire.span("Get active playbooks"):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.status.in_([PlaybookStatus.ACTIVE, PlaybookStatus.RUNNING]),
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps))

                result = self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} active playbooks")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting active playbooks: {str(e)}")
                raise PlaybookError(f"Failed to retrieve active playbooks: {str(e)}")


class AsyncPlaybookRepository(AsyncBaseRepository[Playbook]):
    """
    Async repository for Playbook model operations.

    Extends the async base repository with playbook-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(Playbook, db)

    async def get_by_status(self, status: PlaybookStatus) -> List[Playbook]:
        """
        Get all playbooks with a specific status.

        Args:
            status: Playbook status to filter by

        Returns:
            List of playbooks with the specified status
        """
        with logfire.span("Get playbooks by status async", status=status.value):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.status == status,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps), selectinload(Playbook.variables))

                result = await self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks with status {status.value}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by status: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by status: {str(e)}")

    async def get_by_category(self, category: PlaybookCategory) -> List[Playbook]:
        """
        Get all playbooks in a specific category.

        Args:
            category: Playbook category to filter by

        Returns:
            List of playbooks in the specified category
        """
        with logfire.span("Get playbooks by category async", category=category.value):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.category == category,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps), selectinload(Playbook.variables))

                result = await self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks in category {category.value}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by category: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by category: {str(e)}")

    async def get_execution_history(self, playbook_id: int) -> List[PlaybookExecution]:
        """
        Get execution history for a specific playbook.

        Args:
            playbook_id: Playbook ID

        Returns:
            List of playbook executions

        Raises:
            NotFoundError: If playbook not found
        """
        with logfire.span("Get playbook execution history async", playbook_id=playbook_id):
            try:
                playbook = await self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                stmt = select(PlaybookExecution).where(
                    PlaybookExecution.playbook_id == playbook_id
                ).order_by(desc(PlaybookExecution.created_at))

                result = await self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for playbook {playbook_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbook execution history: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbook execution history: {str(e)}")

    async def update_execution_state(self, playbook_id: int, state: Dict[str, Any]) -> Playbook:
        """
        Update playbook execution state.

        Args:
            playbook_id: Playbook ID
            state: New execution state data

        Returns:
            Updated playbook

        Raises:
            NotFoundError: If playbook not found
            PlaybookError: If update fails
        """
        with logfire.span("Update playbook execution state async", playbook_id=playbook_id):
            try:
                playbook = await self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                # Merge new state with existing state
                current_state = playbook.execution_state or {}
                current_state.update(state)

                playbook.execution_state = current_state
                playbook.updated_at = datetime.utcnow()

                await self.db.commit()
                await self.db.refresh(playbook)

                logfire.info(f"Updated execution state for playbook {playbook_id}")
                return playbook

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error updating playbook execution state: {str(e)}")
                raise PlaybookError(f"Failed to update playbook execution state: {str(e)}")

    async def get_playbook_metrics(self, playbook_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a playbook.

        Args:
            playbook_id: Playbook ID

        Returns:
            Dictionary containing playbook metrics

        Raises:
            NotFoundError: If playbook not found
        """
        with logfire.span("Get playbook metrics async", playbook_id=playbook_id):
            try:
                playbook = await self.get(playbook_id)
                if not playbook:
                    raise NotFoundError(f"Playbook with ID {playbook_id} not found")

                # Get execution statistics
                total_executions_result = await self.db.execute(
                    select(func.count(PlaybookExecution.id)).where(
                        PlaybookExecution.playbook_id == playbook_id
                    )
                )
                total_executions = total_executions_result.scalar()

                successful_executions_result = await self.db.execute(
                    select(func.count(PlaybookExecution.id)).where(
                        and_(
                            PlaybookExecution.playbook_id == playbook_id,
                            PlaybookExecution.status == PlaybookExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = successful_executions_result.scalar()

                avg_execution_time_result = await self.db.execute(
                    select(func.avg(PlaybookExecution.execution_time_ms)).where(
                        and_(
                            PlaybookExecution.playbook_id == playbook_id,
                            PlaybookExecution.status == PlaybookExecutionStatus.COMPLETED
                        )
                    )
                )
                avg_execution_time = avg_execution_time_result.scalar()

                # Get step statistics
                total_steps_result = await self.db.execute(
                    select(func.count(PlaybookStep.id)).where(
                        PlaybookStep.playbook_id == playbook_id
                    )
                )
                total_steps = total_steps_result.scalar()

                # Get recent execution history
                recent_executions_result = await self.db.execute(
                    select(PlaybookExecution).where(
                        PlaybookExecution.playbook_id == playbook_id
                    ).order_by(desc(PlaybookExecution.created_at)).limit(10)
                )
                recent_executions = recent_executions_result.scalars().all()

                metrics = {
                    "playbook_id": playbook_id,
                    "playbook_name": playbook.name,
                    "category": playbook.category.value,
                    "status": playbook.status.value,
                    "version": playbook.version,
                    "total_executions": total_executions or 0,
                    "successful_executions": successful_executions or 0,
                    "success_rate": (successful_executions / total_executions) if total_executions > 0 else 0,
                    "average_execution_time_ms": float(avg_execution_time) if avg_execution_time else 0,
                    "total_steps": total_steps or 0,
                    "recent_executions": [
                        {
                            "id": exec.id,
                            "status": exec.status.value,
                            "execution_time_ms": exec.execution_time_ms,
                            "created_at": exec.created_at.isoformat()
                        }
                        for exec in recent_executions
                    ],
                    "last_executed": playbook.last_executed.isoformat() if playbook.last_executed else None,
                    "execution_count": playbook.execution_count or 0
                }

                logfire.info(f"Retrieved metrics for playbook {playbook_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbook metrics: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbook metrics: {str(e)}")

    async def search_playbooks(
        self,
        query: str,
        category: Optional[PlaybookCategory] = None,
        status: Optional[PlaybookStatus] = None,
        limit: int = 50
    ) -> List[Playbook]:
        """
        Search playbooks by name or description.

        Args:
            query: Search query
            category: Optional category filter
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of matching playbooks
        """
        with logfire.span("Search playbooks async", query=query):
            try:
                conditions = [
                    or_(
                        Playbook.name.ilike(f"%{query}%"),
                        Playbook.description.ilike(f"%{query}%")
                    ),
                    Playbook.deleted_at.is_(None)
                ]

                if category:
                    conditions.append(Playbook.category == category)

                if status:
                    conditions.append(Playbook.status == status)

                stmt = select(Playbook).where(
                    and_(*conditions)
                ).options(selectinload(Playbook.steps)).limit(limit)

                result = await self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Found {len(playbooks)} playbooks matching query '{query}'")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error searching playbooks: {str(e)}")
                raise PlaybookError(f"Failed to search playbooks: {str(e)}")

    async def get_by_template(self, template_id: int) -> List[Playbook]:
        """
        Get all playbooks created from a specific template.

        Args:
            template_id: Template ID

        Returns:
            List of playbooks created from the template
        """
        with logfire.span("Get playbooks by template async", template_id=template_id):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.template_id == template_id,
                        Playbook.deleted_at.is_(None)
                    )
                ).options(selectinload(Playbook.steps))

                result = await self.db.execute(stmt)
                playbooks = result.scalars().all()

                logfire.info(f"Retrieved {len(playbooks)} playbooks from template {template_id}")
                return list(playbooks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting playbooks by template: {str(e)}")
                raise PlaybookError(f"Failed to retrieve playbooks by template: {str(e)}")

    async def create_from_template(self, template_id: int, playbook_data: Dict[str, Any]) -> Playbook:
        """
        Create a new playbook from a template.

        Args:
            template_id: Template ID
            playbook_data: Playbook customization data

        Returns:
            Created playbook

        Raises:
            NotFoundError: If template not found
            PlaybookError: If creation fails
        """
        with logfire.span("Create playbook from template async", template_id=template_id):
            try:
                # Get template
                template_result = await self.db.execute(
                    select(PlaybookTemplate).where(PlaybookTemplate.id == template_id)
                )
                template = template_result.scalars().first()
                if not template:
                    raise NotFoundError(f"Playbook template with ID {template_id} not found")

                # Create playbook from template
                playbook = Playbook(
                    name=playbook_data.get("name", template.name),
                    description=playbook_data.get("description", template.description),
                    category=playbook_data.get("category", template.category),
                    template_id=template_id,
                    version="1.0.0",
                    status=PlaybookStatus.DRAFT,
                    configuration=template.default_configuration or {},
                    metadata=playbook_data.get("metadata", {}),
                    created_at=datetime.utcnow()
                )

                self.db.add(playbook)
                await self.db.commit()
                await self.db.refresh(playbook)

                logfire.info(f"Created playbook {playbook.id} from template {template_id}")
                return playbook

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error creating playbook from template: {str(e)}")
                raise PlaybookError(f"Failed to create playbook from template: {str(e)}")

    async def get_most_used_playbooks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently executed playbooks.

        Args:
            limit: Maximum number of playbooks to return

        Returns:
            List of playbooks with usage statistics
        """
        with logfire.span("Get most used playbooks async", limit=limit):
            try:
                stmt = select(Playbook).where(
                    and_(
                        Playbook.execution_count > 0,
                        Playbook.deleted_at.is_(None)
                    )
                ).order_by(desc(Playbook.execution_count)).limit(limit)

                result = await self.db.execute(stmt)
                playbooks = result.scalars().all()

                playbooks_with_stats = []
                for playbook in playbooks:
                    playbooks_with_stats.append({
                        "id": playbook.id,
                        "name": playbook.name,
                        "category": playbook.category.value,
                        "execution_count": playbook.execution_count or 0,
                        "last_executed": playbook.last_executed.isoformat() if playbook.last_executed else None,
                        "version": playbook.version
                    })

                logfire.info(f"Retrieved {len(playbooks_with_stats)} most used playbooks")
                return playbooks_with_stats

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting most used playbooks: {str(e)}")
                raise PlaybookError(f"Failed to retrieve most used playbooks: {str(e)}")
