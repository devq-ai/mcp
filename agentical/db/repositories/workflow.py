"""
Workflow Repository Implementation

This module provides the repository implementation for Workflow model operations
in the Agentical framework. It extends the base repository with workflow-specific
functionality for workflow lifecycle management, execution tracking, and state persistence.

Features:
- Workflow CRUD operations
- Workflow execution state management and tracking
- Workflow status filtering and discovery
- Workflow step execution monitoring
- Workflow metrics and performance tracking
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
from agentical.db.models.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStepExecution,
    WorkflowType,
    WorkflowStatus,
    WorkflowExecutionStatus,
    WorkflowStepType,
    WorkflowStepStatus
)
from agentical.core.exceptions import (
    NotFoundError,
    ValidationError,
    WorkflowError
)

# Configure logging
logger = logging.getLogger(__name__)


class WorkflowRepository(BaseRepository[Workflow]):
    """
    Repository for Workflow model operations.

    Extends the base repository with workflow-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(Workflow, db)

    def get_by_status(self, status: WorkflowStatus) -> List[Workflow]:
        """
        Get all workflows with a specific status.

        Args:
            status: Workflow status to filter by

        Returns:
            List of workflows with the specified status
        """
        with logfire.span("Get workflows by status", status=status.value):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.status == status,
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} workflows with status {status.value}")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflows by status: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflows by status: {str(e)}")

    def get_by_type(self, workflow_type: WorkflowType) -> List[Workflow]:
        """
        Get all workflows of a specific type.

        Args:
            workflow_type: Type of workflows to retrieve

        Returns:
            List of workflows of the specified type
        """
        with logfire.span("Get workflows by type", workflow_type=workflow_type.value):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.workflow_type == workflow_type,
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} workflows of type {workflow_type.value}")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflows by type: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflows by type: {str(e)}")

    def get_execution_history(self, workflow_id: int) -> List[WorkflowExecution]:
        """
        Get execution history for a specific workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of workflow executions

        Raises:
            NotFoundError: If workflow not found
        """
        with logfire.span("Get workflow execution history", workflow_id=workflow_id):
            try:
                workflow = self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                stmt = select(WorkflowExecution).where(
                    WorkflowExecution.workflow_id == workflow_id
                ).order_by(desc(WorkflowExecution.created_at))

                result = self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for workflow {workflow_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflow execution history: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflow execution history: {str(e)}")

    def update_execution_state(self, workflow_id: int, state: Dict[str, Any]) -> Workflow:
        """
        Update workflow execution state.

        Args:
            workflow_id: Workflow ID
            state: New execution state data

        Returns:
            Updated workflow

        Raises:
            NotFoundError: If workflow not found
            WorkflowError: If update fails
        """
        with logfire.span("Update workflow execution state", workflow_id=workflow_id):
            try:
                workflow = self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                # Merge new state with existing state
                current_state = workflow.execution_state or {}
                current_state.update(state)

                workflow.execution_state = current_state
                workflow.updated_at = datetime.utcnow()

                self.db.commit()
                self.db.refresh(workflow)

                logfire.info(f"Updated execution state for workflow {workflow_id}")
                return workflow

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                self.db.rollback()
                logfire.error(f"Database error updating workflow execution state: {str(e)}")
                raise WorkflowError(f"Failed to update workflow execution state: {str(e)}")

    def get_workflow_metrics(self, workflow_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dictionary containing workflow metrics

        Raises:
            NotFoundError: If workflow not found
        """
        with logfire.span("Get workflow metrics", workflow_id=workflow_id):
            try:
                workflow = self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                # Get execution statistics
                total_executions = self.db.execute(
                    select(func.count(WorkflowExecution.id)).where(
                        WorkflowExecution.workflow_id == workflow_id
                    )
                ).scalar()

                successful_executions = self.db.execute(
                    select(func.count(WorkflowExecution.id)).where(
                        and_(
                            WorkflowExecution.workflow_id == workflow_id,
                            WorkflowExecution.status == WorkflowExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                avg_execution_time = self.db.execute(
                    select(func.avg(WorkflowExecution.execution_time_ms)).where(
                        and_(
                            WorkflowExecution.workflow_id == workflow_id,
                            WorkflowExecution.status == WorkflowExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                # Get step statistics
                total_steps = self.db.execute(
                    select(func.count(WorkflowStep.id)).where(
                        WorkflowStep.workflow_id == workflow_id
                    )
                ).scalar()

                # Get recent execution history
                recent_executions = self.db.execute(
                    select(WorkflowExecution).where(
                        WorkflowExecution.workflow_id == workflow_id
                    ).order_by(desc(WorkflowExecution.created_at)).limit(10)
                ).scalars().all()

                metrics = {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow.name,
                    "workflow_type": workflow.workflow_type.value,
                    "status": workflow.status.value,
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
                    "last_executed": workflow.last_executed.isoformat() if workflow.last_executed else None,
                    "execution_count": workflow.execution_count or 0
                }

                logfire.info(f"Retrieved metrics for workflow {workflow_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflow metrics: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflow metrics: {str(e)}")

    def get_active_workflows(self) -> List[Workflow]:
        """
        Get all active workflows.

        Returns:
            List of active workflows
        """
        with logfire.span("Get active workflows"):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.status.in_([WorkflowStatus.ACTIVE, WorkflowStatus.RUNNING]),
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} active workflows")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting active workflows: {str(e)}")
                raise WorkflowError(f"Failed to retrieve active workflows: {str(e)}")


class AsyncWorkflowRepository(AsyncBaseRepository[Workflow]):
    """
    Async repository for Workflow model operations.

    Extends the async base repository with workflow-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(Workflow, db)

    async def get_by_status(self, status: WorkflowStatus) -> List[Workflow]:
        """
        Get all workflows with a specific status.

        Args:
            status: Workflow status to filter by

        Returns:
            List of workflows with the specified status
        """
        with logfire.span("Get workflows by status async", status=status.value):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.status == status,
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = await self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} workflows with status {status.value}")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflows by status: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflows by status: {str(e)}")

    async def get_by_type(self, workflow_type: WorkflowType) -> List[Workflow]:
        """
        Get all workflows of a specific type.

        Args:
            workflow_type: Type of workflows to retrieve

        Returns:
            List of workflows of the specified type
        """
        with logfire.span("Get workflows by type async", workflow_type=workflow_type.value):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.workflow_type == workflow_type,
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = await self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} workflows of type {workflow_type.value}")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflows by type: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflows by type: {str(e)}")

    async def get_execution_history(self, workflow_id: int) -> List[WorkflowExecution]:
        """
        Get execution history for a specific workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of workflow executions

        Raises:
            NotFoundError: If workflow not found
        """
        with logfire.span("Get workflow execution history async", workflow_id=workflow_id):
            try:
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                stmt = select(WorkflowExecution).where(
                    WorkflowExecution.workflow_id == workflow_id
                ).order_by(desc(WorkflowExecution.created_at))

                result = await self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for workflow {workflow_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflow execution history: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflow execution history: {str(e)}")

    async def update_execution_state(self, workflow_id: int, state: Dict[str, Any]) -> Workflow:
        """
        Update workflow execution state.

        Args:
            workflow_id: Workflow ID
            state: New execution state data

        Returns:
            Updated workflow

        Raises:
            NotFoundError: If workflow not found
            WorkflowError: If update fails
        """
        with logfire.span("Update workflow execution state async", workflow_id=workflow_id):
            try:
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                # Merge new state with existing state
                current_state = workflow.execution_state or {}
                current_state.update(state)

                workflow.execution_state = current_state
                workflow.updated_at = datetime.utcnow()

                await self.db.commit()
                await self.db.refresh(workflow)

                logfire.info(f"Updated execution state for workflow {workflow_id}")
                return workflow

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error updating workflow execution state: {str(e)}")
                raise WorkflowError(f"Failed to update workflow execution state: {str(e)}")

    async def get_workflow_metrics(self, workflow_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dictionary containing workflow metrics

        Raises:
            NotFoundError: If workflow not found
        """
        with logfire.span("Get workflow metrics async", workflow_id=workflow_id):
            try:
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                # Get execution statistics
                total_executions_result = await self.db.execute(
                    select(func.count(WorkflowExecution.id)).where(
                        WorkflowExecution.workflow_id == workflow_id
                    )
                )
                total_executions = total_executions_result.scalar()

                successful_executions_result = await self.db.execute(
                    select(func.count(WorkflowExecution.id)).where(
                        and_(
                            WorkflowExecution.workflow_id == workflow_id,
                            WorkflowExecution.status == WorkflowExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = successful_executions_result.scalar()

                avg_execution_time_result = await self.db.execute(
                    select(func.avg(WorkflowExecution.execution_time_ms)).where(
                        and_(
                            WorkflowExecution.workflow_id == workflow_id,
                            WorkflowExecution.status == WorkflowExecutionStatus.COMPLETED
                        )
                    )
                )
                avg_execution_time = avg_execution_time_result.scalar()

                # Get step statistics
                total_steps_result = await self.db.execute(
                    select(func.count(WorkflowStep.id)).where(
                        WorkflowStep.workflow_id == workflow_id
                    )
                )
                total_steps = total_steps_result.scalar()

                # Get recent execution history
                recent_executions_result = await self.db.execute(
                    select(WorkflowExecution).where(
                        WorkflowExecution.workflow_id == workflow_id
                    ).order_by(desc(WorkflowExecution.created_at)).limit(10)
                )
                recent_executions = recent_executions_result.scalars().all()

                metrics = {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow.name,
                    "workflow_type": workflow.workflow_type.value,
                    "status": workflow.status.value,
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
                    "last_executed": workflow.last_executed.isoformat() if workflow.last_executed else None,
                    "execution_count": workflow.execution_count or 0
                }

                logfire.info(f"Retrieved metrics for workflow {workflow_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflow metrics: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflow metrics: {str(e)}")

    async def get_active_workflows(self) -> List[Workflow]:
        """
        Get all active workflows.

        Returns:
            List of active workflows
        """
        with logfire.span("Get active workflows async"):
            try:
                stmt = select(Workflow).where(
                    and_(
                        Workflow.status.in_([WorkflowStatus.ACTIVE, WorkflowStatus.RUNNING]),
                        Workflow.deleted_at.is_(None)
                    )
                ).options(selectinload(Workflow.steps))

                result = await self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Retrieved {len(workflows)} active workflows")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting active workflows: {str(e)}")
                raise WorkflowError(f"Failed to retrieve active workflows: {str(e)}")

    async def search_workflows(
        self,
        query: str,
        workflow_type: Optional[WorkflowType] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50
    ) -> List[Workflow]:
        """
        Search workflows by name or description.

        Args:
            query: Search query
            workflow_type: Optional workflow type filter
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of matching workflows
        """
        with logfire.span("Search workflows async", query=query):
            try:
                conditions = [
                    or_(
                        Workflow.name.ilike(f"%{query}%"),
                        Workflow.description.ilike(f"%{query}%")
                    ),
                    Workflow.deleted_at.is_(None)
                ]

                if workflow_type:
                    conditions.append(Workflow.workflow_type == workflow_type)

                if status:
                    conditions.append(Workflow.status == status)

                stmt = select(Workflow).where(
                    and_(*conditions)
                ).options(selectinload(Workflow.steps)).limit(limit)

                result = await self.db.execute(stmt)
                workflows = result.scalars().all()

                logfire.info(f"Found {len(workflows)} workflows matching query '{query}'")
                return list(workflows)

            except SQLAlchemyError as e:
                logfire.error(f"Database error searching workflows: {str(e)}")
                raise WorkflowError(f"Failed to search workflows: {str(e)}")

    async def get_workflow_step_executions(self, workflow_id: int, execution_id: int) -> List[WorkflowStepExecution]:
        """
        Get step executions for a specific workflow execution.

        Args:
            workflow_id: Workflow ID
            execution_id: Workflow execution ID

        Returns:
            List of workflow step executions

        Raises:
            NotFoundError: If workflow or execution not found
        """
        with logfire.span("Get workflow step executions async", workflow_id=workflow_id, execution_id=execution_id):
            try:
                # Verify workflow exists
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise NotFoundError(f"Workflow with ID {workflow_id} not found")

                # Verify execution exists
                execution_result = await self.db.execute(
                    select(WorkflowExecution).where(
                        and_(
                            WorkflowExecution.id == execution_id,
                            WorkflowExecution.workflow_id == workflow_id
                        )
                    )
                )
                execution = execution_result.scalars().first()
                if not execution:
                    raise NotFoundError(f"Workflow execution with ID {execution_id} not found")

                # Get step executions
                stmt = select(WorkflowStepExecution).where(
                    WorkflowStepExecution.workflow_execution_id == execution_id
                ).order_by(WorkflowStepExecution.step_order)

                result = await self.db.execute(stmt)
                step_executions = result.scalars().all()

                logfire.info(f"Retrieved {len(step_executions)} step executions for workflow {workflow_id}, execution {execution_id}")
                return list(step_executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting workflow step executions: {str(e)}")
                raise WorkflowError(f"Failed to retrieve workflow step executions: {str(e)}")
