"""
Task Repository Implementation

This module provides the repository implementation for Task model operations
in the Agentical framework. It extends the base repository with task-specific
functionality for task lifecycle management, execution tracking, and dependency resolution.

Features:
- Task CRUD operations
- Task execution state management and tracking
- Task priority and status filtering
- Task dependency resolution and validation
- Task metrics and performance tracking
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
from agentical.db.models.task import (
    Task,
    TaskExecution,
    TaskResult,
    TaskPriority,
    TaskStatus,
    TaskType,
    TaskExecutionStatus
)
from agentical.core.exceptions import (
    NotFoundError,
    ValidationError,
    TaskError
)

# Configure logging
logger = logging.getLogger(__name__)


class TaskRepository(BaseRepository[Task]):
    """
    Repository for Task model operations.

    Extends the base repository with task-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(Task, db)

    def get_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with a specific status.

        Args:
            status: Task status to filter by

        Returns:
            List of tasks with the specified status
        """
        with logfire.span("Get tasks by status", status=status.value):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.status == status,
                        Task.deleted_at.is_(None)
                    )
                )

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks with status {status.value}")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks by status: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks by status: {str(e)}")

    def get_by_priority(self, priority: TaskPriority) -> List[Task]:
        """
        Get all tasks with a specific priority.

        Args:
            priority: Task priority to filter by

        Returns:
            List of tasks with the specified priority
        """
        with logfire.span("Get tasks by priority", priority=priority.value):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.priority == priority,
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.created_at)

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks with priority {priority.value}")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks by priority: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks by priority: {str(e)}")

    def get_by_type(self, task_type: TaskType) -> List[Task]:
        """
        Get all tasks of a specific type.

        Args:
            task_type: Type of tasks to retrieve

        Returns:
            List of tasks of the specified type
        """
        with logfire.span("Get tasks by type", task_type=task_type.value):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.task_type == task_type,
                        Task.deleted_at.is_(None)
                    )
                )

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks of type {task_type.value}")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks by type: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks by type: {str(e)}")

    def get_execution_history(self, task_id: int) -> List[TaskExecution]:
        """
        Get execution history for a specific task.

        Args:
            task_id: Task ID

        Returns:
            List of task executions

        Raises:
            NotFoundError: If task not found
        """
        with logfire.span("Get task execution history", task_id=task_id):
            try:
                task = self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                stmt = select(TaskExecution).where(
                    TaskExecution.task_id == task_id
                ).order_by(desc(TaskExecution.created_at))

                result = self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for task {task_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting task execution history: {str(e)}")
                raise TaskError(f"Failed to retrieve task execution history: {str(e)}")

    def update_execution_state(self, task_id: int, state: Dict[str, Any]) -> Task:
        """
        Update task execution state.

        Args:
            task_id: Task ID
            state: New execution state data

        Returns:
            Updated task

        Raises:
            NotFoundError: If task not found
            TaskError: If update fails
        """
        with logfire.span("Update task execution state", task_id=task_id):
            try:
                task = self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                # Merge new state with existing state
                current_state = task.execution_state or {}
                current_state.update(state)

                task.execution_state = current_state
                task.updated_at = datetime.utcnow()

                self.db.commit()
                self.db.refresh(task)

                logfire.info(f"Updated execution state for task {task_id}")
                return task

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                self.db.rollback()
                logfire.error(f"Database error updating task execution state: {str(e)}")
                raise TaskError(f"Failed to update task execution state: {str(e)}")

    def get_task_metrics(self, task_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary containing task metrics

        Raises:
            NotFoundError: If task not found
        """
        with logfire.span("Get task metrics", task_id=task_id):
            try:
                task = self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                # Get execution statistics
                total_executions = self.db.execute(
                    select(func.count(TaskExecution.id)).where(
                        TaskExecution.task_id == task_id
                    )
                ).scalar()

                successful_executions = self.db.execute(
                    select(func.count(TaskExecution.id)).where(
                        and_(
                            TaskExecution.task_id == task_id,
                            TaskExecution.status == TaskExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                avg_execution_time = self.db.execute(
                    select(func.avg(TaskExecution.execution_time_ms)).where(
                        and_(
                            TaskExecution.task_id == task_id,
                            TaskExecution.status == TaskExecutionStatus.COMPLETED
                        )
                    )
                ).scalar()

                # Get recent execution history
                recent_executions = self.db.execute(
                    select(TaskExecution).where(
                        TaskExecution.task_id == task_id
                    ).order_by(desc(TaskExecution.created_at)).limit(10)
                ).scalars().all()

                metrics = {
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "status": task.status.value,
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
                    "estimated_duration": task.estimated_duration,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "last_executed": task.last_executed.isoformat() if task.last_executed else None
                }

                logfire.info(f"Retrieved metrics for task {task_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting task metrics: {str(e)}")
                raise TaskError(f"Failed to retrieve task metrics: {str(e)}")

    def get_overdue_tasks(self) -> List[Task]:
        """
        Get all overdue tasks.

        Returns:
            List of overdue tasks
        """
        with logfire.span("Get overdue tasks"):
            try:
                current_time = datetime.utcnow()
                stmt = select(Task).where(
                    and_(
                        Task.due_date < current_time,
                        Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS]),
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.due_date)

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} overdue tasks")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting overdue tasks: {str(e)}")
                raise TaskError(f"Failed to retrieve overdue tasks: {str(e)}")

    def get_high_priority_tasks(self) -> List[Task]:
        """
        Get all high priority tasks.

        Returns:
            List of high priority tasks
        """
        with logfire.span("Get high priority tasks"):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.priority == TaskPriority.HIGH,
                        Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS]),
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.created_at)

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} high priority tasks")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting high priority tasks: {str(e)}")
                raise TaskError(f"Failed to retrieve high priority tasks: {str(e)}")

    def get_assigned_tasks(self, assignee_id: Optional[int] = None) -> List[Task]:
        """
        Get tasks assigned to a specific user or all assigned tasks.

        Args:
            assignee_id: Optional user ID to filter by

        Returns:
            List of assigned tasks
        """
        with logfire.span("Get assigned tasks", assignee_id=assignee_id):
            try:
                conditions = [
                    Task.deleted_at.is_(None)
                ]

                if assignee_id:
                    conditions.append(Task.assignee_id == assignee_id)
                else:
                    conditions.append(Task.assignee_id.is_not(None))

                stmt = select(Task).where(
                    and_(*conditions)
                ).order_by(Task.priority.desc(), Task.created_at)

                result = self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} assigned tasks")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting assigned tasks: {str(e)}")
                raise TaskError(f"Failed to retrieve assigned tasks: {str(e)}")


class AsyncTaskRepository(AsyncBaseRepository[Task]):
    """
    Async repository for Task model operations.

    Extends the async base repository with task-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(Task, db)

    async def get_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with a specific status.

        Args:
            status: Task status to filter by

        Returns:
            List of tasks with the specified status
        """
        with logfire.span("Get tasks by status async", status=status.value):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.status == status,
                        Task.deleted_at.is_(None)
                    )
                )

                result = await self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks with status {status.value}")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks by status: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks by status: {str(e)}")

    async def get_by_priority(self, priority: TaskPriority) -> List[Task]:
        """
        Get all tasks with a specific priority.

        Args:
            priority: Task priority to filter by

        Returns:
            List of tasks with the specified priority
        """
        with logfire.span("Get tasks by priority async", priority=priority.value):
            try:
                stmt = select(Task).where(
                    and_(
                        Task.priority == priority,
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.created_at)

                result = await self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks with priority {priority.value}")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks by priority: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks by priority: {str(e)}")

    async def get_execution_history(self, task_id: int) -> List[TaskExecution]:
        """
        Get execution history for a specific task.

        Args:
            task_id: Task ID

        Returns:
            List of task executions

        Raises:
            NotFoundError: If task not found
        """
        with logfire.span("Get task execution history async", task_id=task_id):
            try:
                task = await self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                stmt = select(TaskExecution).where(
                    TaskExecution.task_id == task_id
                ).order_by(desc(TaskExecution.created_at))

                result = await self.db.execute(stmt)
                executions = result.scalars().all()

                logfire.info(f"Retrieved {len(executions)} executions for task {task_id}")
                return list(executions)

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting task execution history: {str(e)}")
                raise TaskError(f"Failed to retrieve task execution history: {str(e)}")

    async def update_execution_state(self, task_id: int, state: Dict[str, Any]) -> Task:
        """
        Update task execution state.

        Args:
            task_id: Task ID
            state: New execution state data

        Returns:
            Updated task

        Raises:
            NotFoundError: If task not found
            TaskError: If update fails
        """
        with logfire.span("Update task execution state async", task_id=task_id):
            try:
                task = await self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                # Merge new state with existing state
                current_state = task.execution_state or {}
                current_state.update(state)

                task.execution_state = current_state
                task.updated_at = datetime.utcnow()

                await self.db.commit()
                await self.db.refresh(task)

                logfire.info(f"Updated execution state for task {task_id}")
                return task

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                await self.db.rollback()
                logfire.error(f"Database error updating task execution state: {str(e)}")
                raise TaskError(f"Failed to update task execution state: {str(e)}")

    async def get_task_metrics(self, task_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary containing task metrics

        Raises:
            NotFoundError: If task not found
        """
        with logfire.span("Get task metrics async", task_id=task_id):
            try:
                task = await self.get(task_id)
                if not task:
                    raise NotFoundError(f"Task with ID {task_id} not found")

                # Get execution statistics
                total_executions_result = await self.db.execute(
                    select(func.count(TaskExecution.id)).where(
                        TaskExecution.task_id == task_id
                    )
                )
                total_executions = total_executions_result.scalar()

                successful_executions_result = await self.db.execute(
                    select(func.count(TaskExecution.id)).where(
                        and_(
                            TaskExecution.task_id == task_id,
                            TaskExecution.status == TaskExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = successful_executions_result.scalar()

                avg_execution_time_result = await self.db.execute(
                    select(func.avg(TaskExecution.execution_time_ms)).where(
                        and_(
                            TaskExecution.task_id == task_id,
                            TaskExecution.status == TaskExecutionStatus.COMPLETED
                        )
                    )
                )
                avg_execution_time = avg_execution_time_result.scalar()

                # Get recent execution history
                recent_executions_result = await self.db.execute(
                    select(TaskExecution).where(
                        TaskExecution.task_id == task_id
                    ).order_by(desc(TaskExecution.created_at)).limit(10)
                )
                recent_executions = recent_executions_result.scalars().all()

                metrics = {
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "status": task.status.value,
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
                    "estimated_duration": task.estimated_duration,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "last_executed": task.last_executed.isoformat() if task.last_executed else None
                }

                logfire.info(f"Retrieved metrics for task {task_id}")
                return metrics

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logfire.error(f"Database error getting task metrics: {str(e)}")
                raise TaskError(f"Failed to retrieve task metrics: {str(e)}")

    async def search_tasks(
        self,
        query: str,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        limit: int = 50
    ) -> List[Task]:
        """
        Search tasks by name or description.

        Args:
            query: Search query
            task_type: Optional task type filter
            status: Optional status filter
            priority: Optional priority filter
            limit: Maximum number of results

        Returns:
            List of matching tasks
        """
        with logfire.span("Search tasks async", query=query):
            try:
                conditions = [
                    or_(
                        Task.name.ilike(f"%{query}%"),
                        Task.description.ilike(f"%{query}%")
                    ),
                    Task.deleted_at.is_(None)
                ]

                if task_type:
                    conditions.append(Task.task_type == task_type)

                if status:
                    conditions.append(Task.status == status)

                if priority:
                    conditions.append(Task.priority == priority)

                stmt = select(Task).where(
                    and_(*conditions)
                ).limit(limit)

                result = await self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Found {len(tasks)} tasks matching query '{query}'")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error searching tasks: {str(e)}")
                raise TaskError(f"Failed to search tasks: {str(e)}")

    async def get_overdue_tasks(self) -> List[Task]:
        """
        Get all overdue tasks.

        Returns:
            List of overdue tasks
        """
        with logfire.span("Get overdue tasks async"):
            try:
                current_time = datetime.utcnow()
                stmt = select(Task).where(
                    and_(
                        Task.due_date < current_time,
                        Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS]),
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.due_date)

                result = await self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} overdue tasks")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting overdue tasks: {str(e)}")
                raise TaskError(f"Failed to retrieve overdue tasks: {str(e)}")

    async def get_tasks_due_soon(self, hours: int = 24) -> List[Task]:
        """
        Get tasks that are due within the specified hours.

        Args:
            hours: Number of hours to look ahead

        Returns:
            List of tasks due soon
        """
        with logfire.span("Get tasks due soon async", hours=hours):
            try:
                current_time = datetime.utcnow()
                future_time = current_time + timedelta(hours=hours)

                stmt = select(Task).where(
                    and_(
                        Task.due_date.between(current_time, future_time),
                        Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS]),
                        Task.deleted_at.is_(None)
                    )
                ).order_by(Task.due_date)

                result = await self.db.execute(stmt)
                tasks = result.scalars().all()

                logfire.info(f"Retrieved {len(tasks)} tasks due within {hours} hours")
                return list(tasks)

            except SQLAlchemyError as e:
                logfire.error(f"Database error getting tasks due soon: {str(e)}")
                raise TaskError(f"Failed to retrieve tasks due soon: {str(e)}")
