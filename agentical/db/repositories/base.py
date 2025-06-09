"""
Base Repository Pattern Implementation

This module provides the base repository pattern implementation for database operations
in the Agentical framework. The repository pattern abstracts the data access layer,
providing a clean separation between business logic and data access code.

Features:
- Generic CRUD operations for any model type
- Type-safe implementation with SQLAlchemy
- Support for both sync and async database operations
- Query optimization through profiling and caching
- Integration with Logfire observability
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast
import logging
from datetime import datetime

import logfire
from sqlalchemy import asc, desc, func, select, update, delete
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.exc import SQLAlchemyError

from agentical.db.models.base import BaseModel
from agentical.db.profiler import profile_sqlalchemy_query
from agentical.db.cache import cached, async_cached

# Type variable for model class
ModelType = TypeVar("ModelType", bound=BaseModel)

# Configure logging
logger = logging.getLogger(__name__)


class BaseRepository(Generic[ModelType]):
    """
    Base repository for database operations.
    
    Generic repository pattern implementation that works with any SQLAlchemy model.
    Provides standard CRUD operations and common query patterns.
    
    Attributes:
        model: The SQLAlchemy model class
        db: Database session
    """
    
    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        """
        Get entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity or None if not found
        """
        with logfire.span(f"Get {self.model.__name__} by ID"):
            return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_by_uuid(self, uuid: str) -> Optional[ModelType]:
        """
        Get entity by UUID.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            Entity or None if not found
        """
        with logfire.span(f"Get {self.model.__name__} by UUID"):
            return self.db.query(self.model).filter(self.model.uuid == uuid).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> List[ModelType]:
        """
        Get all entities with pagination and sorting.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Column to sort by
            sort_desc: Sort in descending order if True
            
        Returns:
            List of entities
        """
        with logfire.span(f"Get all {self.model.__name__}"):
            query = self.db.query(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                query = query.filter(self.model.is_active == True)
            
            # Apply sorting if specified
            if sort_by and hasattr(self.model, sort_by):
                column = getattr(self.model, sort_by)
                query = query.order_by(desc(column) if sort_desc else asc(column))
            else:
                # Default sort by id
                query = query.order_by(desc(self.model.id) if sort_desc else asc(self.model.id))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            return query.all()
    
    def count(self, filter_by: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filtering.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            Count of matching entities
        """
        with logfire.span(f"Count {self.model.__name__}"):
            query = self.db.query(func.count(self.model.id))
            
            # Apply filters if specified
            if filter_by:
                for field, value in filter_by.items():
                    if hasattr(self.model, field):
                        query = query.filter(getattr(self.model, field) == value)
            
            return query.scalar() or 0
    
    def find(self, filter_by: Dict[str, Any]) -> List[ModelType]:
        """
        Find entities by field values.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            List of matching entities
        """
        with logfire.span(f"Find {self.model.__name__}"):
            query = self.db.query(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                query = query.filter(self.model.is_active == True)
            
            # Apply filters
            for field, value in filter_by.items():
                if hasattr(self.model, field):
                    query = query.filter(getattr(self.model, field) == value)
            
            return query.all()
    
    def find_one(self, filter_by: Dict[str, Any]) -> Optional[ModelType]:
        """
        Find a single entity by field values.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            Matching entity or None if not found
        """
        with logfire.span(f"Find one {self.model.__name__}"):
            query = self.db.query(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                query = query.filter(self.model.is_active == True)
            
            # Apply filters
            for field, value in filter_by.items():
                if hasattr(self.model, field):
                    query = query.filter(getattr(self.model, field) == value)
            
            return query.first()
    
    @profile_sqlalchemy_query
    def create(self, data: Dict[str, Any]) -> ModelType:
        """
        Create a new entity.
        
        Args:
            data: Dictionary of field values
            
        Returns:
            Created entity
        """
        with logfire.span(f"Create {self.model.__name__}"):
            try:
                db_obj = self.model(**data)
                self.db.add(db_obj)
                self.db.commit()
                self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error creating {self.model.__name__}: {e}")
                logfire.error(
                    f"Database error creating {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    data=str(data)
                )
                raise
    
    @profile_sqlalchemy_query
    def update(self, id: int, data: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an entity by ID.
        
        Args:
            id: Entity ID
            data: Dictionary of field values to update
            
        Returns:
            Updated entity or None if not found
        """
        with logfire.span(f"Update {self.model.__name__}"):
            try:
                db_obj = self.db.query(self.model).filter(self.model.id == id).first()
                if not db_obj:
                    return None
                
                for key, value in data.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                
                self.db.add(db_obj)
                self.db.commit()
                self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error updating {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error updating {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id,
                    data=str(data)
                )
                raise
    
    @profile_sqlalchemy_query
    def update_by_uuid(self, uuid: str, data: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an entity by UUID.
        
        Args:
            uuid: Entity UUID
            data: Dictionary of field values to update
            
        Returns:
            Updated entity or None if not found
        """
        with logfire.span(f"Update {self.model.__name__} by UUID"):
            try:
                db_obj = self.db.query(self.model).filter(self.model.uuid == uuid).first()
                if not db_obj:
                    return None
                
                for key, value in data.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                
                self.db.add(db_obj)
                self.db.commit()
                self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error updating {self.model.__name__} with UUID {uuid}: {e}")
                logfire.error(
                    f"Database error updating {self.model.__name__} by UUID",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_uuid=uuid,
                    data=str(data)
                )
                raise
    
    @profile_sqlalchemy_query
    def delete(self, id: int) -> bool:
        """
        Delete an entity by ID.
        
        For models with is_active field, this performs a soft delete.
        Otherwise, it performs a hard delete.
        
        Args:
            id: Entity ID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Delete {self.model.__name__}"):
            try:
                db_obj = self.db.query(self.model).filter(self.model.id == id).first()
                if not db_obj:
                    return False
                
                # Perform soft delete if possible
                if hasattr(db_obj, 'is_active'):
                    db_obj.is_active = False
                    self.db.add(db_obj)
                else:
                    # Hard delete
                    self.db.delete(db_obj)
                
                self.db.commit()
                return True
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error deleting {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error deleting {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id
                )
                raise
    
    @profile_sqlalchemy_query
    def delete_by_uuid(self, uuid: str) -> bool:
        """
        Delete an entity by UUID.
        
        For models with is_active field, this performs a soft delete.
        Otherwise, it performs a hard delete.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Delete {self.model.__name__} by UUID"):
            try:
                db_obj = self.db.query(self.model).filter(self.model.uuid == uuid).first()
                if not db_obj:
                    return False
                
                # Perform soft delete if possible
                if hasattr(db_obj, 'is_active'):
                    db_obj.is_active = False
                    self.db.add(db_obj)
                else:
                    # Hard delete
                    self.db.delete(db_obj)
                
                self.db.commit()
                return True
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error deleting {self.model.__name__} with UUID {uuid}: {e}")
                logfire.error(
                    f"Database error deleting {self.model.__name__} by UUID",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_uuid=uuid
                )
                raise
    
    @profile_sqlalchemy_query
    def hard_delete(self, id: int) -> bool:
        """
        Permanently delete an entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Hard delete {self.model.__name__}"):
            try:
                db_obj = self.db.query(self.model).filter(self.model.id == id).first()
                if not db_obj:
                    return False
                
                self.db.delete(db_obj)
                self.db.commit()
                return True
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error hard deleting {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error hard deleting {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id
                )
                raise
    
    @cached(ttl=300)  # Cache for 5 minutes
    def get_cached(self, id: int) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID with caching.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity dictionary or None if not found
        """
        with logfire.span(f"Get cached {self.model.__name__}"):
            obj = self.get(id)
            return obj.to_dict() if obj else None
    
    @cached(ttl=300)  # Cache for 5 minutes
    def get_cached_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by UUID with caching.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            Entity dictionary or None if not found
        """
        with logfire.span(f"Get cached {self.model.__name__} by UUID"):
            obj = self.get_by_uuid(uuid)
            return obj.to_dict() if obj else None


class AsyncBaseRepository(Generic[ModelType]):
    """
    Async base repository for database operations.
    
    Generic repository pattern implementation that works with any SQLAlchemy model
    in an asynchronous context.
    
    Attributes:
        model: The SQLAlchemy model class
        db: Async database session
    """
    
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            db: Async database session
        """
        self.model = model
        self.db = db
    
    async def get(self, id: int) -> Optional[ModelType]:
        """
        Get entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity or None if not found
        """
        with logfire.span(f"Get async {self.model.__name__} by ID"):
            stmt = select(self.model).where(self.model.id == id)
            result = await self.db.execute(stmt)
            return result.scalars().first()
    
    async def get_by_uuid(self, uuid: str) -> Optional[ModelType]:
        """
        Get entity by UUID.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            Entity or None if not found
        """
        with logfire.span(f"Get async {self.model.__name__} by UUID"):
            stmt = select(self.model).where(self.model.uuid == uuid)
            result = await self.db.execute(stmt)
            return result.scalars().first()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> List[ModelType]:
        """
        Get all entities with pagination and sorting.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Column to sort by
            sort_desc: Sort in descending order if True
            
        Returns:
            List of entities
        """
        with logfire.span(f"Get all async {self.model.__name__}"):
            stmt = select(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                stmt = stmt.where(self.model.is_active == True)
            
            # Apply sorting if specified
            if sort_by and hasattr(self.model, sort_by):
                column = getattr(self.model, sort_by)
                stmt = stmt.order_by(desc(column) if sort_desc else asc(column))
            else:
                # Default sort by id
                stmt = stmt.order_by(desc(self.model.id) if sort_desc else asc(self.model.id))
            
            # Apply pagination
            stmt = stmt.offset(skip).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
    
    async def count(self, filter_by: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filtering.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            Count of matching entities
        """
        with logfire.span(f"Count async {self.model.__name__}"):
            stmt = select(func.count(self.model.id))
            
            # Apply filters if specified
            if filter_by:
                for field, value in filter_by.items():
                    if hasattr(self.model, field):
                        stmt = stmt.where(getattr(self.model, field) == value)
            
            result = await self.db.execute(stmt)
            return result.scalar() or 0
    
    async def find(self, filter_by: Dict[str, Any]) -> List[ModelType]:
        """
        Find entities by field values.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            List of matching entities
        """
        with logfire.span(f"Find async {self.model.__name__}"):
            stmt = select(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                stmt = stmt.where(self.model.is_active == True)
            
            # Apply filters
            for field, value in filter_by.items():
                if hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
    
    async def find_one(self, filter_by: Dict[str, Any]) -> Optional[ModelType]:
        """
        Find a single entity by field values.
        
        Args:
            filter_by: Dictionary of field-value pairs to filter by
            
        Returns:
            Matching entity or None if not found
        """
        with logfire.span(f"Find one async {self.model.__name__}"):
            stmt = select(self.model)
            
            # Only include active records by default if model has is_active field
            if hasattr(self.model, 'is_active'):
                stmt = stmt.where(self.model.is_active == True)
            
            # Apply filters
            for field, value in filter_by.items():
                if hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)
            
            result = await self.db.execute(stmt)
            return result.scalars().first()
    
    async def create(self, data: Dict[str, Any]) -> ModelType:
        """
        Create a new entity.
        
        Args:
            data: Dictionary of field values
            
        Returns:
            Created entity
        """
        with logfire.span(f"Create async {self.model.__name__}"):
            try:
                db_obj = self.model(**data)
                self.db.add(db_obj)
                await self.db.commit()
                await self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error creating {self.model.__name__}: {e}")
                logfire.error(
                    f"Database error creating async {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    data=str(data)
                )
                raise
    
    async def update(self, id: int, data: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an entity by ID.
        
        Args:
            id: Entity ID
            data: Dictionary of field values to update
            
        Returns:
            Updated entity or None if not found
        """
        with logfire.span(f"Update async {self.model.__name__}"):
            try:
                stmt = select(self.model).where(self.model.id == id)
                result = await self.db.execute(stmt)
                db_obj = result.scalars().first()
                
                if not db_obj:
                    return None
                
                for key, value in data.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                
                self.db.add(db_obj)
                await self.db.commit()
                await self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error updating {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error updating async {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id,
                    data=str(data)
                )
                raise
    
    async def update_by_uuid(self, uuid: str, data: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an entity by UUID.
        
        Args:
            uuid: Entity UUID
            data: Dictionary of field values to update
            
        Returns:
            Updated entity or None if not found
        """
        with logfire.span(f"Update async {self.model.__name__} by UUID"):
            try:
                stmt = select(self.model).where(self.model.uuid == uuid)
                result = await self.db.execute(stmt)
                db_obj = result.scalars().first()
                
                if not db_obj:
                    return None
                
                for key, value in data.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                
                self.db.add(db_obj)
                await self.db.commit()
                await self.db.refresh(db_obj)
                return db_obj
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error updating {self.model.__name__} with UUID {uuid}: {e}")
                logfire.error(
                    f"Database error updating async {self.model.__name__} by UUID",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_uuid=uuid,
                    data=str(data)
                )
                raise
    
    async def delete(self, id: int) -> bool:
        """
        Delete an entity by ID.
        
        For models with is_active field, this performs a soft delete.
        Otherwise, it performs a hard delete.
        
        Args:
            id: Entity ID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Delete async {self.model.__name__}"):
            try:
                stmt = select(self.model).where(self.model.id == id)
                result = await self.db.execute(stmt)
                db_obj = result.scalars().first()
                
                if not db_obj:
                    return False
                
                # Perform soft delete if possible
                if hasattr(db_obj, 'is_active'):
                    db_obj.is_active = False
                    self.db.add(db_obj)
                else:
                    # Hard delete
                    await self.db.delete(db_obj)
                
                await self.db.commit()
                return True
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error deleting {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error deleting async {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id
                )
                raise
    
    async def delete_by_uuid(self, uuid: str) -> bool:
        """
        Delete an entity by UUID.
        
        For models with is_active field, this performs a soft delete.
        Otherwise, it performs a hard delete.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Delete async {self.model.__name__} by UUID"):
            try:
                stmt = select(self.model).where(self.model.uuid == uuid)
                result = await self.db.execute(stmt)
                db_obj = result.scalars().first()
                
                if not db_obj:
                    return False
                
                # Perform soft delete if possible
                if hasattr(db_obj, 'is_active'):
                    db_obj.is_active = False
                    self.db.add(db_obj)
                else:
                    # Hard delete
                    await self.db.delete(db_obj)
                
                await self.db.commit()
                return True
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error deleting {self.model.__name__} with UUID {uuid}: {e}")
                logfire.error(
                    f"Database error deleting async {self.model.__name__} by UUID",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_uuid=uuid
                )
                raise
    
    async def hard_delete(self, id: int) -> bool:
        """
        Permanently delete an entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            True if entity was deleted, False if not found
        """
        with logfire.span(f"Hard delete async {self.model.__name__}"):
            try:
                stmt = select(self.model).where(self.model.id == id)
                result = await self.db.execute(stmt)
                db_obj = result.scalars().first()
                
                if not db_obj:
                    return False
                
                await self.db.delete(db_obj)
                await self.db.commit()
                return True
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error hard deleting {self.model.__name__} with ID {id}: {e}")
                logfire.error(
                    f"Database error hard deleting async {self.model.__name__}",
                    error=str(e),
                    error_type=type(e).__name__,
                    entity_id=id
                )
                raise
    
    @async_cached(ttl=300)  # Cache for 5 minutes
    async def get_cached(self, id: int) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID with caching.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity dictionary or None if not found
        """
        with logfire.span(f"Get cached async {self.model.__name__}"):
            obj = await self.get(id)
            return obj.to_dict() if obj else None
    
    @async_cached(ttl=300)  # Cache for 5 minutes
    async def get_cached_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by UUID with caching.
        
        Args:
            uuid: Entity UUID
            
        Returns:
            Entity dictionary or None if not found
        """
        with logfire.span(f"Get cached async {self.model.__name__} by UUID"):
            obj = await self.get_by_uuid(uuid)
            return obj.to_dict() if obj else None