"""
Base Database Models for Agentical

This module defines the base models for all database entities in the Agentical
framework, providing common fields, methods, and utilities.

Features:
- Base model with common fields (id, created_at, updated_at)
- Soft delete functionality
- Tracking for creation and modification timestamps
- UUID generation for unique identifiers
- JSON serialization utilities
- SQLAlchemy ORM integration
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Type, Generic, ClassVar
from uuid import uuid4
import json

from sqlalchemy import Column, DateTime, Boolean, String, func, Integer
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .. import Base

# Type variable for model class
T = TypeVar('T', bound='BaseModel')


class BaseModel(Base):
    """Base model for all database entities."""

    __abstract__ = True

    # Common columns for all entities
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid4()))
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, nullable=False, default=True)

    # Store extra_data in a JSON column if available, otherwise use String
    try:
        extra_data = Column(MutableDict.as_mutable(JSONB), nullable=True)
    except ImportError:
        extra_data = Column(String, nullable=True)

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name automatically based on class name."""
        return cls.__name__.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for serialization."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)

            # Convert datetime to ISO format
            if isinstance(value, datetime):
                value = value.isoformat()

            # Handle JSON stored as string
            if column.name == 'extra_data' and isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    value = {}

            result[column.name] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__table__.columns.keys()
        })

    def update(self, data: Dict[str, Any]) -> None:
        """Update model from dictionary."""
        for key, value in data.items():
            if key in self.__table__.columns.keys():
                setattr(self, key, value)

    def soft_delete(self) -> None:
        """Mark record as inactive (soft delete)."""
        self.is_active = False

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get extra data as dictionary."""
        if self.extra_data is None:
            return {}

        if isinstance(self.extra_data, dict):
            return self.extra_data

        try:
            return json.loads(self.extra_data)
        except (json.JSONDecodeError, TypeError):
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]) -> None:
        """Set extra data."""
        if isinstance(self.extra_data, dict):
            self.extra_data = value
        else:
            self.extra_data = json.dumps(value)

    def update_metadata_dict(self, data: Dict[str, Any]) -> None:
        """Update extra data with new values."""
        current = self.metadata_dict
        current.update(data)
        self.metadata_dict = current

    def __repr__(self) -> str:
        """String representation of model."""
        return f"<{self.__class__.__name__}(id={self.id}, uuid={self.uuid})>"


class TimestampMixin:
    """Mixin for timestamp tracking."""

    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class UUIDMixin:
    """Mixin for UUID generation."""

    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid4()))


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""

    is_active = Column(Boolean, nullable=False, default=True)

    def soft_delete(self) -> None:
        """Mark record as inactive (soft delete)."""
        self.is_active = False

    @classmethod
    def get_active(cls, session):
        """Query only active records."""
        return session.query(cls).filter(cls.is_active == True)


class JSONSerializableMixin:
    """Mixin for JSON serialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for serialization."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)

            # Convert datetime to ISO format
            if isinstance(value, datetime):
                value = value.isoformat()

            result[column.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__table__.columns.keys()
        })


class MetadataMixin:
    """Mixin for metadata storage."""

    # Try to use PostgreSQL JSONB if available, otherwise use String
    try:
        extra_data = Column(MutableDict.as_mutable(JSONB), nullable=True)
    except ImportError:
        extra_data = Column(String, nullable=True)

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get extra data as dictionary."""
        if self.extra_data is None:
            return {}

        if isinstance(self.extra_data, dict):
            return self.extra_data

        try:
            return json.loads(self.extra_data)
        except (json.JSONDecodeError, TypeError):
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]) -> None:
        """Set extra data."""
        if isinstance(self.extra_data, dict):
            self.extra_data = value
        else:
            self.extra_data = json.dumps(value)

    def update_metadata_dict(self, data: Dict[str, Any]) -> None:
        """Update extra data with new values."""
        current = self.metadata_dict
        current.update(data)
        self.metadata_dict = current
