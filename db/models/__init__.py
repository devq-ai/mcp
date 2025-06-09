"""
Database Models Package

This package contains all SQLAlchemy ORM models for the Agentical framework,
organized into domain-specific modules. These models define the database schema
and provide object-oriented access to database entities.

Features:
- SQLAlchemy ORM integration
- Base model with common fields
- Domain-specific model implementations
- Type annotations for IDE support
- Integration with Logfire observability
"""

from agentical.db.models.base import (
    BaseModel,
    TimestampMixin,
    UUIDMixin,
    SoftDeleteMixin,
    JSONSerializableMixin,
    MetadataMixin
)

from agentical.db.models.user import (
    User,
    Role,
    user_roles
)

# Export all models for easy imports
__all__ = [
    # Base models and mixins
    "BaseModel",
    "TimestampMixin",
    "UUIDMixin",
    "SoftDeleteMixin",
    "JSONSerializableMixin",
    "MetadataMixin",
    
    # User models
    "User",
    "Role",
    "user_roles",
]