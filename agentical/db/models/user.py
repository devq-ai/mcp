"""
User Model for Agentical

This module defines the User model for authentication and authorization
in the Agentical framework, extending the base model with user-specific
fields and methods.

Features:
- User entity with authentication fields
- Password hashing and verification
- Role-based permissions
- Integration with FastAPI security
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import secrets
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, Table
from sqlalchemy.orm import relationship
from passlib.context import CryptContext

from .base import BaseModel
from .. import Base

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User-role association table
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("user.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("role.id"), primary_key=True)
)


class Role(BaseModel):
    """Role model for user permissions."""

    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=True)
    permissions = Column(String, nullable=True)  # JSON string of permissions

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")

    def __repr__(self) -> str:
        return f"<Role(name={self.name})>"

    @property
    def permission_set(self) -> Set[str]:
        """Get permissions as a set."""
        if not self.permissions:
            return set()

        import json
        try:
            return set(json.loads(self.permissions))
        except json.JSONDecodeError:
            return set()


class User(BaseModel):
    """User model for authentication and authorization."""

    # Authentication fields
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(100), nullable=True)
    verification_token_expires = Column(DateTime, nullable=True)

    # Profile fields
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    display_name = Column(String(100), nullable=True)
    profile_image_url = Column(String(255), nullable=True)

    # Security fields
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    account_locked = Column(Boolean, default=False, nullable=False)
    account_locked_until = Column(DateTime, nullable=True)
    password_reset_token = Column(String(100), nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)

    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")

    def __repr__(self) -> str:
        return f"<User(username={self.username}, email={self.email})>"

    @property
    def password(self) -> str:
        """Password getter for compatibility."""
        raise AttributeError("Password is write-only")

    @password.setter
    def password(self, plain_password: str) -> None:
        """Set password, automatically hashing it."""
        self.hashed_password = pwd_context.hash(plain_password)

    def verify_password(self, plain_password: str) -> bool:
        """Verify password against stored hash."""
        return pwd_context.verify(plain_password, self.hashed_password)

    def generate_verification_token(self, expires_in: int = 86400) -> str:
        """Generate email verification token.

        Args:
            expires_in: Token expiry time in seconds (default: 24 hours)

        Returns:
            Verification token
        """
        token = secrets.token_urlsafe(32)
        self.verification_token = token
        self.verification_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return token

    def verify_email(self, token: str) -> bool:
        """Verify email with token.

        Args:
            token: Verification token

        Returns:
            True if verification successful, False otherwise
        """
        if (
            self.verification_token == token and
            self.verification_token_expires > datetime.utcnow()
        ):
            self.is_verified = True
            self.verification_token = None
            self.verification_token_expires = None
            return True
        return False

    def generate_password_reset_token(self, expires_in: int = 3600) -> str:
        """Generate password reset token.

        Args:
            expires_in: Token expiry time in seconds (default: 1 hour)

        Returns:
            Password reset token
        """
        token = secrets.token_urlsafe(32)
        self.password_reset_token = token
        self.password_reset_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return token

    def verify_password_reset_token(self, token: str) -> bool:
        """Verify password reset token.

        Args:
            token: Password reset token

        Returns:
            True if token is valid, False otherwise
        """
        return (
            self.password_reset_token == token and
            self.password_reset_expires > datetime.utcnow()
        )

    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password with token.

        Args:
            token: Password reset token
            new_password: New password to set

        Returns:
            True if password reset successful, False otherwise
        """
        if self.verify_password_reset_token(token):
            self.password = new_password
            self.password_reset_token = None
            self.password_reset_expires = None
            return True
        return False

    def record_login(self, success: bool) -> None:
        """Record login attempt.

        Args:
            success: Whether login was successful
        """
        if success:
            self.last_login = datetime.utcnow()
            self.failed_login_attempts = 0
            self.account_locked = False
            self.account_locked_until = None
        else:
            self.failed_login_attempts += 1
            # Lock account after 5 failed attempts
            if self.failed_login_attempts >= 5:
                self.account_locked = True
                self.account_locked_until = datetime.utcnow() + timedelta(minutes=15)

    def is_account_locked(self) -> bool:
        """Check if account is locked.

        Returns:
            True if account is locked, False otherwise
        """
        if not self.account_locked:
            return False

        # Check if lock has expired
        if self.account_locked_until and self.account_locked_until <= datetime.utcnow():
            self.account_locked = False
            return False

        return True

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        for role in self.roles:
            if permission in role.permission_set:
                return True
        return False

    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role.

        Args:
            role_name: Role name to check

        Returns:
            True if user has role, False otherwise
        """
        return any(role.name == role_name for role in self.roles)

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary without sensitive fields."""
        result = super().to_dict()

        # Remove sensitive fields
        sensitive_fields = [
            "hashed_password", "verification_token", "password_reset_token",
            "failed_login_attempts", "account_locked", "account_locked_until"
        ]
        for field in sensitive_fields:
            if field in result:
                del result[field]

        # Add derived fields
        result["full_name"] = f"{self.first_name or ''} {self.last_name or ''}".strip()
        result["roles"] = [role.name for role in self.roles]

        return result
