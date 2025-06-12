"""
User Repository Implementation

This module provides the repository implementation for User model operations
in the Agentical framework. It extends the base repository with user-specific
functionality for authentication, role management, and security operations.

Features:
- User CRUD operations
- Authentication and password management
- Role-based authorization
- Account security (locking, verification, etc.)
- Integration with Logfire observability
"""

from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timedelta
import logging
import secrets

import logfire
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from agentical.db.repositories.base import BaseRepository, AsyncBaseRepository
from agentical.db.models.user import User, Role
from agentical.db.cache import cached, async_cached
from agentical.core.exceptions import NotFoundError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User]):
    """
    Repository for User model operations.

    Extends the base repository with user-specific functionality.
    """

    def __init__(self, db: Session):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        super().__init__(User, db)

    def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username to search for

        Returns:
            User or None if not found
        """
        with logfire.span("Get user by username"):
            return self.db.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: Email to search for

        Returns:
            User or None if not found
        """
        with logfire.span("Get user by email"):
            return self.db.query(User).filter(User.email == email).first()

    def authenticate(self, username_or_email: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/email and password.

        Args:
            username_or_email: Username or email
            password: Plain text password

        Returns:
            User if authentication successful, None otherwise
        """
        with logfire.span("Authenticate user"):
            # Find user by username or email
            user = self.db.query(User).filter(
                or_(
                    User.username == username_or_email,
                    User.email == username_or_email
                )
            ).first()

            if not user:
                return None

            # Check if account is locked
            if user.is_account_locked():
                logfire.warning(
                    "Authentication attempt on locked account",
                    username=user.username,
                    account_locked_until=user.account_locked_until
                )
                return None

            # Verify password
            if user.verify_password(password):
                # Record successful login
                user.record_login(success=True)
                self.db.add(user)
                self.db.commit()
                return user
            else:
                # Record failed login
                user.record_login(success=False)
                self.db.add(user)
                self.db.commit()
                return None

    def create_user(self, data: Dict[str, Any], password: str) -> User:
        """
        Create a new user with password.

        Args:
            data: User data
            password: Plain text password

        Returns:
            Created user
        """
        with logfire.span("Create user"):
            try:
                user = User(**data)
                user.password = password  # This will hash the password

                # Generate verification token if not verified
                if not user.is_verified:
                    user.generate_verification_token()

                self.db.add(user)
                self.db.commit()
                self.db.refresh(user)
                return user
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Error creating user: {e}")
                logfire.error(
                    "Database error creating user",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if password changed successfully, False otherwise
        """
        with logfire.span("Change user password"):
            user = self.get(user_id)
            if not user:
                return False

            # Verify current password
            if not user.verify_password(current_password):
                return False

            # Set new password
            user.password = new_password
            self.db.add(user)
            self.db.commit()
            return True

    def request_password_reset(self, email: str) -> Optional[str]:
        """
        Request password reset for user.

        Args:
            email: User email

        Returns:
            Reset token if user found, None otherwise
        """
        with logfire.span("Request password reset"):
            user = self.get_by_email(email)
            if not user:
                return None

            # Generate reset token
            token = user.generate_password_reset_token()
            self.db.add(user)
            self.db.commit()
            return token

    def reset_password(self, email: str, token: str, new_password: str) -> bool:
        """
        Reset user password with token.

        Args:
            email: User email
            token: Reset token
            new_password: New password

        Returns:
            True if password reset successful, False otherwise
        """
        with logfire.span("Reset user password"):
            user = self.get_by_email(email)
            if not user:
                return False

            # Reset password with token
            result = user.reset_password(token, new_password)
            if result:
                self.db.add(user)
                self.db.commit()
            return result

    def verify_email(self, email: str, token: str) -> bool:
        """
        Verify user email with token.

        Args:
            email: User email
            token: Verification token

        Returns:
            True if email verified successfully, False otherwise
        """
        with logfire.span("Verify user email"):
            user = self.get_by_email(email)
            if not user:
                return False

            # Verify email with token
            result = user.verify_email(token)
            if result:
                self.db.add(user)
                self.db.commit()
            return result

    def assign_role(self, user_id: int, role_name: str) -> bool:
        """
        Assign role to user.

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if role assigned successfully, False otherwise
        """
        with logfire.span("Assign role to user"):
            user = self.get(user_id)
            if not user:
                return False

            # Find role
            role = self.db.query(Role).filter(Role.name == role_name).first()
            if not role:
                return False

            # Check if user already has this role
            if role in user.roles:
                return True

            # Assign role
            user.roles.append(role)
            self.db.add(user)
            self.db.commit()
            return True

    def remove_role(self, user_id: int, role_name: str) -> bool:
        """
        Remove role from user.

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if role removed successfully, False otherwise
        """
        with logfire.span("Remove role from user"):
            user = self.get(user_id)
            if not user:
                return False

            # Find role
            role = self.db.query(Role).filter(Role.name == role_name).first()
            if not role:
                return False

            # Check if user has this role
            if role not in user.roles:
                return True

            # Remove role
            user.roles.remove(role)
            self.db.add(user)
            self.db.commit()
            return True

    def unlock_account(self, user_id: int) -> bool:
        """
        Unlock user account.

        Args:
            user_id: User ID

        Returns:
            True if account unlocked successfully, False otherwise
        """
        with logfire.span("Unlock user account"):
            user = self.get(user_id)
            if not user:
                return False

            # Unlock account
            user.account_locked = False
            user.account_locked_until = None
            user.failed_login_attempts = 0
            self.db.add(user)
            self.db.commit()
            return True

    @cached(ttl=60)  # Cache for 1 minute
    def get_user_roles(self, user_id: int) -> List[str]:
        """
        Get user roles with caching.

        Args:
            user_id: User ID

        Returns:
            List of role names
        """
        with logfire.span("Get user roles"):
            user = self.get(user_id)
            if not user:
                return []

            return [role.name for role in user.roles]

    @cached(ttl=60)  # Cache for 1 minute
    def has_permission(self, user_id: int, permission: str) -> bool:
        """
        Check if user has permission with caching.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        with logfire.span("Check user permission"):
            user = self.get(user_id)
            if not user:
                return False

            return user.has_permission(permission)


class AsyncUserRepository(AsyncBaseRepository[User]):
    """
    Async repository for User model operations.

    Extends the async base repository with user-specific functionality.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Async database session
        """
        super().__init__(User, db)

    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username to search for

        Returns:
            User or None if not found
        """
        with logfire.span("Get user by username async"):
            stmt = select(User).where(User.username == username)
            result = await self.db.execute(stmt)
            return result.scalars().first()

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: Email to search for

        Returns:
            User or None if not found
        """
        with logfire.span("Get user by email async"):
            stmt = select(User).where(User.email == email)
            result = await self.db.execute(stmt)
            return result.scalars().first()

    async def authenticate(self, username_or_email: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/email and password.

        Args:
            username_or_email: Username or email
            password: Plain text password

        Returns:
            User if authentication successful, None otherwise
        """
        with logfire.span("Authenticate user async"):
            # Find user by username or email
            stmt = select(User).where(
                or_(
                    User.username == username_or_email,
                    User.email == username_or_email
                )
            )
            result = await self.db.execute(stmt)
            user = result.scalars().first()

            if not user:
                return None

            # Check if account is locked
            if user.is_account_locked():
                logfire.warning(
                    "Authentication attempt on locked account",
                    username=user.username,
                    account_locked_until=user.account_locked_until
                )
                return None

            # Verify password
            if user.verify_password(password):
                # Record successful login
                user.record_login(success=True)
                self.db.add(user)
                await self.db.commit()
                return user
            else:
                # Record failed login
                user.record_login(success=False)
                self.db.add(user)
                await self.db.commit()
                return None

    async def create_user(self, data: Dict[str, Any], password: str) -> User:
        """
        Create a new user with password.

        Args:
            data: User data
            password: Plain text password

        Returns:
            Created user
        """
        with logfire.span("Create user async"):
            try:
                user = User(**data)
                user.password = password  # This will hash the password

                # Generate verification token if not verified
                if not user.is_verified:
                    user.generate_verification_token()

                self.db.add(user)
                await self.db.commit()
                await self.db.refresh(user)
                return user
            except SQLAlchemyError as e:
                await self.db.rollback()
                logger.error(f"Error creating user: {e}")
                logfire.error(
                    "Database error creating user async",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

    @async_cached(ttl=60)  # Cache for 1 minute
    async def get_user_roles(self, user_id: int) -> List[str]:
        """
        Get user roles with caching.

        Args:
            user_id: User ID

        Returns:
            List of role names
        """
        with logfire.span("Get user roles async"):
            user = await self.get(user_id)
            if not user:
                return []

            return [role.name for role in user.roles]

    @async_cached(ttl=60)  # Cache for 1 minute
    async def has_permission(self, user_id: int, permission: str) -> bool:
        """
        Check if user has permission with caching.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        with logfire.span("Check user permission async"):
            user = await self.get(user_id)
            if not user:
                return False

            return user.has_permission(permission)
