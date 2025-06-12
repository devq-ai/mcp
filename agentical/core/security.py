"""
Security and Authorization System for Agentical

This module provides comprehensive authorization and role-based access control (RBAC)
functionality including decorators, dependencies, and permission management for
securing API endpoints and operations.

Features:
- Role-based access control (RBAC) with hierarchical permissions
- Permission decorators for endpoint protection
- Security dependencies for FastAPI integration
- Fine-grained permission checking
- Admin-only operations protection
- Integration with User and Role models
"""

import functools
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Union
import logging

import logfire
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from agentical.db.models.user import User, Role
from agentical.db.session import get_db
from agentical.tools.security.auth_manager import AuthManager, create_auth_manager
from agentical.core.exceptions import SecurityError, ValidationError
from agentical.core.structured_logging import StructuredLogger

# Initialize logger
logger = StructuredLogger("security")
security = HTTPBearer(auto_error=False)

class SystemRole(Enum):
    """System-defined roles with hierarchical permissions."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """System permissions for fine-grained access control."""
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_LIST = "user:list"

    # Role Management
    ROLE_CREATE = "role:create"
    ROLE_READ = "role:read"
    ROLE_UPDATE = "role:update"
    ROLE_DELETE = "role:delete"
    ROLE_ASSIGN = "role:assign"

    # Agent Management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"
    AGENT_MONITOR = "agent:monitor"

    # Playbook Management
    PLAYBOOK_CREATE = "playbook:create"
    PLAYBOOK_READ = "playbook:read"
    PLAYBOOK_UPDATE = "playbook:update"
    PLAYBOOK_DELETE = "playbook:delete"
    PLAYBOOK_EXECUTE = "playbook:execute"
    PLAYBOOK_MONITOR = "playbook:monitor"

    # Workflow Management
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"
    WORKFLOW_MONITOR = "workflow:monitor"

    # Analytics & Monitoring
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_HEALTH = "system:health"

    # Administration
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_LOGS = "admin:logs"
    ADMIN_CONFIG = "admin:config"

# Role-Permission Mapping
ROLE_PERMISSIONS: Dict[SystemRole, Set[Permission]] = {
    SystemRole.SUPER_ADMIN: set(Permission),  # All permissions

    SystemRole.ADMIN: {
        # User management
        Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
        Permission.USER_DELETE, Permission.USER_LIST,
        # Role management
        Permission.ROLE_CREATE, Permission.ROLE_READ, Permission.ROLE_UPDATE,
        Permission.ROLE_DELETE, Permission.ROLE_ASSIGN,
        # All operational permissions
        Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE,
        Permission.AGENT_DELETE, Permission.AGENT_EXECUTE, Permission.AGENT_MONITOR,
        Permission.PLAYBOOK_CREATE, Permission.PLAYBOOK_READ, Permission.PLAYBOOK_UPDATE,
        Permission.PLAYBOOK_DELETE, Permission.PLAYBOOK_EXECUTE, Permission.PLAYBOOK_MONITOR,
        Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ, Permission.WORKFLOW_UPDATE,
        Permission.WORKFLOW_DELETE, Permission.WORKFLOW_EXECUTE, Permission.WORKFLOW_MONITOR,
        # Analytics & monitoring
        Permission.ANALYTICS_READ, Permission.ANALYTICS_EXPORT,
        Permission.SYSTEM_MONITOR, Permission.SYSTEM_HEALTH,
        # Admin operations
        Permission.ADMIN_USERS, Permission.ADMIN_ROLES, Permission.ADMIN_SYSTEM,
        Permission.ADMIN_LOGS, Permission.ADMIN_CONFIG,
    },

    SystemRole.MANAGER: {
        # Limited user management
        Permission.USER_READ, Permission.USER_LIST,
        # All operational permissions
        Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE,
        Permission.AGENT_EXECUTE, Permission.AGENT_MONITOR,
        Permission.PLAYBOOK_CREATE, Permission.PLAYBOOK_READ, Permission.PLAYBOOK_UPDATE,
        Permission.PLAYBOOK_EXECUTE, Permission.PLAYBOOK_MONITOR,
        Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ, Permission.WORKFLOW_UPDATE,
        Permission.WORKFLOW_EXECUTE, Permission.WORKFLOW_MONITOR,
        # Analytics
        Permission.ANALYTICS_READ, Permission.ANALYTICS_EXPORT,
        Permission.SYSTEM_MONITOR, Permission.SYSTEM_HEALTH,
    },

    SystemRole.USER: {
        # Basic operations
        Permission.AGENT_READ, Permission.AGENT_EXECUTE, Permission.AGENT_MONITOR,
        Permission.PLAYBOOK_READ, Permission.PLAYBOOK_EXECUTE, Permission.PLAYBOOK_MONITOR,
        Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE, Permission.WORKFLOW_MONITOR,
        # Limited analytics
        Permission.ANALYTICS_READ, Permission.SYSTEM_HEALTH,
    },

    SystemRole.VIEWER: {
        # Read-only access
        Permission.AGENT_READ, Permission.AGENT_MONITOR,
        Permission.PLAYBOOK_READ, Permission.PLAYBOOK_MONITOR,
        Permission.WORKFLOW_READ, Permission.WORKFLOW_MONITOR,
        Permission.ANALYTICS_READ, Permission.SYSTEM_HEALTH,
    },

    SystemRole.GUEST: {
        # Minimal access
        Permission.SYSTEM_HEALTH,
    }
}

class SecurityContext:
    """Security context for tracking user permissions and access."""

    def __init__(self, user: User, session_id: Optional[str] = None):
        self.user = user
        self.session_id = session_id
        self.permissions = self._calculate_permissions()
        self.roles = [role.name for role in user.roles]

    def _calculate_permissions(self) -> Set[Permission]:
        """Calculate effective permissions from user roles."""
        permissions = set()

        for role in self.user.roles:
            try:
                system_role = SystemRole(role.name)
                permissions.update(ROLE_PERMISSIONS.get(system_role, set()))
            except ValueError:
                # Custom role - parse permissions from role.permissions
                if role.permissions:
                    import json
                    try:
                        role_perms = json.loads(role.permissions)
                        for perm_str in role_perms:
                            try:
                                permissions.add(Permission(perm_str))
                            except ValueError:
                                logger.warning(f"Unknown permission: {perm_str}")
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid permissions JSON for role {role.name}")

        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions."""
        return all(perm in self.permissions for perm in permissions)

    def has_role(self, role: Union[str, SystemRole]) -> bool:
        """Check if user has specific role."""
        role_name = role.value if isinstance(role, SystemRole) else role
        return role_name in self.roles

    def is_admin(self) -> bool:
        """Check if user is admin or super admin."""
        return self.has_role(SystemRole.ADMIN) or self.has_role(SystemRole.SUPER_ADMIN)

    def is_super_admin(self) -> bool:
        """Check if user is super admin."""
        return self.has_role(SystemRole.SUPER_ADMIN)

# Security Dependencies

async def get_current_user_from_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends(create_auth_manager),
    db = Depends(get_db)
) -> User:
    """Get current authenticated user from auth manager."""

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Authenticate using the auth manager
        auth_result = await auth_manager.authenticate(
            token=credentials.credentials,
            auth_type="bearer"
        )

        if not auth_result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user from database
        user = db.query(User).filter(User.id == auth_result.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update last activity
        user.last_login = datetime.utcnow()
        db.commit()

        return user

    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

async def get_security_context(
    current_user: User = Depends(get_current_user_from_auth)
) -> SecurityContext:
    """Get security context for current user."""
    return SecurityContext(current_user)

def require_permissions(*required_permissions: Permission):
    """Decorator to require specific permissions for endpoint access."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from kwargs
            security_context = None
            for key, value in kwargs.items():
                if isinstance(value, SecurityContext):
                    security_context = value
                    break

            if not security_context:
                # Try to get it from dependency injection
                try:
                    security_context = kwargs.get('security_context')
                    if not security_context:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authentication required"
                        )
                except:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )

            # Check permissions
            missing_permissions = [
                perm for perm in required_permissions
                if not security_context.has_permission(perm)
            ]

            if missing_permissions:
                logger.warning(
                    "Access denied due to insufficient permissions",
                    user_id=security_context.user.id,
                    required_permissions=[p.value for p in required_permissions],
                    missing_permissions=[p.value for p in missing_permissions]
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {[p.value for p in missing_permissions]}"
                )

            # Log successful access
            logger.info(
                "Authorized access",
                user_id=security_context.user.id,
                permissions=[p.value for p in required_permissions],
                endpoint=func.__name__
            )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def require_roles(*required_roles: Union[str, SystemRole]):
    """Decorator to require specific roles for endpoint access."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from kwargs
            security_context = None
            for key, value in kwargs.items():
                if isinstance(value, SecurityContext):
                    security_context = value
                    break

            if not security_context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check roles
            role_names = [
                role.value if isinstance(role, SystemRole) else role
                for role in required_roles
            ]

            if not any(security_context.has_role(role) for role in required_roles):
                logger.warning(
                    "Access denied due to insufficient roles",
                    user_id=security_context.user.id,
                    required_roles=role_names,
                    user_roles=security_context.roles
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient roles. Required: {role_names}"
                )

            # Log successful access
            logger.info(
                "Authorized access",
                user_id=security_context.user.id,
                roles=role_names,
                endpoint=func.__name__
            )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def require_admin():
    """Decorator to require admin privileges for endpoint access."""
    return require_roles(SystemRole.ADMIN, SystemRole.SUPER_ADMIN)

def require_super_admin():
    """Decorator to require super admin privileges for endpoint access."""
    return require_roles(SystemRole.SUPER_ADMIN)

# FastAPI Dependencies

def RequirePermissions(*permissions: Permission):
    """FastAPI dependency to require specific permissions."""

    def dependency(security_context: SecurityContext = Depends(get_security_context)):
        missing_permissions = [
            perm for perm in permissions
            if not security_context.has_permission(perm)
        ]

        if missing_permissions:
            logger.warning(
                "Access denied due to insufficient permissions",
                user_id=security_context.user.id,
                required_permissions=[p.value for p in permissions],
                missing_permissions=[p.value for p in missing_permissions]
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[p.value for p in missing_permissions]}"
            )

        return security_context

    return dependency

def RequireRoles(*roles: Union[str, SystemRole]):
    """FastAPI dependency to require specific roles."""

    def dependency(security_context: SecurityContext = Depends(get_security_context)):
        role_names = [
            role.value if isinstance(role, SystemRole) else role
            for role in roles
        ]

        if not any(security_context.has_role(role) for role in roles):
            logger.warning(
                "Access denied due to insufficient roles",
                user_id=security_context.user.id,
                required_roles=role_names,
                user_roles=security_context.roles
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient roles. Required: {role_names}"
            )

        return security_context

    return dependency

def RequireAdmin():
    """FastAPI dependency to require admin privileges."""
    return RequireRoles(SystemRole.ADMIN, SystemRole.SUPER_ADMIN)

def RequireSuperAdmin():
    """FastAPI dependency to require super admin privileges."""
    return RequireRoles(SystemRole.SUPER_ADMIN)

# Utility Functions

def check_resource_ownership(user: User, resource_user_id: Optional[int]) -> bool:
    """Check if user owns the resource or has admin privileges."""
    if not resource_user_id:
        return False

    # User owns the resource
    if user.id == resource_user_id:
        return True

    # User has admin privileges
    security_context = SecurityContext(user)
    return security_context.is_admin()

def get_user_permissions(user: User) -> List[str]:
    """Get list of permission strings for a user."""
    security_context = SecurityContext(user)
    return [perm.value for perm in security_context.permissions]

def get_role_permissions(role_name: str) -> List[str]:
    """Get list of permission strings for a role."""
    try:
        system_role = SystemRole(role_name)
        permissions = ROLE_PERMISSIONS.get(system_role, set())
        return [perm.value for perm in permissions]
    except ValueError:
        return []

def initialize_default_roles(db) -> None:
    """Initialize default system roles in database."""

    with logfire.span("Initialize default roles"):
        try:
            for system_role in SystemRole:
                existing_role = db.query(Role).filter(Role.name == system_role.value).first()

                if not existing_role:
                    # Create new role
                    permissions = ROLE_PERMISSIONS.get(system_role, set())
                    permissions_json = [perm.value for perm in permissions]

                    new_role = Role(
                        name=system_role.value,
                        description=f"System {system_role.value.replace('_', ' ').title()} role",
                        permissions=json.dumps(permissions_json)
                    )

                    db.add(new_role)
                    logger.info(f"Created default role: {system_role.value}")
                else:
                    # Update existing role permissions
                    permissions = ROLE_PERMISSIONS.get(system_role, set())
                    permissions_json = [perm.value for perm in permissions]
                    existing_role.permissions = json.dumps(permissions_json)
                    logger.info(f"Updated role permissions: {system_role.value}")

            db.commit()
            logger.info("Default roles initialized successfully")

        except Exception as e:
            db.rollback()
            logger.error("Failed to initialize default roles", error=str(e))
            raise
