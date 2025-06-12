"""
Security Integration for Agentical API

This module provides comprehensive security integration for all API endpoints,
including authentication middleware, authorization dependencies, and security
policy enforcement for the Agentical framework.

Features:
- Authentication middleware integration
- Authorization dependencies for endpoints
- Security policy enforcement
- Rate limiting integration
- Input validation middleware
- Security audit logging
- API key authentication support
- RBAC enforcement utilities
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
import logging

import logfire
from fastapi import Request, Response, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from sqlalchemy.orm import Session

from agentical.core.security import (
    SecurityContext, Permission, SystemRole, get_current_user_from_auth,
    get_security_context, RequirePermissions, RequireRoles, RequireAdmin
)
from agentical.core.security_config import (
    get_security_config, get_security_validator, get_security_auditor,
    get_security_headers_manager, get_api_key_manager, SecurityPolicyConfig
)
from agentical.core.encryption import get_encryption_manager
from agentical.db.models.user import User, Role
from agentical.db.session import get_db
from agentical.tools.security.auth_manager import AuthManager, create_auth_manager
from agentical.core.exceptions import SecurityError, ValidationError
from agentical.core.structured_logging import StructuredLogger

# Initialize logger and security
logger = StructuredLogger("security_integration")
security = HTTPBearer(auto_error=False)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware for API protection."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.config = get_security_config()
        self.validator = get_security_validator()
        self.auditor = get_security_auditor()
        self.headers_manager = get_security_headers_manager()
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security layers."""

        start_time = datetime.utcnow()

        with logfire.span("Security middleware", method=request.method, url=str(request.url)):
            try:
                # Skip security for health checks and public endpoints
                if self._is_exempt_endpoint(request):
                    response = await call_next(request)
                    return self._add_security_headers(request, response)

                # Validate request size
                content_length = request.headers.get("content-length")
                if content_length and not self.validator.validate_request_size(int(content_length)):
                    self.auditor.log_input_validation_failure(
                        "request_size",
                        [f"Request size exceeds limit: {content_length}"],
                        request
                    )
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Request size exceeds maximum allowed"
                    )

                # Rate limiting
                await self._check_rate_limits(request)

                # Validate headers
                header_errors = self.validator.validate_headers(dict(request.headers))
                if header_errors:
                    self.auditor.log_input_validation_failure(
                        "headers", header_errors, request
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid request headers"
                    )

                # Validate query parameters
                query_params = dict(request.query_params)
                query_errors = self.validator.validate_query_params(query_params)
                if query_errors:
                    self.auditor.log_input_validation_failure(
                        "query_params", query_errors, request
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid query parameters"
                    )

                # Validate JSON payload for POST/PUT requests
                if request.method in ["POST", "PUT", "PATCH"]:
                    await self._validate_json_payload(request)

                # Process request
                response = await call_next(request)

                # Add security headers
                response = self._add_security_headers(request, response)

                # Log successful request
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(
                    "Request processed",
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration=duration
                )

                return response

            except HTTPException:
                raise
            except Exception as e:
                logger.error("Security middleware error", error=str(e))
                self.auditor.log_security_event(
                    "middleware_error",
                    "high",
                    {"error": str(e)},
                    request
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal security error"
                )

    def _is_exempt_endpoint(self, request: Request) -> bool:
        """Check if endpoint is exempt from security checks."""
        exempt_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]

        path = request.url.path
        return any(path.startswith(exempt) for exempt in exempt_paths)

    async def _check_rate_limits(self, request: Request):
        """Check rate limits for request."""
        client_ip = self._get_client_ip(request)
        endpoint = f"{request.method}:{request.url.path}"

        # Check global rate limit
        await self._check_rate_limit(
            f"global:{client_ip}",
            self.config.default_rate_limit_per_minute,
            "global",
            request
        )

        # Check endpoint-specific rate limit
        if request.url.path.startswith("/api/v1/auth/"):
            await self._check_rate_limit(
                f"auth:{client_ip}",
                self.config.auth_rate_limit_per_minute,
                "auth",
                request
            )
        else:
            await self._check_rate_limit(
                f"api:{client_ip}",
                self.config.api_rate_limit_per_minute,
                "api",
                request
            )

    async def _check_rate_limit(
        self,
        key: str,
        limit: int,
        limit_type: str,
        request: Request
    ):
        """Check individual rate limit."""
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)

        if key not in self.rate_limiters:
            self.rate_limiters[key] = {
                "count": 0,
                "window_start": window_start,
                "last_request": now
            }

        limiter = self.rate_limiters[key]

        # Reset window if needed
        if now >= limiter["window_start"] + timedelta(minutes=1):
            limiter["count"] = 0
            limiter["window_start"] = window_start

        limiter["count"] += 1
        limiter["last_request"] = now

        if limiter["count"] > limit:
            self.auditor.log_rate_limit_violation(
                limit_type, limiter["count"], limit, request
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {limit_type}",
                headers={"Retry-After": "60"}
            )

    async def _validate_json_payload(self, request: Request):
        """Validate JSON payload."""
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if body:
                    payload = json.loads(body)
                    json_errors = self.validator.validate_json_payload(payload)
                    if json_errors:
                        self.auditor.log_input_validation_failure(
                            "json_payload", json_errors, request
                        )
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON payload"
                        )
        except json.JSONDecodeError:
            self.auditor.log_input_validation_failure(
                "json_payload", ["Invalid JSON format"], request
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )

    def _add_security_headers(self, request: Request, response: Response) -> Response:
        """Add security headers to response."""
        security_headers = self.headers_manager.get_security_headers(request)

        for name, value in security_headers.items():
            response.headers[name] = value

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

# Enhanced Authentication Dependencies

async def get_current_user_secure(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends(create_auth_manager),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user with comprehensive security checks."""

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with logfire.span("Secure authentication", token_prefix=credentials.credentials[:16]):
        try:
            # Try JWT authentication first
            auth_result = await auth_manager.authenticate(
                token=credentials.credentials,
                auth_type="bearer"
            )

            if not auth_result.success:
                # Try API key authentication
                api_key_manager = get_api_key_manager()
                api_key_data = api_key_manager.validate_api_key(credentials.credentials)

                if api_key_data:
                    user_id = api_key_data["user_id"]
                else:
                    auditor = get_security_auditor()
                    auditor.log_authentication_attempt(
                        "unknown", False, "Invalid token"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            else:
                user_id = auth_result.user_id

            # Get user from database
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check if account is active
            if user.account_locked:
                if user.account_locked_until and user.account_locked_until > datetime.utcnow():
                    raise HTTPException(
                        status_code=status.HTTP_423_LOCKED,
                        detail="Account is temporarily locked"
                    )
                else:
                    # Unlock account if lock period expired
                    user.account_locked = False
                    user.account_locked_until = None
                    db.commit()

            # Update last activity
            user.last_login = datetime.utcnow()
            db.commit()

            logger.info("User authenticated", user_id=user.id, username=user.username)
            return user

        except HTTPException:
            raise
        except SecurityError as e:
            auditor = get_security_auditor()
            auditor.log_authentication_attempt(
                "unknown", False, str(e)
            )
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

async def get_security_context_secure(
    current_user: User = Depends(get_current_user_secure)
) -> SecurityContext:
    """Get security context with audit logging."""

    with logfire.span("Security context", user_id=current_user.id):
        context = SecurityContext(current_user)

        logger.info(
            "Security context created",
            user_id=current_user.id,
            roles=context.roles,
            permissions_count=len(context.permissions)
        )

        return context

async def get_optional_user_secure(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends(create_auth_manager),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise, with security logging."""

    if not credentials:
        return None

    try:
        return await get_current_user_secure(credentials, auth_manager, db)
    except HTTPException:
        return None

# Enhanced Authorization Dependencies

def RequirePermissionsSecure(*permissions: Permission):
    """Enhanced permission dependency with audit logging."""

    def dependency(security_context: SecurityContext = Depends(get_security_context_secure)):
        missing_permissions = [
            perm for perm in permissions
            if not security_context.has_permission(perm)
        ]

        if missing_permissions:
            auditor = get_security_auditor()
            auditor.log_security_event(
                "authorization_denied",
                "medium",
                {
                    "user_id": security_context.user.id,
                    "required_permissions": [p.value for p in permissions],
                    "missing_permissions": [p.value for p in missing_permissions],
                    "user_permissions": [p.value for p in security_context.permissions]
                }
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[p.value for p in missing_permissions]}"
            )

        logger.info(
            "Authorization granted",
            user_id=security_context.user.id,
            permissions=[p.value for p in permissions]
        )

        return security_context

    return dependency

def RequireRolesSecure(*roles: Union[str, SystemRole]):
    """Enhanced role dependency with audit logging."""

    def dependency(security_context: SecurityContext = Depends(get_security_context_secure)):
        role_names = [
            role.value if isinstance(role, SystemRole) else role
            for role in roles
        ]

        if not any(security_context.has_role(role) for role in roles):
            auditor = get_security_auditor()
            auditor.log_security_event(
                "authorization_denied",
                "medium",
                {
                    "user_id": security_context.user.id,
                    "required_roles": role_names,
                    "user_roles": security_context.roles
                }
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient roles. Required: {role_names}"
            )

        logger.info(
            "Authorization granted",
            user_id=security_context.user.id,
            roles=role_names
        )

        return security_context

    return dependency

def RequireAdminSecure():
    """Enhanced admin dependency with audit logging."""
    return RequireRolesSecure(SystemRole.ADMIN, SystemRole.SUPER_ADMIN)

def RequireSuperAdminSecure():
    """Enhanced super admin dependency with audit logging."""
    return RequireRolesSecure(SystemRole.SUPER_ADMIN)

# Resource Ownership Validation

def RequireResourceOwnership(resource_user_id_field: str = "user_id"):
    """Dependency to require resource ownership or admin privileges."""

    def dependency(
        request: Request,
        security_context: SecurityContext = Depends(get_security_context_secure)
    ):
        # Extract resource user ID from path parameters
        path_params = request.path_params
        resource_user_id = path_params.get(resource_user_id_field)

        if resource_user_id is None:
            # Check if it's in query parameters
            query_params = dict(request.query_params)
            resource_user_id = query_params.get(resource_user_id_field)

        if resource_user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing {resource_user_id_field} parameter"
            )

        try:
            resource_user_id = int(resource_user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {resource_user_id_field} format"
            )

        # Check ownership or admin privileges
        if (security_context.user.id != resource_user_id and
            not security_context.is_admin()):

            auditor = get_security_auditor()
            auditor.log_security_event(
                "resource_access_denied",
                "medium",
                {
                    "user_id": security_context.user.id,
                    "resource_user_id": resource_user_id,
                    "resource_field": resource_user_id_field
                }
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient privileges for this resource"
            )

        return security_context

    return dependency

# Data Encryption Dependencies

async def get_encryption_context():
    """Get encryption context for data protection."""
    return get_encryption_manager()

def RequireDataEncryption(classification_level: str = "confidential"):
    """Dependency to ensure data encryption for sensitive operations."""

    def dependency(
        encryption_manager = Depends(get_encryption_context),
        security_context: SecurityContext = Depends(get_security_context_secure)
    ):
        logger.info(
            "Data encryption context",
            user_id=security_context.user.id,
            classification=classification_level
        )

        return {
            "encryption_manager": encryption_manager,
            "security_context": security_context,
            "classification_level": classification_level
        }

    return dependency

# Security Event Logging Utilities

class SecurityEventLogger:
    """Security event logging utilities."""

    def __init__(self):
        self.auditor = get_security_auditor()

    def log_endpoint_access(
        self,
        endpoint: str,
        user_id: int,
        method: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log endpoint access attempt."""
        self.auditor.log_security_event(
            "endpoint_access",
            "low" if success else "medium",
            {
                "endpoint": endpoint,
                "user_id": user_id,
                "method": method,
                "success": success,
                "details": details or {}
            }
        )

    def log_data_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: int,
        action: str,
        success: bool = True
    ):
        """Log data access attempt."""
        self.auditor.log_security_event(
            "data_access",
            "low" if success else "medium",
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "user_id": user_id,
                "action": action,
                "success": success
            }
        )

    def log_privilege_escalation_attempt(
        self,
        user_id: int,
        attempted_action: str,
        required_privileges: List[str]
    ):
        """Log privilege escalation attempt."""
        self.auditor.log_security_event(
            "privilege_escalation_attempt",
            "high",
            {
                "user_id": user_id,
                "attempted_action": attempted_action,
                "required_privileges": required_privileges
            }
        )

# Global security event logger instance
security_event_logger = SecurityEventLogger()

# Utility functions for endpoint protection

def secure_endpoint(
    permissions: Optional[List[Permission]] = None,
    roles: Optional[List[Union[str, SystemRole]]] = None,
    require_admin: bool = False,
    require_super_admin: bool = False,
    log_access: bool = True
):
    """Decorator for endpoint security configuration."""

    def decorator(func: Callable) -> Callable:
        # Add security dependencies based on requirements
        if require_super_admin:
            func.__annotations__["security_context"] = Depends(RequireSuperAdminSecure())
        elif require_admin:
            func.__annotations__["security_context"] = Depends(RequireAdminSecure())
        elif roles:
            func.__annotations__["security_context"] = Depends(RequireRolesSecure(*roles))
        elif permissions:
            func.__annotations__["security_context"] = Depends(RequirePermissionsSecure(*permissions))
        else:
            func.__annotations__["security_context"] = Depends(get_security_context_secure)

        # Add logging if requested
        if log_access:
            original_func = func

            async def logged_func(*args, **kwargs):
                security_context = kwargs.get("security_context")
                if security_context:
                    security_event_logger.log_endpoint_access(
                        func.__name__,
                        security_context.user.id,
                        "API_CALL",
                        True
                    )
                return await original_func(*args, **kwargs)

            return logged_func

        return func

    return decorator

# Initialize security system
def initialize_security_system(db: Session):
    """Initialize security system with default roles and configurations."""

    with logfire.span("Initialize security system"):
        try:
            from agentical.core.security import initialize_default_roles

            # Initialize default roles
            initialize_default_roles(db)

            # Create default admin user if not exists
            admin_user = db.query(User).filter(User.username == "admin").first()
            if not admin_user:
                admin_role = db.query(Role).filter(Role.name == "super_admin").first()
                if admin_role:
                    admin_user = User(
                        username="admin",
                        email="admin@agentical.dev",
                        hashed_password="$2b$12$placeholder_hash_change_in_production",
                        is_verified=True,
                        display_name="System Administrator"
                    )
                    admin_user.password = "ChangeMe123!"  # Will be hashed automatically
                    admin_user.roles.append(admin_role)
                    db.add(admin_user)
                    db.commit()

                    logger.info("Default admin user created")

            logger.info("Security system initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize security system", error=str(e))
            raise SecurityError(f"Security initialization failed: {str(e)}")
