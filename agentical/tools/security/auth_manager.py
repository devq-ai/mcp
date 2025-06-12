"""
Authentication Manager for Agentical

This module provides comprehensive authentication and authorization management
supporting multiple authentication providers, token management, session handling,
and role-based access control with integration to the Agentical framework.

Features:
- Multi-provider authentication (OAuth2, JWT, API keys, Basic, Bearer)
- Role-based access control (RBAC) with granular permissions
- Session management with secure token handling
- Password policies and security enforcement
- Multi-factor authentication (MFA) support
- Rate limiting and brute force protection
- Audit logging for security compliance
- Integration with external identity providers
- Token refresh and rotation capabilities
- Zero-trust security architecture support
"""

import asyncio
import hashlib
import hmac
import jwt
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set
import logging
import json
import re

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    SecurityError
)
from ...core.logging import log_operation


class AuthProvider(Enum):
    """Supported authentication providers."""
    OAUTH2 = "oauth2"
    JWT = "jwt"
    API_KEY = "api_key"
    BASIC = "basic"
    BEARER = "bearer"
    SAML = "saml"
    LDAP = "ldap"


class TokenType(Enum):
    """Types of authentication tokens."""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"


class UserRole(Enum):
    """User roles for role-based access control."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"
    SERVICE_ACCOUNT = "service_account"


class Permission(Enum):
    """System permissions for granular access control."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_SECRETS = "manage_secrets"
    DEPLOY = "deploy"
    MONITOR = "monitor"


class AuthSession:
    """Authentication session with metadata."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        username: str,
        roles: List[UserRole],
        permissions: Set[Permission],
        provider: AuthProvider,
        expires_at: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.username = username
        self.roles = roles
        self.permissions = permissions
        self.provider = provider
        self.expires_at = expires_at
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "roles": [role.value for role in self.roles],
            "permissions": [perm.value for perm in self.permissions],
            "provider": self.provider.value,
            "expires_at": self.expires_at.isoformat(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active
        }

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at

    def has_permission(self, permission: Permission) -> bool:
        """Check if session has specific permission."""
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def has_role(self, role: UserRole) -> bool:
        """Check if session has specific role."""
        return role in self.roles

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class AuthenticationResult:
    """Result of authentication operation."""

    def __init__(
        self,
        auth_id: str,
        success: bool,
        session: Optional[AuthSession] = None,
        tokens: Optional[Dict[str, str]] = None,
        error_message: Optional[str] = None,
        requires_mfa: bool = False,
        lockout_remaining: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.auth_id = auth_id
        self.success = success
        self.session = session
        self.tokens = tokens or {}
        self.error_message = error_message
        self.requires_mfa = requires_mfa
        self.lockout_remaining = lockout_remaining
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "auth_id": self.auth_id,
            "success": self.success,
            "session": self.session.to_dict() if self.session else None,
            "tokens": self.tokens,
            "error_message": self.error_message,
            "requires_mfa": self.requires_mfa,
            "lockout_remaining": self.lockout_remaining,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class AuthManager:
    """
    Comprehensive authentication manager supporting multiple providers
    with session management, RBAC, and security enforcement.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize authentication manager.

        Args:
            config: Configuration for authentication
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.default_provider = AuthProvider(self.config.get("default_provider", "jwt"))
        self.token_expiry_minutes = self.config.get("token_expiry_minutes", 60)
        self.refresh_token_expiry_days = self.config.get("refresh_token_expiry_days", 30)
        self.max_login_attempts = self.config.get("max_login_attempts", 5)
        self.lockout_duration_minutes = self.config.get("lockout_duration_minutes", 15)
        self.password_min_length = self.config.get("password_min_length", 8)
        self.require_special_chars = self.config.get("require_special_chars", True)
        self.session_timeout_minutes = self.config.get("session_timeout_minutes", 30)
        self.jwt_secret = self.config.get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")

        # Internal storage (in production, use proper database)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}

        # Role-permission mapping
        self.role_permissions = {
            UserRole.SUPER_ADMIN: {
                Permission.READ, Permission.WRITE, Permission.DELETE,
                Permission.EXECUTE, Permission.ADMIN, Permission.AUDIT,
                Permission.MANAGE_USERS, Permission.MANAGE_ROLES,
                Permission.MANAGE_SECRETS, Permission.DEPLOY, Permission.MONITOR
            },
            UserRole.ADMIN: {
                Permission.READ, Permission.WRITE, Permission.DELETE,
                Permission.EXECUTE, Permission.MANAGE_USERS,
                Permission.DEPLOY, Permission.MONITOR
            },
            UserRole.USER: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE
            },
            UserRole.VIEWER: {
                Permission.READ
            },
            UserRole.API_CLIENT: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE
            },
            UserRole.SERVICE_ACCOUNT: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE,
                Permission.MONITOR
            }
        }

        # Initialize cryptography if available
        self.fernet = None
        if CRYPTOGRAPHY_AVAILABLE:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption for sensitive data."""
        try:
            # Generate or use provided key
            key = self.config.get("encryption_key")
            if not key:
                key = Fernet.generate_key()
            elif isinstance(key, str):
                key = key.encode()

            self.fernet = Fernet(key)
        except Exception as e:
            self.logger.warning(f"Failed to initialize encryption: {e}")

    @log_operation("authentication")
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        provider: Optional[AuthProvider] = None,
        client_info: Optional[Dict[str, Any]] = None,
        mfa_token: Optional[str] = None
    ) -> AuthenticationResult:
        """
        Authenticate user with provided credentials.

        Args:
            credentials: Authentication credentials
            provider: Authentication provider to use
            client_info: Client information (IP, user agent, etc.)
            mfa_token: Multi-factor authentication token

        Returns:
            AuthenticationResult: Authentication result with session/tokens
        """
        auth_id = str(uuid.uuid4())
        provider = provider or self.default_provider

        try:
            # Extract username/identifier
            username = credentials.get("username") or credentials.get("email")
            if not username:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="Username or email required"
                )

            # Check account lockout
            if self._is_account_locked(username):
                lockout_remaining = self._get_lockout_remaining(username)
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="Account locked due to too many failed attempts",
                    lockout_remaining=lockout_remaining
                )

            # Authenticate based on provider
            if provider == AuthProvider.JWT:
                return await self._authenticate_jwt(auth_id, credentials, client_info, mfa_token)
            elif provider == AuthProvider.OAUTH2:
                return await self._authenticate_oauth2(auth_id, credentials, client_info)
            elif provider == AuthProvider.API_KEY:
                return await self._authenticate_api_key(auth_id, credentials, client_info)
            elif provider == AuthProvider.BASIC:
                return await self._authenticate_basic(auth_id, credentials, client_info, mfa_token)
            elif provider == AuthProvider.BEARER:
                return await self._authenticate_bearer(auth_id, credentials, client_info)
            else:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message=f"Unsupported authentication provider: {provider.value}"
                )

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="Authentication failed due to internal error"
            )

    async def _authenticate_jwt(
        self,
        auth_id: str,
        credentials: Dict[str, Any],
        client_info: Optional[Dict[str, Any]],
        mfa_token: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using JWT tokens."""

        # Check if this is token refresh
        if "refresh_token" in credentials:
            return await self._refresh_jwt_token(auth_id, credentials["refresh_token"])

        # Standard username/password authentication
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="Username and password required"
            )

        # Validate credentials
        user = self._get_user(username)
        if not user or not self._verify_password(password, user["password_hash"]):
            self._record_failed_attempt(username)
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="Invalid username or password"
            )

        # Check if MFA is required
        if user.get("mfa_enabled") and not mfa_token:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                requires_mfa=True,
                error_message="Multi-factor authentication required"
            )

        # Verify MFA if provided
        if user.get("mfa_enabled") and mfa_token:
            if not self._verify_mfa_token(user, mfa_token):
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="Invalid MFA token"
                )

        # Create session and tokens
        session = self._create_session(user, AuthProvider.JWT, client_info)
        tokens = self._generate_jwt_tokens(user, session)

        # Clear failed attempts
        self._clear_failed_attempts(username)

        return AuthenticationResult(
            auth_id=auth_id,
            success=True,
            session=session,
            tokens=tokens
        )

    async def _authenticate_api_key(
        self,
        auth_id: str,
        credentials: Dict[str, Any],
        client_info: Optional[Dict[str, Any]]
    ) -> AuthenticationResult:
        """Authenticate using API key."""

        api_key = credentials.get("api_key")
        if not api_key:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="API key required"
            )

        # Validate API key
        key_info = self.api_keys.get(api_key)
        if not key_info or not key_info.get("active", False):
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="Invalid or inactive API key"
            )

        # Check expiration
        if key_info.get("expires_at"):
            expires_at = datetime.fromisoformat(key_info["expires_at"])
            if datetime.now() > expires_at:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="API key expired"
                )

        # Get associated user
        user = self._get_user(key_info["user_id"])
        if not user:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="User associated with API key not found"
            )

        # Create session
        session = self._create_session(user, AuthProvider.API_KEY, client_info)

        return AuthenticationResult(
            auth_id=auth_id,
            success=True,
            session=session,
            tokens={"api_key": api_key}
        )

    async def _authenticate_basic(
        self,
        auth_id: str,
        credentials: Dict[str, Any],
        client_info: Optional[Dict[str, Any]],
        mfa_token: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using HTTP Basic Authentication."""

        # Basic auth is similar to JWT but with different token format
        return await self._authenticate_jwt(auth_id, credentials, client_info, mfa_token)

    async def _authenticate_bearer(
        self,
        auth_id: str,
        credentials: Dict[str, Any],
        client_info: Optional[Dict[str, Any]]
    ) -> AuthenticationResult:
        """Authenticate using Bearer token."""

        token = credentials.get("token")
        if not token:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message="Bearer token required"
            )

        # Verify JWT token
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("sub")
            session_id = payload.get("session_id")

            if not user_id or not session_id:
                raise jwt.InvalidTokenError("Invalid token payload")

            # Check if session exists and is valid
            session = self.sessions.get(session_id)
            if not session or session.is_expired() or not session.is_active:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="Invalid or expired session"
                )

            # Update session activity
            session.update_activity()

            return AuthenticationResult(
                auth_id=auth_id,
                success=True,
                session=session,
                tokens={"access_token": token}
            )

        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message=f"Invalid token: {e}"
            )

    async def _authenticate_oauth2(
        self,
        auth_id: str,
        credentials: Dict[str, Any],
        client_info: Optional[Dict[str, Any]]
    ) -> AuthenticationResult:
        """Authenticate using OAuth2 flow."""

        # OAuth2 implementation would require external provider integration
        # For now, return not implemented
        return AuthenticationResult(
            auth_id=auth_id,
            success=False,
            error_message="OAuth2 authentication not yet implemented"
        )

    async def _refresh_jwt_token(self, auth_id: str, refresh_token: str) -> AuthenticationResult:
        """Refresh JWT access token using refresh token."""

        try:
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            token_type = payload.get("type")
            user_id = payload.get("sub")
            session_id = payload.get("session_id")

            if token_type != "refresh" or not user_id or not session_id:
                raise jwt.InvalidTokenError("Invalid refresh token")

            # Check if session exists
            session = self.sessions.get(session_id)
            if not session or not session.is_active:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="Invalid session"
                )

            # Get user
            user = self._get_user_by_id(user_id)
            if not user:
                return AuthenticationResult(
                    auth_id=auth_id,
                    success=False,
                    error_message="User not found"
                )

            # Generate new tokens
            tokens = self._generate_jwt_tokens(user, session)

            return AuthenticationResult(
                auth_id=auth_id,
                success=True,
                session=session,
                tokens=tokens
            )

        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                auth_id=auth_id,
                success=False,
                error_message=f"Invalid refresh token: {e}"
            )

    def _create_session(
        self,
        user: Dict[str, Any],
        provider: AuthProvider,
        client_info: Optional[Dict[str, Any]]
    ) -> AuthSession:
        """Create authentication session."""

        session_id = str(uuid.uuid4())
        user_roles = [UserRole(role) for role in user.get("roles", ["user"])]

        # Collect permissions from all roles
        permissions = set()
        for role in user_roles:
            permissions.update(self.role_permissions.get(role, set()))

        # Add custom permissions
        custom_permissions = user.get("permissions", [])
        for perm in custom_permissions:
            try:
                permissions.add(Permission(perm))
            except ValueError:
                continue

        expires_at = datetime.now() + timedelta(minutes=self.session_timeout_minutes)

        session = AuthSession(
            session_id=session_id,
            user_id=user["id"],
            username=user["username"],
            roles=user_roles,
            permissions=permissions,
            provider=provider,
            expires_at=expires_at,
            metadata={
                "client_info": client_info,
                "provider": provider.value
            }
        )

        self.sessions[session_id] = session
        return session

    def _generate_jwt_tokens(self, user: Dict[str, Any], session: AuthSession) -> Dict[str, str]:
        """Generate JWT access and refresh tokens."""

        now = datetime.now()

        # Access token
        access_payload = {
            "sub": user["id"],
            "username": user["username"],
            "session_id": session.session_id,
            "roles": [role.value for role in session.roles],
            "permissions": [perm.value for perm in session.permissions],
            "type": "access",
            "iat": now.timestamp(),
            "exp": (now + timedelta(minutes=self.token_expiry_minutes)).timestamp()
        }

        # Refresh token
        refresh_payload = {
            "sub": user["id"],
            "session_id": session.session_id,
            "type": "refresh",
            "iat": now.timestamp(),
            "exp": (now + timedelta(days=self.refresh_token_expiry_days)).timestamp()
        }

        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": self.token_expiry_minutes * 60
        }

    def _get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username or email."""
        for user in self.users.values():
            if user["username"] == username or user.get("email") == username:
                return user
        return None

    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.users.get(user_id)

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        if BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        else:
            # Fallback to simple hash comparison (not recommended for production)
            return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def _verify_mfa_token(self, user: Dict[str, Any], token: str) -> bool:
        """Verify MFA token."""
        if not TOTP_AVAILABLE:
            return False

        mfa_secret = user.get("mfa_secret")
        if not mfa_secret:
            return False

        try:
            totp = pyotp.TOTP(mfa_secret)
            return totp.verify(token)
        except Exception:
            return False

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.locked_accounts:
            return False

        locked_until = self.locked_accounts[username]
        if datetime.now() > locked_until:
            del self.locked_accounts[username]
            return False

        return True

    def _get_lockout_remaining(self, username: str) -> int:
        """Get remaining lockout time in seconds."""
        if username not in self.locked_accounts:
            return 0

        locked_until = self.locked_accounts[username]
        remaining = (locked_until - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        now = datetime.now()

        if username not in self.login_attempts:
            self.login_attempts[username] = []

        # Clean old attempts (older than lockout duration)
        cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
        self.login_attempts[username] = [
            attempt for attempt in self.login_attempts[username]
            if attempt > cutoff
        ]

        self.login_attempts[username].append(now)

        # Check if lockout threshold reached
        if len(self.login_attempts[username]) >= self.max_login_attempts:
            lockout_until = now + timedelta(minutes=self.lockout_duration_minutes)
            self.locked_accounts[username] = lockout_until
            self.logger.warning(f"Account locked for user: {username}")

    def _clear_failed_attempts(self, username: str) -> None:
        """Clear failed login attempts after successful login."""
        if username in self.login_attempts:
            del self.login_attempts[username]
        if username in self.locked_accounts:
            del self.locked_accounts[username]

    async def create_user(
        self,
        username: str,
        password: str,
        email: str,
        roles: List[UserRole] = None,
        permissions: List[Permission] = None,
        enable_mfa: bool = False
    ) -> Dict[str, Any]:
        """Create new user account."""

        if not self._validate_password(password):
            raise ToolValidationError("Password does not meet security requirements")

        if self._get_user(username) or self._get_user(email):
            raise ToolValidationError("User already exists")

        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)

        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "roles": [role.value for role in (roles or [UserRole.USER])],
            "permissions": [perm.value for perm in (permissions or [])],
            "mfa_enabled": enable_mfa,
            "mfa_secret": secrets.token_urlsafe(16) if enable_mfa else None,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }

        self.users[user_id] = user
        return user

    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy."""
        if len(password) < self.password_min_length:
            return False

        if self.require_special_chars:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                return False

        # Additional checks
        if not re.search(r'[A-Z]', password):  # Uppercase
            return False
        if not re.search(r'[a-z]', password):  # Lowercase
            return False
        if not re.search(r'\d', password):     # Digit
            return False

        return True

    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback (not recommended for production)
            salt = secrets.token_hex(16)
            return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_days: Optional[int] = None,
        permissions: List[Permission] = None
    ) -> Dict[str, Any]:
        """Create API key for user."""

        user = self._get_user_by_id(user_id)
        if not user:
            raise ToolValidationError("User not found")

        api_key = secrets.token_urlsafe(32)
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()

        key_info = {
            "key": api_key,
            "user_id": user_id,
            "name": name,
            "permissions": [perm.value for perm in (permissions or [])],
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
            "active": True,
            "last_used": None
        }

        self.api_keys[api_key] = key_info
        return key_info

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke authentication session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False

    async def validate_session(self, session_id: str) -> Optional[AuthSession]:
        """Validate and return session if valid."""
        session = self.sessions.get(session_id)
        if not session or not session.is_active or session.is_expired():
            return None

        session.update_activity()
        return session

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on authentication manager."""
        health_status = {
            "status": "healthy",
            "active_sessions": len([s for s in self.sessions.values() if s.is_active]),
            "locked_accounts": len(self.locked_accounts),
            "total_users": len(self.users),
            "total_api_keys": len(self.api_keys),
            "configuration": {
                "default_provider": self.default_provider.value,
                "token_expiry_minutes": self.token_expiry_minutes,
                "session_timeout_minutes": self.session_timeout_minutes,
                "max_login_attempts": self.max_login_attempts
            },
            "dependencies": {
                "bcrypt": BCRYPT_AVAILABLE,
                "pyotp": TOTP_AVAILABLE,
                "cryptography": CRYPTOGRAPHY_AVAILABLE
            }
        }

        # Test basic functionality
        try:
            test_user_id = str(uuid.uuid4())
            test_user = {
                "id": test_user_id,
                "username": "health_check_user",
                "email": "health@test.com",
                "password_hash": self._hash_password("test_password"),
                "roles": ["user"],
                "permissions": [],
                "mfa_enabled": False,
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }

            # Test password hashing
            test_hash = self._hash_password("test_password")
            test_verify = self._verify_password("test_password", test_hash)

            health_status["password_hashing"] = test_verify

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["password_hashing"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating authentication manager
def create_auth_manager(config: Optional[Dict[str, Any]] = None) -> AuthManager:
    """
    Create an authentication manager with specified configuration.

    Args:
        config: Configuration for authentication

    Returns:
        AuthManager: Configured authentication manager instance
    """
    return AuthManager(config=config)
