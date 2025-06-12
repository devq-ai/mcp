"""
Authentication API Endpoints

This module provides comprehensive authentication and authorization endpoints
for the Agentical framework, including user registration, login, token management,
and profile operations with full integration to the existing security infrastructure.

Features:
- User registration and email verification
- JWT-based authentication with refresh tokens
- Password reset and account management
- Profile management endpoints
- Session management and logout
- Integration with AuthManager and User models
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import logfire
from fastapi import APIRouter, HTTPException, Depends, status, Body, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, validator
from sqlalchemy.orm import Session

from agentical.tools.security.auth_manager import AuthManager, create_auth_manager
from agentical.db.models.user import User, Role
from agentical.db.session import get_db
from agentical.core.exceptions import SecurityError, ValidationError, NotFoundError
from agentical.core.structured_logging import StructuredLogger

# Initialize router and logger
router = APIRouter(prefix="/auth", tags=["authentication"])
logger = StructuredLogger("auth_api")
security = HTTPBearer(auto_error=False)

# Request/Response Models
class UserRegisterRequest(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")

    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLoginRequest(BaseModel):
    """User login request model."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Remember login session")

class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., description="Refresh token")

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr = Field(..., description="Email address")

class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request model."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")

    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdateRequest(BaseModel):
    """User profile update request model."""
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    display_name: Optional[str] = Field(None, max_length=100)
    profile_image_url: Optional[str] = Field(None, max_length=255)

class AuthResponse(BaseModel):
    """Authentication response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    user: Dict[str, Any] = Field(..., description="User information")

class UserResponse(BaseModel):
    """User response model."""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    display_name: Optional[str] = Field(None, description="Display name")
    profile_image_url: Optional[str] = Field(None, description="Profile image URL")
    is_verified: bool = Field(..., description="Email verified status")
    roles: List[str] = Field(..., description="User roles")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")

class MessageResponse(BaseModel):
    """Generic message response model."""
    message: str = Field(..., description="Response message")
    success: bool = Field(True, description="Operation success status")

# Dependencies
async def get_auth_manager() -> AuthManager:
    """Get authentication manager instance."""
    return create_auth_manager()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""

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

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""

    if not credentials:
        return None

    try:
        return await get_current_user(credentials, auth_manager, db)
    except HTTPException:
        return None

# Authentication Endpoints

@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Register a new user account."""

    with logfire.span("User registration", username=request.username, email=request.email):
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == request.username) | (User.email == request.email)
            ).first()

            if existing_user:
                if existing_user.username == request.username:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Username already exists"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Email already registered"
                    )

            # Create new user using auth manager
            user_data = {
                "username": request.username,
                "email": request.email,
                "password": request.password,
                "first_name": request.first_name,
                "last_name": request.last_name,
                "display_name": f"{request.first_name} {request.last_name}".strip() if request.first_name or request.last_name else request.username
            }

            user_id = await auth_manager.create_user(user_data)

            # Get the created user
            user = db.query(User).filter(User.id == user_id).first()

            # Generate verification token
            verification_token = user.generate_verification_token()
            db.commit()

            logger.info("User registered successfully", user_id=user_id, username=request.username)

            return MessageResponse(
                message="User registered successfully. Please check your email for verification instructions.",
                success=True
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("User registration failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )

@router.post("/login", response_model=AuthResponse)
async def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Authenticate user and return JWT tokens."""

    with logfire.span("User login", username=request.username):
        try:
            # Authenticate user
            auth_result = await auth_manager.authenticate(
                username=request.username,
                password=request.password,
                auth_type="basic"
            )

            if not auth_result.success:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )

            # Get user details
            user = db.query(User).filter(User.id == auth_result.user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

            # Check if account is locked
            if user.account_locked:
                if user.account_locked_until and user.account_locked_until > datetime.utcnow():
                    raise HTTPException(
                        status_code=status.HTTP_423_LOCKED,
                        detail="Account is temporarily locked due to failed login attempts"
                    )
                else:
                    # Unlock account if lock period expired
                    user.account_locked = False
                    user.account_locked_until = None
                    user.failed_login_attempts = 0

            # Update login info
            user.last_login = datetime.utcnow()
            user.failed_login_attempts = 0
            db.commit()

            # Prepare user data for response
            user_data = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": user.display_name,
                "is_verified": user.is_verified,
                "roles": [role.name for role in user.roles]
            }

            logger.info("User logged in successfully", user_id=user.id, username=user.username)

            return AuthResponse(
                access_token=auth_result.access_token,
                refresh_token=auth_result.refresh_token,
                token_type="bearer",
                expires_in=auth_result.expires_in,
                user=user_data
            )

        except HTTPException:
            # Record failed attempt
            user = db.query(User).filter(
                (User.username == request.username) | (User.email == request.username)
            ).first()

            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.account_locked = True
                    user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
                db.commit()

            raise
        except Exception as e:
            logger.error("Login failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed"
            )

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    db: Session = Depends(get_db)
):
    """Refresh JWT access token using refresh token."""

    with logfire.span("Token refresh"):
        try:
            # Refresh tokens using auth manager
            auth_result = await auth_manager._refresh_jwt_token(request.refresh_token)

            if not auth_result.success:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )

            # Get user details
            user = db.query(User).filter(User.id == auth_result.user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

            user_data = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": user.display_name,
                "is_verified": user.is_verified,
                "roles": [role.name for role in user.roles]
            }

            return AuthResponse(
                access_token=auth_result.access_token,
                refresh_token=auth_result.refresh_token,
                token_type="bearer",
                expires_in=auth_result.expires_in,
                user=user_data
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed"
            )

@router.post("/logout", response_model=MessageResponse)
async def logout_user(
    current_user: User = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Logout user and invalidate tokens."""

    with logfire.span("User logout", user_id=current_user.id):
        try:
            # Revoke user session
            await auth_manager.revoke_session(current_user.id)

            logger.info("User logged out successfully", user_id=current_user.id)

            return MessageResponse(
                message="Logged out successfully",
                success=True
            )

        except Exception as e:
            logger.error("Logout failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile information."""

    with logfire.span("Get user profile", user_id=current_user.id):
        return UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            first_name=current_user.first_name,
            last_name=current_user.last_name,
            display_name=current_user.display_name,
            profile_image_url=current_user.profile_image_url,
            is_verified=current_user.is_verified,
            roles=[role.name for role in current_user.roles],
            created_at=current_user.created_at,
            last_login=current_user.last_login
        )

@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    request: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile information."""

    with logfire.span("Update user profile", user_id=current_user.id):
        try:
            # Update user fields
            update_data = request.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(current_user, field, value)

            current_user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(current_user)

            logger.info("User profile updated", user_id=current_user.id)

            return UserResponse(
                id=current_user.id,
                username=current_user.username,
                email=current_user.email,
                first_name=current_user.first_name,
                last_name=current_user.last_name,
                display_name=current_user.display_name,
                profile_image_url=current_user.profile_image_url,
                is_verified=current_user.is_verified,
                roles=[role.name for role in current_user.roles],
                created_at=current_user.created_at,
                last_login=current_user.last_login
            )

        except Exception as e:
            logger.error("Profile update failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Profile update failed"
            )

@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(
    token: str = Query(..., description="Email verification token"),
    db: Session = Depends(get_db)
):
    """Verify user email address."""

    with logfire.span("Email verification", token=token[:8] + "..."):
        try:
            # Find user with verification token
            user = db.query(User).filter(User.verification_token == token).first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid verification token"
                )

            # Check if token expired
            if user.verification_token_expires and user.verification_token_expires < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Verification token expired"
                )

            # Verify email
            user.is_verified = True
            user.verification_token = None
            user.verification_token_expires = None
            user.updated_at = datetime.utcnow()
            db.commit()

            logger.info("Email verified successfully", user_id=user.id)

            return MessageResponse(
                message="Email verified successfully",
                success=True
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Email verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Email verification failed"
            )

@router.post("/reset-password", response_model=MessageResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """Request password reset for user account."""

    with logfire.span("Password reset request", email=request.email):
        try:
            # Find user by email
            user = db.query(User).filter(User.email == request.email).first()

            if not user:
                # Don't reveal if email exists for security
                return MessageResponse(
                    message="If the email exists, a password reset link has been sent",
                    success=True
                )

            # Generate reset token
            import secrets
            reset_token = secrets.token_urlsafe(32)
            user.password_reset_token = reset_token
            user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
            db.commit()

            # TODO: Send password reset email
            logger.info("Password reset requested", user_id=user.id)

            return MessageResponse(
                message="If the email exists, a password reset link has been sent",
                success=True
            )

        except Exception as e:
            logger.error("Password reset request failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password reset request failed"
            )

@router.post("/reset-password/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    db: Session = Depends(get_db)
):
    """Confirm password reset with token and new password."""

    with logfire.span("Password reset confirmation", token=request.token[:8] + "..."):
        try:
            # Find user with reset token
            user = db.query(User).filter(User.password_reset_token == request.token).first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid reset token"
                )

            # Check if token expired
            if user.password_reset_expires and user.password_reset_expires < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Reset token expired"
                )

            # Update password
            user.password = request.new_password  # This will be hashed by the setter
            user.password_reset_token = None
            user.password_reset_expires = None
            user.failed_login_attempts = 0
            user.account_locked = False
            user.account_locked_until = None
            user.updated_at = datetime.utcnow()
            db.commit()

            logger.info("Password reset completed", user_id=user.id)

            return MessageResponse(
                message="Password reset successfully",
                success=True
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Password reset confirmation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password reset confirmation failed"
            )

@router.get("/health", response_model=MessageResponse)
async def auth_health_check(
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Check authentication service health."""

    with logfire.span("Auth health check"):
        try:
            health_status = await auth_manager.health_check()

            if health_status.get("status") == "healthy":
                return MessageResponse(
                    message="Authentication service is healthy",
                    success=True
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication service is unhealthy"
                )

        except Exception as e:
            logger.error("Auth health check failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service error"
            )
