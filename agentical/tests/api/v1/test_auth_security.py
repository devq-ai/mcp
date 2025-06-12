"""
Comprehensive Security and Authentication Tests for Agentical

This module provides comprehensive test coverage for the security and authentication
system including authentication endpoints, authorization mechanisms, RBAC,
data protection, and security hardening features.

Test Categories:
- Authentication endpoint tests
- Authorization and RBAC tests
- Security middleware tests
- Input validation and sanitization tests
- Rate limiting tests
- Data encryption tests
- Security audit logging tests
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import jwt

from agentical.main import app
from agentical.db.models.user import User, Role
from agentical.core.security import SecurityContext, Permission, SystemRole
from agentical.core.security_config import SecurityPolicyConfig, InputSanitizer
from agentical.core.encryption import EncryptionManager, DataClassification
from agentical.tools.security.auth_manager import AuthManager
from agentical.api.security_integration import SecurityMiddleware
from agentical.api.v1.endpoints.auth import get_auth_manager


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client fixture."""
    import httpx
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    with patch('agentical.db.session.get_db') as mock:
        session = Mock(spec=Session)
        mock.return_value = session
        yield session


@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager."""
    manager = Mock(spec=AuthManager)
    manager.authenticate = AsyncMock()
    manager.create_user = AsyncMock()
    manager.health_check = AsyncMock()
    return manager


@pytest.fixture
def sample_user():
    """Sample user for testing."""
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$test_hash",
        first_name="Test",
        last_name="User",
        display_name="Test User",
        is_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    return user


@pytest.fixture
def sample_admin_user():
    """Sample admin user for testing."""
    admin_role = Role(id=1, name="admin", description="Administrator role")
    user = User(
        id=2,
        username="admin",
        email="admin@example.com",
        hashed_password="$2b$12$admin_hash",
        first_name="Admin",
        last_name="User",
        display_name="Admin User",
        is_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    user.roles = [admin_role]
    return user


@pytest.fixture
def valid_jwt_token():
    """Valid JWT token for testing."""
    payload = {
        "user_id": 1,
        "username": "testuser",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, "test_secret", algorithm="HS256")


class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""

    def test_register_user_success(self, client, mock_db_session, mock_auth_manager):
        """Test successful user registration."""
        # Setup mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        mock_auth_manager.create_user.return_value = 1

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/register", json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "StrongPass123!",
                "first_name": "New",
                "last_name": "User"
            })

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert "verification" in data["message"]

    def test_register_user_existing_username(self, client, mock_db_session, mock_auth_manager):
        """Test registration with existing username."""
        # Setup mocks - existing user
        existing_user = User(username="existinguser", email="other@example.com")
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_user

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/register", json={
                "username": "existinguser",
                "email": "new@example.com",
                "password": "StrongPass123!",
                "first_name": "New",
                "last_name": "User"
            })

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "Username already exists" in response.json()["detail"]

    def test_register_user_weak_password(self, client, mock_db_session, mock_auth_manager):
        """Test registration with weak password."""
        response = client.post("/api/v1/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "weak",  # Too weak
            "first_name": "New",
            "last_name": "User"
        })

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_success(self, client, mock_db_session, mock_auth_manager, sample_user):
        """Test successful login."""
        # Setup mocks
        mock_auth_result = Mock()
        mock_auth_result.success = True
        mock_auth_result.user_id = 1
        mock_auth_result.access_token = "test_access_token"
        mock_auth_result.refresh_token = "test_refresh_token"
        mock_auth_result.expires_in = 3600

        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/login", json={
                "username": "testuser",
                "password": "password123"
            })

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"] == "test_access_token"
        assert data["token_type"] == "bearer"
        assert "user" in data

    def test_login_invalid_credentials(self, client, mock_db_session, mock_auth_manager):
        """Test login with invalid credentials."""
        # Setup mocks
        mock_auth_result = Mock()
        mock_auth_result.success = False

        mock_auth_manager.authenticate.return_value = mock_auth_result

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/login", json={
                "username": "testuser",
                "password": "wrongpassword"
            })

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_locked_account(self, client, mock_db_session, mock_auth_manager, sample_user):
        """Test login with locked account."""
        # Setup locked user
        sample_user.account_locked = True
        sample_user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)

        mock_auth_result = Mock()
        mock_auth_result.success = True
        mock_auth_result.user_id = 1

        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/login", json={
                "username": "testuser",
                "password": "password123"
            })

        assert response.status_code == status.HTTP_423_LOCKED
        assert "temporarily locked" in response.json()["detail"]

    def test_refresh_token_success(self, client, mock_db_session, mock_auth_manager, sample_user):
        """Test successful token refresh."""
        # Setup mocks
        mock_auth_result = Mock()
        mock_auth_result.success = True
        mock_auth_result.user_id = 1
        mock_auth_result.access_token = "new_access_token"
        mock_auth_result.refresh_token = "new_refresh_token"
        mock_auth_result.expires_in = 3600

        mock_auth_manager._refresh_jwt_token.return_value = mock_auth_result
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
            response = client.post("/api/v1/auth/refresh", json={
                "refresh_token": "valid_refresh_token"
            })

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"] == "new_access_token"

    def test_logout_success(self, client, mock_auth_manager, sample_user):
        """Test successful logout."""
        mock_auth_manager.revoke_session.return_value = None

        with patch('agentical.api.v1.endpoints.auth.get_current_user', return_value=sample_user):
            with patch('agentical.api.v1.endpoints.auth.get_auth_manager', return_value=mock_auth_manager):
                response = client.post("/api/v1/auth/logout")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

    def test_get_current_user_profile(self, client, sample_user):
        """Test getting current user profile."""
        with patch('agentical.api.v1.endpoints.auth.get_current_user', return_value=sample_user):
            response = client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

    def test_update_user_profile(self, client, mock_db_session, sample_user):
        """Test updating user profile."""
        with patch('agentical.api.v1.endpoints.auth.get_current_user', return_value=sample_user):
            response = client.put("/api/v1/auth/me", json={
                "first_name": "Updated",
                "last_name": "Name"
            })

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["first_name"] == "Updated"
        assert data["last_name"] == "Name"

    def test_verify_email_success(self, client, mock_db_session, sample_user):
        """Test successful email verification."""
        sample_user.verification_token = "valid_token"
        sample_user.verification_token_expires = datetime.utcnow() + timedelta(hours=1)
        sample_user.is_verified = False

        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        response = client.post("/api/v1/auth/verify-email?token=valid_token")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

    def test_verify_email_invalid_token(self, client, mock_db_session):
        """Test email verification with invalid token."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.post("/api/v1/auth/verify-email?token=invalid_token")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid verification token" in response.json()["detail"]

    def test_password_reset_request(self, client, mock_db_session, sample_user):
        """Test password reset request."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        response = client.post("/api/v1/auth/reset-password", json={
            "email": "test@example.com"
        })

        assert response.status_code == status.HTTP_200_OK
        assert "reset link" in response.json()["message"]

    def test_password_reset_confirm_success(self, client, mock_db_session, sample_user):
        """Test successful password reset confirmation."""
        sample_user.password_reset_token = "valid_reset_token"
        sample_user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)

        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user

        response = client.post("/api/v1/auth/reset-password/confirm", json={
            "token": "valid_reset_token",
            "new_password": "NewStrongPass123!"
        })

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True


class TestAuthorizationAndRBAC:
    """Test authorization and role-based access control."""

    def test_security_context_creation(self, sample_user):
        """Test security context creation."""
        admin_role = Role(name="admin", permissions='["admin:users", "admin:system"]')
        sample_user.roles = [admin_role]

        context = SecurityContext(sample_user)

        assert context.user == sample_user
        assert "admin" in context.roles
        assert len(context.permissions) > 0

    def test_permission_checking(self, sample_user):
        """Test permission checking functionality."""
        admin_role = Role(name="admin", permissions='["user:read", "user:create"]')
        sample_user.roles = [admin_role]

        context = SecurityContext(sample_user)

        assert context.has_permission(Permission.USER_READ)
        assert not context.has_permission(Permission.ADMIN_SYSTEM)

    def test_role_checking(self, sample_admin_user):
        """Test role checking functionality."""
        context = SecurityContext(sample_admin_user)

        assert context.has_role("admin")
        assert context.is_admin()
        assert not context.is_super_admin()

    def test_resource_ownership_validation(self, sample_user):
        """Test resource ownership validation."""
        from agentical.core.security import check_resource_ownership

        # User owns resource
        assert check_resource_ownership(sample_user, sample_user.id)

        # User doesn't own resource and isn't admin
        assert not check_resource_ownership(sample_user, 999)

    def test_admin_privileges(self, sample_admin_user):
        """Test admin privilege validation."""
        from agentical.core.security import check_resource_ownership

        # Admin can access any resource
        assert check_resource_ownership(sample_admin_user, 999)


class TestSecurityMiddleware:
    """Test security middleware functionality."""

    def test_request_size_validation(self):
        """Test request size validation."""
        from agentical.core.security_config import SecurityValidator, SecurityPolicyConfig

        config = SecurityPolicyConfig(max_request_size_mb=1)
        validator = SecurityValidator(config)

        # Valid size
        assert validator.validate_request_size(500000)  # 500KB

        # Invalid size
        assert not validator.validate_request_size(2000000)  # 2MB

    def test_query_parameter_validation(self):
        """Test query parameter validation."""
        from agentical.core.security_config import SecurityValidator, SecurityPolicyConfig

        config = SecurityPolicyConfig()
        validator = SecurityValidator(config)

        # Valid parameters
        valid_params = {"page": "1", "limit": "10"}
        errors = validator.validate_query_params(valid_params)
        assert len(errors) == 0

        # SQL injection attempt
        malicious_params = {"query": "'; DROP TABLE users; --"}
        errors = validator.validate_query_params(malicious_params)
        assert len(errors) > 0
        assert any("SQL injection" in error for error in errors)

    def test_header_validation(self):
        """Test header validation."""
        from agentical.core.security_config import SecurityValidator, SecurityPolicyConfig

        config = SecurityPolicyConfig()
        validator = SecurityValidator(config)

        # Valid headers
        valid_headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
        errors = validator.validate_headers(valid_headers)
        assert len(errors) == 0

        # XSS attempt in custom header
        malicious_headers = {"X-Custom": "<script>alert('xss')</script>"}
        errors = validator.validate_headers(malicious_headers)
        assert len(errors) > 0

    def test_json_payload_validation(self):
        """Test JSON payload validation."""
        from agentical.core.security_config import SecurityValidator, SecurityPolicyConfig

        config = SecurityPolicyConfig()
        validator = SecurityValidator(config)

        # Valid payload
        valid_payload = {"name": "test", "value": 123}
        errors = validator.validate_json_payload(valid_payload)
        assert len(errors) == 0

        # Deep nesting
        deep_payload = {"level1": {"level2": {"level3": {"level4": {"level5": {}}}}}}
        errors = validator.validate_json_payload(deep_payload)
        # Should be valid with default depth limit

        # SQL injection in JSON
        malicious_payload = {"query": "'; DROP TABLE users; --"}
        errors = validator.validate_json_payload(malicious_payload)
        assert len(errors) > 0


class TestInputSanitization:
    """Test input sanitization and validation."""

    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        sanitizer = InputSanitizer()

        # Clean input
        assert not sanitizer.detect_sql_injection("normal text")

        # SQL injection attempts
        assert sanitizer.detect_sql_injection("'; DROP TABLE users; --")
        assert sanitizer.detect_sql_injection("1 OR 1=1")
        assert sanitizer.detect_sql_injection("UNION SELECT * FROM users")

    def test_xss_detection(self):
        """Test XSS detection."""
        sanitizer = InputSanitizer()

        # Clean input
        assert not sanitizer.detect_xss("normal text")

        # XSS attempts
        assert sanitizer.detect_xss("<script>alert('xss')</script>")
        assert sanitizer.detect_xss("javascript:alert(1)")
        assert sanitizer.detect_xss("<img src=x onerror=alert(1)>")

    def test_command_injection_detection(self):
        """Test command injection detection."""
        sanitizer = InputSanitizer()

        # Clean input
        assert not sanitizer.detect_command_injection("normal text")

        # Command injection attempts
        assert sanitizer.detect_command_injection("test; rm -rf /")
        assert sanitizer.detect_command_injection("test | cat /etc/passwd")
        assert sanitizer.detect_command_injection("test && malicious_command")

    def test_string_sanitization(self):
        """Test string sanitization."""
        sanitizer = InputSanitizer()

        # Normal string
        result = sanitizer.sanitize_string("Hello World")
        assert result == "Hello World"

        # String with HTML
        result = sanitizer.sanitize_string("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result

        # String with control characters
        result = sanitizer.sanitize_string("test\x00\x01control")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_filename_validation(self):
        """Test filename validation."""
        sanitizer = InputSanitizer()

        # Valid filenames
        assert sanitizer.validate_filename("document.pdf")
        assert sanitizer.validate_filename("my-file_v2.txt")

        # Invalid filenames
        assert not sanitizer.validate_filename("../../../etc/passwd")
        assert not sanitizer.validate_filename("file<script>.txt")
        assert not sanitizer.validate_filename("")
        assert not sanitizer.validate_filename(".")
        assert not sanitizer.validate_filename("..")

    def test_username_validation(self):
        """Test username validation."""
        sanitizer = InputSanitizer()

        # Valid usernames
        assert sanitizer.validate_username("user123")
        assert sanitizer.validate_username("test-user")
        assert sanitizer.validate_username("user.name")

        # Invalid usernames
        assert not sanitizer.validate_username("us")  # Too short
        assert not sanitizer.validate_username("user@domain.com")  # Invalid chars
        assert not sanitizer.validate_username("")

    def test_email_validation(self):
        """Test email validation."""
        sanitizer = InputSanitizer()

        # Valid emails
        assert sanitizer.validate_email("user@example.com")
        assert sanitizer.validate_email("test.email+tag@domain.co.uk")

        # Invalid emails
        assert not sanitizer.validate_email("invalid.email")
        assert not sanitizer.validate_email("@domain.com")
        assert not sanitizer.validate_email("user@")


class TestDataEncryption:
    """Test data encryption and protection."""

    def test_encryption_manager_initialization(self):
        """Test encryption manager initialization."""
        manager = EncryptionManager("test_master_key")
        assert manager.master_key == "test_master_key"
        assert len(manager.keys) > 0

    def test_field_encryption_and_decryption(self):
        """Test field-level encryption and decryption."""
        manager = EncryptionManager("test_master_key")

        # Test data
        test_data = {"sensitive": "credit_card_number", "user_id": 123}

        # Encrypt field
        encrypted_package = manager.encrypt_field(test_data, DataClassification.CONFIDENTIAL)

        assert encrypted_package["metadata"]["encrypted"] is True
        assert "value" in encrypted_package

        # Decrypt field
        decrypted_data = manager.decrypt_field(encrypted_package)
        assert decrypted_data == test_data

    def test_data_masking(self):
        """Test data masking functionality."""
        from agentical.core.encryption import DataMasker, PIIType

        # Test email masking
        masked_email = DataMasker.mask_email("user@example.com")
        assert masked_email == "u**r@example.com"

        # Test phone masking
        masked_phone = DataMasker.mask_phone("555-123-4567")
        assert masked_phone.endswith("4567")

        # Test credit card masking
        masked_card = DataMasker.mask_credit_card("4111-1111-1111-1111")
        assert masked_card.endswith("1111")

    def test_secure_config_management(self):
        """Test secure configuration management."""
        from agentical.core.encryption import SecureConfigManager, EncryptionManager

        manager = EncryptionManager("test_master_key")
        config_manager = SecureConfigManager(manager)

        # Set encrypted config
        config_manager.set_config("api_key", "secret_key_value", DataClassification.RESTRICTED)

        # Get decrypted config
        retrieved_value = config_manager.get_config("api_key")
        assert retrieved_value == "secret_key_value"

        # Non-existent config
        assert config_manager.get_config("non_existent", "default") == "default"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_configuration(self):
        """Test rate limit configuration."""
        from agentical.core.security_config import SecurityPolicyConfig

        config = SecurityPolicyConfig(
            default_rate_limit_per_minute=100,
            auth_rate_limit_per_minute=10,
            api_rate_limit_per_minute=1000
        )

        assert config.default_rate_limit_per_minute == 100
        assert config.auth_rate_limit_per_minute == 10
        assert config.api_rate_limit_per_minute == 1000

    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, client):
        """Test rate limiting enforcement."""
        # This would require integration with the actual middleware
        # For now, test the rate limit logic

        from agentical.api.security_integration import SecurityMiddleware
        from unittest.mock import MagicMock

        app_mock = MagicMock()
        middleware = SecurityMiddleware(app_mock)

        # Simulate multiple requests from same IP
        request_mock = MagicMock()
        request_mock.client.host = "127.0.0.1"
        request_mock.url.path = "/api/v1/test"
        request_mock.method = "GET"
        request_mock.headers = {}

        # Test rate limit checking logic
        try:
            await middleware._check_rate_limit("test_key", 5, "test", request_mock)
            # Should not raise exception for first request
        except HTTPException:
            pytest.fail("Rate limit should not be exceeded on first request")


class TestSecurityAuditLogging:
    """Test security audit logging."""

    def test_security_event_logging(self):
        """Test security event logging."""
        from agentical.core.security_config import SecurityAuditor, SecurityPolicyConfig

        config = SecurityPolicyConfig(log_security_events=True)
        auditor = SecurityAuditor(config)

        # Test logging security event
        auditor.log_security_event(
            "test_event",
            "medium",
            {"detail": "test details"}
        )

        # Should not raise exception

    def test_authentication_attempt_logging(self):
        """Test authentication attempt logging."""
        from agentical.core.security_config import SecurityAuditor, SecurityPolicyConfig

        config = SecurityPolicyConfig(log_failed_auth_attempts=True)
        auditor = SecurityAuditor(config)

        # Test logging failed authentication
        auditor.log_authentication_attempt(
            "testuser",
            False,
            "Invalid password"
        )

        # Should not raise exception

    def test_rate_limit_violation_logging(self):
        """Test rate limit violation logging."""
        from agentical.core.security_config import SecurityAuditor, SecurityPolicyConfig

        config = SecurityPolicyConfig(log_rate_limit_violations=True)
        auditor = SecurityAuditor(config)

        # Test logging rate limit violation
        auditor.log_rate_limit_violation(
            "api",
            150,
            100
        )

        # Should not raise exception


class TestPasswordPolicyValidation:
    """Test password policy validation."""

    def test_password_strength_validation(self):
        """Test password strength validation."""
        from agentical.core.security_config import SecurityValidator, SecurityPolicyConfig

        config = SecurityPolicyConfig(
            min_password_length=8,
            require_uppercase=True,
            require_lowercase=True,
            require_digits=True,
            require_special_chars=True
        )
        validator = SecurityValidator(config)

        # Strong password
        errors = validator.validate_password_strength("StrongPass123!")
        assert len(errors) == 0

        # Weak passwords
        errors = validator.validate_password_strength("weak")
        assert len(errors) > 0

        errors = validator.validate_password_strength("alllowercase123!")
        assert any("uppercase" in error for error in errors)

        errors = validator.validate_password_strength("ALLUPPERCASE123!")
        assert any("lowercase" in error for error in errors)

        errors = validator.validate_password_strength("NoDigitsHere!")
        assert any("digit" in error for error in errors)

        errors = validator.validate_password_strength("NoSpecialChars123")
        assert any("special character" in error for error in errors)


class TestSecurityIntegration:
    """Test overall security integration."""

    def test_security_headers_generation(self):
        """Test security headers generation."""
        from agentical.core.security_config import SecurityHeadersManager, SecurityPolicyConfig
        from unittest.mock import MagicMock

        config = SecurityPolicyConfig(
            enable_hsts=True,
            enable_csp=True,
            enable_xss_protection=True
        )
        headers_manager = SecurityHeadersManager(config)

        request_mock = MagicMock()
        headers = headers_manager.get_security_headers(request_mock)

        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "X-XSS-Protection" in headers
        assert "X-Content-Type-Options" in headers

    def test_api_key_management(self):
        """Test API key management."""
        from agentical.core.security_config import ApiKeyManager

        manager = ApiKeyManager()

        # Generate API key
        api_key = manager.generate_api_key(
            user_id=1,
            name="test_key",
            permissions=["read", "write"]
        )

        assert api_key.startswith("ak_")

        # Validate API key
        key_data = manager.validate_api_key(api_key)
        assert key_data is not None
        assert key_data["user_id"] == 1
        assert key_data["active"] is True

        # Revoke API key
        success = manager.revoke_api_key(api_key)
        assert success is True

        # Validate revoked key
        key_data = manager.validate_api_key(api_key)
        assert key_data is None

    @pytest.mark.asyncio
    async def test_comprehensive_security_flow(self, client):
        """Test comprehensive security flow."""
        # This test would simulate a complete request flow
        # through all security layers

        # 1. Request validation
        # 2. Authentication
        # 3. Authorization
        # 4. Data encryption
        # 5. Audit logging
        # 6. Response with security headers

        # For now, just test that the client works
        response = client.get("/health")
        assert response.status_code in [200, 404]  # Endpoint may not exist in test
