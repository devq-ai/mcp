"""
Security Configuration and Hardening for Agentical

This module provides comprehensive security configuration and hardening utilities
for the Agentical framework, including secure defaults, input validation,
sanitization, and security policy enforcement.

Features:
- Comprehensive security configuration management
- Input validation and sanitization utilities
- Security policy enforcement
- Rate limiting and abuse prevention
- API security hardening
- Secure defaults and best practices
- Security audit logging
- Compliance and regulatory support
"""

import re
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable
from urllib.parse import urlparse
import logging

import logfire
from pydantic import BaseModel, Field, validator
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from agentical.core.exceptions import SecurityError, ValidationError
from agentical.core.structured_logging import StructuredLogger

# Initialize logger
logger = StructuredLogger("security_config")

class SecurityLevel(Enum):
    """Security enforcement levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

class SecurityPolicyConfig(BaseModel):
    """Security policy configuration."""

    # Authentication settings
    jwt_expiry_minutes: int = Field(default=60, ge=1, le=1440)
    refresh_token_expiry_days: int = Field(default=30, ge=1, le=90)
    max_login_attempts: int = Field(default=5, ge=1, le=20)
    account_lockout_minutes: int = Field(default=30, ge=1, le=1440)

    # Password policy
    min_password_length: int = Field(default=8, ge=8, le=128)
    require_uppercase: bool = Field(default=True)
    require_lowercase: bool = Field(default=True)
    require_digits: bool = Field(default=True)
    require_special_chars: bool = Field(default=True)
    password_history_count: int = Field(default=5, ge=0, le=24)

    # Session management
    session_timeout_minutes: int = Field(default=120, ge=5, le=1440)
    concurrent_sessions_limit: int = Field(default=5, ge=1, le=20)
    session_regeneration_interval: int = Field(default=30, ge=5, le=60)

    # Rate limiting
    default_rate_limit_per_minute: int = Field(default=100, ge=1, le=10000)
    auth_rate_limit_per_minute: int = Field(default=10, ge=1, le=100)
    api_rate_limit_per_minute: int = Field(default=1000, ge=1, le=100000)
    burst_allowance: int = Field(default=20, ge=1, le=100)
    rate_limit_strategy: RateLimitStrategy = Field(default=RateLimitStrategy.SLIDING_WINDOW)

    # Input validation
    max_request_size_mb: int = Field(default=10, ge=1, le=100)
    max_query_params: int = Field(default=50, ge=1, le=200)
    max_headers: int = Field(default=50, ge=1, le=200)
    max_json_depth: int = Field(default=10, ge=1, le=50)
    max_string_length: int = Field(default=1000, ge=1, le=100000)

    # Security headers
    enable_hsts: bool = Field(default=True)
    hsts_max_age_seconds: int = Field(default=31536000, ge=1)
    enable_csp: bool = Field(default=True)
    enable_xss_protection: bool = Field(default=True)
    enable_content_type_options: bool = Field(default=True)
    enable_frame_options: bool = Field(default=True)

    # CORS policy
    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allowed_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allowed_headers: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = Field(default=True)
    cors_max_age: int = Field(default=3600, ge=1)

    # Logging and monitoring
    log_security_events: bool = Field(default=True)
    log_failed_auth_attempts: bool = Field(default=True)
    log_rate_limit_violations: bool = Field(default=True)
    log_input_validation_failures: bool = Field(default=True)

    # Compliance settings
    data_retention_days: int = Field(default=90, ge=1, le=2555)  # 7 years max
    audit_log_retention_days: int = Field(default=2555, ge=30, le=2555)
    enable_pii_detection: bool = Field(default=True)
    enable_data_masking: bool = Field(default=True)

    class Config:
        use_enum_values = True

class InputSanitizer:
    """Input sanitization and validation utilities."""

    # Common injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('\s*(OR|AND)\s*')",
        r"(CONCAT\s*\()",
        r"(CHAR\s*\()",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"(;|\||&|`|\$\(|\${)",
        r"(\.\./|\.\.\\)",
        r"(\\x[0-9a-fA-F]{2})",
        r"(%[0-9a-fA-F]{2})",
    ]

    # Allowed patterns
    SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
    SAFE_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
    SAFE_EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    @classmethod
    def detect_sql_injection(cls, input_string: str) -> bool:
        """Detect potential SQL injection attempts."""
        input_lower = input_string.lower()

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return True
        return False

    @classmethod
    def detect_xss(cls, input_string: str) -> bool:
        """Detect potential XSS attempts."""
        input_lower = input_string.lower()

        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return True
        return False

    @classmethod
    def detect_command_injection(cls, input_string: str) -> bool:
        """Detect potential command injection attempts."""
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, input_string):
                return True
        return False

    @classmethod
    def sanitize_string(cls, input_string: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not input_string:
            return ""

        # Truncate to max length
        sanitized = input_string[:max_length]

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Remove control characters except newline and tab
        sanitized = ''.join(char for char in sanitized
                          if ord(char) >= 32 or char in '\n\t')

        # HTML encode dangerous characters
        sanitized = (sanitized
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))

        return sanitized

    @classmethod
    def validate_filename(cls, filename: str) -> bool:
        """Validate filename for safety."""
        if not filename or len(filename) > 255:
            return False

        if filename in ['.', '..']:
            return False

        return bool(cls.SAFE_FILENAME_PATTERN.match(filename))

    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username format."""
        if not username or len(username) < 3 or len(username) > 50:
            return False

        return bool(cls.SAFE_USERNAME_PATTERN.match(username))

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 254:
            return False

        return bool(cls.SAFE_EMAIL_PATTERN.match(email))

    @classmethod
    def validate_json_depth(cls, data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
        """Validate JSON structure depth."""
        if current_depth > max_depth:
            return False

        if isinstance(data, dict):
            for value in data.values():
                if not cls.validate_json_depth(value, max_depth, current_depth + 1):
                    return False
        elif isinstance(data, list):
            for item in data:
                if not cls.validate_json_depth(item, max_depth, current_depth + 1):
                    return False

        return True

class SecurityValidator:
    """Security validation utilities."""

    def __init__(self, config: SecurityPolicyConfig):
        self.config = config
        self.sanitizer = InputSanitizer()

    def validate_request_size(self, content_length: Optional[int]) -> bool:
        """Validate request content length."""
        if content_length is None:
            return True

        max_size_bytes = self.config.max_request_size_mb * 1024 * 1024
        return content_length <= max_size_bytes

    def validate_query_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate query parameters and return validation errors."""
        errors = []

        if len(params) > self.config.max_query_params:
            errors.append(f"Too many query parameters: {len(params)}")

        for key, value in params.items():
            # Validate parameter key
            if len(key) > 100:
                errors.append(f"Parameter key too long: {key[:50]}...")

            if self.sanitizer.detect_sql_injection(key):
                errors.append(f"SQL injection detected in parameter key: {key}")

            if self.sanitizer.detect_xss(key):
                errors.append(f"XSS detected in parameter key: {key}")

            # Validate parameter value
            if isinstance(value, str):
                if len(value) > self.config.max_string_length:
                    errors.append(f"Parameter value too long: {key}")

                if self.sanitizer.detect_sql_injection(value):
                    errors.append(f"SQL injection detected in parameter value: {key}")

                if self.sanitizer.detect_xss(value):
                    errors.append(f"XSS detected in parameter value: {key}")

                if self.sanitizer.detect_command_injection(value):
                    errors.append(f"Command injection detected in parameter value: {key}")

        return errors

    def validate_headers(self, headers: Dict[str, str]) -> List[str]:
        """Validate request headers and return validation errors."""
        errors = []

        if len(headers) > self.config.max_headers:
            errors.append(f"Too many headers: {len(headers)}")

        for name, value in headers.items():
            # Skip standard headers
            if name.lower() in ['authorization', 'content-type', 'user-agent', 'accept']:
                continue

            if len(name) > 100:
                errors.append(f"Header name too long: {name[:50]}...")

            if len(value) > 1000:
                errors.append(f"Header value too long: {name}")

            if self.sanitizer.detect_xss(value):
                errors.append(f"XSS detected in header: {name}")

        return errors

    def validate_json_payload(self, payload: Any) -> List[str]:
        """Validate JSON payload and return validation errors."""
        errors = []

        if not self.sanitizer.validate_json_depth(payload, self.config.max_json_depth):
            errors.append("JSON structure too deep")

        def validate_json_values(data: Any, path: str = ""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key

                    if isinstance(key, str):
                        if len(key) > 100:
                            errors.append(f"JSON key too long at {current_path}")

                        if self.sanitizer.detect_sql_injection(key):
                            errors.append(f"SQL injection detected in JSON key at {current_path}")

                    validate_json_values(value, current_path)

            elif isinstance(data, list):
                for i, item in enumerate(data):
                    validate_json_values(item, f"{path}[{i}]")

            elif isinstance(data, str):
                if len(data) > self.config.max_string_length:
                    errors.append(f"String value too long at {path}")

                if self.sanitizer.detect_sql_injection(data):
                    errors.append(f"SQL injection detected at {path}")

                if self.sanitizer.detect_xss(data):
                    errors.append(f"XSS detected at {path}")

                if self.sanitizer.detect_command_injection(data):
                    errors.append(f"Command injection detected at {path}")

        validate_json_values(payload)
        return errors

    def validate_password_strength(self, password: str) -> List[str]:
        """Validate password strength and return validation errors."""
        errors = []

        if len(password) < self.config.min_password_length:
            errors.append(f"Password must be at least {self.config.min_password_length} characters")

        if self.config.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if self.config.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if self.config.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if self.config.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        # Check for common patterns
        if password.lower() in ['password', '123456', 'qwerty', 'admin']:
            errors.append("Password is too common")

        return errors

class SecurityAuditor:
    """Security audit logging and monitoring."""

    def __init__(self, config: SecurityPolicyConfig):
        self.config = config

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        request: Optional[Request] = None
    ):
        """Log security event with context."""

        if not self.config.log_security_events:
            return

        event_data = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }

        if request:
            event_data.update({
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers)
            })

        with logfire.span("Security audit event", event_type=event_type, severity=severity):
            if severity in ["high", "critical"]:
                logger.error("Security event", **event_data)
            elif severity == "medium":
                logger.warning("Security event", **event_data)
            else:
                logger.info("Security event", **event_data)

    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        failure_reason: Optional[str] = None,
        request: Optional[Request] = None
    ):
        """Log authentication attempt."""

        if not self.config.log_failed_auth_attempts and not success:
            return

        self.log_security_event(
            event_type="authentication_attempt",
            severity="medium" if not success else "low",
            details={
                "username": username,
                "success": success,
                "failure_reason": failure_reason
            },
            request=request
        )

    def log_rate_limit_violation(
        self,
        limit_type: str,
        current_rate: int,
        limit: int,
        request: Optional[Request] = None
    ):
        """Log rate limit violation."""

        if not self.config.log_rate_limit_violations:
            return

        self.log_security_event(
            event_type="rate_limit_violation",
            severity="medium",
            details={
                "limit_type": limit_type,
                "current_rate": current_rate,
                "limit": limit
            },
            request=request
        )

    def log_input_validation_failure(
        self,
        validation_type: str,
        errors: List[str],
        request: Optional[Request] = None
    ):
        """Log input validation failure."""

        if not self.config.log_input_validation_failures:
            return

        self.log_security_event(
            event_type="input_validation_failure",
            severity="medium",
            details={
                "validation_type": validation_type,
                "errors": errors
            },
            request=request
        )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

class SecurityHeadersManager:
    """Security headers management."""

    def __init__(self, config: SecurityPolicyConfig):
        self.config = config

    def get_security_headers(self, request: Request) -> Dict[str, str]:
        """Get security headers for response."""
        headers = {}

        if self.config.enable_hsts:
            headers["Strict-Transport-Security"] = (
                f"max-age={self.config.hsts_max_age_seconds}; "
                "includeSubDomains; preload"
            )

        if self.config.enable_csp:
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
            headers["Content-Security-Policy"] = csp_policy

        if self.config.enable_xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"

        if self.config.enable_content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"

        if self.config.enable_frame_options:
            headers["X-Frame-Options"] = "DENY"

        # Additional security headers
        headers.update({
            "X-Permitted-Cross-Domain-Policies": "none",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            ),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })

        return headers

class ApiKeyManager:
    """API key management and validation."""

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}

    def generate_api_key(
        self,
        user_id: int,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate new API key."""

        api_key = f"ak_{secrets.token_urlsafe(32)}"

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0,
            "active": True
        }

        logger.info("API key generated", user_id=user_id, name=name)
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return metadata."""

        key_data = self.api_keys.get(api_key)
        if not key_data:
            return None

        if not key_data["active"]:
            return None

        if key_data["expires_at"] and key_data["expires_at"] < datetime.utcnow():
            return None

        # Update usage tracking
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1

        return key_data

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""

        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info("API key revoked", api_key=api_key[:16] + "...")
            return True

        return False

# Global instances
_security_config: Optional[SecurityPolicyConfig] = None
_security_validator: Optional[SecurityValidator] = None
_security_auditor: Optional[SecurityAuditor] = None
_security_headers_manager: Optional[SecurityHeadersManager] = None
_api_key_manager: Optional[ApiKeyManager] = None

def get_security_config() -> SecurityPolicyConfig:
    """Get global security configuration."""
    global _security_config

    if _security_config is None:
        _security_config = SecurityPolicyConfig()

    return _security_config

def get_security_validator() -> SecurityValidator:
    """Get global security validator."""
    global _security_validator

    if _security_validator is None:
        _security_validator = SecurityValidator(get_security_config())

    return _security_validator

def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor."""
    global _security_auditor

    if _security_auditor is None:
        _security_auditor = SecurityAuditor(get_security_config())

    return _security_auditor

def get_security_headers_manager() -> SecurityHeadersManager:
    """Get global security headers manager."""
    global _security_headers_manager

    if _security_headers_manager is None:
        _security_headers_manager = SecurityHeadersManager(get_security_config())

    return _security_headers_manager

def get_api_key_manager() -> ApiKeyManager:
    """Get global API key manager."""
    global _api_key_manager

    if _api_key_manager is None:
        _api_key_manager = ApiKeyManager()

    return _api_key_manager
