"""
Middlewares Package for Agentical

This package contains middleware components that handle cross-cutting 
concerns like logging, authentication, security, and request processing.
"""

try:
    from .logging_middleware import (
        RequestLoggingMiddleware,
        HealthCheckLoggingFilter,
        ErrorLoggingMiddleware
    )
except ImportError:
    # Logging middleware not available
    RequestLoggingMiddleware = None
    HealthCheckLoggingFilter = None
    ErrorLoggingMiddleware = None

try:
    from .security import (
        RateLimitMiddleware,
        RateLimitConfig,
        SecurityHeadersMiddleware,
        RequestValidationMiddleware,
        RequestValidationConfig,
        BotProtectionMiddleware,
        BotDetectionConfig
    )
except ImportError:
    # Security middleware not available
    RateLimitMiddleware = None
    RateLimitConfig = None
    SecurityHeadersMiddleware = None
    RequestValidationMiddleware = None
    RequestValidationConfig = None
    BotProtectionMiddleware = None
    BotDetectionConfig = None

__all__ = [
    # Logging middlewares
    "RequestLoggingMiddleware",
    "HealthCheckLoggingFilter", 
    "ErrorLoggingMiddleware",
    
    # Security middlewares
    "RateLimitMiddleware",
    "RateLimitConfig",
    "SecurityHeadersMiddleware",
    "RequestValidationMiddleware",
    "RequestValidationConfig",
    "BotProtectionMiddleware",
    "BotDetectionConfig"
]