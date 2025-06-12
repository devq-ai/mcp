"""
Communication & Integration Tools Module for Agentical

This module provides comprehensive communication and integration capabilities including
email sending, Slack integration, webhook management, external API clients,
and calendar integration for enterprise-grade communication workflows.

Features:
- Multi-provider email sending with templates and tracking
- Slack integration with bot functionality and event handling
- Webhook management with secure signature verification
- Generic external API client with authentication and retries
- Calendar integration with Google Calendar and Outlook
- Enterprise features (audit logging, monitoring, rate limiting)
- Real-time communication and event-driven architectures
"""

from typing import Dict, Any, Optional

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Tool imports (lazy loading to handle optional dependencies)
_TOOLS_REGISTRY: Dict[str, Any] = {}

# Available tools
AVAILABLE_TOOLS = [
    "external_api",
    "email_sender",
    "slack_integration",
    "webhook_manager",
    "calendar_integration"
]

def get_tool(tool_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Get a communication tool instance with lazy loading.

    Args:
        tool_name: Name of the tool to load
        config: Optional configuration dictionary

    Returns:
        Tool instance

    Raises:
        ImportError: If tool dependencies are not available
        ValueError: If tool name is not recognized
    """
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available tools: {AVAILABLE_TOOLS}")

    if tool_name not in _TOOLS_REGISTRY:
        try:
            if tool_name == "external_api":
                from .external_api import ExternalAPIClient
                _TOOLS_REGISTRY[tool_name] = ExternalAPIClient
            elif tool_name == "email_sender":
                from .email_sender import EmailSender
                _TOOLS_REGISTRY[tool_name] = EmailSender
            elif tool_name == "slack_integration":
                from .slack_integration import SlackIntegration
                _TOOLS_REGISTRY[tool_name] = SlackIntegration
            elif tool_name == "webhook_manager":
                from .webhook_manager import WebhookManager
                _TOOLS_REGISTRY[tool_name] = WebhookManager
            elif tool_name == "calendar_integration":
                from .calendar_integration import CalendarIntegration
                _TOOLS_REGISTRY[tool_name] = CalendarIntegration
        except ImportError as e:
            raise ImportError(
                f"Cannot load {tool_name}. Missing dependencies: {e}. "
                f"Please install required packages for {tool_name}."
            )

    tool_class = _TOOLS_REGISTRY[tool_name]
    return tool_class(config or {})

def list_available_tools() -> Dict[str, str]:
    """
    List all available communication tools with descriptions.

    Returns:
        Dictionary mapping tool names to descriptions
    """
    return {
        "external_api": "Generic HTTP API client with authentication and retries",
        "email_sender": "Multi-provider email sending with templates and tracking",
        "slack_integration": "Slack bot integration with messaging and event handling",
        "webhook_manager": "Webhook registration, verification, and event routing",
        "calendar_integration": "Google Calendar and Outlook integration for scheduling"
    }

def check_dependencies() -> Dict[str, bool]:
    """
    Check which tools have their dependencies available.

    Returns:
        Dictionary mapping tool names to availability status
    """
    status = {}

    for tool_name in AVAILABLE_TOOLS:
        try:
            # Try to import the tool to check dependencies
            get_tool(tool_name)
            status[tool_name] = True
        except ImportError:
            status[tool_name] = False

    return status

# Configuration templates for common use cases
CONFIG_TEMPLATES = {
    "development": {
        "external_api": {
            "timeout": 30,
            "max_retries": 3,
            "retry_backoff": "exponential",
            "enable_caching": True
        },
        "email_sender": {
            "provider": "smtp",
            "smtp_server": "localhost",
            "smtp_port": 587,
            "use_tls": True,
            "template_engine": "jinja2"
        },
        "slack_integration": {
            "bot_token": None,  # Must be provided
            "signing_secret": None,  # Must be provided
            "enable_events": True,
            "socket_mode": False
        },
        "webhook_manager": {
            "max_endpoints": 100,
            "signature_verification": True,
            "rate_limiting": True,
            "event_retention_hours": 24
        },
        "calendar_integration": {
            "provider": "google",
            "timezone": "UTC",
            "max_events_per_request": 100,
            "cache_duration_minutes": 15
        }
    },
    "production": {
        "external_api": {
            "timeout": 60,
            "max_retries": 5,
            "retry_backoff": "exponential",
            "enable_caching": True,
            "circuit_breaker": True,
            "rate_limiting": True
        },
        "email_sender": {
            "provider": "sendgrid",  # or "aws_ses"
            "template_engine": "jinja2",
            "bounce_tracking": True,
            "delivery_tracking": True,
            "unsubscribe_handling": True
        },
        "slack_integration": {
            "bot_token": None,  # Must be provided
            "signing_secret": None,  # Must be provided
            "enable_events": True,
            "socket_mode": True,
            "rate_limiting": True,
            "audit_logging": True
        },
        "webhook_manager": {
            "max_endpoints": 1000,
            "signature_verification": True,
            "rate_limiting": True,
            "event_retention_hours": 168,  # 1 week
            "high_availability": True,
            "audit_logging": True
        },
        "calendar_integration": {
            "provider": "google",  # or "outlook"
            "timezone": "UTC",
            "max_events_per_request": 250,
            "cache_duration_minutes": 5,
            "rate_limiting": True,
            "audit_logging": True
        }
    },
    "enterprise": {
        "external_api": {
            "timeout": 120,
            "max_retries": 10,
            "retry_backoff": "exponential",
            "enable_caching": True,
            "circuit_breaker": True,
            "rate_limiting": True,
            "audit_logging": True,
            "encryption_enabled": True,
            "compliance_mode": True
        },
        "email_sender": {
            "provider": "multiple",  # Multi-provider failover
            "template_engine": "jinja2",
            "bounce_tracking": True,
            "delivery_tracking": True,
            "unsubscribe_handling": True,
            "encryption_enabled": True,
            "audit_logging": True,
            "compliance_mode": True
        },
        "slack_integration": {
            "bot_token": None,  # Must be provided
            "signing_secret": None,  # Must be provided
            "enable_events": True,
            "socket_mode": True,
            "rate_limiting": True,
            "audit_logging": True,
            "encryption_enabled": True,
            "compliance_mode": True,
            "high_availability": True
        },
        "webhook_manager": {
            "max_endpoints": 10000,
            "signature_verification": True,
            "rate_limiting": True,
            "event_retention_hours": 720,  # 30 days
            "high_availability": True,
            "audit_logging": True,
            "encryption_enabled": True,
            "compliance_mode": True,
            "disaster_recovery": True
        },
        "calendar_integration": {
            "provider": "multiple",  # Multi-provider support
            "timezone": "UTC",
            "max_events_per_request": 500,
            "cache_duration_minutes": 1,
            "rate_limiting": True,
            "audit_logging": True,
            "encryption_enabled": True,
            "compliance_mode": True,
            "high_availability": True
        }
    }
}

def get_config_template(template_name: str) -> Dict[str, Any]:
    """
    Get a configuration template for common deployment scenarios.

    Args:
        template_name: Name of the template (development, production, enterprise)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If template name is not recognized
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")

    return CONFIG_TEMPLATES[template_name].copy()

# Common configuration constants
DEFAULT_TIMEOUTS = {
    "short": 10,
    "medium": 30,
    "long": 60,
    "extended": 120
}

DEFAULT_RETRY_CONFIGS = {
    "conservative": {"max_retries": 3, "backoff": "linear"},
    "standard": {"max_retries": 5, "backoff": "exponential"},
    "aggressive": {"max_retries": 10, "backoff": "exponential"}
}

RATE_LIMIT_PRESETS = {
    "development": {"requests_per_minute": 60, "burst_limit": 10},
    "production": {"requests_per_minute": 1000, "burst_limit": 100},
    "enterprise": {"requests_per_minute": 10000, "burst_limit": 1000}
}

# Export main classes for direct import
__all__ = [
    "get_tool",
    "list_available_tools",
    "check_dependencies",
    "get_config_template",
    "AVAILABLE_TOOLS",
    "CONFIG_TEMPLATES",
    "DEFAULT_TIMEOUTS",
    "DEFAULT_RETRY_CONFIGS",
    "RATE_LIMIT_PRESETS"
]
