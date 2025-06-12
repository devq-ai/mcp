"""
Security and Infrastructure Tools Package for Agentical

This package provides comprehensive security, authentication, and infrastructure
management tools that form the foundation of the Agentical framework's
security and operational capabilities.

Core Tools Included:
- auth_manager: Authentication and authorization management
- encryption_tool: Data encryption, decryption, and key management
- audit_logging: Security audit logging and compliance
- container_manager: Docker container orchestration and management
- load_balancer: Load balancing and traffic distribution
- monitoring_tool: Infrastructure monitoring and alerting
- secret_manager: Secrets and credentials management

Features:
- Multi-provider authentication (OAuth2, JWT, API keys)
- Enterprise-grade encryption with multiple algorithms
- Comprehensive audit trails for compliance
- Container orchestration with Docker integration
- Intelligent load balancing with health checks
- Real-time infrastructure monitoring
- Secure secrets management with rotation
- Integration with cloud providers and security services
- Performance monitoring and observability
- Zero-trust security architecture support
"""

from .auth_manager import (
    AuthManager,
    AuthenticationResult,
    AuthProvider,
    TokenType,
    UserRole,
    Permission,
    AuthSession
)

from .encryption_tool import (
    EncryptionTool,
    EncryptionResult,
    EncryptionAlgorithm,
    KeyType,
    KeyManager,
    CryptoContext
)

from .audit_logging import (
    AuditLogger,
    AuditEvent,
    AuditLevel,
    AuditCategory,
    ComplianceStandard,
    AuditReport
)

from .container_manager import (
    ContainerManager,
    ContainerInfo,
    ContainerStatus,
    ContainerNetwork,
    DeploymentStrategy,
    ResourceLimits
)

from .load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    HealthCheck,
    BackendServer,
    TrafficRule,
    LoadBalancerMetrics
)

from .monitoring_tool import (
    MonitoringTool,
    MonitoringAlert,
    MetricType,
    AlertLevel,
    MonitoringTarget,
    SystemMetrics
)

from .secret_manager import (
    SecretManager,
    Secret,
    SecretType,
    SecretPolicy,
    RotationStrategy,
    SecretAudit
)

# Security and infrastructure tools
__all__ = [
    # Authentication and authorization
    "AuthManager",
    "AuthenticationResult",
    "AuthProvider",
    "TokenType",
    "UserRole",
    "Permission",
    "AuthSession",

    # Encryption and cryptography
    "EncryptionTool",
    "EncryptionResult",
    "EncryptionAlgorithm",
    "KeyType",
    "KeyManager",
    "CryptoContext",

    # Audit and compliance
    "AuditLogger",
    "AuditEvent",
    "AuditLevel",
    "AuditCategory",
    "ComplianceStandard",
    "AuditReport",

    # Container management
    "ContainerManager",
    "ContainerInfo",
    "ContainerStatus",
    "ContainerNetwork",
    "DeploymentStrategy",
    "ResourceLimits",

    # Load balancing
    "LoadBalancer",
    "LoadBalancingStrategy",
    "HealthCheck",
    "BackendServer",
    "TrafficRule",
    "LoadBalancerMetrics",

    # Monitoring and observability
    "MonitoringTool",
    "MonitoringAlert",
    "MetricType",
    "AlertLevel",
    "MonitoringTarget",
    "SystemMetrics",

    # Secrets management
    "SecretManager",
    "Secret",
    "SecretType",
    "SecretPolicy",
    "RotationStrategy",
    "SecretAudit"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Supported authentication providers
SUPPORTED_AUTH_PROVIDERS = {
    "oauth2": {
        "name": "OAuth 2.0",
        "description": "Industry standard OAuth 2.0 authentication",
        "flows": ["authorization_code", "client_credentials", "password", "implicit"],
        "requires_client_id": True,
        "requires_client_secret": True
    },
    "jwt": {
        "name": "JSON Web Tokens",
        "description": "Stateless JWT-based authentication",
        "algorithms": ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"],
        "requires_secret": True,
        "supports_refresh": True
    },
    "api_key": {
        "name": "API Key Authentication",
        "description": "Simple API key-based authentication",
        "key_types": ["header", "query", "body"],
        "supports_rotation": True,
        "rate_limiting": True
    },
    "basic": {
        "name": "Basic Authentication",
        "description": "HTTP Basic Authentication",
        "requires_username": True,
        "requires_password": True,
        "secure_transport": True
    },
    "bearer": {
        "name": "Bearer Token",
        "description": "Bearer token authentication",
        "token_types": ["access_token", "refresh_token"],
        "supports_expiration": True,
        "secure_storage": True
    }
}

# Supported encryption algorithms
SUPPORTED_ENCRYPTION_ALGORITHMS = {
    "aes": {
        "name": "Advanced Encryption Standard",
        "key_sizes": [128, 192, 256],
        "modes": ["CBC", "GCM", "CTR", "ECB"],
        "iv_required": True,
        "authenticated": ["GCM"]
    },
    "chacha20": {
        "name": "ChaCha20-Poly1305",
        "key_size": 256,
        "authenticated": True,
        "nonce_required": True,
        "high_performance": True
    },
    "rsa": {
        "name": "RSA Encryption",
        "key_sizes": [2048, 3072, 4096],
        "padding": ["OAEP", "PKCS1v15"],
        "use_case": "asymmetric",
        "key_generation": True
    },
    "ecc": {
        "name": "Elliptic Curve Cryptography",
        "curves": ["P-256", "P-384", "P-521", "secp256k1"],
        "use_case": "asymmetric",
        "smaller_keys": True,
        "faster_operations": True
    }
}

# Supported container platforms
SUPPORTED_CONTAINER_PLATFORMS = {
    "docker": {
        "name": "Docker",
        "api_available": True,
        "orchestration": False,
        "registry_support": True,
        "networking": ["bridge", "host", "overlay", "none"]
    },
    "kubernetes": {
        "name": "Kubernetes",
        "api_available": True,
        "orchestration": True,
        "auto_scaling": True,
        "service_discovery": True
    },
    "podman": {
        "name": "Podman",
        "api_available": True,
        "rootless": True,
        "daemonless": True,
        "docker_compatible": True
    }
}

# Monitoring metrics categories
MONITORING_CATEGORIES = {
    "system": {
        "name": "System Metrics",
        "metrics": ["cpu_usage", "memory_usage", "disk_usage", "network_io", "load_average"],
        "collection_interval": 60,
        "retention_days": 30
    },
    "application": {
        "name": "Application Metrics",
        "metrics": ["response_time", "request_rate", "error_rate", "throughput"],
        "collection_interval": 30,
        "retention_days": 90
    },
    "security": {
        "name": "Security Metrics",
        "metrics": ["failed_logins", "unauthorized_access", "suspicious_activity"],
        "collection_interval": 10,
        "retention_days": 365
    },
    "business": {
        "name": "Business Metrics",
        "metrics": ["user_activity", "feature_usage", "conversion_rate"],
        "collection_interval": 300,
        "retention_days": 730
    }
}

# Compliance standards supported
COMPLIANCE_STANDARDS = {
    "sox": {
        "name": "Sarbanes-Oxley Act",
        "description": "Financial reporting and corporate governance",
        "required_controls": ["access_control", "change_management", "audit_trail"],
        "retention_years": 7,
        "geographic_scope": "US"
    },
    "hipaa": {
        "name": "Health Insurance Portability and Accountability Act",
        "description": "Healthcare data protection",
        "required_controls": ["encryption", "access_control", "audit_logging", "incident_response"],
        "retention_years": 6,
        "geographic_scope": "US"
    },
    "gdpr": {
        "name": "General Data Protection Regulation",
        "description": "Data protection and privacy",
        "required_controls": ["consent_management", "data_portability", "right_to_erasure"],
        "retention_varies": True,
        "geographic_scope": "EU"
    },
    "pci_dss": {
        "name": "Payment Card Industry Data Security Standard",
        "description": "Credit card data protection",
        "required_controls": ["encryption", "network_security", "access_control", "monitoring"],
        "compliance_levels": ["Level 1", "Level 2", "Level 3", "Level 4"],
        "geographic_scope": "Global"
    },
    "iso_27001": {
        "name": "ISO/IEC 27001",
        "description": "Information security management systems",
        "required_controls": ["risk_management", "incident_management", "business_continuity"],
        "certification_required": True,
        "geographic_scope": "Global"
    }
}

# Load balancing strategies
LOAD_BALANCING_STRATEGIES = {
    "round_robin": {
        "name": "Round Robin",
        "description": "Distribute requests equally across servers",
        "weighted": False,
        "session_affinity": False,
        "complexity": "low"
    },
    "weighted_round_robin": {
        "name": "Weighted Round Robin",
        "description": "Distribute requests based on server weights",
        "weighted": True,
        "session_affinity": False,
        "complexity": "medium"
    },
    "least_connections": {
        "name": "Least Connections",
        "description": "Route to server with fewest active connections",
        "dynamic": True,
        "session_affinity": False,
        "complexity": "medium"
    },
    "ip_hash": {
        "name": "IP Hash",
        "description": "Route based on client IP hash",
        "session_affinity": True,
        "consistent": True,
        "complexity": "medium"
    },
    "geographic": {
        "name": "Geographic Routing",
        "description": "Route based on client geographic location",
        "latency_optimized": True,
        "requires_geo_data": True,
        "complexity": "high"
    }
}

# Default configurations
DEFAULT_AUTH_CONFIG = {
    "default_provider": "jwt",
    "token_expiry_minutes": 60,
    "refresh_token_expiry_days": 30,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 15,
    "password_min_length": 8,
    "require_special_chars": True,
    "session_timeout_minutes": 30
}

DEFAULT_ENCRYPTION_CONFIG = {
    "default_algorithm": "aes",
    "key_size": 256,
    "mode": "GCM",
    "key_rotation_days": 90,
    "backup_key_count": 3,
    "secure_key_storage": True
}

DEFAULT_AUDIT_CONFIG = {
    "log_level": "INFO",
    "retention_days": 365,
    "encryption_enabled": True,
    "real_time_alerts": True,
    "compliance_standards": ["iso_27001"],
    "include_request_body": False,
    "include_response_body": False
}

DEFAULT_CONTAINER_CONFIG = {
    "platform": "docker",
    "default_network": "bridge",
    "resource_limits": {
        "memory": "512MB",
        "cpu": "0.5",
        "storage": "1GB"
    },
    "health_check_interval": 30,
    "restart_policy": "on-failure",
    "max_restart_count": 3
}

DEFAULT_MONITORING_CONFIG = {
    "collection_interval": 60,
    "retention_days": 30,
    "alert_thresholds": {
        "cpu_usage": 80,
        "memory_usage": 85,
        "disk_usage": 90,
        "error_rate": 5
    },
    "notification_channels": ["email", "slack"],
    "dashboard_refresh": 30
}

DEFAULT_SECRET_CONFIG = {
    "encryption_enabled": True,
    "rotation_enabled": True,
    "rotation_interval_days": 90,
    "backup_count": 5,
    "access_logging": True,
    "expire_unused_days": 180
}

# Security tool error codes
SECURITY_TOOL_ERROR_CODES = {
    "AUTH_FAILED": "S001",
    "TOKEN_EXPIRED": "S002",
    "INSUFFICIENT_PERMISSIONS": "S003",
    "ENCRYPTION_FAILED": "S004",
    "DECRYPTION_FAILED": "S005",
    "KEY_GENERATION_FAILED": "S006",
    "AUDIT_LOG_FAILED": "S007",
    "CONTAINER_START_FAILED": "S008",
    "CONTAINER_STOP_FAILED": "S009",
    "LOAD_BALANCER_CONFIG_ERROR": "S010",
    "HEALTH_CHECK_FAILED": "S011",
    "MONITORING_SETUP_FAILED": "S012",
    "ALERT_DELIVERY_FAILED": "S013",
    "SECRET_NOT_FOUND": "S014",
    "SECRET_ROTATION_FAILED": "S015",
    "COMPLIANCE_VIOLATION": "S016"
}

def get_security_tools_info() -> dict:
    """Get comprehensive information about security tools."""
    return {
        "package_version": __version__,
        "supported_auth_providers": list(SUPPORTED_AUTH_PROVIDERS.keys()),
        "supported_encryption_algorithms": list(SUPPORTED_ENCRYPTION_ALGORITHMS.keys()),
        "supported_container_platforms": list(SUPPORTED_CONTAINER_PLATFORMS.keys()),
        "monitoring_categories": list(MONITORING_CATEGORIES.keys()),
        "compliance_standards": list(COMPLIANCE_STANDARDS.keys()),
        "load_balancing_strategies": list(LOAD_BALANCING_STRATEGIES.keys()),
        "error_codes": SECURITY_TOOL_ERROR_CODES,
        "default_configs": {
            "authentication": DEFAULT_AUTH_CONFIG,
            "encryption": DEFAULT_ENCRYPTION_CONFIG,
            "audit": DEFAULT_AUDIT_CONFIG,
            "container": DEFAULT_CONTAINER_CONFIG,
            "monitoring": DEFAULT_MONITORING_CONFIG,
            "secrets": DEFAULT_SECRET_CONFIG
        }
    }

def validate_security_config(tool_type: str, config: dict) -> tuple[bool, list[str]]:
    """
    Validate security tool configuration.

    Args:
        tool_type: Type of security tool
        config: Configuration dictionary

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    if tool_type == "authentication":
        if "default_provider" in config:
            if config["default_provider"] not in SUPPORTED_AUTH_PROVIDERS:
                errors.append(f"default_provider must be one of: {list(SUPPORTED_AUTH_PROVIDERS.keys())}")

        if "token_expiry_minutes" in config:
            if not isinstance(config["token_expiry_minutes"], int) or config["token_expiry_minutes"] < 1:
                errors.append("token_expiry_minutes must be a positive integer")

        if "max_login_attempts" in config:
            if not isinstance(config["max_login_attempts"], int) or config["max_login_attempts"] < 1:
                errors.append("max_login_attempts must be a positive integer")

    elif tool_type == "encryption":
        if "default_algorithm" in config:
            if config["default_algorithm"] not in SUPPORTED_ENCRYPTION_ALGORITHMS:
                errors.append(f"default_algorithm must be one of: {list(SUPPORTED_ENCRYPTION_ALGORITHMS.keys())}")

        if "key_size" in config:
            if not isinstance(config["key_size"], int) or config["key_size"] not in [128, 192, 256, 2048, 3072, 4096]:
                errors.append("key_size must be a valid encryption key size")

    elif tool_type == "container":
        if "platform" in config:
            if config["platform"] not in SUPPORTED_CONTAINER_PLATFORMS:
                errors.append(f"platform must be one of: {list(SUPPORTED_CONTAINER_PLATFORMS.keys())}")

    elif tool_type == "monitoring":
        if "collection_interval" in config:
            if not isinstance(config["collection_interval"], int) or config["collection_interval"] < 1:
                errors.append("collection_interval must be a positive integer")

    elif tool_type == "load_balancer":
        if "strategy" in config:
            if config["strategy"] not in LOAD_BALANCING_STRATEGIES:
                errors.append(f"strategy must be one of: {list(LOAD_BALANCING_STRATEGIES.keys())}")

    return len(errors) == 0, errors

def create_default_security_manager(config_overrides: dict = None) -> dict:
    """
    Create default security manager configuration.

    Args:
        config_overrides: Override default configurations

    Returns:
        dict: Complete security manager configuration
    """
    config = {
        "auth_manager": DEFAULT_AUTH_CONFIG.copy(),
        "encryption_tool": DEFAULT_ENCRYPTION_CONFIG.copy(),
        "audit_logger": DEFAULT_AUDIT_CONFIG.copy(),
        "container_manager": DEFAULT_CONTAINER_CONFIG.copy(),
        "load_balancer": {},
        "monitoring_tool": DEFAULT_MONITORING_CONFIG.copy(),
        "secret_manager": DEFAULT_SECRET_CONFIG.copy()
    }

    if config_overrides:
        for tool, overrides in config_overrides.items():
            if tool in config:
                config[tool].update(overrides)

    return config
