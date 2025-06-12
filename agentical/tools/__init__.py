"""
Tools Package for Agentical

This package provides comprehensive tool integration capabilities for the
Agentical framework, including MCP server integration, tool discovery,
execution, and management.

Features:
- MCP (Model Context Protocol) server integration
- Dynamic tool discovery and registration
- Unified tool execution framework
- Performance monitoring and observability
- Tool capability mapping and search
- Integration with workflow and agent systems
- Comprehensive error handling and validation
"""

from .core.tool_manager import (
    ToolManager,
    ToolManagerFactory,
    ToolManagerConfig,
    ToolManagerState
)

from .core.tool_registry import (
    ToolRegistry,
    ToolRegistryEntry,
    ToolDiscoveryMode
)

from .execution.tool_executor import (
    ToolExecutor,
    ToolExecutionResult,
    ExecutionContext,
    ExecutionMode,
    ExecutionPriority
)

from .mcp.mcp_client import (
    MCPClient,
    MCPServer,
    MCPToolSchema,
    MCPConnectionStatus
)

# Core tool system components
__all__ = [
    # Manager components
    "ToolManager",
    "ToolManagerFactory",
    "ToolManagerConfig",
    "ToolManagerState",

    # Registry components
    "ToolRegistry",
    "ToolRegistryEntry",
    "ToolDiscoveryMode",

    # Execution components
    "ToolExecutor",
    "ToolExecutionResult",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionPriority",

    # MCP components
    "MCPClient",
    "MCPServer",
    "MCPToolSchema",
    "MCPConnectionStatus"
]

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Package metadata
SUPPORTED_MCP_SERVERS = [
    # Core MCP Servers (NPX-based)
    "filesystem",
    "git",
    "fetch",
    "memory",
    "sequentialthinking",
    "github",
    "inspector",

    # DevQ.ai Python-based Servers
    "taskmaster-ai",
    "ptolemies-mcp",
    "context7-mcp",
    "bayes-mcp",
    "crawl4ai-mcp",
    "dart-mcp",
    "surrealdb-mcp",
    "logfire-mcp",
    "darwin-mcp",

    # Specialized Development Servers
    "agentql-mcp",
    "calendar-mcp",
    "jupyter-mcp",
    "stripe-mcp",
    "shadcn-ui-mcp-server",
    "magic-mcp",

    # Scientific Computing & Solvers
    "solver-z3-mcp",
    "solver-pysat-mcp",
    "solver-mzn-mcp",

    # Registry & Infrastructure
    "registry-mcp",
    "browser-tools-mcp"
]

SUPPORTED_TOOL_TYPES = [
    # Core tool types
    "filesystem",
    "git",
    "memory",
    "fetch",
    "sequential_thinking",

    # Knowledge and analysis
    "ptolemies",
    "context7",
    "bayes",
    "darwin",

    # External integrations
    "github",
    "crawl4ai",
    "calendar",
    "stripe",

    # Development tools
    "jupyter",
    "shadcn_ui",
    "magic",

    # Specialized tools
    "solver_z3",
    "solver_pysat",
    "solver_mzn",

    # Database tools
    "surrealdb",

    # Custom tools
    "custom",
    "api",
    "script"
]

EXECUTION_MODES = [
    "sync",
    "async",
    "batch",
    "stream"
]

EXECUTION_PRIORITIES = [
    "low",
    "normal",
    "high",
    "critical"
]

# Default configuration
DEFAULT_TOOL_MANAGER_CONFIG = {
    "mcp_config_path": "mcp-servers.json",
    "max_concurrent_executions": 50,
    "default_timeout_seconds": 300,
    "discovery_mode": "hybrid",
    "enable_caching": True,
    "cache_ttl_minutes": 30,
    "health_check_interval_minutes": 5,
    "auto_reconnect": True,
    "max_mcp_connections": 20
}

DEFAULT_REGISTRY_CONFIG = {
    "discovery_mode": "hybrid",
    "enable_caching": True,
    "cache_ttl_minutes": 30
}

DEFAULT_EXECUTOR_CONFIG = {
    "max_concurrent_executions": 50,
    "default_timeout_seconds": 300,
    "enable_monitoring": True
}

DEFAULT_MCP_CONFIG = {
    "max_connections": 20
}

# Tool categories for organization
TOOL_CATEGORIES = {
    "development": [
        "filesystem", "git", "github", "jupyter", "shadcn_ui", "magic"
    ],
    "data_analysis": [
        "ptolemies", "context7", "bayes", "darwin", "crawl4ai"
    ],
    "external_services": [
        "fetch", "calendar", "stripe", "github"
    ],
    "scientific_computing": [
        "solver_z3", "solver_pysat", "solver_mzn", "bayes"
    ],
    "infrastructure": [
        "memory", "surrealdb", "logfire", "sequential_thinking"
    ],
    "custom": [
        "custom", "api", "script"
    ]
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "execution_time_warning_seconds": 30.0,
    "execution_time_critical_seconds": 120.0,
    "success_rate_warning_percentage": 85.0,
    "success_rate_critical_percentage": 70.0,
    "concurrent_executions_warning_percentage": 80.0,
    "concurrent_executions_critical_percentage": 95.0
}

# Error codes for tool operations
TOOL_ERROR_CODES = {
    "TOOL_NOT_FOUND": "T001",
    "TOOL_EXECUTION_FAILED": "T002",
    "TOOL_VALIDATION_ERROR": "T003",
    "TOOL_TIMEOUT": "T004",
    "MCP_CONNECTION_ERROR": "T005",
    "MCP_SERVER_ERROR": "T006",
    "REGISTRY_ERROR": "T007",
    "CONFIGURATION_ERROR": "T008"
}

def get_tool_info() -> dict:
    """Get comprehensive information about the tools package."""
    return {
        "package_version": __version__,
        "supported_mcp_servers": len(SUPPORTED_MCP_SERVERS),
        "supported_tool_types": len(SUPPORTED_TOOL_TYPES),
        "execution_modes": EXECUTION_MODES,
        "execution_priorities": EXECUTION_PRIORITIES,
        "tool_categories": TOOL_CATEGORIES,
        "performance_thresholds": PERFORMANCE_THRESHOLDS,
        "error_codes": TOOL_ERROR_CODES,
        "default_configs": {
            "manager": DEFAULT_TOOL_MANAGER_CONFIG,
            "registry": DEFAULT_REGISTRY_CONFIG,
            "executor": DEFAULT_EXECUTOR_CONFIG,
            "mcp": DEFAULT_MCP_CONFIG
        }
    }

def validate_tool_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate tool configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    # Validate required fields
    required_fields = ["mcp_config_path", "max_concurrent_executions"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate data types and ranges
    if "max_concurrent_executions" in config:
        if not isinstance(config["max_concurrent_executions"], int) or config["max_concurrent_executions"] < 1:
            errors.append("max_concurrent_executions must be a positive integer")

    if "default_timeout_seconds" in config:
        if not isinstance(config["default_timeout_seconds"], int) or config["default_timeout_seconds"] < 1:
            errors.append("default_timeout_seconds must be a positive integer")

    if "discovery_mode" in config:
        valid_modes = ["automatic", "manual", "hybrid", "mcp_only"]
        if config["discovery_mode"] not in valid_modes:
            errors.append(f"discovery_mode must be one of: {valid_modes}")

    return len(errors) == 0, errors

def create_default_tool_manager(db_session) -> 'ToolManager':
    """
    Create a tool manager with default configuration.

    Args:
        db_session: Database session

    Returns:
        ToolManager: Configured tool manager instance
    """
    return ToolManagerFactory.create_manager(
        db_session=db_session,
        config=DEFAULT_TOOL_MANAGER_CONFIG
    )
