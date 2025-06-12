"""
MCP (Model Context Protocol) Integration Package for Agentical

This package provides comprehensive MCP server integration capabilities
for the Agentical framework, enabling seamless communication with
external MCP servers and tools.

Features:
- MCP server connection and management
- Tool discovery and schema validation
- Async communication with proper error handling
- Connection pooling and health monitoring
- Integration with tool registry and execution systems
- Support for all standard MCP protocol operations
"""

from .mcp_client import (
    MCPClient,
    MCPServer,
    MCPToolSchema,
    MCPConnectionStatus
)

__all__ = [
    # Client components
    "MCPClient",
    "MCPServer",
    "MCPToolSchema",
    "MCPConnectionStatus"
]

# Package metadata
__version__ = "1.0.0"
__description__ = "MCP server integration for Agentical tool system"

# Supported MCP protocol versions
SUPPORTED_MCP_VERSIONS = ["1.0.0", "1.1.0"]

# Default MCP configuration
DEFAULT_MCP_CONFIG = {
    "max_connections": 20,
    "connection_timeout_seconds": 30,
    "execution_timeout_seconds": 300,
    "retry_attempts": 3,
    "health_check_interval_seconds": 60,
    "auto_reconnect": True
}

# MCP server categories
MCP_SERVER_CATEGORIES = {
    "core": [
        "filesystem",
        "git",
        "fetch",
        "memory",
        "sequentialthinking"
    ],
    "development": [
        "github",
        "jupyter",
        "shadcn-ui",
        "magic"
    ],
    "data_analysis": [
        "ptolemies",
        "context7",
        "bayes",
        "crawl4ai",
        "darwin"
    ],
    "external_services": [
        "calendar",
        "stripe"
    ],
    "scientific": [
        "solver-z3",
        "solver-pysat",
        "solver-mzn"
    ],
    "infrastructure": [
        "surrealdb",
        "logfire",
        "registry"
    ]
}

def get_mcp_info() -> dict:
    """Get information about MCP integration capabilities."""
    return {
        "package_version": __version__,
        "supported_mcp_versions": SUPPORTED_MCP_VERSIONS,
        "default_config": DEFAULT_MCP_CONFIG,
        "server_categories": MCP_SERVER_CATEGORIES,
        "total_supported_servers": sum(len(servers) for servers in MCP_SERVER_CATEGORIES.values())
    }
