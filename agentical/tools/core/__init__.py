"""
Core Tools Package for Agentical

This package contains the core tool management components including
the tool registry, tool manager, and foundational classes for the
Agentical tool system.

Features:
- ToolRegistry: Tool discovery, registration, and management
- ToolManager: High-level tool system coordination
- Tool lifecycle management and configuration
- Integration with MCP servers and databases
- Performance monitoring and caching
"""

from .tool_registry import (
    ToolRegistry,
    ToolRegistryEntry,
    ToolDiscoveryMode
)

from .tool_manager import (
    ToolManager,
    ToolManagerFactory,
    ToolManagerConfig,
    ToolManagerState
)

__all__ = [
    # Registry components
    "ToolRegistry",
    "ToolRegistryEntry",
    "ToolDiscoveryMode",

    # Manager components
    "ToolManager",
    "ToolManagerFactory",
    "ToolManagerConfig",
    "ToolManagerState"
]

# Package metadata
__version__ = "1.0.0"
__description__ = "Core tool management system for Agentical"
