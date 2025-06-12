"""
Tool Registry for Agentical

This module provides the ToolRegistry class that manages tool discovery,
registration, and lifecycle management with comprehensive MCP integration.

Features:
- Dynamic tool discovery from MCP servers
- Tool registration and validation
- Tool capability mapping and search
- Performance monitoring and caching
- Integration with database and workflow systems
- Tool template and schema management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Type, Union, Callable
from enum import Enum
from pathlib import Path
import importlib
import inspect

from ...core.exceptions import (
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
    ConfigurationError
)
from ...core.logging import log_operation
from ...db.models.tool import (
    Tool,
    ToolType,
    ToolStatus,
    ToolCapability,
    ToolParameter
)
from ...db.repositories.tool import AsyncToolRepository
from ..mcp.mcp_client import MCPClient, MCPToolSchema


class ToolDiscoveryMode(Enum):
    """Tool discovery modes."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"
    MCP_ONLY = "mcp_only"


class ToolRegistryEntry:
    """Registry entry for a tool."""

    def __init__(
        self,
        tool: Tool,
        schema: Optional[MCPToolSchema],
        mcp_server: Optional[str],
        capabilities: List[ToolCapability],
        parameters: List[ToolParameter],
        metadata: Dict[str, Any]
    ):
        self.tool = tool
        self.schema = schema
        self.mcp_server = mcp_server
        self.capabilities = capabilities
        self.parameters = parameters
        self.metadata = metadata
        self.registered_at = datetime.utcnow()
        self.usage_count = 0
        self.last_used = None
        self.is_enabled = True
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "last_execution": None
        }

    def update_usage(self, execution_time: float, success: bool) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

        metrics = self.performance_metrics
        metrics["total_executions"] += 1
        metrics["last_execution"] = self.last_used.isoformat()

        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1

        # Update average execution time
        if metrics["total_executions"] == 1:
            metrics["average_execution_time"] = execution_time
        else:
            current_avg = metrics["average_execution_time"]
            new_avg = (current_avg * (metrics["total_executions"] - 1) + execution_time) / metrics["total_executions"]
            metrics["average_execution_time"] = new_avg

    def get_success_rate(self) -> float:
        """Get tool success rate percentage."""
        if self.performance_metrics["total_executions"] == 0:
            return 100.0
        return (self.performance_metrics["successful_executions"] / self.performance_metrics["total_executions"]) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry entry to dictionary."""
        return {
            "tool_id": self.tool.id,
            "tool_name": self.tool.name,
            "tool_type": self.tool.tool_type.value,
            "mcp_server": self.mcp_server,
            "capabilities": [cap.capability_name for cap in self.capabilities],
            "parameters": [param.name for param in self.parameters],
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_enabled": self.is_enabled,
            "performance_metrics": self.performance_metrics,
            "success_rate": self.get_success_rate()
        }


class ToolRegistry:
    """
    Central registry for tool discovery, registration, and management.

    Manages tool discovery from MCP servers, registration, validation,
    and provides search and access capabilities for the tool ecosystem.
    """

    def __init__(
        self,
        db_session,
        mcp_client: MCPClient,
        discovery_mode: ToolDiscoveryMode = ToolDiscoveryMode.HYBRID,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 30
    ):
        """Initialize the tool registry."""
        self.db_session = db_session
        self.mcp_client = mcp_client
        self.discovery_mode = discovery_mode
        self.enable_caching = enable_caching
        self.cache_ttl_minutes = cache_ttl_minutes

        # Registry storage
        self._registry: Dict[str, ToolRegistryEntry] = {}
        self._tool_index: Dict[str, str] = {}  # tool_name -> tool_id mapping
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> tool_ids
        self._mcp_server_index: Dict[str, Set[str]] = {}  # server -> tool_ids

        # Repository for database operations
        self.tool_repo = AsyncToolRepository(db_session)

        # Performance tracking
        self._metrics = {
            "registrations": 0,
            "discoveries": 0,
            "lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "mcp_integrations": 0,
            "errors": 0
        }

        # Cache for frequently accessed tools
        self._tool_cache: Dict[str, ToolRegistryEntry] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        # Tool templates and schemas
        self._tool_templates: Dict[str, Dict[str, Any]] = {}
        self._schema_validators: Dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Initialize the registry with tool discovery."""
        # Load existing tools from database
        await self._load_existing_tools()

        # Discover tools from MCP servers
        if self.discovery_mode in [ToolDiscoveryMode.AUTOMATIC, ToolDiscoveryMode.HYBRID, ToolDiscoveryMode.MCP_ONLY]:
            await self._discover_mcp_tools()

        # Load tool templates
        await self._load_tool_templates()

        # Build indices
        await self._rebuild_indices()

    async def _load_existing_tools(self) -> None:
        """Load existing tools from database."""
        try:
            existing_tools = await self.tool_repo.get_available_tools()

            for tool in existing_tools:
                # Load capabilities and parameters
                capabilities = await self.tool_repo.get_tool_capabilities(tool.id)
                parameters = tool.parameters if hasattr(tool, 'parameters') else []

                entry = ToolRegistryEntry(
                    tool=tool,
                    schema=None,  # Will be populated from MCP if available
                    mcp_server=None,
                    capabilities=capabilities,
                    parameters=parameters,
                    metadata=tool.get_configuration()
                )

                self._registry[str(tool.id)] = entry
                self._tool_index[tool.name] = str(tool.id)

        except Exception as e:
            raise ConfigurationError(f"Failed to load existing tools: {str(e)}")

    async def _discover_mcp_tools(self) -> None:
        """Discover tools from MCP servers."""
        try:
            discovered_tools = await self.mcp_client.discover_tools()

            for server_name, tool_names in discovered_tools.items():
                if server_name in self.mcp_client.servers:
                    server = self.mcp_client.servers[server_name]

                    for tool_name in tool_names:
                        if tool_name in server.tools:
                            await self._register_mcp_tool(server_name, tool_name, server.tools[tool_name])

            self._metrics["discoveries"] += len([tool for tools in discovered_tools.values() for tool in tools])

        except Exception as e:
            self._metrics["errors"] += 1
            raise ToolError(f"MCP tool discovery failed: {str(e)}")

    async def _register_mcp_tool(self, server_name: str, tool_name: str, schema: MCPToolSchema) -> str:
        """Register a tool discovered from MCP server."""
        # Check if tool already exists
        existing_tool_id = self._tool_index.get(f"{server_name}.{tool_name}")
        if existing_tool_id:
            # Update existing tool with MCP schema
            entry = self._registry[existing_tool_id]
            entry.schema = schema
            entry.mcp_server = server_name
            return existing_tool_id

        # Create new tool in database
        tool_data = {
            "name": f"{server_name}.{tool_name}",
            "description": schema.description,
            "tool_type": self._map_tool_type(server_name),
            "status": ToolStatus.AVAILABLE,
            "endpoint_url": f"mcp://{server_name}/{tool_name}",
            "configuration": {
                "mcp_server": server_name,
                "mcp_tool": tool_name,
                "parameters": schema.parameters,
                "required": schema.required
            }
        }

        # Create tool in database
        tool = await self.tool_repo.create(tool_data)

        # Create capabilities based on tool schema
        capabilities = await self._create_tool_capabilities(tool.id, schema)

        # Create parameters
        parameters = await self._create_tool_parameters(tool.id, schema)

        # Create registry entry
        entry = ToolRegistryEntry(
            tool=tool,
            schema=schema,
            mcp_server=server_name,
            capabilities=capabilities,
            parameters=parameters,
            metadata={"source": "mcp", "server": server_name}
        )

        # Register in memory
        tool_id = str(tool.id)
        self._registry[tool_id] = entry
        self._tool_index[tool.name] = tool_id

        # Update indices
        if server_name not in self._mcp_server_index:
            self._mcp_server_index[server_name] = set()
        self._mcp_server_index[server_name].add(tool_id)

        self._metrics["registrations"] += 1
        self._metrics["mcp_integrations"] += 1

        return tool_id

    def _map_tool_type(self, server_name: str) -> ToolType:
        """Map MCP server name to tool type."""
        mapping = {
            "filesystem": ToolType.FILESYSTEM,
            "git": ToolType.GIT,
            "memory": ToolType.MEMORY,
            "fetch": ToolType.FETCH,
            "ptolemies": ToolType.PTOLEMIES,
            "context7": ToolType.CONTEXT7,
            "bayes": ToolType.BAYES,
            "github": ToolType.GITHUB,
            "crawl4ai": ToolType.CRAWL4AI,
            "calendar": ToolType.CALENDAR,
            "stripe": ToolType.STRIPE,
            "jupyter": ToolType.JUPYTER,
            "shadcn-ui": ToolType.SHADCN_UI,
            "magic": ToolType.MAGIC,
            "surrealdb": ToolType.SURREALDB,
            "solver-z3": ToolType.SOLVER_Z3,
            "solver-pysat": ToolType.SOLVER_PYSAT,
            "solver-mzn": ToolType.SOLVER_MZN
        }

        return mapping.get(server_name, ToolType.CUSTOM)

    async def _create_tool_capabilities(self, tool_id: int, schema: MCPToolSchema) -> List[ToolCapability]:
        """Create tool capabilities based on schema."""
        capabilities = []

        # Infer capabilities from tool name and parameters
        capability_name = f"{schema.name}_capability"
        input_types = list(schema.parameters.keys())

        capability_data = {
            "tool_id": tool_id,
            "capability_name": capability_name,
            "description": f"Capability for {schema.name}",
            "input_types": input_types,
            "output_type": "object",
            "data_formats": ["json"]
        }

        # In a real implementation, this would create database records
        # For now, create mock capability objects
        capability = ToolCapability(**capability_data)
        capabilities.append(capability)

        return capabilities

    async def _create_tool_parameters(self, tool_id: int, schema: MCPToolSchema) -> List[ToolParameter]:
        """Create tool parameters based on schema."""
        parameters = []

        for param_name, param_config in schema.parameters.items():
            parameter_data = {
                "tool_id": tool_id,
                "name": param_name,
                "data_type": param_config.get("type", "string"),
                "is_required": param_name in schema.required,
                "description": param_config.get("description", ""),
                "default_value": param_config.get("default"),
                "validation_rules": param_config.get("validation", {})
            }

            # Create mock parameter objects
            parameter = ToolParameter(**parameter_data)
            parameters.append(parameter)

        return parameters

    async def register_tool(
        self,
        tool: Tool,
        capabilities: List[ToolCapability] = None,
        parameters: List[ToolParameter] = None,
        metadata: Dict[str, Any] = None,
        force: bool = False
    ) -> str:
        """Register a tool in the registry."""
        tool_id = str(tool.id)

        if tool_id in self._registry and not force:
            raise ToolValidationError(f"Tool {tool.name} is already registered")

        entry = ToolRegistryEntry(
            tool=tool,
            schema=None,
            mcp_server=None,
            capabilities=capabilities or [],
            parameters=parameters or [],
            metadata=metadata or {}
        )

        self._registry[tool_id] = entry
        self._tool_index[tool.name] = tool_id

        self._metrics["registrations"] += 1
        return tool_id

    async def get_tool(self, tool_identifier: Union[str, int]) -> Optional[ToolRegistryEntry]:
        """Get a tool by ID or name."""
        self._metrics["lookups"] += 1

        # Handle caching
        cache_key = str(tool_identifier)
        if self.enable_caching and cache_key in self._tool_cache:
            expiry = self._cache_expiry.get(cache_key)
            if expiry and datetime.utcnow() < expiry:
                self._metrics["cache_hits"] += 1
                return self._tool_cache[cache_key]

        self._metrics["cache_misses"] += 1

        # Look up tool
        tool_id = None
        if isinstance(tool_identifier, int) or tool_identifier.isdigit():
            tool_id = str(tool_identifier)
        else:
            tool_id = self._tool_index.get(tool_identifier)

        if tool_id and tool_id in self._registry:
            entry = self._registry[tool_id]

            # Cache the result
            if self.enable_caching:
                self._tool_cache[cache_key] = entry
                self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=self.cache_ttl_minutes)

            return entry

        return None

    async def search_tools(
        self,
        query: str = None,
        tool_type: ToolType = None,
        capability: str = None,
        mcp_server: str = None,
        status: ToolStatus = None,
        limit: int = 50
    ) -> List[ToolRegistryEntry]:
        """Search for tools based on criteria."""
        results = []

        for entry in self._registry.values():
            # Apply filters
            if tool_type and entry.tool.tool_type != tool_type:
                continue

            if capability and capability not in [cap.capability_name for cap in entry.capabilities]:
                continue

            if mcp_server and entry.mcp_server != mcp_server:
                continue

            if status and entry.tool.status != status:
                continue

            if query:
                # Simple text search in name and description
                search_text = f"{entry.tool.name} {entry.tool.description}".lower()
                if query.lower() not in search_text:
                    continue

            results.append(entry)

            if len(results) >= limit:
                break

        # Sort by usage count and success rate
        results.sort(key=lambda x: (x.usage_count, x.get_success_rate()), reverse=True)
        return results

    async def get_tools_by_capability(self, capability: str) -> List[ToolRegistryEntry]:
        """Get tools that have a specific capability."""
        tool_ids = self._capability_index.get(capability, set())
        return [self._registry[tool_id] for tool_id in tool_ids if tool_id in self._registry]

    async def get_tools_by_server(self, server_name: str) -> List[ToolRegistryEntry]:
        """Get tools from a specific MCP server."""
        tool_ids = self._mcp_server_index.get(server_name, set())
        return [self._registry[tool_id] for tool_id in tool_ids if tool_id in self._registry]

    async def update_tool_usage(self, tool_identifier: Union[str, int], execution_time: float, success: bool) -> None:
        """Update tool usage statistics."""
        entry = await self.get_tool(tool_identifier)
        if entry:
            entry.update_usage(execution_time, success)

            # Update database
            await self.tool_repo.track_usage(
                tool_id=entry.tool.id,
                execution_time=execution_time,
                success=success,
                context={"registry_update": True}
            )

    async def enable_tool(self, tool_identifier: Union[str, int]) -> bool:
        """Enable a tool."""
        entry = await self.get_tool(tool_identifier)
        if entry:
            entry.is_enabled = True
            entry.tool.status = ToolStatus.AVAILABLE
            await self.tool_repo.update(entry.tool.id, {"status": ToolStatus.AVAILABLE})
            return True
        return False

    async def disable_tool(self, tool_identifier: Union[str, int]) -> bool:
        """Disable a tool."""
        entry = await self.get_tool(tool_identifier)
        if entry:
            entry.is_enabled = False
            entry.tool.status = ToolStatus.UNAVAILABLE
            await self.tool_repo.update(entry.tool.id, {"status": ToolStatus.UNAVAILABLE})
            return True
        return False

    async def refresh_mcp_tools(self, server_name: str = None) -> int:
        """Refresh tools from MCP servers."""
        if server_name:
            # Refresh specific server
            if server_name in self.mcp_client.servers:
                await self.mcp_client.servers[server_name]._discover_tools()
                await self._register_server_tools(server_name)
                return len(self.mcp_client.servers[server_name].tools)
        else:
            # Refresh all servers
            total_refreshed = 0
            for srv_name in self.mcp_client.servers.keys():
                await self.mcp_client.servers[srv_name]._discover_tools()
                await self._register_server_tools(srv_name)
                total_refreshed += len(self.mcp_client.servers[srv_name].tools)
            return total_refreshed

        return 0

    async def _register_server_tools(self, server_name: str) -> None:
        """Register tools from a specific server."""
        if server_name in self.mcp_client.servers:
            server = self.mcp_client.servers[server_name]
            for tool_name, schema in server.tools.items():
                await self._register_mcp_tool(server_name, tool_name, schema)

    async def _rebuild_indices(self) -> None:
        """Rebuild search indices."""
        self._capability_index.clear()
        self._mcp_server_index.clear()

        for tool_id, entry in self._registry.items():
            # Build capability index
            for capability in entry.capabilities:
                if capability.capability_name not in self._capability_index:
                    self._capability_index[capability.capability_name] = set()
                self._capability_index[capability.capability_name].add(tool_id)

            # Build MCP server index
            if entry.mcp_server:
                if entry.mcp_server not in self._mcp_server_index:
                    self._mcp_server_index[entry.mcp_server] = set()
                self._mcp_server_index[entry.mcp_server].add(tool_id)

    async def _load_tool_templates(self) -> None:
        """Load tool templates for common patterns."""
        default_templates = {
            "file_operation": {
                "description": "Template for file operation tools",
                "capabilities": ["file_read", "file_write"],
                "common_parameters": ["path", "content", "encoding"]
            },
            "api_client": {
                "description": "Template for API client tools",
                "capabilities": ["http_request", "data_processing"],
                "common_parameters": ["url", "method", "headers", "payload"]
            },
            "data_processor": {
                "description": "Template for data processing tools",
                "capabilities": ["data_transform", "data_analysis"],
                "common_parameters": ["input_data", "format", "options"]
            },
            "ai_service": {
                "description": "Template for AI service tools",
                "capabilities": ["ai_processing", "model_inference"],
                "common_parameters": ["prompt", "model", "parameters"]
            }
        }

        self._tool_templates.update(default_templates)

    async def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry performance metrics."""
        total_tools = len(self._registry)
        enabled_tools = sum(1 for entry in self._registry.values() if entry.is_enabled)
        mcp_tools = sum(1 for entry in self._registry.values() if entry.mcp_server)

        return {
            **self._metrics,
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "mcp_tools": mcp_tools,
            "cache_size": len(self._tool_cache),
            "cache_hit_rate": (
                self._metrics["cache_hits"] / max(self._metrics["lookups"], 1)
            ) * 100,
            "success_rate": self._calculate_overall_success_rate(),
            "avg_execution_time": self._calculate_average_execution_time()
        }

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all tools."""
        total_executions = sum(entry.performance_metrics["total_executions"] for entry in self._registry.values())
        successful_executions = sum(entry.performance_metrics["successful_executions"] for entry in self._registry.values())

        if total_executions == 0:
            return 100.0
        return (successful_executions / total_executions) * 100

    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time across all tools."""
        execution_times = [
            entry.performance_metrics["average_execution_time"]
            for entry in self._registry.values()
            if entry.performance_metrics["total_executions"] > 0
        ]

        if not execution_times:
            return 0.0
        return sum(execution_times) / len(execution_times)

    async def cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        if not self.enable_caching:
            return

        now = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._cache_expiry.items()
            if expiry < now
        ]

        for key in expired_keys:
            self._tool_cache.pop(key, None)
            self._cache_expiry.pop(key, None)

    async def export_registry(self) -> Dict[str, Any]:
        """Export registry data for backup or analysis."""
        return {
            "tools": [entry.to_dict() for entry in self._registry.values()],
            "metrics": await self.get_registry_metrics(),
            "templates": self._tool_templates,
            "indices": {
                "capabilities": {cap: list(tool_ids) for cap, tool_ids in self._capability_index.items()},
                "mcp_servers": {server: list(tool_ids) for server, tool_ids in self._mcp_server_index.items()}
            },
            "exported_at": datetime.utcnow().isoformat()
        }

    async def shutdown(self) -> None:
        """Shutdown the registry and cleanup resources."""
        # Clear caches
        self._tool_cache.clear()
        self._cache_expiry.clear()

        # Clear indices
        self._capability_index.clear()
        self._mcp_server_index.clear()
        self._tool_index.clear()

        # Clear registry
        self._registry.clear()

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"ToolRegistry(tools={len(self._registry)}, "
            f"enabled={sum(1 for e in self._registry.values() if e.is_enabled)}, "
            f"mcp_servers={len(self._mcp_server_index)}, "
            f"discovery_mode={self.discovery_mode.value})"
        )
