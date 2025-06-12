"""
Tool Manager for Agentical

This module provides the ToolManager class that coordinates all tool-related
operations, managing the lifecycle and integration between tool registry,
MCP client, and tool executor.

Features:
- High-level tool system coordination
- Tool lifecycle management (discovery, registration, execution)
- MCP server integration and management
- Performance monitoring and health checks
- Configuration management and templates
- Integration with workflow and agent systems
- Scheduling and queue management for tool operations
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set
from enum import Enum
from contextlib import asynccontextmanager
import logging

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    ConfigurationError
)
from ...core.logging import log_operation
from ...db.models.tool import Tool, ToolType, ToolStatus
from ...db.repositories.tool import AsyncToolRepository
from .tool_registry import ToolRegistry, ToolRegistryEntry, ToolDiscoveryMode
from ..mcp.mcp_client import MCPClient
from ..execution.tool_executor import ToolExecutor, ToolExecutionResult, ExecutionMode, ExecutionPriority


class ToolManagerState(Enum):
    """Tool manager operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class ToolManagerConfig:
    """Configuration for tool manager."""

    def __init__(
        self,
        mcp_config_path: str = "mcp-servers.json",
        max_concurrent_executions: int = 50,
        default_timeout_seconds: int = 300,
        discovery_mode: ToolDiscoveryMode = ToolDiscoveryMode.HYBRID,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 30,
        health_check_interval_minutes: int = 5,
        auto_reconnect: bool = True,
        max_mcp_connections: int = 20
    ):
        self.mcp_config_path = mcp_config_path
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout_seconds = default_timeout_seconds
        self.discovery_mode = discovery_mode
        self.enable_caching = enable_caching
        self.cache_ttl_minutes = cache_ttl_minutes
        self.health_check_interval_minutes = health_check_interval_minutes
        self.auto_reconnect = auto_reconnect
        self.max_mcp_connections = max_mcp_connections

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mcp_config_path": self.mcp_config_path,
            "max_concurrent_executions": self.max_concurrent_executions,
            "default_timeout_seconds": self.default_timeout_seconds,
            "discovery_mode": self.discovery_mode.value,
            "enable_caching": self.enable_caching,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "health_check_interval_minutes": self.health_check_interval_minutes,
            "auto_reconnect": self.auto_reconnect,
            "max_mcp_connections": self.max_mcp_connections
        }


class ToolManager:
    """
    High-level tool system coordinator.

    Manages the complete tool ecosystem including MCP integration,
    tool discovery, execution, and performance monitoring.
    """

    def __init__(
        self,
        db_session,
        config: Optional[ToolManagerConfig] = None
    ):
        """Initialize the tool manager."""
        self.db_session = db_session
        self.config = config or ToolManagerConfig()

        # Manager state
        self.state = ToolManagerState.INITIALIZING
        self.start_time = None
        self.shutdown_requested = False

        # Core components (initialized in initialize())
        self.mcp_client: Optional[MCPClient] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_executor: Optional[ToolExecutor] = None
        self.tool_repo = AsyncToolRepository(db_session)

        # Performance tracking
        self.metrics = {
            "tools_discovered": 0,
            "tools_registered": 0,
            "tools_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "mcp_servers_connected": 0,
            "uptime_seconds": 0,
            "health_checks_performed": 0
        }

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        # Tool operation queue
        self.operation_queue: asyncio.Queue = asyncio.Queue()
        self.queue_processors = 3  # Number of queue processing tasks

        logging.info(f"Tool manager initialized with config: {self.config.to_dict()}")

    async def initialize(self) -> None:
        """Initialize the tool manager and all components."""
        try:
            self.state = ToolManagerState.INITIALIZING
            logging.info("Initializing tool manager")

            # Initialize MCP client
            self.mcp_client = MCPClient(max_connections=self.config.max_mcp_connections)

            # Load MCP servers from configuration
            servers_loaded = await self.mcp_client.load_servers_from_config(self.config.mcp_config_path)
            self.metrics["mcp_servers_connected"] = servers_loaded

            logging.info(f"Loaded {servers_loaded} MCP servers")

            # Initialize tool registry
            self.tool_registry = ToolRegistry(
                db_session=self.db_session,
                mcp_client=self.mcp_client,
                discovery_mode=self.config.discovery_mode,
                enable_caching=self.config.enable_caching,
                cache_ttl_minutes=self.config.cache_ttl_minutes
            )

            await self.tool_registry.initialize()

            # Get discovery metrics
            registry_metrics = await self.tool_registry.get_registry_metrics()
            self.metrics["tools_discovered"] = registry_metrics.get("mcp_tools", 0)
            self.metrics["tools_registered"] = registry_metrics.get("total_tools", 0)

            # Initialize tool executor
            self.tool_executor = ToolExecutor(
                tool_registry=self.tool_registry,
                mcp_client=self.mcp_client,
                max_concurrent_executions=self.config.max_concurrent_executions,
                default_timeout_seconds=self.config.default_timeout_seconds,
                enable_monitoring=True
            )

            # Start background tasks
            await self._start_background_tasks()

            # Set state to running
            self.state = ToolManagerState.RUNNING
            self.start_time = datetime.utcnow()

            logging.info("Tool manager initialization complete")

        except Exception as e:
            logging.error(f"Tool manager initialization failed: {str(e)}")
            self.state = ToolManagerState.STOPPED
            raise ConfigurationError(f"Failed to initialize tool manager: {str(e)}")

    async def execute_tool(
        self,
        tool_identifier: Union[str, int],
        parameters: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.ASYNC,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout_seconds: Optional[int] = None,
        retry_attempts: int = 3,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """
        Execute a tool through the tool executor.

        Args:
            tool_identifier: Tool ID or name
            parameters: Tool execution parameters
            mode: Execution mode
            priority: Execution priority
            timeout_seconds: Custom timeout
            retry_attempts: Number of retry attempts
            user_context: Additional context

        Returns:
            ToolExecutionResult: Execution result

        Raises:
            ToolError: If tool system is not ready
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If execution fails
        """
        if self.state != ToolManagerState.RUNNING:
            raise ToolError("Tool manager is not running")

        try:
            result = await self.tool_executor.execute_tool(
                tool_identifier=tool_identifier,
                parameters=parameters,
                mode=mode,
                priority=priority,
                timeout_seconds=timeout_seconds,
                retry_attempts=retry_attempts,
                user_context=user_context
            )

            # Update metrics
            self.metrics["tools_executed"] += 1
            if result.success:
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1

            return result

        except Exception as e:
            self.metrics["failed_executions"] += 1
            logging.error(f"Tool execution failed: {tool_identifier}: {str(e)}")
            raise

    async def batch_execute_tools(
        self,
        tool_requests: List[Dict[str, Any]],
        max_parallel: int = 10,
        fail_fast: bool = False
    ) -> List[ToolExecutionResult]:
        """Execute multiple tools in batch."""
        if self.state != ToolManagerState.RUNNING:
            raise ToolError("Tool manager is not running")

        results = await self.tool_executor.batch_execute(
            tool_requests=tool_requests,
            max_parallel=max_parallel,
            fail_fast=fail_fast
        )

        # Update metrics
        for result in results:
            self.metrics["tools_executed"] += 1
            if result.success:
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1

        return results

    async def discover_tools(self, server_name: Optional[str] = None) -> Dict[str, List[str]]:
        """Discover tools from MCP servers."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        return await self.tool_registry.refresh_mcp_tools(server_name)

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
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        return await self.tool_registry.search_tools(
            query=query,
            tool_type=tool_type,
            capability=capability,
            mcp_server=mcp_server,
            status=status,
            limit=limit
        )

    async def get_tool_info(self, tool_identifier: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        entry = await self.tool_registry.get_tool(tool_identifier)
        return entry.to_dict() if entry else None

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        results = await self.tool_registry.search_tools(status=ToolStatus.AVAILABLE)
        return [entry.to_dict() for entry in results]

    async def get_tools_by_server(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools from a specific MCP server."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        results = await self.tool_registry.get_tools_by_server(server_name)
        return [entry.to_dict() for entry in results]

    async def get_tools_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get tools that have a specific capability."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        results = await self.tool_registry.get_tools_by_capability(capability)
        return [entry.to_dict() for entry in results]

    async def enable_tool(self, tool_identifier: Union[str, int]) -> bool:
        """Enable a tool."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        return await self.tool_registry.enable_tool(tool_identifier)

    async def disable_tool(self, tool_identifier: Union[str, int]) -> bool:
        """Disable a tool."""
        if not self.tool_registry:
            raise ToolError("Tool registry not initialized")

        return await self.tool_registry.disable_tool(tool_identifier)

    async def get_mcp_server_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers."""
        if not self.mcp_client:
            raise ToolError("MCP client not initialized")

        health_check = await self.mcp_client.health_check_all()
        metrics = self.mcp_client.get_client_metrics()

        return {
            "health_status": health_check,
            "client_metrics": metrics,
            "server_details": {
                name: server.get_metrics()
                for name, server in self.mcp_client.servers.items()
            }
        }

    async def reconnect_mcp_servers(self) -> int:
        """Attempt to reconnect failed MCP servers."""
        if not self.mcp_client:
            raise ToolError("MCP client not initialized")

        return await self.mcp_client.reconnect_failed_servers()

    async def cancel_tool_execution(self, execution_id: str) -> bool:
        """Cancel a running tool execution."""
        if not self.tool_executor:
            raise ToolError("Tool executor not initialized")

        return await self.tool_executor.cancel_execution(execution_id)

    async def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get all active tool executions."""
        if not self.tool_executor:
            raise ToolError("Tool executor not initialized")

        return await self.tool_executor.get_active_executions()

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution."""
        if not self.tool_executor:
            raise ToolError("Tool executor not initialized")

        return await self.tool_executor.get_execution_status(execution_id)

    async def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive tool manager status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        self.metrics["uptime_seconds"] = uptime

        status = {
            "state": self.state.value,
            "uptime_seconds": uptime,
            "metrics": self.metrics,
            "config": self.config.to_dict(),
            "background_tasks": len(self.background_tasks),
            "queue_size": self.operation_queue.qsize()
        }

        # Add component status if initialized
        if self.mcp_client:
            status["mcp_status"] = await self.get_mcp_server_status()

        if self.tool_registry:
            status["registry_metrics"] = await self.tool_registry.get_registry_metrics()

        if self.tool_executor:
            status["executor_metrics"] = await self.tool_executor.get_executor_metrics()

        return status

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {
                "manager_state": self.state == ToolManagerState.RUNNING,
                "mcp_client": self.mcp_client is not None,
                "tool_registry": self.tool_registry is not None,
                "tool_executor": self.tool_executor is not None,
                "background_tasks": len(self.background_tasks) > 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add component health checks
        if self.tool_executor:
            executor_health = await self.tool_executor.health_check()
            health["executor_health"] = executor_health
            if executor_health["status"] != "healthy":
                health["status"] = "degraded"

        # Check MCP server health
        if self.mcp_client:
            mcp_health = await self.mcp_client.health_check_all()
            healthy_servers = sum(1 for h in mcp_health.values() if h)
            total_servers = len(mcp_health)

            health["mcp_health"] = {
                "healthy_servers": healthy_servers,
                "total_servers": total_servers,
                "details": mcp_health
            }

            if healthy_servers == 0 and total_servers > 0:
                health["status"] = "unhealthy"
            elif healthy_servers < total_servers:
                health["status"] = "degraded"

        # Overall health determination
        if not all(health["checks"].values()):
            health["status"] = "unhealthy"

        self.metrics["health_checks_performed"] += 1
        return health

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)

        # Queue processing tasks
        for i in range(self.queue_processors):
            queue_task = asyncio.create_task(self._queue_processor_loop(f"processor_{i}"))
            self.background_tasks.add(queue_task)

        # Cache cleanup task
        cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self.background_tasks.add(cleanup_task)

        # MCP reconnection task (if auto-reconnect enabled)
        if self.config.auto_reconnect:
            reconnect_task = asyncio.create_task(self._mcp_reconnection_loop())
            self.background_tasks.add(reconnect_task)

        logging.info(f"Started {len(self.background_tasks)} background tasks")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self.shutdown_requested:
            try:
                await self.health_check()
                await asyncio.sleep(self.config.health_check_interval_minutes * 60)
            except Exception as e:
                logging.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(30)  # Back off on error

    async def _queue_processor_loop(self, processor_name: str) -> None:
        """Background queue processing loop."""
        while not self.shutdown_requested:
            try:
                # Get operation from queue (with timeout)
                try:
                    operation = await asyncio.wait_for(self.operation_queue.get(), timeout=10.0)
                    await self._process_queue_operation(operation)
                    self.operation_queue.task_done()
                except asyncio.TimeoutError:
                    continue  # No operations in queue
            except Exception as e:
                logging.error(f"Queue processor {processor_name} error: {str(e)}")
                await asyncio.sleep(5)

    async def _process_queue_operation(self, operation: Dict[str, Any]) -> None:
        """Process a queued operation."""
        operation_type = operation.get("type")

        if operation_type == "tool_discovery":
            server_name = operation.get("server_name")
            await self.discover_tools(server_name)
        elif operation_type == "cache_cleanup":
            if self.tool_registry:
                await self.tool_registry.cleanup_cache()
        elif operation_type == "metrics_update":
            # Update internal metrics
            pass
        else:
            logging.warning(f"Unknown queue operation type: {operation_type}")

    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop."""
        while not self.shutdown_requested:
            try:
                # Queue cache cleanup operation
                await self.operation_queue.put({"type": "cache_cleanup"})
                await asyncio.sleep(1800)  # Clean every 30 minutes
            except Exception as e:
                logging.error(f"Cache cleanup loop error: {str(e)}")
                await asyncio.sleep(600)

    async def _mcp_reconnection_loop(self) -> None:
        """Background MCP server reconnection loop."""
        while not self.shutdown_requested:
            try:
                if self.mcp_client:
                    reconnected = await self.mcp_client.reconnect_failed_servers()
                    if reconnected > 0:
                        logging.info(f"Reconnected {reconnected} MCP servers")
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logging.error(f"MCP reconnection loop error: {str(e)}")
                await asyncio.sleep(600)

    async def queue_operation(self, operation: Dict[str, Any]) -> None:
        """Queue an operation for background processing."""
        await self.operation_queue.put(operation)

    async def pause(self) -> None:
        """Pause the tool manager."""
        self.state = ToolManagerState.PAUSED
        logging.info("Tool manager paused")

    async def resume(self) -> None:
        """Resume the tool manager."""
        self.state = ToolManagerState.RUNNING
        logging.info("Tool manager resumed")

    async def shutdown(self) -> None:
        """Shutdown the tool manager gracefully."""
        logging.info("Tool manager shutting down")
        self.state = ToolManagerState.SHUTTING_DOWN
        self.shutdown_requested = True

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Shutdown components
        if self.tool_executor:
            await self.tool_executor.shutdown()

        if self.tool_registry:
            await self.tool_registry.shutdown()

        if self.mcp_client:
            await self.mcp_client.shutdown()

        self.state = ToolManagerState.STOPPED
        logging.info("Tool manager shutdown complete")

    def __repr__(self) -> str:
        """String representation of the tool manager."""
        return (
            f"ToolManager(state={self.state.value}, "
            f"tools_executed={self.metrics['tools_executed']}, "
            f"mcp_servers={self.metrics['mcp_servers_connected']})"
        )


class ToolManagerFactory:
    """Factory for creating tool manager instances."""

    @staticmethod
    def create_manager(
        db_session,
        config: Optional[Dict[str, Any]] = None
    ) -> ToolManager:
        """Create a tool manager with the given configuration."""
        manager_config = ToolManagerConfig()

        if config:
            # Update config with provided values
            for key, value in config.items():
                if hasattr(manager_config, key):
                    setattr(manager_config, key, value)

        manager = ToolManager(
            db_session=db_session,
            config=manager_config
        )

        logging.info("Tool manager created via factory")
        return manager

    @staticmethod
    def create_from_file(
        db_session,
        config_file_path: str
    ) -> ToolManager:
        """Create a tool manager from configuration file."""
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)

            return ToolManagerFactory.create_manager(db_session, config_data)

        except Exception as e:
            raise ConfigurationError(f"Failed to load tool manager config from {config_file_path}: {str(e)}")
