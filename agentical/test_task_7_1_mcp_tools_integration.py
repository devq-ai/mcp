"""
Test Suite for Task 7.1: MCP Server Tools Integration

This module contains comprehensive tests for the MCP tools integration functionality,
including MCP client, tool registry, tool executor, and tool manager components.

Test Coverage:
- MCPClient connection and tool discovery
- ToolRegistry registration and search functionality
- ToolExecutor execution with various modes and priorities
- ToolManager high-level coordination
- Error handling and performance monitoring
- Integration between all components
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from tools.mcp.mcp_client import MCPClient, MCPServer, MCPToolSchema, MCPConnectionStatus
from tools.core.tool_registry import ToolRegistry, ToolRegistryEntry, ToolDiscoveryMode
from tools.execution.tool_executor import ToolExecutor, ToolExecutionResult, ExecutionMode, ExecutionPriority
from tools.core.tool_manager import ToolManager, ToolManagerConfig, ToolManagerState
from db.models.tool import Tool, ToolType, ToolStatus
from core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError
)


class TestMCPToolSchema:
    """Test suite for MCPToolSchema."""

    def test_schema_creation(self):
        """Test MCPToolSchema creation."""
        schema = MCPToolSchema(
            name="test_tool",
            description="Test tool description",
            parameters={
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"}
            },
            required=["param1"],
            returns={"type": "object"}
        )

        assert schema.name == "test_tool"
        assert schema.description == "Test tool description"
        assert "param1" in schema.parameters
        assert "param1" in schema.required
        assert "param2" not in schema.required

    def test_parameter_validation_success(self):
        """Test successful parameter validation."""
        schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            parameters={
                "name": {"type": "string"},
                "count": {"type": "integer"}
            },
            required=["name"]
        )

        # Valid parameters
        params = {"name": "test", "count": 5}
        assert schema.validate_parameters(params) is True

    def test_parameter_validation_missing_required(self):
        """Test parameter validation with missing required parameter."""
        schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            parameters={
                "name": {"type": "string"},
                "count": {"type": "integer"}
            },
            required=["name"]
        )

        # Missing required parameter
        params = {"count": 5}
        with pytest.raises(ToolValidationError, match="Required parameter 'name' missing"):
            schema.validate_parameters(params)

    def test_parameter_validation_wrong_type(self):
        """Test parameter validation with wrong type."""
        schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            parameters={
                "name": {"type": "string"},
                "count": {"type": "integer"}
            },
            required=["name"]
        )

        # Wrong type for count
        params = {"name": "test", "count": "not_a_number"}
        with pytest.raises(ToolValidationError, match="Parameter 'count' has invalid type"):
            schema.validate_parameters(params)

    def test_schema_serialization(self):
        """Test schema serialization to dictionary."""
        schema = MCPToolSchema(
            name="test_tool",
            description="Test tool",
            parameters={"param1": {"type": "string"}},
            required=["param1"]
        )

        schema_dict = schema.to_dict()
        assert "name" in schema_dict
        assert "description" in schema_dict
        assert "parameters" in schema_dict
        assert "required" in schema_dict
        assert "created_at" in schema_dict


class TestMCPServer:
    """Test suite for MCPServer."""

    def test_server_creation(self):
        """Test MCP server creation."""
        server = MCPServer(
            name="test_server",
            command="python",
            args=["-m", "test_module"],
            description="Test server",
            env={"TEST_ENV": "value"}
        )

        assert server.name == "test_server"
        assert server.command == "python"
        assert server.args == ["-m", "test_module"]
        assert server.status == MCPConnectionStatus.DISCONNECTED
        assert server.env["TEST_ENV"] == "value"

    def test_server_metrics(self):
        """Test server metrics collection."""
        server = MCPServer(
            name="test_server",
            command="python",
            args=["-m", "test_module"]
        )

        # Initial metrics
        metrics = server.get_metrics()
        assert metrics["total_executions"] == 0
        assert metrics["successful_executions"] == 0
        assert metrics["success_rate"] == 0

        # Update metrics
        server.successful_executions = 8
        server.total_executions = 10
        metrics = server.get_metrics()
        assert metrics["success_rate"] == 80.0

    def test_server_serialization(self):
        """Test server serialization."""
        server = MCPServer(
            name="test_server",
            command="python",
            args=["-m", "test_module"],
            description="Test server"
        )

        server_dict = server.to_dict()
        assert "name" in server_dict
        assert "command" in server_dict
        assert "args" in server_dict
        assert "status" in server_dict
        assert "tools" in server_dict
        assert "metrics" in server_dict


class TestMCPClient:
    """Test suite for MCPClient."""

    @pytest.fixture
    def mcp_client(self):
        """Create an MCP client for testing."""
        return MCPClient(max_connections=5)

    def test_client_initialization(self, mcp_client):
        """Test MCP client initialization."""
        assert mcp_client.max_connections == 5
        assert len(mcp_client.servers) == 0
        assert len(mcp_client.connection_pool) == 0

    @pytest.mark.asyncio
    async def test_add_server(self, mcp_client):
        """Test adding a server to the client."""
        success = await mcp_client.add_server(
            name="test_server",
            command="echo",
            args=["hello"],
            description="Test server",
            auto_connect=False
        )

        assert success is True
        assert "test_server" in mcp_client.servers
        assert mcp_client.servers["test_server"].name == "test_server"

    @pytest.mark.asyncio
    async def test_duplicate_server(self, mcp_client):
        """Test adding duplicate server."""
        await mcp_client.add_server(
            name="test_server",
            command="echo",
            args=["hello"],
            auto_connect=False
        )

        # Try to add duplicate
        success = await mcp_client.add_server(
            name="test_server",
            command="echo",
            args=["hello"],
            auto_connect=False
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_discover_tools(self, mcp_client):
        """Test tool discovery."""
        # Add mock server
        await mcp_client.add_server(
            name="filesystem",
            command="echo",
            args=["hello"],
            auto_connect=False
        )

        # Mock server tools
        server = mcp_client.servers["filesystem"]
        server.tools = {
            "read_file": MCPToolSchema("read_file", "Read a file", {"path": {"type": "string"}}, ["path"]),
            "write_file": MCPToolSchema("write_file", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
        }
        server.status = MCPConnectionStatus.CONNECTED

        discovered = await mcp_client.discover_tools()
        assert "filesystem" in discovered
        assert "read_file" in discovered["filesystem"]
        assert "write_file" in discovered["filesystem"]

    def test_client_metrics(self, mcp_client):
        """Test client metrics."""
        metrics = mcp_client.get_client_metrics()
        assert "total_servers" in metrics
        assert "connected_servers" in metrics
        assert "total_tools" in metrics
        assert "success_rate" in metrics

    @pytest.mark.asyncio
    async def test_client_shutdown(self, mcp_client):
        """Test client shutdown."""
        await mcp_client.add_server(
            name="test_server",
            command="echo",
            args=["hello"],
            auto_connect=False
        )

        await mcp_client.shutdown()
        assert len(mcp_client.servers) == 0
        assert len(mcp_client.connection_pool) == 0


class TestToolRegistry:
    """Test suite for ToolRegistry."""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    async def mock_mcp_client(self):
        """Create mock MCP client."""
        client = AsyncMock(spec=MCPClient)
        client.servers = {}
        client.discover_tools.return_value = {}
        return client

    @pytest.fixture
    async def tool_registry(self, mock_db_session, mock_mcp_client):
        """Create tool registry for testing."""
        registry = ToolRegistry(
            db_session=mock_db_session,
            mcp_client=mock_mcp_client,
            discovery_mode=ToolDiscoveryMode.MANUAL,
            enable_caching=True
        )
        return registry

    @pytest.mark.asyncio
    async def test_registry_initialization(self, tool_registry):
        """Test registry initialization."""
        # Mock repository methods
        tool_registry.tool_repo.get_available_tools = AsyncMock(return_value=[])

        await tool_registry.initialize()
        assert len(tool_registry._registry) == 0

    @pytest.mark.asyncio
    async def test_register_tool(self, tool_registry, mock_db_session):
        """Test tool registration."""
        # Create mock tool
        tool = Tool(
            id=1,
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.CUSTOM,
            status=ToolStatus.AVAILABLE
        )

        # Mock repository methods
        tool_registry.tool_repo.get_available_tools = AsyncMock(return_value=[])
        await tool_registry.initialize()

        tool_id = await tool_registry.register_tool(tool)
        assert tool_id == "1"
        assert "1" in tool_registry._registry
        assert tool_registry._tool_index["test_tool"] == "1"

    @pytest.mark.asyncio
    async def test_get_tool(self, tool_registry, mock_db_session):
        """Test getting a tool from registry."""
        # Create and register tool
        tool = Tool(
            id=1,
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.CUSTOM,
            status=ToolStatus.AVAILABLE
        )

        tool_registry.tool_repo.get_available_tools = AsyncMock(return_value=[])
        await tool_registry.initialize()
        await tool_registry.register_tool(tool)

        # Get by ID
        entry = await tool_registry.get_tool(1)
        assert entry is not None
        assert entry.tool.name == "test_tool"

        # Get by name
        entry = await tool_registry.get_tool("test_tool")
        assert entry is not None
        assert entry.tool.id == 1

    @pytest.mark.asyncio
    async def test_search_tools(self, tool_registry, mock_db_session):
        """Test tool search functionality."""
        # Create and register multiple tools
        tools = [
            Tool(id=1, name="file_reader", tool_type=ToolType.FILESYSTEM, status=ToolStatus.AVAILABLE),
            Tool(id=2, name="git_status", tool_type=ToolType.GIT, status=ToolStatus.AVAILABLE),
            Tool(id=3, name="file_writer", tool_type=ToolType.FILESYSTEM, status=ToolStatus.AVAILABLE)
        ]

        tool_registry.tool_repo.get_available_tools = AsyncMock(return_value=[])
        await tool_registry.initialize()

        for tool in tools:
            await tool_registry.register_tool(tool)

        # Search by tool type
        results = await tool_registry.search_tools(tool_type=ToolType.FILESYSTEM)
        assert len(results) == 2
        assert all(entry.tool.tool_type == ToolType.FILESYSTEM for entry in results)

        # Search by query
        results = await tool_registry.search_tools(query="file")
        assert len(results) == 2
        assert all("file" in entry.tool.name for entry in results)

    @pytest.mark.asyncio
    async def test_registry_metrics(self, tool_registry, mock_db_session):
        """Test registry metrics collection."""
        tool_registry.tool_repo.get_available_tools = AsyncMock(return_value=[])
        await tool_registry.initialize()

        metrics = await tool_registry.get_registry_metrics()
        assert "total_tools" in metrics
        assert "enabled_tools" in metrics
        assert "mcp_tools" in metrics
        assert "cache_hit_rate" in metrics


class TestToolExecutor:
    """Test suite for ToolExecutor."""

    @pytest.fixture
    async def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = AsyncMock(spec=ToolRegistry)
        return registry

    @pytest.fixture
    async def mock_mcp_client(self):
        """Create mock MCP client."""
        client = AsyncMock(spec=MCPClient)
        return client

    @pytest.fixture
    async def tool_executor(self, mock_tool_registry, mock_mcp_client):
        """Create tool executor for testing."""
        return ToolExecutor(
            tool_registry=mock_tool_registry,
            mcp_client=mock_mcp_client,
            max_concurrent_executions=10
        )

    @pytest.fixture
    def mock_tool_entry(self):
        """Create mock tool registry entry."""
        tool = Tool(
            id=1,
            name="test_tool",
            tool_type=ToolType.CUSTOM,
            status=ToolStatus.AVAILABLE
        )

        entry = ToolRegistryEntry(
            tool=tool,
            schema=None,
            mcp_server=None,
            capabilities=[],
            parameters=[],
            metadata={}
        )
        entry.is_enabled = True
        return entry

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tool_executor, mock_tool_registry, mock_tool_entry):
        """Test successful tool execution."""
        # Mock registry response
        mock_tool_registry.get_tool.return_value = mock_tool_entry
        mock_tool_registry.update_tool_usage = AsyncMock()

        # Execute tool
        result = await tool_executor.execute_tool(
            tool_identifier=1,
            parameters={"param1": "value1"},
            mode=ExecutionMode.ASYNC
        )

        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.execution_id is not None
        mock_tool_registry.update_tool_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, tool_executor, mock_tool_registry):
        """Test tool execution with tool not found."""
        mock_tool_registry.get_tool.return_value = None

        with pytest.raises(ToolNotFoundError):
            await tool_executor.execute_tool(
                tool_identifier="nonexistent_tool",
                parameters={}
            )

    @pytest.mark.asyncio
    async def test_execute_tool_disabled(self, tool_executor, mock_tool_registry, mock_tool_entry):
        """Test tool execution with disabled tool."""
        mock_tool_entry.is_enabled = False
        mock_tool_registry.get_tool.return_value = mock_tool_entry

        with pytest.raises(ToolExecutionError, match="is disabled"):
            await tool_executor.execute_tool(
                tool_identifier=1,
                parameters={}
            )

    @pytest.mark.asyncio
    async def test_batch_execute(self, tool_executor, mock_tool_registry, mock_tool_entry):
        """Test batch tool execution."""
        mock_tool_registry.get_tool.return_value = mock_tool_entry
        mock_tool_registry.update_tool_usage = AsyncMock()

        requests = [
            {"tool_identifier": 1, "parameters": {"param": "value1"}},
            {"tool_identifier": 1, "parameters": {"param": "value2"}},
            {"tool_identifier": 1, "parameters": {"param": "value3"}}
        ]

        results = await tool_executor.batch_execute(requests, max_parallel=2)
        assert len(results) == 3
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_cancel_execution(self, tool_executor):
        """Test execution cancellation."""
        execution_id = "test_execution_123"

        # Mock active execution
        mock_context = MagicMock()
        mock_context.is_cancelled = False
        tool_executor._active_executions[execution_id] = mock_context

        success = await tool_executor.cancel_execution(execution_id)
        assert success is True
        assert mock_context.is_cancelled is True

    @pytest.mark.asyncio
    async def test_executor_metrics(self, tool_executor):
        """Test executor metrics collection."""
        metrics = await tool_executor.get_executor_metrics()
        assert "total_executions" in metrics
        assert "successful_executions" in metrics
        assert "failed_executions" in metrics
        assert "active_executions" in metrics
        assert "success_rate" in metrics

    @pytest.mark.asyncio
    async def test_health_check(self, tool_executor, mock_mcp_client, mock_tool_registry):
        """Test executor health check."""
        mock_mcp_client.health_check_all.return_value = {"server1": True, "server2": False}
        mock_tool_registry.get_registry_metrics.return_value = {"total_tools": 10, "enabled_tools": 8}

        health = await tool_executor.health_check()
        assert "status" in health
        assert "mcp_servers" in health
        assert "registry_status" in health
        assert "executor_metrics" in health


class TestToolManager:
    """Test suite for ToolManager."""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def tool_manager_config(self):
        """Create tool manager configuration."""
        return ToolManagerConfig(
            mcp_config_path="test-mcp-servers.json",
            max_concurrent_executions=10,
            discovery_mode=ToolDiscoveryMode.MANUAL
        )

    @pytest.mark.asyncio
    async def test_manager_initialization(self, mock_db_session, tool_manager_config):
        """Test tool manager initialization."""
        with patch('tools.core.tool_manager.MCPClient') as mock_mcp_class, \
             patch('tools.core.tool_manager.ToolRegistry') as mock_registry_class, \
             patch('tools.core.tool_manager.ToolExecutor') as mock_executor_class:

            # Mock MCP client
            mock_mcp_client = AsyncMock()
            mock_mcp_client.load_servers_from_config.return_value = 5
            mock_mcp_class.return_value = mock_mcp_client

            # Mock registry
            mock_registry = AsyncMock()
            mock_registry.get_registry_metrics.return_value = {"total_tools": 10, "mcp_tools": 5}
            mock_registry_class.return_value = mock_registry

            # Mock executor
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor

            manager = ToolManager(mock_db_session, tool_manager_config)

            # Mock background tasks
            with patch.object(manager, '_start_background_tasks', new_callable=AsyncMock):
                await manager.initialize()

            assert manager.state == ToolManagerState.RUNNING
            assert manager.mcp_client == mock_mcp_client
            assert manager.tool_registry == mock_registry
            assert manager.tool_executor == mock_executor

    @pytest.mark.asyncio
    async def test_execute_tool_through_manager(self, mock_db_session, tool_manager_config):
        """Test tool execution through manager."""
        with patch('tools.core.tool_manager.MCPClient') as mock_mcp_class, \
             patch('tools.core.tool_manager.ToolRegistry') as mock_registry_class, \
             patch('tools.core.tool_manager.ToolExecutor') as mock_executor_class:

            # Setup mocks
            mock_mcp_client = AsyncMock()
            mock_mcp_client.load_servers_from_config.return_value = 0
            mock_mcp_class.return_value = mock_mcp_client

            mock_registry = AsyncMock()
            mock_registry.get_registry_metrics.return_value = {"total_tools": 0, "mcp_tools": 0}
            mock_registry_class.return_value = mock_registry

            mock_executor = AsyncMock()
            mock_result = ToolExecutionResult("exec_123", "test_tool", True, {"result": "success"})
            mock_executor.execute_tool.return_value = mock_result
            mock_executor_class.return_value = mock_executor

            manager = ToolManager(mock_db_session, tool_manager_config)

            with patch.object(manager, '_start_background_tasks', new_callable=AsyncMock):
                await manager.initialize()

            # Execute tool
            result = await manager.execute_tool(
                tool_identifier="test_tool",
                parameters={"param": "value"}
            )

            assert result.success is True
            assert result.tool_name == "test_tool"
            mock_executor.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_status(self, mock_db_session, tool_manager_config):
        """Test manager status reporting."""
        manager = ToolManager(mock_db_session, tool_manager_config)
        manager.state = ToolManagerState.RUNNING
        manager.start_time = datetime.utcnow()

        status = await manager.get_manager_status()
        assert "state" in status
        assert "uptime_seconds" in status
        assert "metrics" in status
        assert "config" in status
        assert status["state"] == "running"

    @pytest.mark.asyncio
    async def test_manager_health_check(self, mock_db_session, tool_manager_config):
        """Test manager health check."""
        manager = ToolManager(mock_db_session, tool_manager_config)
        manager.state = ToolManagerState.RUNNING

        health = await manager.health_check()
        assert "status" in health
        assert "checks" in health
        assert "timestamp" in health


class TestIntegration:
    """Integration tests for the complete tool system."""

    @pytest.mark.asyncio
    async def test_end_to_end_tool_execution(self):
        """Test complete end-to-end tool execution flow."""
        # This would be a comprehensive integration test
        # For now, we'll simulate the flow

        # 1. Initialize MCP client
        mcp_client = MCPClient(max_connections=5)

        # 2. Add mock server
        await mcp_client.add_server(
            name="test_server",
            command="echo",
            args=["hello"],
            auto_connect=False
        )

        # 3. Simulate tool discovery
        server = mcp_client.servers["test_server"]
        server.tools = {
            "echo_tool": MCPToolSchema(
                "echo_tool",
                "Echo text",
                {"text": {"type": "string"}},
                ["text"]
            )
        }
        server.status = MCPConnectionStatus.CONNECTED

        # 4. Create mock database session
        mock_db_session = AsyncMock()

        # 5. Initialize registry
        registry = ToolRegistry(
            db_session=mock_db_session,
            mcp_client=mcp_client,
            discovery_mode=ToolDiscoveryMode.MCP_ONLY
        )

        # Mock repository methods
        registry.tool_repo.get_available_tools = AsyncMock(return_value=[])
        registry.tool_repo.create = AsyncMock(return_value=Tool(
            id=1, name="test_server.echo_tool", tool_type=ToolType.CUSTOM, status=ToolStatus.AVAILABLE
        ))

        await registry.initialize()

        # 6. Create executor
        executor = ToolExecutor(
            tool_registry=registry,
            mcp_client=mcp_client,
            max_concurrent_executions=10
        )

        # Verify the integration
        assert len(mcp_client.servers) == 1
        assert "test_server" in mcp_client.servers
        assert len(server.tools) == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
