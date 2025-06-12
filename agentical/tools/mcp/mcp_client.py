"""
MCP Client for Tool Integration

This module provides the MCP (Model Context Protocol) client for integrating
with external MCP servers and tools within the Agentical framework.

Features:
- MCP server connection and communication
- Tool discovery and registration from MCP servers
- Async tool execution with proper error handling
- Tool parameter validation and schema management
- Performance monitoring and observability
- Connection pooling and health checking
"""

import asyncio
import json
import subprocess
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set, Callable
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


class MCPConnectionStatus(Enum):
    """MCP server connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    TIMEOUT = "timeout"


class MCPToolSchema:
    """Schema definition for MCP tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str] = None,
        returns: Dict[str, Any] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or []
        self.returns = returns or {}
        self.created_at = datetime.utcnow()

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameters against schema."""
        # Check required parameters
        for req_param in self.required:
            if req_param not in params:
                raise ToolValidationError(f"Required parameter '{req_param}' missing")

        # Validate parameter types (simplified validation)
        for param_name, param_value in params.items():
            if param_name in self.parameters:
                expected_type = self.parameters[param_name].get('type')
                if expected_type and not self._validate_type(param_value, expected_type):
                    raise ToolValidationError(
                        f"Parameter '{param_name}' has invalid type. "
                        f"Expected {expected_type}, got {type(param_value).__name__}"
                    )

        return True

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type against expected type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "returns": self.returns,
            "created_at": self.created_at.isoformat()
        }


class MCPServer:
    """Represents an MCP server instance."""

    def __init__(
        self,
        name: str,
        command: str,
        args: List[str],
        description: str = "",
        env: Dict[str, str] = None,
        cwd: str = None
    ):
        self.name = name
        self.command = command
        self.args = args
        self.description = description
        self.env = env or {}
        self.cwd = cwd

        # Runtime state
        self.status = MCPConnectionStatus.DISCONNECTED
        self.process: Optional[subprocess.Popen] = None
        self.tools: Dict[str, MCPToolSchema] = {}
        self.last_health_check = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.average_execution_time = 0.0

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        if self.status == MCPConnectionStatus.CONNECTED:
            return True

        self.status = MCPConnectionStatus.CONNECTING
        self.connection_attempts += 1

        try:
            # Start MCP server process
            env = {**self.env} if self.env else None

            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.cwd,
                text=True
            )

            # Wait for process to start
            await asyncio.sleep(1)

            # Check if process is still running
            if self.process.poll() is None:
                self.status = MCPConnectionStatus.CONNECTED
                await self._discover_tools()
                logging.info(f"MCP server '{self.name}' connected successfully")
                return True
            else:
                error_output = self.process.stderr.read() if self.process.stderr else "Unknown error"
                logging.error(f"MCP server '{self.name}' failed to start: {error_output}")
                self.status = MCPConnectionStatus.ERROR
                return False

        except Exception as e:
            logging.error(f"Failed to connect to MCP server '{self.name}': {str(e)}")
            self.status = MCPConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                logging.error(f"Error disconnecting from MCP server '{self.name}': {str(e)}")

        self.status = MCPConnectionStatus.DISCONNECTED
        self.process = None
        logging.info(f"MCP server '{self.name}' disconnected")

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        # For now, we'll use a simplified tool discovery
        # In a real implementation, this would use the MCP protocol to query available tools

        # Simulate tool discovery based on server type
        mock_tools = self._get_mock_tools_for_server()

        for tool_name, tool_config in mock_tools.items():
            schema = MCPToolSchema(
                name=tool_name,
                description=tool_config.get('description', ''),
                parameters=tool_config.get('parameters', {}),
                required=tool_config.get('required', []),
                returns=tool_config.get('returns', {})
            )
            self.tools[tool_name] = schema

        logging.info(f"Discovered {len(self.tools)} tools from MCP server '{self.name}'")

    def _get_mock_tools_for_server(self) -> Dict[str, Dict[str, Any]]:
        """Get mock tools based on server name (for demonstration)."""
        tool_definitions = {
            "filesystem": {
                "read_file": {
                    "description": "Read content from a file",
                    "parameters": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                },
                "write_file": {
                    "description": "Write content to a file",
                    "parameters": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                },
                "list_directory": {
                    "description": "List directory contents",
                    "parameters": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            },
            "git": {
                "git_status": {
                    "description": "Get git repository status",
                    "parameters": {},
                    "required": []
                },
                "git_commit": {
                    "description": "Create a git commit",
                    "parameters": {
                        "message": {"type": "string", "description": "Commit message"}
                    },
                    "required": ["message"]
                },
                "git_push": {
                    "description": "Push commits to remote repository",
                    "parameters": {
                        "remote": {"type": "string", "description": "Remote name", "default": "origin"},
                        "branch": {"type": "string", "description": "Branch name"}
                    },
                    "required": ["branch"]
                }
            },
            "fetch": {
                "fetch_url": {
                    "description": "Fetch content from a URL",
                    "parameters": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                        "headers": {"type": "object", "description": "HTTP headers"}
                    },
                    "required": ["url"]
                }
            },
            "memory": {
                "store_memory": {
                    "description": "Store information in memory",
                    "parameters": {
                        "key": {"type": "string", "description": "Memory key"},
                        "value": {"type": "string", "description": "Value to store"}
                    },
                    "required": ["key", "value"]
                },
                "retrieve_memory": {
                    "description": "Retrieve information from memory",
                    "parameters": {
                        "key": {"type": "string", "description": "Memory key"}
                    },
                    "required": ["key"]
                }
            },
            "ptolemies": {
                "search_knowledge": {
                    "description": "Search the knowledge base",
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                    },
                    "required": ["query"]
                },
                "store_knowledge": {
                    "description": "Store knowledge in the database",
                    "parameters": {
                        "content": {"type": "string", "description": "Knowledge content"},
                        "tags": {"type": "array", "description": "Knowledge tags"}
                    },
                    "required": ["content"]
                }
            },
            "context7": {
                "analyze_context": {
                    "description": "Analyze contextual information",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to analyze"},
                        "context_type": {"type": "string", "description": "Type of context analysis"}
                    },
                    "required": ["text"]
                }
            }
        }

        return tool_definitions.get(self.name, {})

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on this MCP server."""
        if self.status != MCPConnectionStatus.CONNECTED:
            raise ToolExecutionError(f"MCP server '{self.name}' is not connected")

        if tool_name not in self.tools:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found on server '{self.name}'")

        # Validate parameters
        tool_schema = self.tools[tool_name]
        tool_schema.validate_parameters(parameters)

        start_time = datetime.utcnow()

        try:
            # In a real implementation, this would send the tool execution request
            # via the MCP protocol. For now, we'll simulate tool execution.
            result = await self._simulate_tool_execution(tool_name, parameters)

            # Track success
            self.successful_executions += 1
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_performance_metrics(execution_time)

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool": tool_name,
                "server": self.name
            }

        except Exception as e:
            self.failed_executions += 1
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_performance_metrics(execution_time)

            logging.error(f"Tool execution failed: {tool_name} on {self.name}: {str(e)}")
            raise ToolExecutionError(f"Tool execution failed: {str(e)}")

        finally:
            self.total_executions += 1

    async def _simulate_tool_execution(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Simulate tool execution (for demonstration purposes)."""
        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Return mock results based on tool type
        if tool_name == "read_file":
            return f"Content of file: {parameters.get('path', 'unknown')}"
        elif tool_name == "write_file":
            return f"Successfully wrote to: {parameters.get('path', 'unknown')}"
        elif tool_name == "list_directory":
            return ["file1.txt", "file2.py", "subdirectory/"]
        elif tool_name == "git_status":
            return "On branch main\nnothing to commit, working tree clean"
        elif tool_name == "git_commit":
            return f"Committed with message: {parameters.get('message', 'No message')}"
        elif tool_name == "fetch_url":
            return f"Fetched content from: {parameters.get('url', 'unknown')}"
        elif tool_name == "search_knowledge":
            return [
                {"title": "Knowledge Item 1", "content": "Sample content 1"},
                {"title": "Knowledge Item 2", "content": "Sample content 2"}
            ]
        else:
            return f"Tool {tool_name} executed successfully with parameters: {parameters}"

    def _update_performance_metrics(self, execution_time: float) -> None:
        """Update performance metrics."""
        if self.total_executions == 0:
            self.average_execution_time = execution_time
        else:
            # Calculate rolling average
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1) + execution_time) /
                self.total_executions
            )

    async def health_check(self) -> bool:
        """Perform health check on the MCP server."""
        if self.status != MCPConnectionStatus.CONNECTED:
            return False

        try:
            # Check if process is still running
            if self.process and self.process.poll() is None:
                self.last_health_check = datetime.utcnow()
                return True
            else:
                self.status = MCPConnectionStatus.ERROR
                return False
        except Exception:
            self.status = MCPConnectionStatus.ERROR
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get server performance metrics."""
        success_rate = (
            (self.successful_executions / self.total_executions * 100)
            if self.total_executions > 0 else 0
        )

        return {
            "name": self.name,
            "status": self.status.value,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "average_execution_time": self.average_execution_time,
            "tools_count": len(self.tools),
            "connection_attempts": self.connection_attempts,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert server to dictionary."""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "description": self.description,
            "status": self.status.value,
            "tools": [tool.to_dict() for tool in self.tools.values()],
            "metrics": self.get_metrics()
        }


class MCPClient:
    """Client for managing MCP server connections and tool execution."""

    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self.servers: Dict[str, MCPServer] = {}
        self.connection_pool: Set[str] = set()

        # Performance tracking
        self.total_tool_executions = 0
        self.successful_tool_executions = 0

        logging.info(f"MCP Client initialized with max connections: {max_connections}")

    async def add_server(
        self,
        name: str,
        command: str,
        args: List[str],
        description: str = "",
        env: Dict[str, str] = None,
        cwd: str = None,
        auto_connect: bool = True
    ) -> bool:
        """Add an MCP server to the client."""
        if name in self.servers:
            logging.warning(f"MCP server '{name}' already exists")
            return False

        server = MCPServer(
            name=name,
            command=command,
            args=args,
            description=description,
            env=env,
            cwd=cwd
        )

        self.servers[name] = server

        if auto_connect:
            success = await server.connect()
            if success:
                self.connection_pool.add(name)
            return success

        return True

    async def load_servers_from_config(self, config_path: str) -> int:
        """Load MCP servers from configuration file."""
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            servers_added = 0
            mcp_servers = config.get('mcp_servers', {})

            for server_name, server_config in mcp_servers.items():
                success = await self.add_server(
                    name=server_name,
                    command=server_config['command'],
                    args=server_config['args'],
                    description=server_config.get('description', ''),
                    env=server_config.get('env'),
                    cwd=server_config.get('cwd'),
                    auto_connect=True
                )

                if success:
                    servers_added += 1

            logging.info(f"Loaded {servers_added} MCP servers from configuration")
            return servers_added

        except Exception as e:
            logging.error(f"Failed to load MCP servers from config: {str(e)}")
            raise ConfigurationError(f"Failed to load MCP server configuration: {str(e)}")

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on a specific MCP server."""
        if server_name not in self.servers:
            raise ToolNotFoundError(f"MCP server '{server_name}' not found")

        server = self.servers[server_name]

        try:
            result = await server.execute_tool(tool_name, parameters)
            self.successful_tool_executions += 1
            return result
        except Exception as e:
            logging.error(f"Tool execution failed: {server_name}.{tool_name}: {str(e)}")
            raise
        finally:
            self.total_tool_executions += 1

    async def discover_tools(self, server_name: Optional[str] = None) -> Dict[str, List[str]]:
        """Discover available tools from MCP servers."""
        discovered_tools = {}

        servers_to_check = [server_name] if server_name else list(self.servers.keys())

        for srv_name in servers_to_check:
            if srv_name in self.servers:
                server = self.servers[srv_name]
                if server.status == MCPConnectionStatus.CONNECTED:
                    discovered_tools[srv_name] = list(server.tools.keys())
                else:
                    discovered_tools[srv_name] = []

        return discovered_tools

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all connected servers."""
        health_results = {}

        for server_name, server in self.servers.items():
            health_results[server_name] = await server.health_check()

        return health_results

    async def reconnect_failed_servers(self) -> int:
        """Attempt to reconnect failed servers."""
        reconnected = 0

        for server_name, server in self.servers.items():
            if server.status == MCPConnectionStatus.ERROR:
                if server.connection_attempts < server.max_connection_attempts:
                    logging.info(f"Attempting to reconnect MCP server '{server_name}'")
                    if await server.connect():
                        self.connection_pool.add(server_name)
                        reconnected += 1

        return reconnected

    def get_server_metrics(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for specific server or all servers."""
        if server_name:
            if server_name in self.servers:
                return self.servers[server_name].get_metrics()
            else:
                return {}

        return {
            srv_name: server.get_metrics()
            for srv_name, server in self.servers.items()
        }

    def get_client_metrics(self) -> Dict[str, Any]:
        """Get overall client metrics."""
        connected_servers = sum(
            1 for server in self.servers.values()
            if server.status == MCPConnectionStatus.CONNECTED
        )

        total_tools = sum(len(server.tools) for server in self.servers.values())

        success_rate = (
            (self.successful_tool_executions / self.total_tool_executions * 100)
            if self.total_tool_executions > 0 else 0
        )

        return {
            "total_servers": len(self.servers),
            "connected_servers": connected_servers,
            "total_tools": total_tools,
            "total_tool_executions": self.total_tool_executions,
            "successful_tool_executions": self.successful_tool_executions,
            "success_rate": success_rate,
            "connection_pool_size": len(self.connection_pool)
        }

    async def shutdown(self) -> None:
        """Shutdown all MCP server connections."""
        logging.info("Shutting down MCP client")

        for server in self.servers.values():
            await server.disconnect()

        self.servers.clear()
        self.connection_pool.clear()

        logging.info("MCP client shutdown complete")

    def __repr__(self) -> str:
        """String representation of MCP client."""
        return (
            f"MCPClient(servers={len(self.servers)}, "
            f"connected={len(self.connection_pool)}, "
            f"executions={self.total_tool_executions})"
        )
