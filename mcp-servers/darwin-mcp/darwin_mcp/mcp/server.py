#!/usr/bin/env python3
"""
Darwin MCP Server

This module implements the Model Context Protocol (MCP) server for Darwin,
providing genetic algorithm optimization capabilities to AI agents.

The server exposes tools for:
- Creating and configuring optimization problems
- Running single and multi-objective optimizations
- Monitoring optimization progress
- Analyzing and visualizing results
- Managing optimization templates

Example usage:
    $ python -m darwin_mcp.mcp.server
    
    Or programmatically:
    
    from darwin_mcp.mcp.server import create_server
    server = create_server()
    await server.run()
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import typer
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from .handlers import DarwinMCPHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CLI application
app = typer.Typer(
    name="darwin-mcp",
    help="Darwin Genetic Algorithm MCP Server",
    add_completion=False,
)


class DarwinMCPServer:
    """Darwin MCP Server implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize the Darwin MCP server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = Server("darwin-mcp")
        self.handlers = DarwinMCPHandlers()
        
        # Register MCP handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Darwin optimization tools."""
            return [
                Tool(
                    name="create_optimization",
                    description="Create a new genetic algorithm optimization problem",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name for the optimization problem"
                            },
                            "variables": {
                                "type": "array",
                                "description": "Optimization variables definition",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string", "enum": ["continuous", "discrete", "categorical", "permutation"]},
                                        "bounds": {"type": "array", "items": {"type": "number"}},
                                        "values": {"type": "array"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["name", "type"]
                                }
                            },
                            "objectives": {
                                "type": "array",
                                "description": "Optimization objectives",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string", "enum": ["minimize", "maximize"]},
                                        "function": {"type": "string"},
                                        "weight": {"type": "number"}
                                    },
                                    "required": ["name", "type"]
                                }
                            },
                            "constraints": {
                                "type": "array",
                                "description": "Optimization constraints",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["equality", "inequality"]},
                                        "expression": {"type": "string"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["type", "expression"]
                                }
                            },
                            "config": {
                                "type": "object",
                                "description": "Algorithm configuration",
                                "properties": {
                                    "algorithm": {"type": "string", "default": "genetic"},
                                    "population_size": {"type": "integer", "default": 100},
                                    "max_generations": {"type": "integer", "default": 200},
                                    "mutation_rate": {"type": "number", "default": 0.1},
                                    "crossover_rate": {"type": "number", "default": 0.8}
                                }
                            }
                        },
                        "required": ["name", "variables", "objectives"]
                    }
                ),
                
                Tool(
                    name="run_optimization",
                    description="Execute a genetic algorithm optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "optimization_id": {
                                "type": "string",
                                "description": "ID of the optimization to run"
                            },
                            "async_mode": {
                                "type": "boolean",
                                "default": False,
                                "description": "Run optimization asynchronously"
                            }
                        },
                        "required": ["optimization_id"]
                    }
                ),
                
                Tool(
                    name="get_optimization_status",
                    description="Get the current status of an optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "optimization_id": {
                                "type": "string",
                                "description": "ID of the optimization to check"
                            }
                        },
                        "required": ["optimization_id"]
                    }
                ),
                
                Tool(
                    name="get_optimization_results",
                    description="Get results from a completed optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "optimization_id": {
                                "type": "string",
                                "description": "ID of the optimization"
                            },
                            "include_history": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include evolution history"
                            },
                            "include_visualization": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include result visualizations"
                            }
                        },
                        "required": ["optimization_id"]
                    }
                ),
                
                Tool(
                    name="list_optimizations",
                    description="List all optimization problems",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["all", "created", "running", "completed", "failed"],
                                "default": "all",
                                "description": "Filter by optimization status"
                            },
                            "limit": {
                                "type": "integer",
                                "default": 20,
                                "description": "Maximum number of results"
                            }
                        }
                    }
                ),
                
                Tool(
                    name="create_template",
                    description="Create an optimization template for common problems",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_type": {
                                "type": "string",
                                "enum": ["function_optimization", "portfolio_optimization", "neural_network_tuning", "scheduling", "design_optimization"],
                                "description": "Type of optimization template"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Template-specific parameters"
                            }
                        },
                        "required": ["template_type"]
                    }
                ),
                
                Tool(
                    name="analyze_problem",
                    description="Analyze optimization problem complexity and provide recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "variables": {
                                "type": "array",
                                "description": "Problem variables for analysis"
                            },
                            "objectives": {
                                "type": "array",
                                "description": "Problem objectives for analysis"
                            },
                            "constraints": {
                                "type": "array",
                                "description": "Problem constraints for analysis"
                            }
                        },
                        "required": ["variables", "objectives"]
                    }
                ),
                
                Tool(
                    name="visualize_results",
                    description="Create visualizations of optimization results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "optimization_id": {
                                "type": "string",
                                "description": "ID of the optimization"
                            },
                            "chart_type": {
                                "type": "string",
                                "enum": ["convergence", "pareto_frontier", "variable_distribution", "objective_space", "diversity"],
                                "description": "Type of visualization to create"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["png", "svg", "html"],
                                "default": "png",
                                "description": "Output format for visualization"
                            }
                        },
                        "required": ["optimization_id", "chart_type"]
                    }
                ),
                
                Tool(
                    name="compare_algorithms",
                    description="Compare performance of different genetic algorithms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "problem": {
                                "type": "object",
                                "description": "Optimization problem definition"
                            },
                            "algorithms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of algorithms to compare"
                            },
                            "runs": {
                                "type": "integer",
                                "default": 5,
                                "description": "Number of runs per algorithm"
                            }
                        },
                        "required": ["problem", "algorithms"]
                    }
                ),
                
                Tool(
                    name="get_system_status",
                    description="Get Darwin system status and health metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_metrics": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include performance metrics"
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> CallToolResult:
            """Handle tool calls."""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                # Route to appropriate handler
                if name == "create_optimization":
                    result = await self.handlers.create_optimization(arguments or {})
                elif name == "run_optimization":
                    result = await self.handlers.run_optimization(arguments or {})
                elif name == "get_optimization_status":
                    result = await self.handlers.get_optimization_status(arguments or {})
                elif name == "get_optimization_results":
                    result = await self.handlers.get_optimization_results(arguments or {})
                elif name == "list_optimizations":
                    result = await self.handlers.list_optimizations(arguments or {})
                elif name == "create_template":
                    result = await self.handlers.create_template(arguments or {})
                elif name == "analyze_problem":
                    result = await self.handlers.analyze_problem(arguments or {})
                elif name == "visualize_results":
                    result = await self.handlers.visualize_results(arguments or {})
                elif name == "compare_algorithms":
                    result = await self.handlers.compare_algorithms(arguments or {})
                elif name == "get_system_status":
                    result = await self.handlers.get_system_status(arguments or {})
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Format result for MCP
                if isinstance(result, dict):
                    content = [TextContent(type="text", text=json.dumps(result, indent=2))]
                elif isinstance(result, str):
                    content = [TextContent(type="text", text=result)]
                else:
                    content = [TextContent(type="text", text=str(result))]
                
                return CallToolResult(content=content)
                
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                error_content = [TextContent(
                    type="text", 
                    text=f"Error: {str(e)}"
                )]
                return CallToolResult(content=error_content, isError=True)
    
    async def run_stdio(self):
        """Run the server using stdio transport."""
        logger.info("Starting Darwin MCP server with stdio transport")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="darwin-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities()
                )
            )


def create_server(host: str = "localhost", port: int = 8000) -> DarwinMCPServer:
    """Create a Darwin MCP server instance.
    
    Args:
        host: Server host address
        port: Server port number
        
    Returns:
        DarwinMCPServer instance
    """
    return DarwinMCPServer(host=host, port=port)


@app.command()
def serve(
    host: str = typer.Option("localhost", help="Server host address"),
    port: int = typer.Option(8000, help="Server port number"),
    stdio: bool = typer.Option(True, help="Use stdio transport"),
):
    """Start the Darwin MCP server."""
    server = create_server(host=host, port=port)
    
    if stdio:
        asyncio.run(server.run_stdio())
    else:
        typer.echo(f"Starting Darwin MCP server on {host}:{port}")
        # For future HTTP transport implementation
        typer.echo("HTTP transport not yet implemented. Use --stdio for now.")


@app.command()
def version():
    """Show version information."""
    from darwin_mcp import __version__
    typer.echo(f"Darwin MCP Server v{__version__}")


@app.command() 
def health():
    """Check server health."""
    typer.echo("Darwin MCP Server: OK")
    # Add more health checks as needed


def main():
    """Main entry point for the CLI."""
    try:
        # If no arguments provided, start with stdio by default
        if len(sys.argv) == 1:
            sys.argv.append("serve")
            sys.argv.append("--stdio")
        
        app()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()