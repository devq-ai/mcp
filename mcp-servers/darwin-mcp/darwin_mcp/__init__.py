"""
Darwin MCP Server Package

This package provides a Model Context Protocol (MCP) server that exposes
Darwin's genetic algorithm optimization capabilities to AI agents and applications.

The Darwin MCP server enables AI agents to:
- Create and configure genetic algorithm optimizations
- Run single and multi-objective optimizations
- Monitor optimization progress in real-time
- Analyze and visualize optimization results
- Access pre-built optimization templates
- Integrate with existing optimization workflows

Key Features:
- Complete genetic algorithm toolkit
- Multi-objective optimization (NSGA-II, NSGA-III)
- Constraint handling and penalty methods
- Real-time progress monitoring
- Interactive visualization capabilities
- Production-ready deployment options
- Comprehensive logging and metrics

Usage:
    The server can be started using the CLI:
    
    $ darwin-mcp
    
    Or programmatically:
    
    from darwin_mcp.mcp.server import create_server
    server = create_server()

Components:
- mcp/: MCP server implementation and handlers
- optimization/: Core optimization engine and algorithms
- algorithms/: Genetic algorithm implementations
- utils/: Utility functions and helpers

For more information, see the Darwin documentation at:
https://darwin.devq.ai/docs
"""

__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "team@devq.ai"
__license__ = "BSD-3-Clause"

# Import key classes for easy access
from darwin_mcp.optimization.engine import OptimizationEngine
from darwin_mcp.optimization.problem import OptimizationProblem
from darwin_mcp.algorithms.genetic import GeneticAlgorithm
from darwin_mcp.mcp.server import create_server

__all__ = [
    "OptimizationEngine",
    "OptimizationProblem", 
    "GeneticAlgorithm",
    "create_server",
    "__version__",
]