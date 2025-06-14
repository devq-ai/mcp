"""
Darwin MCP Server Utilities

This package provides utility classes and helper functions for the Darwin
MCP server implementation. It includes template management, problem analysis,
visualization, and other supporting functionality.

Components:
- templates: Optimization problem templates for common scenarios
- analyzer: Problem complexity analysis and recommendations
- visualizer: Result visualization and chart generation
- helpers: Common utility functions and helpers

These utilities support the core MCP server functionality by providing
reusable components for problem setup, analysis, and result processing.
"""

from .templates import TemplateManager
from .analyzer import ProblemAnalyzer
from .visualizer import ResultVisualizer
from .helpers import ValidationHelper, ConversionHelper

__all__ = [
    "TemplateManager",
    "ProblemAnalyzer", 
    "ResultVisualizer",
    "ValidationHelper",
    "ConversionHelper",
]