"""
Darwin Optimization Engine Package

This package provides the core optimization engine and problem definition
capabilities for the Darwin genetic algorithm platform.

Components:
- engine: Core optimization execution engine
- problem: Optimization problem definition and validation
- config: Algorithm configuration and parameters
- results: Result processing and analysis

The optimization engine coordinates the execution of genetic algorithms,
manages problem definitions, and processes results for analysis and
visualization.
"""

from .engine import OptimizationEngine
from .problem import OptimizationProblem, Variable, Objective, Constraint
from .config import OptimizationConfig
from .results import OptimizationResult

__all__ = [
    "OptimizationEngine",
    "OptimizationProblem",
    "Variable", 
    "Objective",
    "Constraint",
    "OptimizationConfig",
    "OptimizationResult",
]