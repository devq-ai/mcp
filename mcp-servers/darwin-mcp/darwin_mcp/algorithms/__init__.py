"""
Darwin Genetic Algorithm Implementations

This package provides genetic algorithm implementations for the Darwin
optimization platform. It includes various algorithm variants and
genetic operators for different optimization scenarios.

Components:
- genetic: Core genetic algorithm implementation
- operators: Genetic operators (selection, crossover, mutation)
- multi_objective: Multi-objective optimization algorithms
- constraints: Constraint handling methods

The algorithms are designed to work with the Darwin optimization engine
and support various problem types including single/multi-objective,
constrained/unconstrained, and different variable types.
"""

from .genetic import GeneticAlgorithm
from .operators import SelectionOperator, CrossoverOperator, MutationOperator
from .multi_objective import NSGAII, NSGAIII
from .constraints import ConstraintHandler

__all__ = [
    "GeneticAlgorithm",
    "SelectionOperator",
    "CrossoverOperator", 
    "MutationOperator",
    "NSGAII",
    "NSGAIII",
    "ConstraintHandler",
]