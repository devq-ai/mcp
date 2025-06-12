"""
Optimization Module for Agentical Framework

This module provides advanced optimization capabilities for the Agentical framework,
implementing genetic algorithms, evolutionary strategies, and multi-objective
optimization for sophisticated agent decision-making and problem-solving.

Components:
- GeneticAlgorithmEngine: Core genetic algorithm implementation
- PopulationManager: Population initialization and management
- FitnessEvaluator: Fitness function evaluation framework
- SelectionStrategies: Tournament, roulette, rank-based selection
- MutationOperators: Uniform, gaussian, polynomial mutations
- CrossoverOperators: Single-point, uniform, arithmetic crossover
- ConvergenceDetector: Solution quality and diversity monitoring
- DarwinMCPClient: Integration with darwin-mcp server

Features:
- Multi-objective optimization support
- Adaptive parameter control
- Parallel fitness evaluation
- Real-time convergence monitoring
- Integration with Bayesian inference
- Performance optimization for large solution spaces
- Comprehensive logging and observability
"""

from .genetic_algorithm import (
    GeneticAlgorithmEngine,
    GeneticAlgorithmConfig,
    OptimizationResult,
    Solution,
    Population
)

from .fitness_evaluator import (
    FitnessEvaluator,
    FitnessFunction,
    MultiObjectiveEvaluator,
    FitnessResult,
    ObjectiveType
)

from .population_manager import (
    PopulationManager,
    PopulationConfig,
    InitializationStrategy,
    PopulationMetrics,
    DiversityMeasure
)

from .selection_strategies import (
    SelectionStrategy,
    TournamentSelection,
    RouletteWheelSelection,
    RankBasedSelection,
    ElitismSelection,
    SelectionConfig
)

from .mutation_operators import (
    MutationOperator,
    UniformMutation,
    GaussianMutation,
    PolynomialMutation,
    AdaptiveMutation,
    MutationConfig
)

from .crossover_operators import (
    CrossoverOperator,
    SinglePointCrossover,
    UniformCrossover,
    ArithmeticCrossover,
    BlendCrossover,
    CrossoverConfig
)

from .convergence_detector import (
    ConvergenceDetector,
    ConvergenceConfig,
    ConvergenceMetrics,
    ConvergenceStatus,
    DiversityAnalyzer
)

from .darwin_mcp_client import (
    DarwinMCPClient,
    DarwinConfig,
    OptimizationRequest,
    OptimizationResponse,
    ServerStatus
)

from .multi_objective import (
    MultiObjectiveOptimizer,
    ParetoFront,
    DominanceRelation,
    HypervolumeCalculator,
    MOOConfig
)

from .adaptive_control import (
    AdaptiveController,
    ParameterScheduler,
    PerformanceMonitor,
    AdaptiveConfig,
    ControlStrategy
)

__all__ = [
    # Core Engine
    "GeneticAlgorithmEngine",
    "GeneticAlgorithmConfig",
    "OptimizationResult",
    "Solution",
    "Population",

    # Fitness Evaluation
    "FitnessEvaluator",
    "FitnessFunction",
    "MultiObjectiveEvaluator",
    "FitnessResult",
    "ObjectiveType",

    # Population Management
    "PopulationManager",
    "PopulationConfig",
    "InitializationStrategy",
    "PopulationMetrics",
    "DiversityMeasure",

    # Selection Strategies
    "SelectionStrategy",
    "TournamentSelection",
    "RouletteWheelSelection",
    "RankBasedSelection",
    "ElitismSelection",
    "SelectionConfig",

    # Mutation Operators
    "MutationOperator",
    "UniformMutation",
    "GaussianMutation",
    "PolynomialMutation",
    "AdaptiveMutation",
    "MutationConfig",

    # Crossover Operators
    "CrossoverOperator",
    "SinglePointCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
    "BlendCrossover",
    "CrossoverConfig",

    # Convergence Detection
    "ConvergenceDetector",
    "ConvergenceConfig",
    "ConvergenceMetrics",
    "ConvergenceStatus",
    "DiversityAnalyzer",

    # Darwin MCP Integration
    "DarwinMCPClient",
    "DarwinConfig",
    "OptimizationRequest",
    "OptimizationResponse",
    "ServerStatus",

    # Multi-Objective Optimization
    "MultiObjectiveOptimizer",
    "ParetoFront",
    "DominanceRelation",
    "HypervolumeCalculator",
    "MOOConfig",

    # Adaptive Control
    "AdaptiveController",
    "ParameterScheduler",
    "PerformanceMonitor",
    "AdaptiveConfig",
    "ControlStrategy"
]

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__description__ = "Genetic algorithm and optimization capabilities for Agentical framework"

# Module-level configuration
DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 1000
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_SELECTION_PRESSURE = 2.0
DEFAULT_ELITISM_RATE = 0.1

# Performance settings
DEFAULT_PARALLEL_EVALUATION = True
DEFAULT_MAX_WORKERS = 4
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CONVERGENCE_PATIENCE = 50

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
