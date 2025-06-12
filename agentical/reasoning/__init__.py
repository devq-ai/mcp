"""
Agent Reasoning System for Agentical Framework

This module provides comprehensive reasoning capabilities for intelligent agent decision-making,
including Bayesian inference, genetic optimization, simulation, and probabilistic analysis.

Components:
- BayesianInferenceEngine: Core inference engine with belief updating
- GeneticAlgorithmEngine: Evolutionary optimization for complex problems
- SimulationEngine: Monte Carlo simulation and scenario analysis
- BeliefUpdater: Dynamic probability updates based on new evidence
- DecisionTree: Probabilistic decision-making framework
- UncertaintyQuantifier: Confidence and uncertainty measurement
- ProbabilisticModel: Base classes for Bayesian models

Features:
- Multi-objective optimization with genetic algorithms
- Monte Carlo simulation with uncertainty quantification
- Integration with Darwin-MCP and Bayes-MCP servers
- Real-time belief updating and learning
- Comprehensive statistical analysis
- Performance optimization and parallel processing
- Full observability and logging
"""

from .bayesian_engine import (
    BayesianInferenceEngine,
    BayesianConfig,
    InferenceResult,
    Evidence,
    Hypothesis
)

from .genetic_optimizer import (
    GeneticAlgorithmEngine,
    GeneticConfig,
    Individual,
    FitnessFunction,
    OptimizationResult,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod,
    OptimizationObjective
)

from .simulation_engine import (
    SimulationEngine,
    SimulationConfig,
    Scenario,
    Parameter,
    SimulationRun,
    SimulationResult,
    SamplingMethod,
    DistributionType,
    SimulationType,
    AnalysisType
)

from .belief_updater import (
    BeliefUpdater,
    BeliefState,
    BeliefUpdate,
    EvidenceType,
    UpdateStrategy
)

from .decision_tree import (
    DecisionTree,
    DecisionNode,
    ProbabilisticBranch,
    DecisionOutcome,
    TreeMetrics
)

from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    ConfidenceLevel,
    UncertaintyMeasure,
    DistributionType,
    QuantificationMethod
)

from .probabilistic_models import (
    ProbabilisticModel,
    BayesianNetwork,
    MarkovChain,
    HiddenMarkovModel,
    GaussianProcess
)

from .mcp_integration import (
    BayesMCPClient,
    MCPConfig,
    ServerStatus,
    InferenceRequest,
    InferenceResponse
)

__all__ = [
    # Core Bayesian Engine
    "BayesianInferenceEngine",
    "BayesianConfig",
    "InferenceResult",
    "Evidence",
    "Hypothesis",

    # Genetic Algorithm Optimization
    "GeneticAlgorithmEngine",
    "GeneticConfig",
    "Individual",
    "FitnessFunction",
    "OptimizationResult",
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod",
    "OptimizationObjective",

    # Simulation Engine
    "SimulationEngine",
    "SimulationConfig",
    "Scenario",
    "Parameter",
    "SimulationRun",
    "SimulationResult",
    "SamplingMethod",
    "DistributionType",
    "SimulationType",
    "AnalysisType",

    # Belief Updating
    "BeliefUpdater",
    "BeliefState",
    "BeliefUpdate",
    "EvidenceType",
    "UpdateStrategy",

    # Decision Trees
    "DecisionTree",
    "DecisionNode",
    "ProbabilisticBranch",
    "DecisionOutcome",
    "TreeMetrics",

    # Uncertainty Quantification
    "UncertaintyQuantifier",
    "ConfidenceLevel",
    "UncertaintyMeasure",
    "DistributionType",
    "QuantificationMethod",

    # Probabilistic Models
    "ProbabilisticModel",
    "BayesianNetwork",
    "MarkovChain",
    "HiddenMarkovModel",
    "GaussianProcess",

    # MCP Integration
    "BayesMCPClient",
    "MCPConfig",
    "ServerStatus",
    "InferenceRequest",
    "InferenceResponse"
]

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__description__ = "Comprehensive reasoning system with Bayesian inference, genetic optimization, and simulation for Agentical framework"

# Module-level configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_UNCERTAINTY_TOLERANCE = 0.25
DEFAULT_INFERENCE_TIMEOUT = 30.0
DEFAULT_MAX_ITERATIONS = 1000

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
