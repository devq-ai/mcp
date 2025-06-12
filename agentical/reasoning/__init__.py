"""
Bayesian Inference Engine Module for Agentical Framework

This module provides advanced Bayesian reasoning capabilities for agent decision-making,
including belief updating, uncertainty quantification, and probabilistic decision trees.

Components:
- BayesianInferenceEngine: Core inference engine with belief updating
- BeliefUpdater: Dynamic probability updates based on new evidence
- DecisionTree: Probabilistic decision-making framework
- UncertaintyQuantifier: Confidence and uncertainty measurement
- ProbabilisticModel: Base classes for Bayesian models

Features:
- Integration with bayes-mcp server
- Real-time belief updating
- Uncertainty quantification
- Decision tree framework
- Performance optimization
- Comprehensive logging
"""

from .bayesian_engine import (
    BayesianInferenceEngine,
    BayesianConfig,
    InferenceResult,
    Evidence,
    Hypothesis
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
    # Core Engine
    "BayesianInferenceEngine",
    "BayesianConfig",
    "InferenceResult",
    "Evidence",
    "Hypothesis",

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
__description__ = "Bayesian inference and reasoning capabilities for Agentical framework"

# Module-level configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_UNCERTAINTY_TOLERANCE = 0.25
DEFAULT_INFERENCE_TIMEOUT = 30.0
DEFAULT_MAX_ITERATIONS = 1000

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
