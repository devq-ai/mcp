"""
Decision Tree Component for Probabilistic Decision-Making

This module provides sophisticated decision tree capabilities for the Agentical
framework's Bayesian reasoning system, enabling structured probabilistic
decision-making with uncertainty quantification.

Features:
- Probabilistic decision tree construction and evaluation
- Multi-criteria decision analysis with Bayesian updates
- Uncertainty propagation through decision paths
- Dynamic tree modification and pruning
- Performance optimization for real-time decisions
- Comprehensive logging and observability
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import asyncio
import json
import math
import numpy as np
from collections import defaultdict, deque

import logfire
from pydantic import BaseModel, Field, validator

from agentical.core.exceptions import (
    AgentError,
    ValidationError,
    ConfigurationError
)
from agentical.core.structured_logging import (
    StructuredLogger,
    CorrelationContext,
    LogLevel,
    OperationType
)


class NodeType(str, Enum):
    """Types of decision tree nodes."""
    ROOT = "root"
    DECISION = "decision"
    CHANCE = "chance"
    OUTCOME = "outcome"
    TERMINAL = "terminal"


class DecisionCriteria(str, Enum):
    """Criteria for decision evaluation."""
    MAXIMUM_EXPECTED_VALUE = "maximum_expected_value"
    MAXIMUM_UTILITY = "maximum_utility"
    MINIMAX_REGRET = "minimax_regret"
    EXPECTED_UTILITY = "expected_utility"
    SATISFICING = "satisficing"


class UncertaintyType(str, Enum):
    """Types of uncertainty in decision making."""
    ALEATORY = "aleatory"  # Natural randomness
    EPISTEMIC = "epistemic"  # Knowledge uncertainty
    PARAMETRIC = "parametric"  # Parameter uncertainty
    MODEL = "model"  # Model uncertainty


@dataclass
class DecisionOutcome:
    """Represents an outcome of a decision path."""
    outcome_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Value metrics
    expected_value: float = 0.0
    utility: float = 0.0
    probability: float = 0.0

    # Uncertainty measures
    variance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    uncertainty_type: UncertaintyType = UncertaintyType.EPISTEMIC

    # Additional metrics
    regret_value: float = 0.0
    satisfaction_level: float = 0.0
    risk_score: float = 0.0

    # Metadata
    path_cost: float = 0.0
    computation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProbabilisticBranch:
    """Represents a probabilistic branch in the decision tree."""
    branch_id: str = field(default_factory=lambda: str(uuid4()))
    parent_node_id: str = ""
    child_node_id: str = ""

    # Branch properties
    condition: str = ""
    probability: float = 0.0
    likelihood_function: Optional[Callable] = None

    # Value propagation
    expected_value: float = 0.0
    cumulative_probability: float = 0.0
    path_utility: float = 0.0

    # Uncertainty propagation
    uncertainty_contribution: float = 0.0
    confidence_degradation: float = 0.0

    # Branch metadata
    weight: float = 1.0
    active: bool = True
    pruned: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecisionNode:
    """Represents a node in the probabilistic decision tree."""
    node_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    node_type: NodeType = NodeType.DECISION

    # Node state
    is_expanded: bool = False
    is_evaluated: bool = False
    depth: int = 0

    # Decision properties
    decision_criteria: DecisionCriteria = DecisionCriteria.MAXIMUM_EXPECTED_VALUE
    available_actions: List[str] = field(default_factory=list)
    selected_action: Optional[str] = None

    # Probabilistic properties
    prior_probability: float = 1.0
    posterior_probability: float = 1.0
    likelihood_cache: Dict[str, float] = field(default_factory=dict)

    # Value calculations
    expected_value: float = 0.0
    utility_value: float = 0.0
    regret_value: float = 0.0

    # Uncertainty metrics
    uncertainty_measure: float = 0.0
    confidence_level: float = 0.0
    variance: float = 0.0

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    branches: Dict[str, ProbabilisticBranch] = field(default_factory=dict)

    # Evaluation results
    outcomes: List[DecisionOutcome] = field(default_factory=list)
    best_outcome: Optional[DecisionOutcome] = None

    # Performance metrics
    evaluation_count: int = 0
    last_evaluation: Optional[datetime] = None
    computation_time_ms: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TreeMetrics:
    """Comprehensive metrics for decision tree analysis."""
    tree_id: str = ""

    # Structure metrics
    total_nodes: int = 0
    decision_nodes: int = 0
    chance_nodes: int = 0
    terminal_nodes: int = 0
    max_depth: int = 0

    # Performance metrics
    total_evaluations: int = 0
    average_evaluation_time_ms: float = 0.0
    total_computation_time_ms: float = 0.0

    # Quality metrics
    tree_balance_score: float = 0.0
    pruning_efficiency: float = 0.0
    convergence_rate: float = 0.0

    # Decision quality
    average_confidence: float = 0.0
    uncertainty_propagation: float = 0.0
    regret_minimization: float = 0.0

    # Resource utilization
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class DecisionTreeConfig(BaseModel):
    """Configuration for probabilistic decision tree."""

    # Tree construction
    max_depth: int = Field(default=10, ge=1, le=50)
    max_branches_per_node: int = Field(default=10, ge=2)
    pruning_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    # Decision criteria
    default_criteria: DecisionCriteria = DecisionCriteria.MAXIMUM_EXPECTED_VALUE
    utility_function: Optional[str] = None
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Uncertainty handling
    uncertainty_propagation: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    uncertainty_discount_factor: float = Field(default=0.95, ge=0.0, le=1.0)

    # Performance settings
    enable_caching: bool = True
    cache_size: int = Field(default=1000, ge=10)
    parallel_evaluation: bool = True
    max_workers: int = Field(default=4, ge=1)

    # Pruning and optimization
    enable_dynamic_pruning: bool = True
    pruning_frequency: int = Field(default=100, ge=1)
    memory_limit_mb: float = Field(default=100.0, gt=0.0)

    # Logging and monitoring
    detailed_logging: bool = True
    performance_monitoring: bool = True
    log_evaluation_frequency: int = Field(default=50, ge=1)


class DecisionTree:
    """
    Sophisticated probabilistic decision tree for agent decision-making.

    Provides advanced decision tree capabilities including:
    - Multi-criteria decision analysis with Bayesian updates
    - Uncertainty quantification and propagation
    - Dynamic tree construction and pruning
    - Performance optimization for real-time decisions
    - Comprehensive evaluation metrics
    """

    def __init__(
        self,
        config: DecisionTreeConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the probabilistic decision tree.

        Args:
            config: Tree configuration
            logger: Optional structured logger
        """
        self.config = config
        self.logger = logger or StructuredLogger("decision_tree")

        # Tree structure
        self.tree_id = str(uuid4())
        self.nodes: Dict[str, DecisionNode] = {}
        self.root_node_id: Optional[str] = None

        # Evaluation cache
        self.evaluation_cache: Dict[str, DecisionOutcome] = {}
        self.likelihood_cache: Dict[str, float] = {}

        # Performance tracking
        self.evaluation_count = 0
        self.cache_hits = 0
        self.total_computation_time = 0.0
        self.pruning_count = 0

        # State management
        self.is_built = False
        self.last_evaluation = datetime.utcnow()

        logfire.info(
            "Decision tree initialized",
            tree_id=self.tree_id,
            max_depth=config.max_depth,
            criteria=config.default_criteria
        )

    async def create_root_node(
        self,
        name: str,
        description: str = "",
        available_actions: Optional[List[str]] = None
    ) -> str:
        """
        Create the root node of the decision tree.

        Args:
            name: Root node name
            description: Node description
            available_actions: Available decision actions

        Returns:
            Root node ID
        """
        with logfire.span("Create Root Node", name=name):
            if self.root_node_id is not None:
                raise ValidationError("Root node already exists")

            root_node = DecisionNode(
                name=name,
                description=description,
                node_type=NodeType.ROOT,
                depth=0,
                available_actions=available_actions or [],
                decision_criteria=self.config.default_criteria
            )

            self.nodes[root_node.node_id] = root_node
            self.root_node_id = root_node.node_id

            self.logger.log(
                LogLevel.INFO,
                f"Created root node: {name}",
                operation_type=OperationType.CREATE,
                node_id=root_node.node_id,
                tree_id=self.tree_id
            )

            return root_node.node_id

    async def add_decision_node(
        self,
        parent_id: str,
        name: str,
        available_actions: List[str],
        condition: str = "",
        probability: float = 1.0,
        description: str = ""
    ) -> str:
        """
        Add a decision node to the tree.

        Args:
            parent_id: Parent node ID
            name: Node name
            available_actions: Available decision actions
            condition: Condition for reaching this node
            probability: Probability of reaching this node
            description: Node description

        Returns:
            New node ID
        """
        with logfire.span("Add Decision Node", name=name, parent_id=parent_id):
            if parent_id not in self.nodes:
                raise ValidationError(f"Parent node {parent_id} not found")

            parent_node = self.nodes[parent_id]

            if parent_node.depth >= self.config.max_depth - 1:
                raise ValidationError(f"Maximum tree depth {self.config.max_depth} reached")

            if len(parent_node.children_ids) >= self.config.max_branches_per_node:
                raise ValidationError(f"Maximum branches per node {self.config.max_branches_per_node} reached")

            # Create new decision node
            decision_node = DecisionNode(
                name=name,
                description=description,
                node_type=NodeType.DECISION,
                depth=parent_node.depth + 1,
                parent_id=parent_id,
                available_actions=available_actions,
                decision_criteria=self.config.default_criteria
            )

            # Create branch connecting parent to child
            branch = ProbabilisticBranch(
                parent_node_id=parent_id,
                child_node_id=decision_node.node_id,
                condition=condition,
                probability=probability
            )

            # Update tree structure
            self.nodes[decision_node.node_id] = decision_node
            parent_node.children_ids.append(decision_node.node_id)
            parent_node.branches[decision_node.node_id] = branch

            self.logger.log(
                LogLevel.INFO,
                f"Added decision node: {name}",
                operation_type=OperationType.CREATE,
                node_id=decision_node.node_id,
                parent_id=parent_id,
                depth=decision_node.depth
            )

            return decision_node.node_id

    async def add_chance_node(
        self,
        parent_id: str,
        name: str,
        outcomes: List[Tuple[str, float]],  # (outcome_name, probability)
        condition: str = "",
        description: str = ""
    ) -> str:
        """
        Add a chance node with probabilistic outcomes.

        Args:
            parent_id: Parent node ID
            name: Node name
            outcomes: List of (outcome_name, probability) tuples
            condition: Condition for reaching this node
            description: Node description

        Returns:
            New node ID
        """
        with logfire.span("Add Chance Node", name=name, parent_id=parent_id):
            if parent_id not in self.nodes:
                raise ValidationError(f"Parent node {parent_id} not found")

            # Validate probabilities sum to 1
            total_probability = sum(prob for _, prob in outcomes)
            if not math.isclose(total_probability, 1.0, rel_tol=1e-6):
                raise ValidationError(f"Outcome probabilities must sum to 1.0, got {total_probability}")

            parent_node = self.nodes[parent_id]

            # Create chance node
            chance_node = DecisionNode(
                name=name,
                description=description,
                node_type=NodeType.CHANCE,
                depth=parent_node.depth + 1,
                parent_id=parent_id
            )

            # Add outcome nodes
            for outcome_name, outcome_prob in outcomes:
                outcome_node = DecisionNode(
                    name=f"{name}_{outcome_name}",
                    description=f"Outcome: {outcome_name}",
                    node_type=NodeType.OUTCOME,
                    depth=chance_node.depth + 1,
                    parent_id=chance_node.node_id,
                    prior_probability=outcome_prob
                )

                # Create branch for outcome
                outcome_branch = ProbabilisticBranch(
                    parent_node_id=chance_node.node_id,
                    child_node_id=outcome_node.node_id,
                    condition=f"outcome_{outcome_name}",
                    probability=outcome_prob
                )

                self.nodes[outcome_node.node_id] = outcome_node
                chance_node.children_ids.append(outcome_node.node_id)
                chance_node.branches[outcome_node.node_id] = outcome_branch

            # Connect to parent
            parent_branch = ProbabilisticBranch(
                parent_node_id=parent_id,
                child_node_id=chance_node.node_id,
                condition=condition,
                probability=1.0
            )

            self.nodes[chance_node.node_id] = chance_node
            parent_node.children_ids.append(chance_node.node_id)
            parent_node.branches[chance_node.node_id] = parent_branch

            self.logger.log(
                LogLevel.INFO,
                f"Added chance node: {name} with {len(outcomes)} outcomes",
                operation_type=OperationType.CREATE,
                node_id=chance_node.node_id,
                parent_id=parent_id,
                outcome_count=len(outcomes)
            )

            return chance_node.node_id

    async def add_terminal_node(
        self,
        parent_id: str,
        name: str,
        value: float,
        utility: float = 0.0,
        condition: str = "",
        probability: float = 1.0,
        description: str = ""
    ) -> str:
        """
        Add a terminal node with final values.

        Args:
            parent_id: Parent node ID
            name: Node name
            value: Terminal value
            utility: Utility score
            condition: Condition for reaching this node
            probability: Probability of reaching this node
            description: Node description

        Returns:
            New node ID
        """
        with logfire.span("Add Terminal Node", name=name, parent_id=parent_id):
            if parent_id not in self.nodes:
                raise ValidationError(f"Parent node {parent_id} not found")

            parent_node = self.nodes[parent_id]

            # Create terminal node
            terminal_node = DecisionNode(
                name=name,
                description=description,
                node_type=NodeType.TERMINAL,
                depth=parent_node.depth + 1,
                parent_id=parent_id,
                expected_value=value,
                utility_value=utility,
                is_evaluated=True
            )

            # Create outcome for terminal node
            outcome = DecisionOutcome(
                name=f"terminal_{name}",
                description=f"Terminal outcome: {name}",
                expected_value=value,
                utility=utility,
                probability=probability
            )
            terminal_node.outcomes.append(outcome)
            terminal_node.best_outcome = outcome

            # Create branch
            branch = ProbabilisticBranch(
                parent_node_id=parent_id,
                child_node_id=terminal_node.node_id,
                condition=condition,
                probability=probability,
                expected_value=value
            )

            # Update tree structure
            self.nodes[terminal_node.node_id] = terminal_node
            parent_node.children_ids.append(terminal_node.node_id)
            parent_node.branches[terminal_node.node_id] = branch

            self.logger.log(
                LogLevel.INFO,
                f"Added terminal node: {name}",
                operation_type=OperationType.CREATE,
                node_id=terminal_node.node_id,
                parent_id=parent_id,
                value=value,
                utility=utility
            )

            return terminal_node.node_id

    async def evaluate_tree(
        self,
        evidence: Optional[Dict[str, Any]] = None
    ) -> DecisionOutcome:
        """
        Evaluate the entire decision tree and find optimal decision path.

        Args:
            evidence: Optional evidence for Bayesian updates

        Returns:
            Best decision outcome
        """
        start_time = datetime.utcnow()

        with logfire.span("Evaluate Tree", tree_id=self.tree_id):
            try:
                if self.root_node_id is None:
                    raise ValidationError("Tree has no root node")

                # Update probabilities with evidence if provided
                if evidence:
                    await self._update_probabilities_with_evidence(evidence)

                # Perform backward induction evaluation
                best_outcome = await self._backward_induction()

                # Update metrics
                computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_computation_time += computation_time
                self.evaluation_count += 1
                self.last_evaluation = datetime.utcnow()

                # Prune tree if configured
                if self.config.enable_dynamic_pruning and self.evaluation_count % self.config.pruning_frequency == 0:
                    await self._prune_tree()

                self.logger.log(
                    LogLevel.INFO,
                    f"Tree evaluation completed",
                    operation_type=OperationType.EVALUATION,
                    tree_id=self.tree_id,
                    best_value=best_outcome.expected_value,
                    computation_time_ms=computation_time
                )

                return best_outcome

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Tree evaluation failed: {str(e)}",
                    operation_type=OperationType.EVALUATION,
                    tree_id=self.tree_id,
                    error=str(e)
                )
                raise AgentError(f"Tree evaluation failed: {str(e)}")

    async def get_decision_path(self, node_id: Optional[str] = None) -> List[DecisionNode]:
        """
        Get the optimal decision path from root to specified node.

        Args:
            node_id: Target node ID (defaults to best terminal node)

        Returns:
            List of nodes in optimal path
        """
        with logfire.span("Get Decision Path", node_id=node_id):
            if self.root_node_id is None:
                return []

            # If no target specified, find best terminal node
            if node_id is None:
                await self.evaluate_tree()
                node_id = self._find_best_terminal_node()

            # Trace path back to root
            path = []
            current_id = node_id

            while current_id is not None:
                if current_id in self.nodes:
                    node = self.nodes[current_id]
                    path.insert(0, node)
                    current_id = node.parent_id
                else:
                    break

            return path

    async def get_tree_metrics(self) -> TreeMetrics:
        """Get comprehensive tree analysis metrics."""
        with logfire.span("Get Tree Metrics"):
            # Count node types
            decision_nodes = sum(1 for node in self.nodes.values() if node.node_type == NodeType.DECISION)
            chance_nodes = sum(1 for node in self.nodes.values() if node.node_type == NodeType.CHANCE)
            terminal_nodes = sum(1 for node in self.nodes.values() if node.node_type == NodeType.TERMINAL)

            # Calculate depth
            max_depth = max((node.depth for node in self.nodes.values()), default=0)

            # Performance metrics
            avg_eval_time = (
                self.total_computation_time / max(self.evaluation_count, 1)
                if self.evaluation_count > 0 else 0.0
            )

            # Quality metrics
            avg_confidence = np.mean([
                node.confidence_level for node in self.nodes.values()
                if node.confidence_level > 0
            ]) if any(node.confidence_level > 0 for node in self.nodes.values()) else 0.0

            cache_hit_rate = (
                self.cache_hits / max(self.evaluation_count, 1)
                if self.config.enable_caching and self.evaluation_count > 0 else 0.0
            )

            metrics = TreeMetrics(
                tree_id=self.tree_id,
                total_nodes=len(self.nodes),
                decision_nodes=decision_nodes,
                chance_nodes=chance_nodes,
                terminal_nodes=terminal_nodes,
                max_depth=max_depth,
                total_evaluations=self.evaluation_count,
                average_evaluation_time_ms=avg_eval_time,
                total_computation_time_ms=self.total_computation_time,
                average_confidence=avg_confidence,
                cache_hit_rate=cache_hit_rate
            )

            return metrics

    # Private helper methods

    async def _backward_induction(self) -> DecisionOutcome:
        """Perform backward induction to find optimal decision path."""
        if self.root_node_id is None:
            raise ValidationError("No root node for evaluation")

        # Evaluate nodes from leaves to root
        evaluated_nodes = set()

        # First, evaluate all terminal nodes
        for node in self.nodes.values():
            if node.node_type == NodeType.TERMINAL:
                await self._evaluate_node(node.node_id, evaluated_nodes)

        # Then evaluate internal nodes bottom-up
        max_iterations = len(self.nodes) * 2  # Prevent infinite loops
        iteration = 0

        while self.root_node_id not in evaluated_nodes and iteration < max_iterations:
            for node in self.nodes.values():
                if node.node_id not in evaluated_nodes:
                    # Check if all children are evaluated
                    if all(child_id in evaluated_nodes for child_id in node.children_ids):
                        await self._evaluate_node(node.node_id, evaluated_nodes)
            iteration += 1

        # Get root node result
        root_node = self.nodes[self.root_node_id]
        if root_node.best_outcome is None:
            raise AgentError("Failed to evaluate root node")

        return root_node.best_outcome

    async def _evaluate_node(self, node_id: str, evaluated_nodes: set) -> None:
        """Evaluate a single node based on its type and children."""
        node = self.nodes[node_id]

        if node.node_type == NodeType.TERMINAL:
            # Terminal nodes are already evaluated
            evaluated_nodes.add(node_id)
            return

        if node.node_type == NodeType.DECISION:
            # For decision nodes, choose best action
            best_outcome = None
            best_value = float('-inf')

            for child_id in node.children_ids:
                child_node = self.nodes[child_id]
                branch = node.branches[child_id]

                if child_node.best_outcome is not None:
                    # Apply decision criteria
                    value = await self._apply_decision_criteria(
                        child_node.best_outcome, node.decision_criteria
                    )

                    if value > best_value:
                        best_value = value
                        best_outcome = child_node.best_outcome

            if best_outcome is not None:
                node.best_outcome = best_outcome
                node.expected_value = best_outcome.expected_value
                node.utility_value = best_outcome.utility
                node.is_evaluated = True

        elif node.node_type == NodeType.CHANCE:
            # For chance nodes, compute expected value
            expected_value = 0.0
            expected_utility = 0.0
            total_probability = 0.0

            for child_id in node.children_ids:
                child_node = self.nodes[child_id]
                branch = node.branches[child_id]

                if child_node.best_outcome is not None:
                    expected_value += branch.probability * child_node.best_outcome.expected_value
                    expected_utility += branch.probability * child_node.best_outcome.utility
                    total_probability += branch.probability

            # Normalize if probabilities don't sum to 1
            if total_probability > 0:
                expected_value /= total_probability
                expected_utility /= total_probability

            # Create outcome for chance node
            outcome = DecisionOutcome(
                name=f"chance_{node.name}",
                description=f"Expected outcome for chance node {node.name}",
                expected_value=expected_value,
                utility=expected_utility,
                probability=1.0
            )

            node.best_outcome = outcome
            node.expected_value = expected_value
            node.utility_value = expected_utility
            node.is_evaluated = True

        elif node.node_type in [NodeType.ROOT, NodeType.OUTCOME]:
            # Root and outcome nodes follow decision node logic
            await self._evaluate_node_as_decision(node)

        evaluated_nodes.add(node_id)
        node.evaluation_count += 1
        node.last_evaluation = datetime.utcnow()

    async def _evaluate_node_as_decision(self, node: DecisionNode) -> None:
        """Evaluate a node using decision node logic."""
        best_outcome = None
        best_value = float('-inf')

        for child_id in node.children_ids:
            child_node = self.nodes[child_id]

            if child_node.best_outcome is not None:
                value = await self._apply_decision_criteria(
                    child_node.best_outcome, node.decision_criteria
                )

                if value > best_value:
                    best_value = value
                    best_outcome = child_node.best_outcome

        if best_outcome is not None:
            node.best_outcome = best_outcome
            node.expected_value = best_outcome.expected_value
            node.utility_value = best_outcome.utility
            node.is_evaluated = True

    async def _apply_decision_criteria(
        self,
        outcome: DecisionOutcome,
        criteria: DecisionCriteria
    ) -> float:
        """Apply decision criteria to evaluate outcome value."""
        if criteria == DecisionCriteria.MAXIMUM_EXPECTED_VALUE:
            return outcome.expected_value
        elif criteria == DecisionCriteria.MAXIMUM_UTILITY:
            return outcome.utility
        elif criteria == DecisionCriteria.EXPECTED_UTILITY:
            return outcome.expected_value * outcome.utility
        elif criteria == DecisionCriteria.MINIMAX_REGRET:
            return -outcome.regret_value
        elif criteria == DecisionCriteria.SATISFICING:
            # Return 1 if satisfactory, 0 otherwise
            return 1.0 if outcome.satisfaction_level >= self.config.risk_tolerance else 0.0
        else:
            return outcome.expected_value

    async def _update_probabilities_with_evidence(self, evidence: Dict[str, Any]) -> None:
        """Update node probabilities based on new evidence."""
        for node in self.nodes.values():
            for branch in node.branches.values():
                # Simple evidence updating - in practice, this would use
                # more sophisticated Bayesian updating methods
                if branch.condition in evidence:
                    evidence_value = evidence[branch.condition]
                    if isinstance(evidence_value, (int, float)):
                        # Update probability based on evidence strength
                        branch.probability *= evidence_value

    def _find_best_terminal_node(self) -> Optional[str]:
        """Find the terminal node with the best outcome."""
        best_node_id = None
        best_value = float('-inf')

        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.TERMINAL and node.best_outcome:
                if node.best_outcome.expected_value > best_value:
                    best_value = node.best_outcome.expected_value
                    best_node_id = node_id

        return best_node_id

    async def _prune_tree(self) -> None:
        """Prune branches with low probability or value."""
        pruned_count = 0

        for node in self.nodes.values():
            branches_to_remove = []

            for child_id, branch in node.branches.items():
                if (branch.probability < self.config.pruning_threshold or
                    branch.expected_value < -1000):  # Very negative values
                    branches_to_remove.append(child_id)

            for child_id in branches_to_remove:
                if child_id in node.children_ids:
                    node.children_ids.remove(child_id)
                del node.branches[child_id]
                branch = node.branches.get(child_id)
                if branch:
                    branch.pruned = True
                pruned_count += 1

        self.pruning_count += pruned_count

        if pruned_count > 0:
            self.logger.log(
                LogLevel.INFO,
                f"Pruned {pruned_count} branches from tree",
                operation_type=OperationType.OPTIMIZATION,
                tree_id=self.tree_id,
                pruned_count=pruned_count
            )
