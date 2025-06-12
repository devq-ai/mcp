"""
Bayesian Inference Engine Core Implementation

This module provides the core Bayesian inference engine for the Agentical framework,
implementing sophisticated probabilistic reasoning capabilities for agent decision-making.

Features:
- Core Bayesian inference engine with belief updating
- Evidence processing and hypothesis evaluation
- Prior and posterior probability management
- Integration with bayes-mcp server
- Performance optimization and caching
- Comprehensive observability and logging
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
from collections import defaultdict

import logfire
from pydantic import BaseModel, Field, validator
from scipy import stats
from scipy.special import logsumexp

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


class InferenceMethod(str, Enum):
    """Supported Bayesian inference methods."""
    EXACT = "exact"
    VARIATIONAL = "variational"
    MCMC = "mcmc"
    APPROXIMATE = "approximate"
    BELIEF_PROPAGATION = "belief_propagation"


class HypothesisStatus(str, Enum):
    """Status of hypothesis evaluation."""
    PENDING = "pending"
    EVALUATING = "evaluating"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class Evidence:
    """Represents a piece of evidence for Bayesian inference."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    value: Union[float, bool, str, Dict[str, Any]] = None
    likelihood: float = 0.0
    reliability: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = f"evidence_{self.id[:8]}"

        if not 0.0 <= self.likelihood <= 1.0:
            raise ValueError("Likelihood must be between 0 and 1")

        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Reliability must be between 0 and 1")


@dataclass
class Hypothesis:
    """Represents a hypothesis for Bayesian evaluation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    likelihood_function: Optional[Callable] = None
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence_level: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = f"hypothesis_{self.id[:8]}"

        if not 0.0 <= self.prior_probability <= 1.0:
            raise ValueError("Prior probability must be between 0 and 1")


class BayesianConfig(BaseModel):
    """Configuration for Bayesian inference engine."""

    # Core inference settings
    inference_method: InferenceMethod = InferenceMethod.EXACT
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    uncertainty_tolerance: float = Field(default=0.25, ge=0.0, le=1.0)
    max_iterations: int = Field(default=1000, ge=1)
    convergence_threshold: float = Field(default=1e-6, gt=0.0)

    # Performance settings
    enable_caching: bool = True
    cache_size: int = Field(default=1000, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1)

    # Evidence processing
    evidence_decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_evidence_count: int = Field(default=1, ge=1)

    # Logging and observability
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    metrics_collection: bool = True

    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v <= 0.5:
            raise ValueError("Confidence threshold must be greater than 0.5")
        return v


@dataclass
class InferenceResult:
    """Result of Bayesian inference computation."""
    inference_id: str = field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str = ""
    method_used: InferenceMethod = InferenceMethod.EXACT

    # Probability results
    posterior_probability: float = 0.0
    confidence_level: float = 0.0
    uncertainty_measure: float = 0.0

    # Evidence analysis
    evidence_count: int = 0
    supporting_evidence_strength: float = 0.0
    contradicting_evidence_strength: float = 0.0

    # Computation metadata
    iterations_performed: int = 0
    convergence_achieved: bool = False
    computation_time_ms: float = 0.0

    # Status and quality metrics
    success: bool = False
    confidence_achieved: bool = False
    quality_score: float = 0.0

    # Additional data
    distribution_parameters: Dict[str, float] = field(default_factory=dict)
    likelihood_values: Dict[str, float] = field(default_factory=dict)
    marginal_likelihood: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BayesianInferenceEngine:
    """
    Core Bayesian inference engine for probabilistic reasoning.

    Provides sophisticated Bayesian inference capabilities including:
    - Belief updating using Bayes' theorem
    - Evidence processing and likelihood computation
    - Hypothesis evaluation and ranking
    - Uncertainty quantification
    - Performance optimization through caching
    """

    def __init__(
        self,
        config: BayesianConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the Bayesian inference engine.

        Args:
            config: Engine configuration
            logger: Optional structured logger
        """
        self.config = config
        self.logger = logger or StructuredLogger("bayesian_engine")

        # Core components
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.evidence_store: Dict[str, Evidence] = {}
        self.inference_cache: Dict[str, InferenceResult] = {}

        # Performance tracking
        self.inference_count = 0
        self.cache_hits = 0
        self.total_computation_time = 0.0

        # State management
        self.is_initialized = False
        self.last_update = datetime.utcnow()

        logfire.info(
            "Bayesian inference engine initialized",
            method=config.inference_method,
            confidence_threshold=config.confidence_threshold
        )

    async def initialize(self) -> None:
        """Initialize the inference engine and prepare for operations."""
        with logfire.span("Initialize Bayesian Engine"):
            try:
                # Validate configuration
                await self._validate_configuration()

                # Setup inference components
                await self._setup_inference_components()

                # Initialize caching if enabled
                if self.config.enable_caching:
                    await self._initialize_cache()

                self.is_initialized = True
                self.logger.log(
                    LogLevel.INFO,
                    "Bayesian inference engine initialized successfully",
                    operation_type=OperationType.INITIALIZATION
                )

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Failed to initialize Bayesian engine: {str(e)}",
                    operation_type=OperationType.INITIALIZATION,
                    error=str(e)
                )
                raise ConfigurationError(f"Engine initialization failed: {str(e)}")

    async def add_hypothesis(
        self,
        name: str,
        description: str,
        prior_probability: float = 0.5,
        likelihood_function: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new hypothesis for evaluation.

        Args:
            name: Hypothesis name
            description: Detailed description
            prior_probability: Initial probability belief
            likelihood_function: Custom likelihood computation function
            metadata: Additional metadata

        Returns:
            Hypothesis ID
        """
        with logfire.span("Add Hypothesis", name=name):
            hypothesis = Hypothesis(
                name=name,
                description=description,
                prior_probability=prior_probability,
                likelihood_function=likelihood_function,
                metadata=metadata or {}
            )

            self.hypotheses[hypothesis.id] = hypothesis

            self.logger.log(
                LogLevel.INFO,
                f"Added hypothesis: {name}",
                operation_type=OperationType.CREATE,
                hypothesis_id=hypothesis.id,
                prior_probability=prior_probability
            )

            return hypothesis.id

    async def add_evidence(
        self,
        name: str,
        value: Union[float, bool, str, Dict[str, Any]],
        likelihood: float,
        reliability: float = 1.0,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add evidence for hypothesis evaluation.

        Args:
            name: Evidence name
            value: Evidence value
            likelihood: Likelihood of this evidence given hypothesis
            reliability: Reliability of the evidence source
            source: Evidence source identifier
            metadata: Additional metadata

        Returns:
            Evidence ID
        """
        with logfire.span("Add Evidence", name=name):
            evidence = Evidence(
                name=name,
                value=value,
                likelihood=likelihood,
                reliability=reliability,
                source=source,
                metadata=metadata or {}
            )

            self.evidence_store[evidence.id] = evidence

            self.logger.log(
                LogLevel.INFO,
                f"Added evidence: {name}",
                operation_type=OperationType.CREATE,
                evidence_id=evidence.id,
                likelihood=likelihood,
                reliability=reliability
            )

            return evidence.id

    async def compute_inference(
        self,
        hypothesis_id: str,
        evidence_ids: Optional[List[str]] = None
    ) -> InferenceResult:
        """
        Perform Bayesian inference for a hypothesis.

        Args:
            hypothesis_id: ID of hypothesis to evaluate
            evidence_ids: Optional list of specific evidence to consider

        Returns:
            Inference result with updated probabilities
        """
        start_time = datetime.utcnow()

        with logfire.span("Compute Inference", hypothesis_id=hypothesis_id):
            try:
                # Validate inputs
                if hypothesis_id not in self.hypotheses:
                    raise ValidationError(f"Hypothesis {hypothesis_id} not found")

                hypothesis = self.hypotheses[hypothesis_id]

                # Check cache if enabled
                cache_key = self._generate_cache_key(hypothesis_id, evidence_ids)
                if self.config.enable_caching and cache_key in self.inference_cache:
                    self.cache_hits += 1
                    return self.inference_cache[cache_key]

                # Gather evidence
                evidence_list = await self._gather_evidence(evidence_ids)

                # Perform inference computation
                result = await self._perform_inference(hypothesis, evidence_list)

                # Update hypothesis with results
                await self._update_hypothesis(hypothesis, result)

                # Cache result if enabled
                if self.config.enable_caching:
                    self.inference_cache[cache_key] = result

                # Update metrics
                computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                result.computation_time_ms = computation_time
                self.total_computation_time += computation_time
                self.inference_count += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"Inference completed for hypothesis {hypothesis.name}",
                    operation_type=OperationType.COMPUTATION,
                    hypothesis_id=hypothesis_id,
                    posterior_probability=result.posterior_probability,
                    confidence_level=result.confidence_level,
                    computation_time_ms=computation_time
                )

                return result

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Inference computation failed: {str(e)}",
                    operation_type=OperationType.COMPUTATION,
                    hypothesis_id=hypothesis_id,
                    error=str(e)
                )
                raise AgentError(f"Inference computation failed: {str(e)}")

    async def update_belief(
        self,
        hypothesis_id: str,
        new_evidence_id: str
    ) -> InferenceResult:
        """
        Update belief for a hypothesis with new evidence.

        Args:
            hypothesis_id: ID of hypothesis to update
            new_evidence_id: ID of new evidence

        Returns:
            Updated inference result
        """
        with logfire.span("Update Belief", hypothesis_id=hypothesis_id):
            # Get current evidence for hypothesis
            current_evidence = [
                eid for eid, evidence in self.evidence_store.items()
                if eid in self.hypotheses[hypothesis_id].supporting_evidence
            ]

            # Add new evidence
            current_evidence.append(new_evidence_id)

            # Recompute inference with updated evidence
            return await self.compute_inference(hypothesis_id, current_evidence)

    async def get_hypothesis_ranking(self) -> List[Tuple[str, float]]:
        """
        Get hypotheses ranked by posterior probability.

        Returns:
            List of (hypothesis_id, posterior_probability) tuples
        """
        with logfire.span("Get Hypothesis Ranking"):
            ranking = [
                (h_id, h.posterior_probability)
                for h_id, h in self.hypotheses.items()
            ]

            ranking.sort(key=lambda x: x[1], reverse=True)

            self.logger.log(
                LogLevel.INFO,
                f"Generated hypothesis ranking for {len(ranking)} hypotheses",
                operation_type=OperationType.QUERY,
                hypothesis_count=len(ranking)
            )

            return ranking

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine performance metrics."""
        with logfire.span("Get Engine Metrics"):
            cache_hit_rate = (
                self.cache_hits / max(self.inference_count, 1)
                if self.config.enable_caching else 0.0
            )

            avg_computation_time = (
                self.total_computation_time / max(self.inference_count, 1)
                if self.inference_count > 0 else 0.0
            )

            metrics = {
                "inference_count": self.inference_count,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "total_computation_time_ms": self.total_computation_time,
                "average_computation_time_ms": avg_computation_time,
                "hypothesis_count": len(self.hypotheses),
                "evidence_count": len(self.evidence_store),
                "cache_size": len(self.inference_cache),
                "last_update": self.last_update.isoformat()
            }

            return metrics

    # Private helper methods

    async def _validate_configuration(self) -> None:
        """Validate engine configuration parameters."""
        if self.config.confidence_threshold <= 0.5:
            raise ConfigurationError("Confidence threshold must be > 0.5")

        if self.config.uncertainty_tolerance >= 0.5:
            raise ConfigurationError("Uncertainty tolerance must be < 0.5")

    async def _setup_inference_components(self) -> None:
        """Setup core inference computation components."""
        # Initialize computation methods based on selected inference method
        if self.config.inference_method == InferenceMethod.EXACT:
            self._inference_func = self._exact_inference
        elif self.config.inference_method == InferenceMethod.VARIATIONAL:
            self._inference_func = self._variational_inference
        elif self.config.inference_method == InferenceMethod.MCMC:
            self._inference_func = self._mcmc_inference
        elif self.config.inference_method == InferenceMethod.APPROXIMATE:
            self._inference_func = self._approximate_inference
        else:
            self._inference_func = self._exact_inference

    async def _initialize_cache(self) -> None:
        """Initialize inference result caching system."""
        self.inference_cache = {}
        self.cache_hits = 0

    def _generate_cache_key(
        self,
        hypothesis_id: str,
        evidence_ids: Optional[List[str]]
    ) -> str:
        """Generate cache key for inference result."""
        evidence_key = "_".join(sorted(evidence_ids or []))
        return f"{hypothesis_id}_{evidence_key}"

    async def _gather_evidence(
        self,
        evidence_ids: Optional[List[str]]
    ) -> List[Evidence]:
        """Gather evidence for inference computation."""
        if evidence_ids is None:
            return list(self.evidence_store.values())

        evidence_list = []
        for eid in evidence_ids:
            if eid in self.evidence_store:
                evidence_list.append(self.evidence_store[eid])

        return evidence_list

    async def _perform_inference(
        self,
        hypothesis: Hypothesis,
        evidence_list: List[Evidence]
    ) -> InferenceResult:
        """Perform the actual inference computation."""
        # Call the selected inference method
        return await self._inference_func(hypothesis, evidence_list)

    async def _exact_inference(
        self,
        hypothesis: Hypothesis,
        evidence_list: List[Evidence]
    ) -> InferenceResult:
        """Perform exact Bayesian inference using Bayes' theorem."""
        if not evidence_list:
            return InferenceResult(
                hypothesis_id=hypothesis.id,
                method_used=InferenceMethod.EXACT,
                posterior_probability=hypothesis.prior_probability,
                confidence_level=0.0,
                success=True
            )

        # Compute likelihood of evidence given hypothesis
        likelihood = 1.0
        for evidence in evidence_list:
            evidence_likelihood = evidence.likelihood * evidence.reliability
            likelihood *= evidence_likelihood

        # Apply evidence decay if configured
        if self.config.evidence_decay_rate > 0:
            time_factor = self._compute_time_decay(evidence_list)
            likelihood *= time_factor

        # Compute marginal likelihood (normalization constant)
        # For simplicity, assume binary hypothesis space
        marginal_likelihood = (
            likelihood * hypothesis.prior_probability +
            (1 - likelihood) * (1 - hypothesis.prior_probability)
        )

        # Compute posterior probability using Bayes' theorem
        if marginal_likelihood > 0:
            posterior = (likelihood * hypothesis.prior_probability) / marginal_likelihood
        else:
            posterior = hypothesis.prior_probability

        # Compute confidence based on evidence strength
        evidence_strength = sum(e.likelihood * e.reliability for e in evidence_list)
        confidence = min(evidence_strength / len(evidence_list), 1.0)

        # Compute uncertainty measure
        uncertainty = 1.0 - abs(posterior - 0.5) * 2

        return InferenceResult(
            hypothesis_id=hypothesis.id,
            method_used=InferenceMethod.EXACT,
            posterior_probability=posterior,
            confidence_level=confidence,
            uncertainty_measure=uncertainty,
            evidence_count=len(evidence_list),
            marginal_likelihood=marginal_likelihood,
            success=True,
            confidence_achieved=confidence >= self.config.confidence_threshold,
            quality_score=confidence * (1 - uncertainty)
        )

    async def _variational_inference(
        self,
        hypothesis: Hypothesis,
        evidence_list: List[Evidence]
    ) -> InferenceResult:
        """Perform variational Bayesian inference."""
        # Simplified variational inference implementation
        # In practice, this would use more sophisticated variational methods
        return await self._exact_inference(hypothesis, evidence_list)

    async def _mcmc_inference(
        self,
        hypothesis: Hypothesis,
        evidence_list: List[Evidence]
    ) -> InferenceResult:
        """Perform MCMC-based Bayesian inference."""
        # Simplified MCMC implementation
        # In practice, this would use Markov Chain Monte Carlo sampling
        return await self._exact_inference(hypothesis, evidence_list)

    async def _approximate_inference(
        self,
        hypothesis: Hypothesis,
        evidence_list: List[Evidence]
    ) -> InferenceResult:
        """Perform approximate Bayesian inference."""
        # Use exact inference as approximation for now
        return await self._exact_inference(hypothesis, evidence_list)

    def _compute_time_decay(self, evidence_list: List[Evidence]) -> float:
        """Compute time-based decay factor for evidence."""
        now = datetime.utcnow()
        total_decay = 0.0

        for evidence in evidence_list:
            time_diff = (now - evidence.timestamp).total_seconds() / 3600  # hours
            decay_factor = math.exp(-self.config.evidence_decay_rate * time_diff)
            total_decay += decay_factor

        return total_decay / len(evidence_list) if evidence_list else 1.0

    async def _update_hypothesis(
        self,
        hypothesis: Hypothesis,
        result: InferenceResult
    ) -> None:
        """Update hypothesis with inference results."""
        hypothesis.posterior_probability = result.posterior_probability
        hypothesis.confidence_level = result.confidence_level
        hypothesis.updated_at = datetime.utcnow()

        # Update status based on confidence and threshold
        if result.confidence_achieved and result.posterior_probability >= self.config.confidence_threshold:
            hypothesis.status = HypothesisStatus.CONFIRMED
        elif result.confidence_achieved and result.posterior_probability <= (1 - self.config.confidence_threshold):
            hypothesis.status = HypothesisStatus.REJECTED
        elif result.uncertainty_measure <= self.config.uncertainty_tolerance:
            hypothesis.status = HypothesisStatus.UNCERTAIN
        else:
            hypothesis.status = HypothesisStatus.EVALUATING

        self.last_update = datetime.utcnow()
