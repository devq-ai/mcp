"""
Belief Updater Component for Dynamic Probability Updates

This module provides sophisticated belief updating mechanisms for the Agentical
framework's Bayesian inference system, enabling real-time probability updates
based on new evidence and changing conditions.

Features:
- Dynamic belief state management
- Multiple update strategies (sequential, batch, weighted)
- Evidence integration with temporal considerations
- Belief persistence and recovery
- Performance optimization for real-time updates
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


class EvidenceType(str, Enum):
    """Types of evidence for belief updating."""
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    CONTEXTUAL = "contextual"


class UpdateStrategy(str, Enum):
    """Strategies for belief updating."""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    KALMAN_FILTER = "kalman_filter"


class BeliefState(str, Enum):
    """States of belief during updating process."""
    STABLE = "stable"
    UPDATING = "updating"
    CONVERGING = "converging"
    OSCILLATING = "oscillating"
    DIVERGING = "diverging"


@dataclass
class BeliefUpdate:
    """Represents a single belief update operation."""
    update_id: str = field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str = ""
    evidence_id: str = ""

    # Update details
    prior_belief: float = 0.0
    posterior_belief: float = 0.0
    belief_change: float = 0.0
    confidence_change: float = 0.0

    # Evidence information
    evidence_type: EvidenceType = EvidenceType.NEUTRAL
    evidence_strength: float = 0.0
    evidence_reliability: float = 1.0

    # Update metadata
    update_strategy: UpdateStrategy = UpdateStrategy.SEQUENTIAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0

    # Quality metrics
    convergence_indicator: float = 0.0
    stability_score: float = 0.0
    update_quality: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefHistory:
    """Tracks belief update history for analysis."""
    hypothesis_id: str = ""
    belief_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    update_count: int = 0

    # Stability metrics
    variance: float = 0.0
    trend: float = 0.0
    oscillation_frequency: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class BeliefUpdaterConfig(BaseModel):
    """Configuration for belief updater component."""

    # Update strategy settings
    default_strategy: UpdateStrategy = UpdateStrategy.SEQUENTIAL
    enable_adaptive_strategy: bool = True
    convergence_threshold: float = Field(default=1e-4, gt=0.0)
    stability_window: int = Field(default=10, ge=1)

    # Evidence processing
    evidence_decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_discount_factor: float = Field(default=0.95, ge=0.0, le=1.0)

    # Performance settings
    max_history_size: int = Field(default=1000, ge=10)
    batch_size: int = Field(default=10, ge=1)
    update_timeout_ms: float = Field(default=100.0, gt=0.0)
    enable_parallel_updates: bool = True

    # Stability detection
    oscillation_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    divergence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_stability_period: int = Field(default=5, ge=1)

    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_update_frequency: int = Field(default=10, ge=1)
    performance_monitoring: bool = True


class BeliefUpdater:
    """
    Advanced belief updater for dynamic probability management.

    Provides sophisticated belief updating capabilities including:
    - Real-time belief state management
    - Multiple update strategies with adaptive selection
    - Evidence integration with temporal considerations
    - Stability detection and oscillation prevention
    - Performance optimization for high-frequency updates
    """

    def __init__(
        self,
        config: BeliefUpdaterConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the belief updater.

        Args:
            config: Updater configuration
            logger: Optional structured logger
        """
        self.config = config
        self.logger = logger or StructuredLogger("belief_updater")

        # Core state management
        self.belief_states: Dict[str, float] = {}
        self.confidence_states: Dict[str, float] = {}
        self.update_histories: Dict[str, BeliefHistory] = {}
        self.current_states: Dict[str, BeliefState] = {}

        # Update tracking
        self.pending_updates: Dict[str, List[BeliefUpdate]] = defaultdict(list)
        self.update_count = 0
        self.last_update_time = datetime.utcnow()

        # Performance metrics
        self.total_update_time = 0.0
        self.convergence_count = 0
        self.oscillation_count = 0

        logfire.info(
            "Belief updater initialized",
            strategy=config.default_strategy,
            convergence_threshold=config.convergence_threshold
        )

    async def initialize_belief(
        self,
        hypothesis_id: str,
        initial_belief: float = 0.5,
        initial_confidence: float = 0.0
    ) -> None:
        """
        Initialize belief state for a hypothesis.

        Args:
            hypothesis_id: Unique hypothesis identifier
            initial_belief: Initial belief probability
            initial_confidence: Initial confidence level
        """
        with logfire.span("Initialize Belief", hypothesis_id=hypothesis_id):
            if not 0.0 <= initial_belief <= 1.0:
                raise ValidationError("Initial belief must be between 0 and 1")

            if not 0.0 <= initial_confidence <= 1.0:
                raise ValidationError("Initial confidence must be between 0 and 1")

            self.belief_states[hypothesis_id] = initial_belief
            self.confidence_states[hypothesis_id] = initial_confidence
            self.current_states[hypothesis_id] = BeliefState.STABLE

            # Initialize history tracking
            history = BeliefHistory(hypothesis_id=hypothesis_id)
            history.belief_values.append(initial_belief)
            history.confidence_values.append(initial_confidence)
            history.timestamps.append(datetime.utcnow())
            self.update_histories[hypothesis_id] = history

            self.logger.log(
                LogLevel.INFO,
                f"Initialized belief for hypothesis {hypothesis_id}",
                operation_type=OperationType.INITIALIZATION,
                hypothesis_id=hypothesis_id,
                initial_belief=initial_belief,
                initial_confidence=initial_confidence
            )

    async def update_belief(
        self,
        hypothesis_id: str,
        evidence_id: str,
        evidence_type: EvidenceType,
        evidence_strength: float,
        evidence_reliability: float = 1.0,
        strategy: Optional[UpdateStrategy] = None
    ) -> BeliefUpdate:
        """
        Update belief based on new evidence.

        Args:
            hypothesis_id: Hypothesis to update
            evidence_id: Evidence identifier
            evidence_type: Type of evidence
            evidence_strength: Strength of evidence (0.0 to 1.0)
            evidence_reliability: Reliability of evidence source
            strategy: Optional update strategy override

        Returns:
            Belief update result
        """
        start_time = datetime.utcnow()

        with logfire.span("Update Belief", hypothesis_id=hypothesis_id):
            try:
                # Validate inputs
                if hypothesis_id not in self.belief_states:
                    raise ValidationError(f"Hypothesis {hypothesis_id} not initialized")

                if not 0.0 <= evidence_strength <= 1.0:
                    raise ValidationError("Evidence strength must be between 0 and 1")

                if not 0.0 <= evidence_reliability <= 1.0:
                    raise ValidationError("Evidence reliability must be between 0 and 1")

                # Get current belief state
                prior_belief = self.belief_states[hypothesis_id]
                prior_confidence = self.confidence_states[hypothesis_id]

                # Select update strategy
                update_strategy = strategy or self._select_update_strategy(hypothesis_id)

                # Perform belief update
                posterior_belief, posterior_confidence = await self._perform_update(
                    hypothesis_id,
                    prior_belief,
                    prior_confidence,
                    evidence_type,
                    evidence_strength,
                    evidence_reliability,
                    update_strategy
                )

                # Create update record
                update = BeliefUpdate(
                    hypothesis_id=hypothesis_id,
                    evidence_id=evidence_id,
                    prior_belief=prior_belief,
                    posterior_belief=posterior_belief,
                    belief_change=abs(posterior_belief - prior_belief),
                    confidence_change=abs(posterior_confidence - prior_confidence),
                    evidence_type=evidence_type,
                    evidence_strength=evidence_strength,
                    evidence_reliability=evidence_reliability,
                    update_strategy=update_strategy
                )

                # Update states
                self.belief_states[hypothesis_id] = posterior_belief
                self.confidence_states[hypothesis_id] = posterior_confidence

                # Update history
                await self._update_history(hypothesis_id, posterior_belief, posterior_confidence)

                # Analyze stability
                await self._analyze_stability(hypothesis_id)

                # Record metrics
                computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                update.computation_time_ms = computation_time
                self.total_update_time += computation_time
                self.update_count += 1

                # Calculate quality metrics
                update.convergence_indicator = await self._calculate_convergence(hypothesis_id)
                update.stability_score = await self._calculate_stability(hypothesis_id)
                update.update_quality = (update.stability_score + (1 - update.convergence_indicator)) / 2

                self.logger.log(
                    LogLevel.INFO,
                    f"Updated belief for hypothesis {hypothesis_id}",
                    operation_type=OperationType.UPDATE,
                    hypothesis_id=hypothesis_id,
                    prior_belief=prior_belief,
                    posterior_belief=posterior_belief,
                    belief_change=update.belief_change,
                    evidence_type=evidence_type.value,
                    computation_time_ms=computation_time
                )

                return update

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Belief update failed: {str(e)}",
                    operation_type=OperationType.UPDATE,
                    hypothesis_id=hypothesis_id,
                    error=str(e)
                )
                raise AgentError(f"Belief update failed: {str(e)}")

    async def batch_update(
        self,
        updates: List[Tuple[str, str, EvidenceType, float, float]]
    ) -> List[BeliefUpdate]:
        """
        Perform batch belief updates for efficiency.

        Args:
            updates: List of (hypothesis_id, evidence_id, evidence_type, strength, reliability)

        Returns:
            List of belief update results
        """
        with logfire.span("Batch Update", update_count=len(updates)):
            results = []

            if self.config.enable_parallel_updates:
                # Parallel processing for independent updates
                tasks = []
                for hypothesis_id, evidence_id, evidence_type, strength, reliability in updates:
                    task = self.update_belief(
                        hypothesis_id, evidence_id, evidence_type, strength, reliability,
                        UpdateStrategy.BATCH
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions
                successful_results = [r for r in results if isinstance(r, BeliefUpdate)]

            else:
                # Sequential processing
                for hypothesis_id, evidence_id, evidence_type, strength, reliability in updates:
                    try:
                        result = await self.update_belief(
                            hypothesis_id, evidence_id, evidence_type, strength, reliability,
                            UpdateStrategy.BATCH
                        )
                        results.append(result)
                    except Exception as e:
                        self.logger.log(
                            LogLevel.WARNING,
                            f"Batch update failed for hypothesis {hypothesis_id}: {str(e)}",
                            operation_type=OperationType.UPDATE,
                            hypothesis_id=hypothesis_id,
                            error=str(e)
                        )

                successful_results = results

            self.logger.log(
                LogLevel.INFO,
                f"Batch update completed: {len(successful_results)}/{len(updates)} successful",
                operation_type=OperationType.BATCH_UPDATE,
                total_updates=len(updates),
                successful_updates=len(successful_results)
            )

            return successful_results

    async def get_belief_state(self, hypothesis_id: str) -> Tuple[float, float, BeliefState]:
        """
        Get current belief state for a hypothesis.

        Args:
            hypothesis_id: Hypothesis identifier

        Returns:
            Tuple of (belief, confidence, state)
        """
        if hypothesis_id not in self.belief_states:
            raise ValidationError(f"Hypothesis {hypothesis_id} not found")

        return (
            self.belief_states[hypothesis_id],
            self.confidence_states[hypothesis_id],
            self.current_states[hypothesis_id]
        )

    async def get_belief_history(self, hypothesis_id: str) -> BeliefHistory:
        """
        Get belief update history for a hypothesis.

        Args:
            hypothesis_id: Hypothesis identifier

        Returns:
            Belief history object
        """
        if hypothesis_id not in self.update_histories:
            raise ValidationError(f"No history found for hypothesis {hypothesis_id}")

        return self.update_histories[hypothesis_id]

    async def get_convergence_analysis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Analyze convergence properties of belief updates.

        Args:
            hypothesis_id: Hypothesis identifier

        Returns:
            Convergence analysis results
        """
        with logfire.span("Convergence Analysis", hypothesis_id=hypothesis_id):
            if hypothesis_id not in self.update_histories:
                raise ValidationError(f"No history found for hypothesis {hypothesis_id}")

            history = self.update_histories[hypothesis_id]

            if len(history.belief_values) < 2:
                return {"status": "insufficient_data", "analysis": {}}

            beliefs = list(history.belief_values)

            # Calculate convergence metrics
            recent_variance = np.var(beliefs[-self.config.stability_window:])
            overall_variance = np.var(beliefs)

            # Trend analysis
            if len(beliefs) >= 3:
                x = np.arange(len(beliefs))
                trend_slope = np.polyfit(x, beliefs, 1)[0]
            else:
                trend_slope = 0.0

            # Oscillation detection
            oscillation_count = 0
            for i in range(1, len(beliefs) - 1):
                if (beliefs[i] > beliefs[i-1] and beliefs[i] > beliefs[i+1]) or \
                   (beliefs[i] < beliefs[i-1] and beliefs[i] < beliefs[i+1]):
                    oscillation_count += 1

            oscillation_frequency = oscillation_count / max(len(beliefs) - 2, 1)

            # Convergence status
            is_converging = recent_variance < self.config.convergence_threshold
            is_stable = recent_variance < overall_variance * 0.5
            is_oscillating = oscillation_frequency > self.config.oscillation_threshold

            analysis = {
                "status": "converging" if is_converging else "diverging",
                "analysis": {
                    "recent_variance": recent_variance,
                    "overall_variance": overall_variance,
                    "trend_slope": trend_slope,
                    "oscillation_frequency": oscillation_frequency,
                    "is_converging": is_converging,
                    "is_stable": is_stable,
                    "is_oscillating": is_oscillating,
                    "update_count": len(beliefs),
                    "stability_window": self.config.stability_window
                }
            }

            return analysis

    async def get_updater_metrics(self) -> Dict[str, Any]:
        """Get comprehensive updater performance metrics."""
        with logfire.span("Get Updater Metrics"):
            avg_update_time = (
                self.total_update_time / max(self.update_count, 1)
                if self.update_count > 0 else 0.0
            )

            metrics = {
                "update_count": self.update_count,
                "convergence_count": self.convergence_count,
                "oscillation_count": self.oscillation_count,
                "total_update_time_ms": self.total_update_time,
                "average_update_time_ms": avg_update_time,
                "tracked_hypotheses": len(self.belief_states),
                "last_update_time": self.last_update_time.isoformat(),
                "convergence_rate": self.convergence_count / max(self.update_count, 1),
                "oscillation_rate": self.oscillation_count / max(self.update_count, 1)
            }

            return metrics

    # Private helper methods

    def _select_update_strategy(self, hypothesis_id: str) -> UpdateStrategy:
        """Select optimal update strategy based on hypothesis state."""
        if not self.config.enable_adaptive_strategy:
            return self.config.default_strategy

        current_state = self.current_states.get(hypothesis_id, BeliefState.STABLE)

        if current_state == BeliefState.OSCILLATING:
            return UpdateStrategy.EXPONENTIAL_SMOOTHING
        elif current_state == BeliefState.DIVERGING:
            return UpdateStrategy.WEIGHTED_AVERAGE
        elif current_state == BeliefState.CONVERGING:
            return UpdateStrategy.SEQUENTIAL
        else:
            return self.config.default_strategy

    async def _perform_update(
        self,
        hypothesis_id: str,
        prior_belief: float,
        prior_confidence: float,
        evidence_type: EvidenceType,
        evidence_strength: float,
        evidence_reliability: float,
        strategy: UpdateStrategy
    ) -> Tuple[float, float]:
        """Perform the actual belief update computation."""

        # Apply reliability weighting
        weighted_strength = evidence_strength * evidence_reliability * self.config.reliability_weight

        # Apply temporal discount if configured
        temporal_factor = self.config.temporal_discount_factor
        effective_strength = weighted_strength * temporal_factor

        if strategy == UpdateStrategy.SEQUENTIAL:
            return await self._sequential_update(
                prior_belief, prior_confidence, evidence_type, effective_strength
            )
        elif strategy == UpdateStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_update(
                hypothesis_id, prior_belief, prior_confidence, evidence_type, effective_strength
            )
        elif strategy == UpdateStrategy.EXPONENTIAL_SMOOTHING:
            return await self._exponential_smoothing_update(
                hypothesis_id, prior_belief, prior_confidence, evidence_type, effective_strength
            )
        else:
            # Default to sequential
            return await self._sequential_update(
                prior_belief, prior_confidence, evidence_type, effective_strength
            )

    async def _sequential_update(
        self,
        prior_belief: float,
        prior_confidence: float,
        evidence_type: EvidenceType,
        evidence_strength: float
    ) -> Tuple[float, float]:
        """Perform sequential Bayesian update."""

        if evidence_type == EvidenceType.SUPPORTING:
            # Evidence supports the hypothesis
            likelihood = evidence_strength
            posterior = (likelihood * prior_belief) / (
                likelihood * prior_belief + (1 - likelihood) * (1 - prior_belief)
            )
        elif evidence_type == EvidenceType.CONTRADICTING:
            # Evidence contradicts the hypothesis
            likelihood = 1 - evidence_strength
            posterior = (likelihood * prior_belief) / (
                likelihood * prior_belief + (1 - likelihood) * (1 - prior_belief)
            )
        else:
            # Neutral or contextual evidence
            posterior = prior_belief

        # Update confidence based on evidence strength
        confidence_increase = evidence_strength * (1 - prior_confidence)
        posterior_confidence = min(prior_confidence + confidence_increase, 1.0)

        return posterior, posterior_confidence

    async def _weighted_average_update(
        self,
        hypothesis_id: str,
        prior_belief: float,
        prior_confidence: float,
        evidence_type: EvidenceType,
        evidence_strength: float
    ) -> Tuple[float, float]:
        """Perform weighted average update to reduce oscillation."""

        # Get recent belief values for averaging
        history = self.update_histories.get(hypothesis_id)
        if history and len(history.belief_values) > 1:
            recent_beliefs = list(history.belief_values)[-self.config.stability_window:]
            weight = 0.7  # Weight for historical average
            historical_average = np.mean(recent_beliefs)
        else:
            historical_average = prior_belief
            weight = 0.0

        # Compute new belief using sequential update
        new_belief, new_confidence = await self._sequential_update(
            prior_belief, prior_confidence, evidence_type, evidence_strength
        )

        # Combine with historical average
        posterior = weight * historical_average + (1 - weight) * new_belief

        return posterior, new_confidence

    async def _exponential_smoothing_update(
        self,
        hypothesis_id: str,
        prior_belief: float,
        prior_confidence: float,
        evidence_type: EvidenceType,
        evidence_strength: float
    ) -> Tuple[float, float]:
        """Perform exponential smoothing update for stability."""

        # Exponential smoothing parameter (higher = more responsive)
        alpha = 0.3

        # Compute new belief using sequential update
        new_belief, new_confidence = await self._sequential_update(
            prior_belief, prior_confidence, evidence_type, evidence_strength
        )

        # Apply exponential smoothing
        posterior = alpha * new_belief + (1 - alpha) * prior_belief
        posterior_confidence = alpha * new_confidence + (1 - alpha) * prior_confidence

        return posterior, posterior_confidence

    async def _update_history(
        self,
        hypothesis_id: str,
        belief: float,
        confidence: float
    ) -> None:
        """Update belief history for a hypothesis."""
        history = self.update_histories[hypothesis_id]

        history.belief_values.append(belief)
        history.confidence_values.append(confidence)
        history.timestamps.append(datetime.utcnow())
        history.update_count += 1
        history.last_updated = datetime.utcnow()

        # Update variance calculation
        if len(history.belief_values) > 1:
            history.variance = np.var(list(history.belief_values))

    async def _analyze_stability(self, hypothesis_id: str) -> None:
        """Analyze and update stability state for a hypothesis."""
        history = self.update_histories[hypothesis_id]

        if len(history.belief_values) < self.config.min_stability_period:
            self.current_states[hypothesis_id] = BeliefState.UPDATING
            return

        recent_beliefs = list(history.belief_values)[-self.config.stability_window:]
        recent_variance = np.var(recent_beliefs)

        # Check for convergence
        if recent_variance < self.config.convergence_threshold:
            self.current_states[hypothesis_id] = BeliefState.CONVERGING
            self.convergence_count += 1

        # Check for oscillation
        elif recent_variance > self.config.oscillation_threshold:
            # Count direction changes
            direction_changes = 0
            for i in range(1, len(recent_beliefs) - 1):
                if (recent_beliefs[i] > recent_beliefs[i-1] and recent_beliefs[i] > recent_beliefs[i+1]) or \
                   (recent_beliefs[i] < recent_beliefs[i-1] and recent_beliefs[i] < recent_beliefs[i+1]):
                    direction_changes += 1

            if direction_changes >= len(recent_beliefs) * 0.3:
                self.current_states[hypothesis_id] = BeliefState.OSCILLATING
                self.oscillation_count += 1
            else:
                self.current_states[hypothesis_id] = BeliefState.UPDATING

        # Check for divergence
        elif recent_variance > self.config.divergence_threshold:
            self.current_states[hypothesis_id] = BeliefState.DIVERGING

        else:
            self.current_states[hypothesis_id] = BeliefState.STABLE

    async def _calculate_convergence(self, hypothesis_id: str) -> float:
        """Calculate convergence indicator for a hypothesis."""
        history = self.update_histories[hypothesis_id]

        if len(history.belief_values) < 2:
            return 1.0  # No convergence data

        recent_beliefs = list(history.belief_values)[-self.config.stability_window:]
        variance = np.var(recent_beliefs)

        # Normalize variance to 0-1 scale (lower variance = higher convergence)
        convergence = max(0.0, min(1.0, variance / self.config.convergence_threshold))

        return convergence

    async def _calculate_stability(self, hypothesis_id: str) -> float:
        """Calculate stability score for a hypothesis."""
        history = self.update_histories[hypothesis_id]

        if len(history.belief_values) < 3:
            return 0.5  # Neutral stability for insufficient data

        beliefs = list(history.belief_values)

        # Calculate stability as inverse of normalized variance
        variance = np.var(beliefs)
        max_possible_variance = 0.25  # Maximum variance for uniform distribution

        stability = 1.0 - min(1.0, variance / max_possible_variance)

        return stability
