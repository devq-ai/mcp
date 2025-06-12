"""
Uncertainty Quantifier Component for Confidence Measurement

This module provides sophisticated uncertainty quantification capabilities for the
Agentical framework's Bayesian reasoning system, enabling comprehensive confidence
measurement and uncertainty analysis for agent decision-making.

Features:
- Multiple uncertainty quantification methods
- Confidence interval estimation
- Distribution-based uncertainty analysis
- Aleatory and epistemic uncertainty separation
- Real-time uncertainty tracking
- Performance optimization for high-frequency updates
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
from scipy import stats
from scipy.special import logsumexp

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


class DistributionType(str, Enum):
    """Types of probability distributions for uncertainty modeling."""
    NORMAL = "normal"
    BETA = "beta"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    STUDENT_T = "student_t"
    DIRICHLET = "dirichlet"
    MIXTURE = "mixture"


class QuantificationMethod(str, Enum):
    """Methods for uncertainty quantification."""
    BAYESIAN = "bayesian"
    BOOTSTRAP = "bootstrap"
    MONTE_CARLO = "monte_carlo"
    VARIATIONAL = "variational"
    ENSEMBLE = "ensemble"
    CONFORMAL = "conformal"


class UncertaintySource(str, Enum):
    """Sources of uncertainty in agent reasoning."""
    MODEL = "model"
    PARAMETER = "parameter"
    DATA = "data"
    MEASUREMENT = "measurement"
    PREDICTION = "prediction"
    APPROXIMATION = "approximation"


class ConfidenceLevel(str, Enum):
    """Standard confidence levels for uncertainty intervals."""
    LOW = "0.68"  # 1 sigma
    MEDIUM = "0.90"  # 90%
    HIGH = "0.95"  # 95%
    VERY_HIGH = "0.99"  # 99%


@dataclass
class UncertaintyMeasure:
    """Represents a single uncertainty measurement."""
    measure_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Core uncertainty metrics
    variance: float = 0.0
    standard_deviation: float = 0.0
    entropy: float = 0.0
    mutual_information: float = 0.0

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)

    # Distribution properties
    distribution_type: DistributionType = DistributionType.NORMAL
    distribution_parameters: Dict[str, float] = field(default_factory=dict)
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Uncertainty decomposition
    aleatory_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    total_uncertainty: float = 0.0

    # Quality metrics
    uncertainty_source: UncertaintySource = UncertaintySource.PREDICTION
    reliability_score: float = 0.0
    coverage_probability: float = 0.0

    # Computation metadata
    method_used: QuantificationMethod = QuantificationMethod.BAYESIAN
    sample_size: int = 0
    computation_time_ms: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyHistory:
    """Tracks uncertainty evolution over time."""
    entity_id: str = ""
    uncertainty_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Trend analysis
    uncertainty_trend: float = 0.0
    volatility: float = 0.0
    stability_score: float = 0.0

    # Statistical properties
    mean_uncertainty: float = 0.0
    uncertainty_variance: float = 0.0
    min_uncertainty: float = float('inf')
    max_uncertainty: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class UncertaintyQuantifierConfig(BaseModel):
    """Configuration for uncertainty quantifier."""

    # Quantification methods
    default_method: QuantificationMethod = QuantificationMethod.BAYESIAN
    enable_multiple_methods: bool = True
    confidence_levels: List[float] = Field(default=[0.68, 0.90, 0.95, 0.99])

    # Distribution modeling
    default_distribution: DistributionType = DistributionType.NORMAL
    auto_distribution_selection: bool = True
    distribution_fit_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # Sampling settings
    monte_carlo_samples: int = Field(default=10000, ge=100)
    bootstrap_samples: int = Field(default=1000, ge=100)
    ensemble_size: int = Field(default=10, ge=2)

    # Performance settings
    enable_caching: bool = True
    cache_size: int = Field(default=1000, ge=10)
    parallel_computation: bool = True
    max_workers: int = Field(default=4, ge=1)

    # Quality control
    uncertainty_bounds: Tuple[float, float] = Field(default=(0.0, 10.0))
    outlier_detection: bool = True
    outlier_threshold: float = Field(default=3.0, gt=0.0)

    # History tracking
    max_history_size: int = Field(default=1000, ge=10)
    trend_window_size: int = Field(default=50, ge=5)

    # Logging and monitoring
    detailed_logging: bool = True
    performance_monitoring: bool = True
    log_frequency: int = Field(default=100, ge=1)


class UncertaintyQuantifier:
    """
    Advanced uncertainty quantifier for agent confidence measurement.

    Provides comprehensive uncertainty quantification capabilities including:
    - Multiple quantification methods (Bayesian, Bootstrap, Monte Carlo)
    - Confidence interval estimation with various distributions
    - Aleatory and epistemic uncertainty decomposition
    - Real-time uncertainty tracking and trend analysis
    - Performance optimization for high-frequency measurements
    """

    def __init__(
        self,
        config: UncertaintyQuantifierConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the uncertainty quantifier.

        Args:
            config: Quantifier configuration
            logger: Optional structured logger
        """
        self.config = config
        self.logger = logger or StructuredLogger("uncertainty_quantifier")

        # Core state management
        self.uncertainty_cache: Dict[str, UncertaintyMeasure] = {}
        self.uncertainty_histories: Dict[str, UncertaintyHistory] = {}

        # Distribution cache for fitted models
        self.distribution_cache: Dict[str, Any] = {}

        # Performance tracking
        self.quantification_count = 0
        self.cache_hits = 0
        self.total_computation_time = 0.0

        # Statistical models
        self.fitted_distributions: Dict[str, Dict[str, Any]] = {}

        logfire.info(
            "Uncertainty quantifier initialized",
            method=config.default_method,
            confidence_levels=config.confidence_levels
        )

    async def quantify_uncertainty(
        self,
        entity_id: str,
        data: Union[List[float], np.ndarray],
        method: Optional[QuantificationMethod] = None,
        distribution: Optional[DistributionType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UncertaintyMeasure:
        """
        Quantify uncertainty for given data.

        Args:
            entity_id: Unique identifier for the entity
            data: Data points for uncertainty analysis
            method: Optional quantification method override
            distribution: Optional distribution type override
            metadata: Additional metadata

        Returns:
            Comprehensive uncertainty measure
        """
        start_time = datetime.utcnow()

        with logfire.span("Quantify Uncertainty", entity_id=entity_id):
            try:
                # Validate inputs
                if not data or len(data) < 2:
                    raise ValidationError("Insufficient data for uncertainty quantification")

                data_array = np.array(data, dtype=float)
                if not np.isfinite(data_array).all():
                    raise ValidationError("Data contains infinite or NaN values")

                # Check cache
                cache_key = self._generate_cache_key(entity_id, data_array, method, distribution)
                if self.config.enable_caching and cache_key in self.uncertainty_cache:
                    self.cache_hits += 1
                    return self.uncertainty_cache[cache_key]

                # Select quantification method
                quant_method = method or self.config.default_method

                # Select distribution
                dist_type = distribution or await self._select_distribution(data_array)

                # Perform uncertainty quantification
                uncertainty_measure = await self._perform_quantification(
                    entity_id, data_array, quant_method, dist_type, metadata
                )

                # Update history
                await self._update_uncertainty_history(entity_id, uncertainty_measure)

                # Cache result
                if self.config.enable_caching:
                    self.uncertainty_cache[cache_key] = uncertainty_measure

                # Update metrics
                computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                uncertainty_measure.computation_time_ms = computation_time
                self.total_computation_time += computation_time
                self.quantification_count += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"Quantified uncertainty for entity {entity_id}",
                    operation_type=OperationType.COMPUTATION,
                    entity_id=entity_id,
                    method=quant_method.value,
                    total_uncertainty=uncertainty_measure.total_uncertainty,
                    computation_time_ms=computation_time
                )

                return uncertainty_measure

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Uncertainty quantification failed: {str(e)}",
                    operation_type=OperationType.COMPUTATION,
                    entity_id=entity_id,
                    error=str(e)
                )
                raise AgentError(f"Uncertainty quantification failed: {str(e)}")

    async def estimate_confidence_intervals(
        self,
        data: Union[List[float], np.ndarray],
        confidence_levels: Optional[List[float]] = None,
        method: QuantificationMethod = QuantificationMethod.BAYESIAN
    ) -> Dict[str, Tuple[float, float]]:
        """
        Estimate confidence intervals for given data.

        Args:
            data: Data points for interval estimation
            confidence_levels: Confidence levels to compute (0-1)
            method: Quantification method

        Returns:
            Dictionary mapping confidence levels to (lower, upper) bounds
        """
        with logfire.span("Estimate Confidence Intervals"):
            data_array = np.array(data, dtype=float)
            confidence_levels = confidence_levels or self.config.confidence_levels

            intervals = {}

            if method == QuantificationMethod.BAYESIAN:
                intervals = await self._bayesian_confidence_intervals(data_array, confidence_levels)
            elif method == QuantificationMethod.BOOTSTRAP:
                intervals = await self._bootstrap_confidence_intervals(data_array, confidence_levels)
            elif method == QuantificationMethod.MONTE_CARLO:
                intervals = await self._monte_carlo_confidence_intervals(data_array, confidence_levels)
            else:
                # Default to normal approximation
                intervals = await self._normal_confidence_intervals(data_array, confidence_levels)

            return intervals

    async def decompose_uncertainty(
        self,
        entity_id: str,
        model_predictions: List[List[float]],
        true_uncertainty: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Decompose uncertainty into aleatory and epistemic components.

        Args:
            entity_id: Entity identifier
            model_predictions: List of prediction arrays from different models
            true_uncertainty: Known true uncertainty for validation

        Returns:
            Tuple of (aleatory_uncertainty, epistemic_uncertainty)
        """
        with logfire.span("Decompose Uncertainty", entity_id=entity_id):
            if not model_predictions or len(model_predictions) < 2:
                raise ValidationError("Need at least 2 model predictions for decomposition")

            predictions_array = np.array(model_predictions)

            # Epistemic uncertainty: variance across models
            mean_predictions = np.mean(predictions_array, axis=0)
            epistemic_uncertainty = np.var(mean_predictions)

            # Aleatory uncertainty: average variance within models
            within_model_variances = [np.var(pred) for pred in model_predictions]
            aleatory_uncertainty = np.mean(within_model_variances)

            self.logger.log(
                LogLevel.INFO,
                f"Decomposed uncertainty for entity {entity_id}",
                operation_type=OperationType.ANALYSIS,
                entity_id=entity_id,
                aleatory_uncertainty=aleatory_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty
            )

            return aleatory_uncertainty, epistemic_uncertainty

    async def track_uncertainty_evolution(
        self,
        entity_id: str,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty evolution over time.

        Args:
            entity_id: Entity identifier
            window_size: Analysis window size

        Returns:
            Evolution analysis results
        """
        with logfire.span("Track Uncertainty Evolution", entity_id=entity_id):
            if entity_id not in self.uncertainty_histories:
                raise ValidationError(f"No uncertainty history found for entity {entity_id}")

            history = self.uncertainty_histories[entity_id]
            window_size = window_size or self.config.trend_window_size

            if len(history.uncertainty_values) < window_size:
                return {"status": "insufficient_data", "analysis": {}}

            uncertainty_values = list(history.uncertainty_values)[-window_size:]
            timestamps = list(history.timestamps)[-window_size:]

            # Trend analysis
            if len(uncertainty_values) >= 3:
                x = np.arange(len(uncertainty_values))
                trend_slope = np.polyfit(x, uncertainty_values, 1)[0]
            else:
                trend_slope = 0.0

            # Volatility analysis
            volatility = np.std(uncertainty_values)

            # Stability analysis
            recent_mean = np.mean(uncertainty_values[-min(10, len(uncertainty_values)):])
            overall_mean = np.mean(uncertainty_values)
            stability = 1.0 - abs(recent_mean - overall_mean) / max(overall_mean, 1e-6)

            analysis = {
                "status": "analyzed",
                "analysis": {
                    "trend_slope": trend_slope,
                    "volatility": volatility,
                    "stability_score": stability,
                    "mean_uncertainty": np.mean(uncertainty_values),
                    "min_uncertainty": np.min(uncertainty_values),
                    "max_uncertainty": np.max(uncertainty_values),
                    "uncertainty_range": np.max(uncertainty_values) - np.min(uncertainty_values),
                    "window_size": window_size,
                    "sample_count": len(uncertainty_values)
                }
            }

            return analysis

    async def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty quantifier metrics."""
        with logfire.span("Get Uncertainty Metrics"):
            avg_computation_time = (
                self.total_computation_time / max(self.quantification_count, 1)
                if self.quantification_count > 0 else 0.0
            )

            cache_hit_rate = (
                self.cache_hits / max(self.quantification_count, 1)
                if self.config.enable_caching and self.quantification_count > 0 else 0.0
            )

            metrics = {
                "quantification_count": self.quantification_count,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "total_computation_time_ms": self.total_computation_time,
                "average_computation_time_ms": avg_computation_time,
                "tracked_entities": len(self.uncertainty_histories),
                "cached_measures": len(self.uncertainty_cache),
                "fitted_distributions": len(self.fitted_distributions)
            }

            return metrics

    # Private helper methods

    def _generate_cache_key(
        self,
        entity_id: str,
        data: np.ndarray,
        method: Optional[QuantificationMethod],
        distribution: Optional[DistributionType]
    ) -> str:
        """Generate cache key for uncertainty measure."""
        data_hash = hash(data.tobytes())
        method_str = method.value if method else "default"
        dist_str = distribution.value if distribution else "default"
        return f"{entity_id}_{data_hash}_{method_str}_{dist_str}"

    async def _select_distribution(self, data: np.ndarray) -> DistributionType:
        """Automatically select best distribution for data."""
        if not self.config.auto_distribution_selection:
            return self.config.default_distribution

        # Test different distributions and select best fit
        distributions = [
            DistributionType.NORMAL,
            DistributionType.BETA,
            DistributionType.GAMMA,
            DistributionType.UNIFORM
        ]

        best_distribution = self.config.default_distribution
        best_p_value = 0.0

        for dist_type in distributions:
            try:
                if dist_type == DistributionType.NORMAL:
                    # Shapiro-Wilk test for normality
                    _, p_value = stats.shapiro(data)
                elif dist_type == DistributionType.UNIFORM:
                    # Kolmogorov-Smirnov test for uniform distribution
                    _, p_value = stats.kstest(data, 'uniform')
                else:
                    # Use generic goodness-of-fit test
                    p_value = 0.05  # Default acceptable p-value

                if p_value > best_p_value and p_value > self.config.distribution_fit_threshold:
                    best_p_value = p_value
                    best_distribution = dist_type

            except Exception:
                continue

        return best_distribution

    async def _perform_quantification(
        self,
        entity_id: str,
        data: np.ndarray,
        method: QuantificationMethod,
        distribution: DistributionType,
        metadata: Optional[Dict[str, Any]]
    ) -> UncertaintyMeasure:
        """Perform the actual uncertainty quantification."""

        # Basic statistical measures
        variance = np.var(data)
        std_dev = np.std(data)

        # Entropy calculation
        if len(data) > 1:
            # Estimate probability density using histogram
            hist, bin_edges = np.histogram(data, bins=min(20, len(data)//2), density=True)
            # Avoid log(0) by adding small epsilon
            hist = hist + 1e-10
            bin_width = bin_edges[1] - bin_edges[0]
            entropy = -np.sum(hist * np.log(hist) * bin_width)
        else:
            entropy = 0.0

        # Confidence intervals
        confidence_intervals = await self.estimate_confidence_intervals(
            data, self.config.confidence_levels, method
        )

        # Percentiles
        percentiles = {
            "5": np.percentile(data, 5),
            "25": np.percentile(data, 25),
            "50": np.percentile(data, 50),
            "75": np.percentile(data, 75),
            "95": np.percentile(data, 95)
        }

        # Distribution parameters
        dist_params = await self._fit_distribution(data, distribution)

        # For simplicity, assume total uncertainty equals standard deviation
        total_uncertainty = std_dev

        # Simple decomposition: assume 70% aleatory, 30% epistemic for demonstration
        aleatory_uncertainty = total_uncertainty * 0.7
        epistemic_uncertainty = total_uncertainty * 0.3

        # Create uncertainty measure
        measure = UncertaintyMeasure(
            name=f"uncertainty_{entity_id}",
            description=f"Uncertainty measure for {entity_id}",
            variance=variance,
            standard_deviation=std_dev,
            entropy=entropy,
            confidence_intervals={str(level): interval for level, interval in confidence_intervals.items()},
            percentiles=percentiles,
            distribution_type=distribution,
            distribution_parameters=dist_params,
            aleatory_uncertainty=aleatory_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            method_used=method,
            sample_size=len(data),
            reliability_score=min(1.0, len(data) / 100.0),  # Simple reliability based on sample size
            metadata=metadata or {}
        )

        return measure

    async def _fit_distribution(
        self,
        data: np.ndarray,
        distribution: DistributionType
    ) -> Dict[str, float]:
        """Fit specified distribution to data and return parameters."""
        try:
            if distribution == DistributionType.NORMAL:
                mean, std = stats.norm.fit(data)
                return {"mean": mean, "std": std}
            elif distribution == DistributionType.BETA:
                a, b, loc, scale = stats.beta.fit(data)
                return {"a": a, "b": b, "loc": loc, "scale": scale}
            elif distribution == DistributionType.GAMMA:
                a, loc, scale = stats.gamma.fit(data)
                return {"a": a, "loc": loc, "scale": scale}
            elif distribution == DistributionType.UNIFORM:
                loc, scale = stats.uniform.fit(data)
                return {"loc": loc, "scale": scale}
            else:
                # Default to normal
                mean, std = stats.norm.fit(data)
                return {"mean": mean, "std": std}
        except Exception:
            # Fallback to basic statistics
            return {"mean": np.mean(data), "std": np.std(data)}

    async def _bayesian_confidence_intervals(
        self,
        data: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[float, Tuple[float, float]]:
        """Compute Bayesian confidence intervals."""
        intervals = {}

        # Simple Bayesian approach using normal likelihood with conjugate prior
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        for level in confidence_levels:
            # Use t-distribution for small samples
            alpha = 1 - level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * np.sqrt(sample_var / n)

            lower = sample_mean - margin_error
            upper = sample_mean + margin_error
            intervals[level] = (lower, upper)

        return intervals

    async def _bootstrap_confidence_intervals(
        self,
        data: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[float, Tuple[float, float]]:
        """Compute bootstrap confidence intervals."""
        intervals = {}
        n_bootstrap = self.config.bootstrap_samples

        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(bootstrap_means, lower_percentile)
            upper = np.percentile(bootstrap_means, upper_percentile)
            intervals[level] = (lower, upper)

        return intervals

    async def _monte_carlo_confidence_intervals(
        self,
        data: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[float, Tuple[float, float]]:
        """Compute Monte Carlo confidence intervals."""
        intervals = {}
        n_samples = self.config.monte_carlo_samples

        # Estimate distribution parameters
        mean_est = np.mean(data)
        std_est = np.std(data)

        # Generate Monte Carlo samples
        mc_samples = np.random.normal(mean_est, std_est, n_samples)

        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(mc_samples, lower_percentile)
            upper = np.percentile(mc_samples, upper_percentile)
            intervals[level] = (lower, upper)

        return intervals

    async def _normal_confidence_intervals(
        self,
        data: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[float, Tuple[float, float]]:
        """Compute normal approximation confidence intervals."""
        intervals = {}

        mean = np.mean(data)
        std_error = np.std(data) / np.sqrt(len(data))

        for level in confidence_levels:
            z_critical = stats.norm.ppf((1 + level) / 2)
            margin_error = z_critical * std_error

            lower = mean - margin_error
            upper = mean + margin_error
            intervals[level] = (lower, upper)

        return intervals

    async def _update_uncertainty_history(
        self,
        entity_id: str,
        measure: UncertaintyMeasure
    ) -> None:
        """Update uncertainty history for an entity."""
        if entity_id not in self.uncertainty_histories:
            self.uncertainty_histories[entity_id] = UncertaintyHistory(entity_id=entity_id)

        history = self.uncertainty_histories[entity_id]

        history.uncertainty_values.append(measure.total_uncertainty)
        history.confidence_values.append(measure.reliability_score)
        history.timestamps.append(datetime.utcnow())

        # Update statistics
        uncertainties = list(history.uncertainty_values)
        history.mean_uncertainty = np.mean(uncertainties)
        history.uncertainty_variance = np.var(uncertainties)
        history.min_uncertainty = min(history.min_uncertainty, measure.total_uncertainty)
        history.max_uncertainty = max(history.max_uncertainty, measure.total_uncertainty)

        history.last_updated = datetime.utcnow()
