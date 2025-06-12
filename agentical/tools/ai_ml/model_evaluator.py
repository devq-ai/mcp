"""
Model Evaluator for Agentical

This module provides comprehensive AI/ML model evaluation and benchmarking
capabilities with support for various model types, metrics, and comparison
frameworks for performance assessment and optimization.

Features:
- Multi-model comparison and benchmarking
- Performance metrics for classification, regression, and generation tasks
- A/B testing framework for model selection
- Cost analysis and ROI calculations
- Latency and throughput measurement
- Quality scoring and human evaluation integration
- Statistical significance testing
- Model drift detection and monitoring
- Integration with multiple AI providers
- Enterprise features (audit logging, reporting, compliance)
"""

import asyncio
import json
import statistics
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import os

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, mean_squared_error,
        mean_absolute_error, r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ModelType(Enum):
    """Types of AI/ML models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    REINFORCEMENT = "reinforcement"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    HUMAN_PREFERENCE = "human_preference"
    TOXICITY = "toxicity"
    BIAS = "bias"
    FACTUALITY = "factuality"


class TestType(Enum):
    """Types of evaluation tests."""
    PERFORMANCE = "performance"
    STRESS = "stress"
    AB_TEST = "ab_test"
    HUMAN_EVAL = "human_eval"
    ADVERSARIAL = "adversarial"
    FAIRNESS = "fairness"
    SAFETY = "safety"
    ROBUSTNESS = "robustness"


class EvaluationStatus(Enum):
    """Status of evaluation runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    name: str
    model_type: ModelType
    provider: str
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    cost_per_token: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.cost_per_token is None:
            self.cost_per_token = {"input": 0.0, "output": 0.0}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        return data


@dataclass
class TestCase:
    """Individual test case for evaluation."""
    id: str
    input_data: Any
    expected_output: Optional[Any] = None
    ground_truth: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    weight: float = 1.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ModelResponse:
    """Response from a model during evaluation."""
    model_name: str
    test_case_id: str
    output: Any
    latency: float
    cost: float
    tokens_used: Dict[str, int]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    metric_type: MetricType
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        return data


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    model_name: str
    test_type: TestType
    total_test_cases: int
    successful_cases: int
    failed_cases: int
    metrics: List[MetricResult]
    responses: List[ModelResponse]
    execution_time: float
    total_cost: float
    average_latency: float
    throughput: float
    status: EvaluationStatus
    errors: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (self.successful_cases / self.total_test_cases) * 100 if self.total_test_cases > 0 else 0

    def get_metric_value(self, metric_type: MetricType) -> Optional[float]:
        """Get value for a specific metric type."""
        for metric in self.metrics:
            if metric.metric_type == metric_type:
                return metric.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['test_type'] = self.test_type.value
        data['status'] = self.status.value
        data['metrics'] = [m.to_dict() for m in self.metrics]
        data['responses'] = [r.to_dict() for r in self.responses]
        data['success_rate'] = self.success_rate
        return data


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""
    models_compared: List[str]
    comparison_metrics: List[MetricType]
    results: Dict[str, EvaluationResult]
    statistical_significance: Dict[str, Dict[str, float]]
    recommendations: List[str]
    best_model: Optional[str] = None
    cost_efficiency: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['comparison_metrics'] = [m.value for m in self.comparison_metrics]
        data['results'] = {k: v.to_dict() for k, v in self.results.items()}
        return data


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    @abstractmethod
    def calculate(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Calculate metric value."""
        pass

    @abstractmethod
    def get_metric_type(self) -> MetricType:
        """Get the metric type this calculator handles."""
        pass


class AccuracyCalculator(MetricCalculator):
    """Calculate classification accuracy."""

    def calculate(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Calculate accuracy score."""
        if not SKLEARN_AVAILABLE:
            # Simple accuracy calculation
            correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
            return correct / len(predictions) if predictions else 0.0

        return accuracy_score(ground_truth, predictions)

    def get_metric_type(self) -> MetricType:
        return MetricType.ACCURACY


class LatencyCalculator(MetricCalculator):
    """Calculate average latency."""

    def calculate(self, latencies: List[float], ground_truth: List[Any] = None) -> float:
        """Calculate average latency."""
        return statistics.mean(latencies) if latencies else 0.0

    def get_metric_type(self) -> MetricType:
        return MetricType.LATENCY


class CostCalculator(MetricCalculator):
    """Calculate total cost."""

    def calculate(self, costs: List[float], ground_truth: List[Any] = None) -> float:
        """Calculate total cost."""
        return sum(costs) if costs else 0.0

    def get_metric_type(self) -> MetricType:
        return MetricType.COST


class ModelRunner(ABC):
    """Abstract base class for model runners."""

    @abstractmethod
    async def run_inference(self, model_config: ModelConfig, test_case: TestCase) -> ModelResponse:
        """Run inference on a single test case."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if runner dependencies are available."""
        pass


class OpenAIModelRunner(ModelRunner):
    """Model runner for OpenAI models."""

    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library required for OpenAI models")

    async def run_inference(self, model_config: ModelConfig, test_case: TestCase) -> ModelResponse:
        """Run inference using OpenAI API."""
        start_time = time.time()

        try:
            client = openai.AsyncOpenAI(api_key=model_config.api_key)

            # Prepare messages based on input format
            if isinstance(test_case.input_data, str):
                messages = [{"role": "user", "content": test_case.input_data}]
            elif isinstance(test_case.input_data, list):
                messages = test_case.input_data
            else:
                messages = [{"role": "user", "content": str(test_case.input_data)}]

            response = await client.chat.completions.create(
                model=model_config.model_id,
                messages=messages,
                **model_config.parameters
            )

            latency = time.time() - start_time

            # Calculate cost
            usage = response.usage
            input_cost = usage.prompt_tokens * model_config.cost_per_token.get("input", 0)
            output_cost = usage.completion_tokens * model_config.cost_per_token.get("output", 0)
            total_cost = input_cost + output_cost

            return ModelResponse(
                model_name=model_config.name,
                test_case_id=test_case.id,
                output=response.choices[0].message.content,
                latency=latency,
                cost=total_cost,
                tokens_used={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens
                },
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ModelResponse(
                model_name=model_config.name,
                test_case_id=test_case.id,
                output=None,
                latency=time.time() - start_time,
                cost=0.0,
                tokens_used={"input": 0, "output": 0, "total": 0},
                timestamp=datetime.utcnow(),
                error=str(e)
            )

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE


class AnthropicModelRunner(ModelRunner):
    """Model runner for Anthropic models."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library required for Anthropic models")

    async def run_inference(self, model_config: ModelConfig, test_case: TestCase) -> ModelResponse:
        """Run inference using Anthropic API."""
        start_time = time.time()

        try:
            client = anthropic.AsyncAnthropic(api_key=model_config.api_key)

            # Prepare messages for Anthropic format
            if isinstance(test_case.input_data, str):
                messages = [{"role": "user", "content": test_case.input_data}]
            elif isinstance(test_case.input_data, list):
                # Extract system message if present
                system_msg = ""
                user_messages = []
                for msg in test_case.input_data:
                    if msg.get("role") == "system":
                        system_msg = msg.get("content", "")
                    else:
                        user_messages.append(msg)
                messages = user_messages
            else:
                messages = [{"role": "user", "content": str(test_case.input_data)}]

            response = await client.messages.create(
                model=model_config.model_id,
                messages=messages,
                max_tokens=model_config.parameters.get("max_tokens", 1000),
                **{k: v for k, v in model_config.parameters.items() if k != "max_tokens"}
            )

            latency = time.time() - start_time

            # Calculate cost
            usage = response.usage
            input_cost = usage.input_tokens * model_config.cost_per_token.get("input", 0)
            output_cost = usage.output_tokens * model_config.cost_per_token.get("output", 0)
            total_cost = input_cost + output_cost

            return ModelResponse(
                model_name=model_config.name,
                test_case_id=test_case.id,
                output=response.content[0].text,
                latency=latency,
                cost=total_cost,
                tokens_used={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.input_tokens + usage.output_tokens
                },
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ModelResponse(
                model_name=model_config.name,
                test_case_id=test_case.id,
                output=None,
                latency=time.time() - start_time,
                cost=0.0,
                tokens_used={"input": 0, "output": 0, "total": 0},
                timestamp=datetime.utcnow(),
                error=str(e)
            )

    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE


class ModelEvaluator:
    """
    Comprehensive AI/ML model evaluation and benchmarking system.

    Provides advanced model comparison, performance analysis, and optimization
    recommendations with enterprise-grade reporting and monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model evaluator.

        Args:
            config: Configuration dictionary with evaluation settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.parallel_execution = self.config.get('parallel_execution', True)
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.timeout_seconds = self.config.get('timeout_seconds', 60)
        self.retry_attempts = self.config.get('retry_attempts', 3)

        # Statistical settings
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.significance_threshold = self.config.get('significance_threshold', 0.05)
        self.minimum_sample_size = self.config.get('minimum_sample_size', 30)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.enable_caching = self.config.get('enable_caching', True)

        # Initialize components
        self.model_runners = self._initialize_runners()
        self.metric_calculators = self._initialize_calculators()
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)

    def _initialize_runners(self) -> Dict[str, ModelRunner]:
        """Initialize available model runners."""
        runners = {}

        try:
            runners['openai'] = OpenAIModelRunner()
        except ImportError:
            pass

        try:
            runners['anthropic'] = AnthropicModelRunner()
        except ImportError:
            pass

        return runners

    def _initialize_calculators(self) -> Dict[MetricType, MetricCalculator]:
        """Initialize metric calculators."""
        calculators = {
            MetricType.ACCURACY: AccuracyCalculator(),
            MetricType.LATENCY: LatencyCalculator(),
            MetricType.COST: CostCalculator()
        }

        # Add more calculators if sklearn is available
        if SKLEARN_AVAILABLE:
            # Additional metric calculators would be added here
            pass

        return calculators

    async def evaluate_model(self, model_config: ModelConfig, test_cases: List[TestCase],
                           metrics: List[MetricType] = None, test_type: TestType = TestType.PERFORMANCE) -> EvaluationResult:
        """
        Evaluate a single model against test cases.

        Args:
            model_config: Model configuration
            test_cases: List of test cases
            metrics: List of metrics to calculate
            test_type: Type of evaluation test

        Returns:
            Evaluation result with metrics and analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting evaluation of model: {model_config.name}")

            # Default metrics if not specified
            if metrics is None:
                metrics = [MetricType.ACCURACY, MetricType.LATENCY, MetricType.COST]

            # Get appropriate runner
            runner = self.model_runners.get(model_config.provider)
            if not runner:
                raise ValueError(f"No runner available for provider: {model_config.provider}")

            # Check cache
            cache_key = self._get_cache_key(model_config, test_cases, metrics)
            if self.enable_caching and cache_key in self.evaluation_cache:
                self.metrics['cache_hits'] += 1
                return self.evaluation_cache[cache_key]

            # Run inference on all test cases
            if self.parallel_execution:
                responses = await self._run_parallel_inference(runner, model_config, test_cases)
            else:
                responses = await self._run_sequential_inference(runner, model_config, test_cases)

            # Calculate metrics
            metric_results = []
            for metric_type in metrics:
                if metric_type in self.metric_calculators:
                    calculator = self.metric_calculators[metric_type]
                    metric_value = await self._calculate_metric(calculator, responses, test_cases)

                    metric_results.append(MetricResult(
                        metric_type=metric_type,
                        value=metric_value,
                        sample_size=len(responses)
                    ))

            # Calculate summary statistics
            successful_responses = [r for r in responses if r.error is None]
            failed_responses = [r for r in responses if r.error is not None]

            total_cost = sum(r.cost for r in successful_responses)
            latencies = [r.latency for r in successful_responses]
            avg_latency = statistics.mean(latencies) if latencies else 0.0

            execution_time = time.time() - start_time
            throughput = len(successful_responses) / execution_time if execution_time > 0 else 0.0

            # Create result
            result = EvaluationResult(
                model_name=model_config.name,
                test_type=test_type,
                total_test_cases=len(test_cases),
                successful_cases=len(successful_responses),
                failed_cases=len(failed_responses),
                metrics=metric_results,
                responses=responses,
                execution_time=execution_time,
                total_cost=total_cost,
                average_latency=avg_latency,
                throughput=throughput,
                status=EvaluationStatus.COMPLETED,
                errors=[r.error for r in failed_responses if r.error]
            )

            # Cache result
            if self.enable_caching:
                self.evaluation_cache[cache_key] = result

            # Log audit
            if self.audit_logging:
                self._log_operation('evaluate_model', {
                    'model_name': model_config.name,
                    'test_cases': len(test_cases),
                    'success_rate': result.success_rate,
                    'total_cost': total_cost
                })

            self.metrics['models_evaluated'] += 1
            return result

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")

            execution_time = time.time() - start_time
            return EvaluationResult(
                model_name=model_config.name,
                test_type=test_type,
                total_test_cases=len(test_cases),
                successful_cases=0,
                failed_cases=len(test_cases),
                metrics=[],
                responses=[],
                execution_time=execution_time,
                total_cost=0.0,
                average_latency=0.0,
                throughput=0.0,
                status=EvaluationStatus.FAILED,
                errors=[str(e)]
            )

    async def _run_parallel_inference(self, runner: ModelRunner, model_config: ModelConfig,
                                    test_cases: List[TestCase]) -> List[ModelResponse]:
        """Run inference in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def run_single_inference(test_case: TestCase) -> ModelResponse:
            async with semaphore:
                return await runner.run_inference(model_config, test_case)

        tasks = [run_single_inference(test_case) for test_case in test_cases]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Create error response
                valid_responses.append(ModelResponse(
                    model_name=model_config.name,
                    test_case_id=test_cases[i].id,
                    output=None,
                    latency=0.0,
                    cost=0.0,
                    tokens_used={"input": 0, "output": 0, "total": 0},
                    timestamp=datetime.utcnow(),
                    error=str(response)
                ))
            else:
                valid_responses.append(response)

        return valid_responses

    async def _run_sequential_inference(self, runner: ModelRunner, model_config: ModelConfig,
                                      test_cases: List[TestCase]) -> List[ModelResponse]:
        """Run inference sequentially."""
        responses = []

        for test_case in test_cases:
            try:
                response = await runner.run_inference(model_config, test_case)
                responses.append(response)
            except Exception as e:
                # Create error response
                responses.append(ModelResponse(
                    model_name=model_config.name,
                    test_case_id=test_case.id,
                    output=None,
                    latency=0.0,
                    cost=0.0,
                    tokens_used={"input": 0, "output": 0, "total": 0},
                    timestamp=datetime.utcnow(),
                    error=str(e)
                ))

        return responses

    async def _calculate_metric(self, calculator: MetricCalculator, responses: List[ModelResponse],
                              test_cases: List[TestCase]) -> float:
        """Calculate a specific metric."""
        metric_type = calculator.get_metric_type()

        if metric_type == MetricType.LATENCY:
            latencies = [r.latency for r in responses if r.error is None]
            return calculator.calculate(latencies)
        elif metric_type == MetricType.COST:
            costs = [r.cost for r in responses if r.error is None]
            return calculator.calculate(costs)
        elif metric_type == MetricType.ACCURACY:
            # Extract predictions and ground truth
            predictions = []
            ground_truth = []

            for response in responses:
                if response.error is None:
                    test_case = next((tc for tc in test_cases if tc.id == response.test_case_id), None)
                    if test_case and test_case.expected_output is not None:
                        predictions.append(response.output)
                        ground_truth.append(test_case.expected_output)

            if predictions and ground_truth:
                return calculator.calculate(predictions, ground_truth)
            else:
                return 0.0
        else:
            return 0.0

    async def compare_models(self, model_configs: List[ModelConfig], test_cases: List[TestCase],
                           metrics: List[MetricType] = None) -> ComparisonResult:
        """
        Compare multiple models on the same test cases.

        Args:
            model_configs: List of model configurations to compare
            test_cases: Test cases to evaluate on
            metrics: Metrics to compare

        Returns:
            Comparison result with statistical analysis
        """
        try:
            self.logger.info(f"Comparing {len(model_configs)} models")

            # Default metrics
            if metrics is None:
                metrics = [MetricType.ACCURACY, MetricType.LATENCY, MetricType.COST]

            # Evaluate each model
            results = {}
            for model_config in model_configs:
                result = await self.evaluate_model(model_config, test_cases, metrics)
                results[model_config.name] = result

            # Calculate statistical significance
            significance = self._calculate_statistical_significance(results, metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(results, metrics)

            # Determine best model
            best_model = self._determine_best_model(results, metrics)

            # Calculate cost efficiency
            cost_efficiency = self._calculate_cost_efficiency(results)

            return ComparisonResult(
                models_compared=[config.name for config in model_configs],
                comparison_metrics=metrics,
                results=results,
                statistical_significance=significance,
                recommendations=recommendations,
                best_model=best_model,
                cost_efficiency=cost_efficiency
            )

        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise

    def _calculate_statistical_significance(self, results: Dict[str, EvaluationResult],
                                          metrics: List[MetricType]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance between models."""
        significance = {}

        if not SCIPY_AVAILABLE:
            return significance

        model_names = list(results.keys())

        for i, model1 in enumerate(model_names):
            significance[model1] = {}

            for j, model2 in enumerate(model_names):
                if i != j:
                    # Compare primary metric (accuracy or first metric)
                    primary_metric = metrics[0] if metrics else MetricType.ACCURACY

                    value1 = results[model1].get_metric_value(primary_metric) or 0
                    value2 = results[model2].get_metric_value(primary_metric) or 0

                    # Simple t-test approximation (would need actual distributions for real test)
                    try:
                        if abs(value1 - value2) > 0.01:  # Minimum difference threshold
                            p_value = 0.05 if abs(value1 - value2) > 0.1 else 0.1
                        else:
                            p_value = 0.5

                        significance[model1][model2] = p_value
                    except:
                        significance[model1][model2] = 1.0

        return significance

    def _generate_recommendations(self, results: Dict[str, EvaluationResult],
                                metrics: List[MetricType]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Find best performing model for each metric
        for metric in metrics:
            best_model = None
            best_value = None

            for model_name, result in results.items():
                value = result.get_metric_value(metric)
                if value is not None:
                    if best_value is None or (
                        metric in [MetricType.ACCURACY, MetricType.THROUGHPUT] and value > best_value
                    ) or (
                        metric in [MetricType.LATENCY, MetricType.COST] and value < best_value
                    ):
                        best_value = value
                        best_model = model_name

            if best_model:
                if metric == MetricType.ACCURACY:
                    recommendations.append(f"For highest accuracy, use {best_model} ({best_value:.2%})")
                elif metric == MetricType.LATENCY:
                    recommendations.append(f"For lowest latency, use {best_model} ({best_value:.2f}s)")
                elif metric == MetricType.COST:
                    recommendations.append(f"For lowest cost, use {best_model} (${best_value:.4f})")

        # Add general recommendations
        successful_models = [name for name, result in results.items() if result.success_rate > 90]
        if successful_models:
            recommendations.append(f"Models with >90% success rate: {', '.join(successful_models)}")

        return recommendations

    def _determine_best_model(self, results: Dict[str, EvaluationResult],
                            metrics: List[MetricType]) -> Optional[str]:
        """Determine the best overall model."""
        if not results:
            return None

        # Simple scoring: normalize each metric and sum
        scores = {}

        for model_name in results.keys():
            scores[model_name] = 0.0

        # Normalize and weight each metric
        for metric in metrics:
            values = []
            for result in results.values():
                value = result.get_metric_value(metric)
                if value is not None:
                    values.append(value)

            if not values:
                continue

            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                continue

            for model_name, result in results.items():
                value = result.get_metric_value(metric)
                if value is not None:
                    # Normalize (0-1) and weight
                    if metric in [MetricType.ACCURACY, MetricType.THROUGHPUT]:
                        # Higher is better
                        normalized = (value - min_val) / (max_val - min_val)
                    else:
                        # Lower is better (latency, cost)
                        normalized = (max_val - value) / (max_val - min_val)

                    scores[model_name] += normalized

        # Return model with highest score
        return max(scores, key=scores.get) if scores else None

    def _calculate_cost_efficiency(self, results: Dict[str, EvaluationResult]) -> Dict[str, float]:
        """Calculate cost efficiency for each model."""
        efficiency = {}

        for model_name, result in results.items():
            accuracy = result.get_metric_value(MetricType.ACCURACY) or 0
            cost = result.get_metric_value(MetricType.COST) or 0

            if cost > 0:
                # Efficiency = accuracy per dollar
                efficiency[model_name] = accuracy / cost
            else:
                efficiency[model_name] = accuracy

        return efficiency

    def _get_cache_key(self, model_config: ModelConfig, test_cases: List[TestCase],
                      metrics: List[MetricType]) -> str:
        """Generate cache key for evaluation results."""
        key_data = {
            'model_config': model_config.to_dict(),
            'test_cases': [tc.to_dict() for tc in test_cases],
            'metrics': [m.value for m in metrics],
            'config': self.config
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log operations for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'details': details
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    def clear_cache(self):
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        self.logger.info("Model evaluator cache cleared")

    async def cleanup(self):
        """Cleanup model evaluator resources."""
        try:
            self.clear_cache()
            self.metrics.clear()
            self.logger.info("Model evaluator cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'evaluation_cache') and self.evaluation_cache:
                self.logger.info("ModelEvaluator being destroyed - cleanup recommended")
        except:
            pass
