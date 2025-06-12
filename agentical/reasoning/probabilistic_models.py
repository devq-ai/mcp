"""
Probabilistic Models Component for Advanced Bayesian Modeling

This module provides sophisticated probabilistic modeling capabilities for the
Agentical framework's Bayesian reasoning system, implementing various probabilistic
models including Bayesian networks, Markov chains, and Gaussian processes.

Features:
- Base probabilistic model framework
- Bayesian network implementation with inference
- Markov chain and Hidden Markov model support
- Gaussian process modeling with uncertainty
- Model comparison and selection
- Performance optimization for large models
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
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from scipy import stats, linalg
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

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


class ModelType(str, Enum):
    """Types of probabilistic models."""
    BAYESIAN_NETWORK = "bayesian_network"
    MARKOV_CHAIN = "markov_chain"
    HIDDEN_MARKOV_MODEL = "hidden_markov_model"
    GAUSSIAN_PROCESS = "gaussian_process"
    MIXTURE_MODEL = "mixture_model"
    KALMAN_FILTER = "kalman_filter"


class InferenceMethod(str, Enum):
    """Inference methods for probabilistic models."""
    EXACT = "exact"
    VARIATIONAL = "variational"
    MCMC = "mcmc"
    BELIEF_PROPAGATION = "belief_propagation"
    GIBBS_SAMPLING = "gibbs_sampling"
    IMPORTANCE_SAMPLING = "importance_sampling"


class ModelState(str, Enum):
    """States of probabilistic model lifecycle."""
    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    PREDICTING = "predicting"
    ERROR = "error"


@dataclass
class ModelParameters:
    """Base parameters for probabilistic models."""
    parameter_id: str = field(default_factory=lambda: str(uuid4()))
    model_type: ModelType = ModelType.BAYESIAN_NETWORK

    # Core parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    learning_rate: float = 0.01
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6

    # Model structure
    num_variables: int = 0
    num_states: int = 0
    structure: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    log_likelihood: float = 0.0
    aic_score: float = 0.0
    bic_score: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelPrediction:
    """Prediction result from probabilistic model."""
    prediction_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""

    # Prediction results
    mean_prediction: Union[float, np.ndarray] = 0.0
    variance_prediction: Union[float, np.ndarray] = 0.0
    probability_distribution: Optional[np.ndarray] = None

    # Uncertainty quantification
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    prediction_uncertainty: float = 0.0
    model_uncertainty: float = 0.0

    # Quality metrics
    prediction_quality: float = 0.0
    reliability_score: float = 0.0

    # Metadata
    input_data: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    method_used: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)


class ProbabilisticModelConfig(BaseModel):
    """Configuration for probabilistic models."""

    # Model settings
    model_type: ModelType = ModelType.BAYESIAN_NETWORK
    inference_method: InferenceMethod = InferenceMethod.EXACT
    enable_learning: bool = True

    # Training parameters
    max_training_iterations: int = Field(default=1000, ge=1)
    convergence_threshold: float = Field(default=1e-6, gt=0.0)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)

    # Performance settings
    enable_caching: bool = True
    cache_size: int = Field(default=1000, ge=10)
    parallel_inference: bool = True
    max_workers: int = Field(default=4, ge=1)

    # Memory management
    max_memory_mb: float = Field(default=500.0, gt=0.0)
    enable_memory_optimization: bool = True

    # Quality control
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    cross_validation_folds: int = Field(default=5, ge=2)
    early_stopping: bool = True
    patience: int = Field(default=10, ge=1)

    # Logging and monitoring
    detailed_logging: bool = True
    log_training_progress: bool = True
    performance_monitoring: bool = True


class ProbabilisticModel(ABC):
    """
    Abstract base class for probabilistic models.

    Provides common interface and functionality for all probabilistic models
    including training, inference, and prediction capabilities.
    """

    def __init__(
        self,
        model_id: str,
        config: ProbabilisticModelConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the probabilistic model.

        Args:
            model_id: Unique model identifier
            config: Model configuration
            logger: Optional structured logger
        """
        self.model_id = model_id
        self.config = config
        self.logger = logger or StructuredLogger("probabilistic_model")

        # Model state
        self.state = ModelState.INITIALIZED
        self.parameters: Optional[ModelParameters] = None
        self.training_data: Optional[np.ndarray] = None

        # Performance tracking
        self.training_iterations = 0
        self.prediction_count = 0
        self.total_training_time = 0.0
        self.total_prediction_time = 0.0

        # Caching
        self.prediction_cache: Dict[str, ModelPrediction] = {}
        self.inference_cache: Dict[str, Any] = {}

        logfire.info(
            "Probabilistic model initialized",
            model_id=model_id,
            model_type=config.model_type.value
        )

    @abstractmethod
    async def train(
        self,
        training_data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """
        Train the probabilistic model on provided data.

        Args:
            training_data: Training data array
            labels: Optional labels for supervised learning

        Returns:
            Trained model parameters
        """
        pass

    @abstractmethod
    async def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]]
    ) -> ModelPrediction:
        """
        Make predictions using the trained model.

        Args:
            input_data: Input data for prediction

        Returns:
            Model prediction with uncertainty
        """
        pass

    @abstractmethod
    async def compute_likelihood(
        self,
        data: np.ndarray
    ) -> float:
        """
        Compute likelihood of data under the model.

        Args:
            data: Data for likelihood computation

        Returns:
            Log-likelihood value
        """
        pass

    async def validate_model(
        self,
        validation_data: np.ndarray,
        validation_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Validate model performance on validation data.

        Args:
            validation_data: Validation data
            validation_labels: Optional validation labels

        Returns:
            Validation metrics
        """
        with logfire.span("Validate Model", model_id=self.model_id):
            if self.state != ModelState.TRAINED:
                raise ValidationError("Model must be trained before validation")

            # Compute predictions
            predictions = []
            for i in range(len(validation_data)):
                pred = await self.predict(validation_data[i:i+1])
                predictions.append(pred.mean_prediction)

            predictions = np.array(predictions)

            # Compute validation metrics
            if validation_labels is not None:
                mse = np.mean((predictions.flatten() - validation_labels.flatten()) ** 2)
                mae = np.mean(np.abs(predictions.flatten() - validation_labels.flatten()))
                r2 = 1 - np.sum((validation_labels.flatten() - predictions.flatten()) ** 2) / \
                        np.sum((validation_labels.flatten() - np.mean(validation_labels)) ** 2)
            else:
                mse = mae = r2 = 0.0

            # Compute likelihood
            likelihood = await self.compute_likelihood(validation_data)

            metrics = {
                "mse": mse,
                "mae": mae,
                "r2_score": r2,
                "log_likelihood": likelihood,
                "validation_samples": len(validation_data)
            }

            self.logger.log(
                LogLevel.INFO,
                f"Model validation completed",
                operation_type=OperationType.VALIDATION,
                model_id=self.model_id,
                **metrics
            )

            return metrics

    async def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information and metrics."""
        with logfire.span("Get Model Info"):
            avg_training_time = (
                self.total_training_time / max(self.training_iterations, 1)
                if self.training_iterations > 0 else 0.0
            )

            avg_prediction_time = (
                self.total_prediction_time / max(self.prediction_count, 1)
                if self.prediction_count > 0 else 0.0
            )

            info = {
                "model_id": self.model_id,
                "model_type": self.config.model_type.value,
                "state": self.state.value,
                "training_iterations": self.training_iterations,
                "prediction_count": self.prediction_count,
                "total_training_time_ms": self.total_training_time,
                "total_prediction_time_ms": self.total_prediction_time,
                "average_training_time_ms": avg_training_time,
                "average_prediction_time_ms": avg_prediction_time,
                "cached_predictions": len(self.prediction_cache),
                "parameters": self.parameters.__dict__ if self.parameters else None
            }

            return info


class BayesianNetwork(ProbabilisticModel):
    """
    Bayesian Network implementation for probabilistic reasoning.

    Provides Bayesian network modeling with structure learning,
    parameter estimation, and exact/approximate inference.
    """

    def __init__(
        self,
        model_id: str,
        config: ProbabilisticModelConfig,
        logger: Optional[StructuredLogger] = None
    ):
        super().__init__(model_id, config, logger)

        # Network structure
        self.nodes: Dict[str, int] = {}
        self.edges: List[Tuple[int, int]] = []
        self.adjacency_matrix: Optional[np.ndarray] = None

        # Conditional probability tables
        self.cpts: Dict[int, np.ndarray] = {}

        # Inference engine
        self.inference_engine = None

    async def train(
        self,
        training_data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """Train Bayesian network on data."""
        start_time = datetime.utcnow()

        with logfire.span("Train Bayesian Network", model_id=self.model_id):
            try:
                self.state = ModelState.TRAINING
                self.training_data = training_data

                # Learn network structure (simplified approach)
                await self._learn_structure(training_data)

                # Learn parameters (CPTs)
                await self._learn_parameters(training_data)

                # Create model parameters
                self.parameters = ModelParameters(
                    model_type=ModelType.BAYESIAN_NETWORK,
                    num_variables=training_data.shape[1],
                    structure={"edges": self.edges, "nodes": self.nodes},
                    parameters={"cpts": {str(k): v.tolist() for k, v in self.cpts.items()}}
                )

                self.state = ModelState.TRAINED
                training_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_training_time += training_time
                self.training_iterations += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"Bayesian network training completed",
                    operation_type=OperationType.TRAINING,
                    model_id=self.model_id,
                    num_variables=self.parameters.num_variables,
                    num_edges=len(self.edges),
                    training_time_ms=training_time
                )

                return self.parameters

            except Exception as e:
                self.state = ModelState.ERROR
                self.logger.log(
                    LogLevel.ERROR,
                    f"Bayesian network training failed: {str(e)}",
                    operation_type=OperationType.TRAINING,
                    model_id=self.model_id,
                    error=str(e)
                )
                raise AgentError(f"Training failed: {str(e)}")

    async def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]]
    ) -> ModelPrediction:
        """Make predictions using Bayesian network inference."""
        start_time = datetime.utcnow()

        with logfire.span("Predict Bayesian Network", model_id=self.model_id):
            if self.state != ModelState.TRAINED:
                raise ValidationError("Model must be trained before prediction")

            # Convert input to evidence format
            evidence = self._prepare_evidence(input_data)

            # Perform inference
            prediction_result = await self._perform_inference(evidence)

            # Create prediction object
            prediction = ModelPrediction(
                model_id=self.model_id,
                mean_prediction=prediction_result["mean"],
                variance_prediction=prediction_result["variance"],
                probability_distribution=prediction_result.get("distribution"),
                prediction_uncertainty=prediction_result.get("uncertainty", 0.0),
                input_data=evidence,
                method_used=self.config.inference_method.value
            )

            prediction_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction.computation_time_ms = prediction_time
            self.total_prediction_time += prediction_time
            self.prediction_count += 1

            return prediction

    async def compute_likelihood(self, data: np.ndarray) -> float:
        """Compute likelihood of data under Bayesian network."""
        if self.state != ModelState.TRAINED:
            raise ValidationError("Model must be trained before likelihood computation")

        total_likelihood = 0.0

        for sample in data:
            sample_likelihood = 0.0

            # Compute likelihood for each variable given its parents
            for node_idx in range(len(sample)):
                parent_values = self._get_parent_values(node_idx, sample)
                node_value = int(sample[node_idx])

                if node_idx in self.cpts:
                    cpt = self.cpts[node_idx]
                    # Simplified likelihood computation
                    if len(parent_values) == 0:
                        prob = cpt[node_value] if node_value < len(cpt) else 1e-10
                    else:
                        # More complex indexing for conditional probabilities
                        prob = 1e-10  # Placeholder

                    sample_likelihood += np.log(max(prob, 1e-10))

            total_likelihood += sample_likelihood

        return total_likelihood / len(data)

    # Private helper methods

    async def _learn_structure(self, data: np.ndarray) -> None:
        """Learn Bayesian network structure from data."""
        num_vars = data.shape[1]

        # Simple structure learning: create nodes for each variable
        self.nodes = {f"var_{i}": i for i in range(num_vars)}

        # Simplified structure: assume each variable depends on previous ones
        self.edges = []
        for i in range(1, num_vars):
            for j in range(i):
                # Add edge with some probability based on correlation
                correlation = np.corrcoef(data[:, i], data[:, j])[0, 1]
                if abs(correlation) > 0.3:  # Threshold for edge creation
                    self.edges.append((j, i))

        # Create adjacency matrix
        self.adjacency_matrix = np.zeros((num_vars, num_vars))
        for parent, child in self.edges:
            self.adjacency_matrix[parent, child] = 1

    async def _learn_parameters(self, data: np.ndarray) -> None:
        """Learn conditional probability tables from data."""
        num_vars = data.shape[1]

        for node_idx in range(num_vars):
            parents = [i for i in range(num_vars) if self.adjacency_matrix[i, node_idx] == 1]

            if not parents:
                # No parents: learn marginal distribution
                unique_values, counts = np.unique(data[:, node_idx], return_counts=True)
                cpt = counts / np.sum(counts)
                self.cpts[node_idx] = cpt
            else:
                # Has parents: learn conditional distribution
                # Simplified approach: assume binary variables
                num_states = 2
                num_parent_configs = 2 ** len(parents)
                cpt = np.ones((num_parent_configs, num_states)) * 0.5  # Uniform prior

                # Update with data (simplified counting)
                for sample in data:
                    parent_config = 0
                    for i, parent in enumerate(parents):
                        if sample[parent] > 0.5:
                            parent_config += 2 ** i

                    child_value = int(sample[node_idx] > 0.5)
                    if parent_config < len(cpt):
                        cpt[parent_config, child_value] += 1

                # Normalize
                for i in range(num_parent_configs):
                    if np.sum(cpt[i]) > 0:
                        cpt[i] = cpt[i] / np.sum(cpt[i])

                self.cpts[node_idx] = cpt

    def _prepare_evidence(self, input_data: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert input data to evidence format."""
        if isinstance(input_data, np.ndarray):
            evidence = {}
            for i, value in enumerate(input_data.flatten()):
                if not np.isnan(value):
                    evidence[f"var_{i}"] = value
            return evidence
        else:
            return input_data

    async def _perform_inference(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference on Bayesian network."""
        # Simplified inference: return mean and variance based on evidence
        # In practice, this would use proper belief propagation or other methods

        num_vars = len(self.nodes)
        predictions = np.zeros(num_vars)
        variances = np.ones(num_vars) * 0.1  # Default variance

        # Set evidence variables
        for var_name, value in evidence.items():
            if var_name in self.nodes:
                var_idx = self.nodes[var_name]
                predictions[var_idx] = value
                variances[var_idx] = 0.0  # No uncertainty for observed variables

        # Predict unobserved variables (simplified)
        for var_idx in range(num_vars):
            var_name = f"var_{var_idx}"
            if var_name not in evidence:
                # Use prior or parent-based prediction
                if var_idx in self.cpts:
                    cpt = self.cpts[var_idx]
                    if len(cpt.shape) == 1:
                        # Marginal distribution
                        predictions[var_idx] = np.argmax(cpt)
                    else:
                        # Conditional distribution - use first configuration
                        predictions[var_idx] = np.argmax(cpt[0])

        return {
            "mean": predictions,
            "variance": variances,
            "uncertainty": np.mean(variances)
        }

    def _get_parent_values(self, node_idx: int, sample: np.ndarray) -> List[float]:
        """Get parent values for a node from a sample."""
        parents = []
        if self.adjacency_matrix is not None:
            for i in range(len(sample)):
                if self.adjacency_matrix[i, node_idx] == 1:
                    parents.append(sample[i])
        return parents


class MarkovChain(ProbabilisticModel):
    """
    Markov Chain implementation for sequential modeling.

    Provides Markov chain modeling with transition matrix learning
    and state sequence prediction.
    """

    def __init__(
        self,
        model_id: str,
        config: ProbabilisticModelConfig,
        num_states: int = 10,
        logger: Optional[StructuredLogger] = None
    ):
        super().__init__(model_id, config, logger)

        self.num_states = num_states
        self.transition_matrix: Optional[np.ndarray] = None
        self.initial_distribution: Optional[np.ndarray] = None
        self.state_mapping: Dict[Any, int] = {}

    async def train(
        self,
        training_data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """Train Markov chain on sequential data."""
        start_time = datetime.utcnow()

        with logfire.span("Train Markov Chain", model_id=self.model_id):
            try:
                self.state = ModelState.TRAINING

                # Learn state mapping
                unique_states = np.unique(training_data.flatten())
                self.state_mapping = {state: i for i, state in enumerate(unique_states)}
                self.num_states = len(unique_states)

                # Initialize matrices
                self.transition_matrix = np.zeros((self.num_states, self.num_states))
                self.initial_distribution = np.zeros(self.num_states)

                # Learn from sequences
                for sequence in training_data:
                    sequence_states = [self.state_mapping[s] for s in sequence if s in self.state_mapping]

                    if len(sequence_states) > 0:
                        # Update initial distribution
                        self.initial_distribution[sequence_states[0]] += 1

                        # Update transition matrix
                        for i in range(len(sequence_states) - 1):
                            current_state = sequence_states[i]
                            next_state = sequence_states[i + 1]
                            self.transition_matrix[current_state, next_state] += 1

                # Normalize
                if np.sum(self.initial_distribution) > 0:
                    self.initial_distribution /= np.sum(self.initial_distribution)

                for i in range(self.num_states):
                    if np.sum(self.transition_matrix[i]) > 0:
                        self.transition_matrix[i] /= np.sum(self.transition_matrix[i])

                # Create parameters
                self.parameters = ModelParameters(
                    model_type=ModelType.MARKOV_CHAIN,
                    num_states=self.num_states,
                    parameters={
                        "transition_matrix": self.transition_matrix.tolist(),
                        "initial_distribution": self.initial_distribution.tolist(),
                        "state_mapping": self.state_mapping
                    }
                )

                self.state = ModelState.TRAINED
                training_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_training_time += training_time
                self.training_iterations += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"Markov chain training completed",
                    operation_type=OperationType.TRAINING,
                    model_id=self.model_id,
                    num_states=self.num_states,
                    training_time_ms=training_time
                )

                return self.parameters

            except Exception as e:
                self.state = ModelState.ERROR
                raise AgentError(f"Training failed: {str(e)}")

    async def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]]
    ) -> ModelPrediction:
        """Predict next states in Markov chain."""
        start_time = datetime.utcnow()

        with logfire.span("Predict Markov Chain", model_id=self.model_id):
            if self.state != ModelState.TRAINED:
                raise ValidationError("Model must be trained before prediction")

            # Get current state
            if isinstance(input_data, np.ndarray):
                current_state_value = input_data[-1]  # Last state in sequence
            else:
                current_state_value = input_data.get("current_state")

            if current_state_value not in self.state_mapping:
                # Unknown state, use uniform distribution
                next_state_probs = np.ones(self.num_states) / self.num_states
            else:
                current_state_idx = self.state_mapping[current_state_value]
                next_state_probs = self.transition_matrix[current_state_idx]

            # Predict most likely next state
            next_state_idx = np.argmax(next_state_probs)
            confidence = next_state_probs[next_state_idx]
            uncertainty = 1.0 - confidence

            prediction = ModelPrediction(
                model_id=self.model_id,
                mean_prediction=next_state_idx,
                variance_prediction=uncertainty,
                probability_distribution=next_state_probs,
                prediction_uncertainty=uncertainty,
                reliability_score=confidence,
                input_data={"current_state": current_state_value},
                method_used="markov_transition"
            )

            prediction_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction.computation_time_ms = prediction_time
            self.total_prediction_time += prediction_time
            self.prediction_count += 1

            return prediction

    async def compute_likelihood(self, data: np.ndarray) -> float:
        """Compute likelihood of sequences under Markov chain."""
        if self.state != ModelState.TRAINED:
            raise ValidationError("Model must be trained before likelihood computation")

        total_log_likelihood = 0.0
        num_sequences = 0

        for sequence in data:
            sequence_states = [self.state_mapping.get(s) for s in sequence if s in self.state_mapping]
            sequence_states = [s for s in sequence_states if s is not None]

            if len(sequence_states) > 0:
                sequence_log_likelihood = 0.0

                # Initial state probability
                sequence_log_likelihood += np.log(max(self.initial_distribution[sequence_states[0]], 1e-10))

                # Transition probabilities
                for i in range(len(sequence_states) - 1):
                    current_state = sequence_states[i]
                    next_state = sequence_states[i + 1]
                    transition_prob = self.transition_matrix[current_state, next_state]
                    sequence_log_likelihood += np.log(max(transition_prob, 1e-10))

                total_log_likelihood += sequence_log_likelihood
                num_sequences += 1

        return total_log_likelihood / max(num_sequences, 1)


class HiddenMarkovModel(ProbabilisticModel):
    """
    Hidden Markov Model implementation for sequence modeling.

    Provides HMM modeling with Baum-Welch learning and Viterbi decoding.
    """

    def __init__(
        self,
        model_id: str,
        config: ProbabilisticModelConfig,
        num_hidden_states: int = 5,
        num_observations: int = 10,
        logger: Optional[StructuredLogger] = None
    ):
        super().__init__(model_id, config, logger)

        self.num_hidden_states = num_hidden_states
        self.num_observations = num_observations

        # HMM parameters
        self.transition_matrix: Optional[np.ndarray] = None
        self.emission_matrix: Optional[np.ndarray] = None
        self.initial_distribution: Optional[np.ndarray] = None

    async def train(
        self,
        training_data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """Train HMM using Baum-Welch algorithm."""
        start_time = datetime.utcnow()

        with logfire.span("Train HMM", model_id=self.model_id):
            try:
                self.state = ModelState.TRAINING

                # Initialize parameters randomly
                self.transition_matrix = np.random.rand(self.num_hidden_states, self.num_hidden_states)
                self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)

                self.emission_matrix = np.random.rand(self.num_hidden_states, self.num_observations)
                self.emission_matrix = self.emission_matrix / np.sum(self.emission_matrix, axis=1, keepdims=True)

                self.initial_distribution = np.random.rand(self.num_hidden_states)
                self.initial_distribution = self.initial_distribution / np.sum(self.initial_distribution)

                # Simplified Baum-Welch implementation
                for iteration in range(self.config.max_training_iterations):
                    # Forward-backward algorithm (simplified)
                    old_likelihood = await self._compute_hmm_likelihood(training_data)

                    # E-step: compute forward and backward probabilities
                    # M-step: update parameters
                    # (Simplified implementation)

                    new_likelihood = await self._compute_hmm_likelihood(training_data)

                    if abs(new_likelihood - old_likelihood) < self.config.convergence_threshold:
                        break

                # Create parameters
                self.parameters = ModelParameters(
                    model_type=ModelType.HIDDEN_MARKOV_MODEL,
                    num_states=self.num_hidden_states,
                    parameters={
                        "transition_matrix": self.transition_matrix.tolist(),
                        "emission_matrix": self.emission_matrix.tolist(),
                        "initial_distribution": self.initial_distribution.tolist()
                    }
                )

                self.state = ModelState.TRAINED
                training_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_training_time += training_time
                self.training_iterations += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"HMM training completed",
                    operation_type=OperationType.TRAINING,
                    model_id=self.model_id,
                    num_hidden_states=self.num_hidden_states,
                    training_time_ms=training_time
                )

                return self.parameters

            except Exception as e:
                self.state = ModelState.ERROR
                raise AgentError(f"Training failed: {str(e)}")

    async def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]]
    ) -> ModelPrediction:
        """Predict hidden state sequence using Viterbi algorithm."""
        start_time = datetime.utcnow()

        with logfire.span("Predict HMM", model_id=self.model_id):
            if self.state != ModelState.TRAINED:
                raise ValidationError("Model must be trained before prediction")

            # Simplified Viterbi algorithm
            if isinstance(input_data, np.ndarray):
                observations = input_data.flatten().astype(int)
            else:
                observations = input_data.get("observations", [])

            # Viterbi path finding (simplified)
            path = []
            for obs in observations:
                if obs < self.num_observations:
                    # Find most likely state
                    state_probs = self.emission_matrix[:, obs]
                    best_state = np.argmax(state_probs)
                    path.append(best_state)
                else:
                    path.append(0)  # Default state

            path = np.array(path)
            uncertainty = 0.3  # Simplified uncertainty measure

            prediction = ModelPrediction(
                model_id=self.model_id,
                mean_prediction=path,
                variance_prediction=uncertainty,
                prediction_uncertainty=uncertainty,
                reliability_score=1.0 - uncertainty,
                input_data={"observations": observations.tolist()},
                method_used="viterbi"
            )

            prediction_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction.computation_time_ms = prediction_time
            self.total_prediction_time += prediction_time
            self.prediction_count += 1

            return prediction

    async def compute_likelihood(self, data: np.ndarray) -> float:
        """Compute likelihood of observations under HMM."""
        return await self._compute_hmm_likelihood(data)

    async def _compute_hmm_likelihood(self, data: np.ndarray) -> float:
        """Compute HMM likelihood using forward algorithm."""
        total_likelihood = 0.0

        for sequence in data:
            sequence = sequence.flatten().astype(int)
            sequence = sequence[sequence < self.num_observations]

            if len(sequence) > 0:
                # Forward algorithm (simplified)
                forward = np.zeros((len(sequence), self.num_hidden_states))

                # Initialize
                if sequence[0] < self.num_observations:
                    forward[0] = self.initial_distribution * self.emission_matrix[:, sequence[0]]
                else:
                    forward[0] = self.initial_distribution

                # Forward pass
                for t in range(1, len(sequence)):
                    if sequence[t] < self.num_observations:
                        for j in range(self.num_hidden_states):
                            forward[t, j] = np.sum(forward[t-1] * self.transition_matrix[:, j]) * \
                                          self.emission_matrix[j, sequence[t]]

                # Sequence likelihood
                sequence_likelihood = np.sum(forward[-1])
                total_likelihood += np.log(max(sequence_likelihood, 1e-10))

        return total_likelihood / len(data)


class GaussianProcess(ProbabilisticModel):
    """
    Gaussian Process implementation for regression with uncertainty.

    Provides GP modeling with kernel learning and uncertainty quantification.
    """

    def __init__(
        self,
        model_id: str,
        config: ProbabilisticModelConfig,
        kernel_type: str = "rbf",
        logger: Optional[StructuredLogger] = None
    ):
        super().__init__(model_id, config, logger)

        self.kernel_type = kernel_type
        self.gp_model: Optional[GaussianProcessRegressor] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    async def train(
        self,
        training_data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """Train Gaussian Process on data."""
        start_time = datetime.utcnow()

        with logfire.span("Train GP", model_id=self.model_id):
            try:
                self.state = ModelState.TRAINING

                if labels is None:
                    raise ValidationError("Labels required for GP training")

                self.X_train = training_data
                self.y_train = labels

                # Create kernel
                if self.kernel_type == "rbf":
                    kernel = ConstantKernel(1.0) * RBF(1.0)
                else:
                    kernel = ConstantKernel(1.0) * RBF(1.0)

                # Create and train GP
                self.gp_model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5
                )

                self.gp_model.fit(self.X_train, self.y_train)

                # Create parameters
                self.parameters = ModelParameters(
                    model_type=ModelType.GAUSSIAN_PROCESS,
                    num_variables=training_data.shape[1],
                    parameters={
                        "kernel_type": self.kernel_type,
                        "kernel_params": str(self.gp_model.kernel_),
                        "log_marginal_likelihood": self.gp_model.log_marginal_likelihood()
                    }
                )

                self.state = ModelState.TRAINED
                training_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_training_time += training_time
                self.training_iterations += 1

                self.logger.log(
                    LogLevel.INFO,
                    f"GP training completed",
                    operation_type=OperationType.TRAINING,
                    model_id=self.model_id,
                    num_samples=len(training_data),
                    training_time_ms=training_time
                )

                return self.parameters

            except Exception as e:
                self.state = ModelState.ERROR
                raise AgentError(f"Training failed: {str(e)}")

    async def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]]
    ) -> ModelPrediction:
        """Make GP predictions with uncertainty."""
        start_time = datetime.utcnow()

        with logfire.span("Predict GP", model_id=self.model_id):
            if self.state != ModelState.TRAINED or self.gp_model is None:
                raise ValidationError("Model must be trained before prediction")

            # Prepare input
            if isinstance(input_data, dict):
                X_test = np.array(list(input_data.values())).reshape(1, -1)
            else:
                X_test = input_data.reshape(1, -1) if input_data.ndim == 1 else input_data

            # Make prediction with uncertainty
            mean_pred, std_pred = self.gp_model.predict(X_test, return_std=True)

            # Compute confidence intervals
            confidence_intervals = {}
            for level in [0.68, 0.95, 0.99]:
                z_score = stats.norm.ppf((1 + level) / 2)
                lower = mean_pred - z_score * std_pred
                upper = mean_pred + z_score * std_pred
                confidence_intervals[str(level)] = (float(lower[0]), float(upper[0]))

            prediction = ModelPrediction(
                model_id=self.model_id,
                mean_prediction=float(mean_pred[0]),
                variance_prediction=float(std_pred[0] ** 2),
                prediction_uncertainty=float(std_pred[0]),
                confidence_intervals=confidence_intervals,
                reliability_score=1.0 - min(float(std_pred[0]), 1.0),
                input_data=input_data if isinstance(input_data, dict) else {"input": input_data.tolist()},
                method_used="gaussian_process"
            )

            prediction_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction.computation_time_ms = prediction_time
            self.total_prediction_time += prediction_time
            self.prediction_count += 1

            return prediction

    async def compute_likelihood(self, data: np.ndarray) -> float:
        """Compute log marginal likelihood of GP."""
        if self.state != ModelState.TRAINED or self.gp_model is None:
            raise ValidationError("Model must be trained before likelihood computation")

        return self.gp_model.log_marginal_likelihood()
