"""
Comprehensive Test Suite for Task 5.1: Bayesian Inference Integration

This test suite validates all components of the Bayesian inference system including:
- BayesianInferenceEngine core functionality
- BeliefUpdater dynamic probability updates
- DecisionTree probabilistic decision-making
- UncertaintyQuantifier confidence measurement
- MCP integration with bayes-mcp server
- Probabilistic models (Bayesian networks, Markov chains, etc.)

Test Categories:
- Unit tests for individual components
- Integration tests for component interaction
- Performance tests for optimization validation
- Error handling and edge case tests
- End-to-end workflow tests
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import all Bayesian inference components
from agentical.reasoning import (
    BayesianInferenceEngine,
    BayesianConfig,
    InferenceResult,
    Evidence,
    Hypothesis,
    BeliefUpdater,
    BeliefUpdaterConfig,
    BeliefUpdate,
    EvidenceType,
    UpdateStrategy,
    DecisionTree,
    DecisionTreeConfig,
    DecisionNode,
    NodeType,
    DecisionCriteria,
    UncertaintyQuantifier,
    UncertaintyQuantifierConfig,
    UncertaintyMeasure,
    DistributionType,
    QuantificationMethod,
    BayesMCPClient,
    MCPConfig,
    InferenceRequest,
    InferenceResponse,
    BayesianNetwork,
    MarkovChain,
    HiddenMarkovModel,
    GaussianProcess,
    ProbabilisticModelConfig,
    ModelType
)


class TestBayesianInferenceEngine:
    """Test suite for BayesianInferenceEngine core functionality."""

    @pytest.fixture
    async def engine(self):
        """Create a configured Bayesian inference engine."""
        config = BayesianConfig(
            confidence_threshold=0.75,
            uncertainty_tolerance=0.25,
            max_iterations=100,
            enable_caching=True
        )
        engine = BayesianInferenceEngine(config)
        await engine.initialize()
        return engine

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization and configuration."""
        assert engine.is_initialized
        assert engine.config.confidence_threshold == 0.75
        assert engine.config.uncertainty_tolerance == 0.25
        assert len(engine.hypotheses) == 0
        assert len(engine.evidence_store) == 0

    @pytest.mark.asyncio
    async def test_add_hypothesis(self, engine):
        """Test adding hypotheses to the engine."""
        hypothesis_id = await engine.add_hypothesis(
            name="test_hypothesis",
            description="Test hypothesis for validation",
            prior_probability=0.6
        )

        assert hypothesis_id in engine.hypotheses
        hypothesis = engine.hypotheses[hypothesis_id]
        assert hypothesis.name == "test_hypothesis"
        assert hypothesis.prior_probability == 0.6
        assert hypothesis.posterior_probability == 0.6

    @pytest.mark.asyncio
    async def test_add_evidence(self, engine):
        """Test adding evidence to the engine."""
        evidence_id = await engine.add_evidence(
            name="test_evidence",
            value=True,
            likelihood=0.8,
            reliability=0.9,
            source="test_source"
        )

        assert evidence_id in engine.evidence_store
        evidence = engine.evidence_store[evidence_id]
        assert evidence.name == "test_evidence"
        assert evidence.likelihood == 0.8
        assert evidence.reliability == 0.9

    @pytest.mark.asyncio
    async def test_compute_inference(self, engine):
        """Test Bayesian inference computation."""
        # Add hypothesis
        hypothesis_id = await engine.add_hypothesis(
            name="rain_hypothesis",
            description="It will rain today",
            prior_probability=0.3
        )

        # Add supporting evidence
        evidence_id = await engine.add_evidence(
            name="clouds_observed",
            value=True,
            likelihood=0.9,
            reliability=0.95
        )

        # Perform inference
        result = await engine.compute_inference(hypothesis_id, [evidence_id])

        assert isinstance(result, InferenceResult)
        assert result.success
        assert result.hypothesis_id == hypothesis_id
        assert 0.0 <= result.posterior_probability <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0
        assert result.evidence_count == 1

    @pytest.mark.asyncio
    async def test_belief_updating(self, engine):
        """Test dynamic belief updating with new evidence."""
        # Add hypothesis
        hypothesis_id = await engine.add_hypothesis(
            name="market_crash",
            description="Stock market will crash",
            prior_probability=0.1
        )

        # Add first evidence
        evidence1_id = await engine.add_evidence(
            name="volatility_high",
            value=True,
            likelihood=0.7,
            reliability=0.8
        )

        # Initial inference
        result1 = await engine.compute_inference(hypothesis_id, [evidence1_id])
        initial_posterior = result1.posterior_probability

        # Add second evidence
        evidence2_id = await engine.add_evidence(
            name="economic_indicators",
            value=True,
            likelihood=0.8,
            reliability=0.9
        )

        # Updated inference
        result2 = await engine.update_belief(hypothesis_id, evidence2_id)

        assert result2.posterior_probability != initial_posterior
        assert result2.evidence_count == 2

    @pytest.mark.asyncio
    async def test_hypothesis_ranking(self, engine):
        """Test hypothesis ranking by posterior probability."""
        # Add multiple hypotheses
        h1_id = await engine.add_hypothesis("hypothesis_1", "First hypothesis", 0.3)
        h2_id = await engine.add_hypothesis("hypothesis_2", "Second hypothesis", 0.7)
        h3_id = await engine.add_hypothesis("hypothesis_3", "Third hypothesis", 0.5)

        # Get ranking
        ranking = await engine.get_hypothesis_ranking()

        assert len(ranking) == 3
        assert ranking[0][0] == h2_id  # Highest probability first
        assert ranking[0][1] == 0.7
        assert ranking[2][0] == h1_id  # Lowest probability last

    @pytest.mark.asyncio
    async def test_engine_metrics(self, engine):
        """Test engine performance metrics collection."""
        # Perform some operations
        h_id = await engine.add_hypothesis("test", "test", 0.5)
        e_id = await engine.add_evidence("test", True, 0.8, 0.9)
        await engine.compute_inference(h_id, [e_id])

        metrics = await engine.get_engine_metrics()

        assert "inference_count" in metrics
        assert "hypothesis_count" in metrics
        assert "evidence_count" in metrics
        assert metrics["inference_count"] >= 1
        assert metrics["hypothesis_count"] >= 1
        assert metrics["evidence_count"] >= 1


class TestBeliefUpdater:
    """Test suite for BeliefUpdater component."""

    @pytest.fixture
    async def updater(self):
        """Create a configured belief updater."""
        config = BeliefUpdaterConfig(
            default_strategy=UpdateStrategy.SEQUENTIAL,
            convergence_threshold=1e-4,
            stability_window=10
        )
        return BeliefUpdater(config)

    @pytest.mark.asyncio
    async def test_initialize_belief(self, updater):
        """Test belief state initialization."""
        hypothesis_id = "test_hypothesis"
        await updater.initialize_belief(hypothesis_id, 0.6, 0.8)

        belief, confidence, state = await updater.get_belief_state(hypothesis_id)
        assert belief == 0.6
        assert confidence == 0.8
        assert hypothesis_id in updater.update_histories

    @pytest.mark.asyncio
    async def test_update_belief_supporting(self, updater):
        """Test belief update with supporting evidence."""
        hypothesis_id = "test_hypothesis"
        await updater.initialize_belief(hypothesis_id, 0.5, 0.5)

        update = await updater.update_belief(
            hypothesis_id,
            "evidence_1",
            EvidenceType.SUPPORTING,
            0.8,
            0.9
        )

        assert isinstance(update, BeliefUpdate)
        assert update.hypothesis_id == hypothesis_id
        assert update.evidence_type == EvidenceType.SUPPORTING
        assert update.posterior_belief > update.prior_belief

    @pytest.mark.asyncio
    async def test_update_belief_contradicting(self, updater):
        """Test belief update with contradicting evidence."""
        hypothesis_id = "test_hypothesis"
        await updater.initialize_belief(hypothesis_id, 0.7, 0.5)

        update = await updater.update_belief(
            hypothesis_id,
            "evidence_1",
            EvidenceType.CONTRADICTING,
            0.8,
            0.9
        )

        assert update.posterior_belief < update.prior_belief

    @pytest.mark.asyncio
    async def test_batch_update(self, updater):
        """Test batch belief updates."""
        # Initialize multiple hypotheses
        for i in range(3):
            await updater.initialize_belief(f"hypothesis_{i}", 0.5, 0.5)

        # Prepare batch updates
        updates = [
            ("hypothesis_0", "evidence_0", EvidenceType.SUPPORTING, 0.8, 0.9),
            ("hypothesis_1", "evidence_1", EvidenceType.CONTRADICTING, 0.7, 0.8),
            ("hypothesis_2", "evidence_2", EvidenceType.NEUTRAL, 0.5, 0.9)
        ]

        results = await updater.batch_update(updates)

        assert len(results) == 3
        assert all(isinstance(r, BeliefUpdate) for r in results)

    @pytest.mark.asyncio
    async def test_convergence_analysis(self, updater):
        """Test convergence analysis functionality."""
        hypothesis_id = "test_hypothesis"
        await updater.initialize_belief(hypothesis_id, 0.5, 0.5)

        # Perform multiple updates to create history
        for i in range(15):
            await updater.update_belief(
                hypothesis_id,
                f"evidence_{i}",
                EvidenceType.SUPPORTING,
                0.6 + (i % 3) * 0.1,
                0.9
            )

        analysis = await updater.get_convergence_analysis(hypothesis_id)

        assert "status" in analysis
        assert "analysis" in analysis
        assert "recent_variance" in analysis["analysis"]
        assert "is_converging" in analysis["analysis"]


class TestDecisionTree:
    """Test suite for DecisionTree component."""

    @pytest.fixture
    async def decision_tree(self):
        """Create a configured decision tree."""
        config = DecisionTreeConfig(
            max_depth=5,
            max_branches_per_node=4,
            confidence_threshold=0.7
        )
        return DecisionTree(config)

    @pytest.mark.asyncio
    async def test_create_root_node(self, decision_tree):
        """Test root node creation."""
        root_id = await decision_tree.create_root_node(
            name="investment_decision",
            description="Should we invest in this opportunity?",
            available_actions=["invest", "hold", "reject"]
        )

        assert decision_tree.root_node_id == root_id
        assert root_id in decision_tree.nodes
        root_node = decision_tree.nodes[root_id]
        assert root_node.node_type == NodeType.ROOT
        assert len(root_node.available_actions) == 3

    @pytest.mark.asyncio
    async def test_add_decision_node(self, decision_tree):
        """Test adding decision nodes to the tree."""
        root_id = await decision_tree.create_root_node("root", "Root decision")

        decision_id = await decision_tree.add_decision_node(
            parent_id=root_id,
            name="market_analysis",
            available_actions=["bullish", "bearish", "neutral"],
            condition="market_conditions_evaluated",
            probability=0.8
        )

        assert decision_id in decision_tree.nodes
        decision_node = decision_tree.nodes[decision_id]
        assert decision_node.node_type == NodeType.DECISION
        assert decision_node.parent_id == root_id
        assert decision_id in decision_tree.nodes[root_id].children_ids

    @pytest.mark.asyncio
    async def test_add_chance_node(self, decision_tree):
        """Test adding chance nodes with probabilistic outcomes."""
        root_id = await decision_tree.create_root_node("root", "Root decision")

        chance_id = await decision_tree.add_chance_node(
            parent_id=root_id,
            name="market_volatility",
            outcomes=[("high", 0.3), ("medium", 0.5), ("low", 0.2)],
            condition="market_opens"
        )

        assert chance_id in decision_tree.nodes
        chance_node = decision_tree.nodes[chance_id]
        assert chance_node.node_type == NodeType.CHANCE
        assert len(chance_node.children_ids) == 3

    @pytest.mark.asyncio
    async def test_add_terminal_node(self, decision_tree):
        """Test adding terminal nodes with values."""
        root_id = await decision_tree.create_root_node("root", "Root decision")

        terminal_id = await decision_tree.add_terminal_node(
            parent_id=root_id,
            name="profit_outcome",
            value=100.0,
            utility=0.8,
            condition="investment_succeeds",
            probability=0.6
        )

        assert terminal_id in decision_tree.nodes
        terminal_node = decision_tree.nodes[terminal_id]
        assert terminal_node.node_type == NodeType.TERMINAL
        assert terminal_node.expected_value == 100.0
        assert terminal_node.utility_value == 0.8

    @pytest.mark.asyncio
    async def test_tree_evaluation(self, decision_tree):
        """Test complete tree evaluation and optimization."""
        # Build a simple decision tree
        root_id = await decision_tree.create_root_node(
            "investment_decision",
            "Investment decision tree",
            ["invest", "hold"]
        )

        # Add terminal outcomes
        success_id = await decision_tree.add_terminal_node(
            root_id, "success", 200.0, 0.9, "invest_and_succeed", 0.7
        )

        failure_id = await decision_tree.add_terminal_node(
            root_id, "failure", -50.0, 0.1, "invest_and_fail", 0.3
        )

        hold_id = await decision_tree.add_terminal_node(
            root_id, "hold", 0.0, 0.5, "hold_position", 1.0
        )

        # Evaluate the tree
        best_outcome = await decision_tree.evaluate_tree()

        assert best_outcome is not None
        assert hasattr(best_outcome, 'expected_value')
        assert hasattr(best_outcome, 'utility')

    @pytest.mark.asyncio
    async def test_decision_path(self, decision_tree):
        """Test optimal decision path extraction."""
        root_id = await decision_tree.create_root_node("root", "Root")
        terminal_id = await decision_tree.add_terminal_node(
            root_id, "outcome", 50.0, 0.7
        )

        path = await decision_tree.get_decision_path()

        assert len(path) >= 1
        assert path[0].node_id == root_id

    @pytest.mark.asyncio
    async def test_tree_metrics(self, decision_tree):
        """Test tree metrics collection."""
        root_id = await decision_tree.create_root_node("root", "Root")
        await decision_tree.add_terminal_node(root_id, "outcome", 50.0, 0.7)
        await decision_tree.evaluate_tree()

        metrics = await decision_tree.get_tree_metrics()

        assert "total_nodes" in metrics.__dict__
        assert "total_evaluations" in metrics.__dict__
        assert metrics.total_nodes >= 2


class TestUncertaintyQuantifier:
    """Test suite for UncertaintyQuantifier component."""

    @pytest.fixture
    async def quantifier(self):
        """Create a configured uncertainty quantifier."""
        config = UncertaintyQuantifierConfig(
            default_method=QuantificationMethod.BAYESIAN,
            confidence_levels=[0.68, 0.95, 0.99],
            monte_carlo_samples=1000
        )
        return UncertaintyQuantifier(config)

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_normal(self, quantifier):
        """Test uncertainty quantification with normal data."""
        # Generate normal data
        data = np.random.normal(10, 2, 100)

        measure = await quantifier.quantify_uncertainty(
            entity_id="test_entity",
            data=data,
            method=QuantificationMethod.BAYESIAN
        )

        assert isinstance(measure, UncertaintyMeasure)
        assert measure.variance > 0
        assert measure.standard_deviation > 0
        assert measure.entropy >= 0
        assert len(measure.confidence_intervals) > 0

    @pytest.mark.asyncio
    async def test_estimate_confidence_intervals(self, quantifier):
        """Test confidence interval estimation."""
        data = np.random.normal(5, 1, 50)

        intervals = await quantifier.estimate_confidence_intervals(
            data=data,
            confidence_levels=[0.68, 0.95],
            method=QuantificationMethod.BOOTSTRAP
        )

        assert len(intervals) == 2
        assert 0.68 in intervals
        assert 0.95 in intervals

        # Check interval properties
        lower_68, upper_68 = intervals[0.68]
        lower_95, upper_95 = intervals[0.95]

        assert lower_68 < upper_68
        assert lower_95 < upper_95
        assert upper_95 - lower_95 > upper_68 - lower_68  # 95% interval should be wider

    @pytest.mark.asyncio
    async def test_decompose_uncertainty(self, quantifier):
        """Test aleatory vs epistemic uncertainty decomposition."""
        # Create model predictions with different uncertainties
        predictions = [
            np.random.normal(10, 1, 20),  # Model 1
            np.random.normal(10.5, 1.2, 20),  # Model 2
            np.random.normal(9.8, 0.8, 20)   # Model 3
        ]

        aleatory, epistemic = await quantifier.decompose_uncertainty(
            entity_id="test_entity",
            model_predictions=predictions
        )

        assert aleatory >= 0
        assert epistemic >= 0
        assert isinstance(aleatory, float)
        assert isinstance(epistemic, float)

    @pytest.mark.asyncio
    async def test_uncertainty_evolution(self, quantifier):
        """Test uncertainty tracking over time."""
        entity_id = "evolving_entity"

        # Add multiple uncertainty measurements
        for i in range(15):
            data = np.random.normal(i, 1 + i*0.1, 20)
            await quantifier.quantify_uncertainty(entity_id, data)

        evolution = await quantifier.track_uncertainty_evolution(entity_id)

        assert evolution["status"] == "analyzed"
        assert "trend_slope" in evolution["analysis"]
        assert "volatility" in evolution["analysis"]
        assert "stability_score" in evolution["analysis"]


class TestBayesMCPClient:
    """Test suite for MCP integration component."""

    @pytest.fixture
    async def mcp_client(self):
        """Create a mock MCP client for testing."""
        config = MCPConfig(
            server_url="http://localhost:3000",
            max_retries=2,
            timeout_seconds=5.0
        )
        return BayesMCPClient(config)

    @pytest.mark.asyncio
    async def test_client_initialization(self, mcp_client):
        """Test MCP client initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()

            await mcp_client.initialize()

            assert mcp_client.session is not None

    @pytest.mark.asyncio
    async def test_inference_request(self, mcp_client):
        """Test sending inference requests to MCP server."""
        # Mock the HTTP session
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "request_id": "test_request",
            "success": True,
            "posterior_belief": 0.75,
            "confidence_level": 0.8,
            "uncertainty_measure": 0.2
        })

        with patch.object(mcp_client, 'session') as mock_session:
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()

            request = InferenceRequest(
                hypothesis_id="test_hypothesis",
                prior_belief=0.5,
                evidence_likelihood=0.8,
                evidence_reliability=0.9
            )

            response = await mcp_client.send_inference_request(request)

            assert isinstance(response, InferenceResponse)
            assert response.success
            assert response.posterior_belief == 0.75

    @pytest.mark.asyncio
    async def test_batch_requests(self, mcp_client):
        """Test batch processing of inference requests."""
        requests = [
            InferenceRequest(hypothesis_id=f"hyp_{i}", prior_belief=0.5)
            for i in range(3)
        ]

        # Mock batch response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "responses": [
                {"request_id": f"test_request_{i}", "success": True, "posterior_belief": 0.6}
                for i in range(3)
            ]
        })

        with patch.object(mcp_client, 'session') as mock_session:
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()

            responses = await mcp_client.batch_inference_requests(requests)

            assert len(responses) == 3
            assert all(isinstance(r, InferenceResponse) for r in responses)


class TestProbabilisticModels:
    """Test suite for probabilistic models."""

    @pytest.fixture
    async def model_config(self):
        """Create model configuration."""
        return ProbabilisticModelConfig(
            model_type=ModelType.BAYESIAN_NETWORK,
            max_training_iterations=50,
            convergence_threshold=1e-4
        )

    @pytest.mark.asyncio
    async def test_bayesian_network(self, model_config):
        """Test Bayesian network implementation."""
        model = BayesianNetwork("bn_test", model_config)

        # Generate synthetic data
        training_data = np.random.randint(0, 2, (100, 4))

        # Train model
        parameters = await model.train(training_data)

        assert model.state.value == "trained"
        assert parameters.model_type == ModelType.BAYESIAN_NETWORK
        assert parameters.num_variables == 4

        # Test prediction
        test_input = np.array([[1, 0, 1, 0]])
        prediction = await model.predict(test_input)

        assert isinstance(prediction.mean_prediction, np.ndarray)
        assert prediction.model_id == "bn_test"

    @pytest.mark.asyncio
    async def test_markov_chain(self, model_config):
        """Test Markov chain implementation."""
        model_config.model_type = ModelType.MARKOV_CHAIN
        model = MarkovChain("mc_test", model_config, num_states=3)

        # Generate sequence data
        sequences = [
            np.array([0, 1, 2, 1, 0]),
            np.array([1, 2, 2, 0, 1]),
            np.array([0, 0, 1, 2, 2])
        ]
        training_data = np.array(sequences)

        # Train model
        parameters = await model.train(training_data)

        assert model.state.value == "trained"
        assert parameters.model_type == ModelType.MARKOV_CHAIN

        # Test prediction
        prediction = await model.predict(np.array([1]))

        assert prediction.model_id == "mc_test"
        assert hasattr(prediction, 'probability_distribution')

    @pytest.mark.asyncio
    async def test_gaussian_process(self, model_config):
        """Test Gaussian process implementation."""
        model_config.model_type = ModelType.GAUSSIAN_PROCESS
        model = GaussianProcess("gp_test", model_config)

        # Generate regression data
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = np.sin(X).flatten() + np.random.normal(0, 0.1, 50)

        # Train model
        parameters = await model.train(X, y)

        assert model.state.value == "trained"
        assert parameters.model_type == ModelType.GAUSSIAN_PROCESS

        # Test prediction
        X_test = np.array([[5.0]])
        prediction = await model.predict(X_test)

        assert isinstance(prediction.mean_prediction, float)
        assert prediction.variance_prediction > 0
        assert len(prediction.confidence_intervals) > 0


class TestIntegrationWorkflows:
    """Integration tests for complete Bayesian inference workflows."""

    @pytest.mark.asyncio
    async def test_complete_inference_workflow(self):
        """Test complete inference workflow from evidence to decision."""
        # Initialize components
        engine_config = BayesianConfig(confidence_threshold=0.75)
        engine = BayesianInferenceEngine(engine_config)
        await engine.initialize()

        updater_config = BeliefUpdaterConfig()
        updater = BeliefUpdater(updater_config)

        tree_config = DecisionTreeConfig(max_depth=3)
        tree = DecisionTree(tree_config)

        # Workflow: Medical diagnosis scenario
        # 1. Add hypothesis
        disease_hypothesis = await engine.add_hypothesis(
            "has_disease",
            "Patient has the target disease",
            0.1  # Low prior probability
        )

        # 2. Initialize belief tracking
        await updater.initialize_belief(disease_hypothesis, 0.1, 0.5)

        # 3. Add evidence sequentially
        symptoms = [
            ("fever", 0.8, 0.9),
            ("cough", 0.7, 0.8),
            ("fatigue", 0.6, 0.7)
        ]

        evidence_ids = []
        for symptom, likelihood, reliability in symptoms:
            evidence_id = await engine.add_evidence(
                symptom, True, likelihood, reliability, "clinical_observation"
            )
            evidence_ids.append(evidence_id)

            # Update belief
            await updater.update_belief(
                disease_hypothesis,
                evidence_id,
                EvidenceType.SUPPORTING,
                likelihood,
                reliability
            )

        # 4. Perform inference
        result = await engine.compute_inference(disease_hypothesis, evidence_ids)

        # 5. Build decision tree for treatment
        root_id = await tree.create_root_node(
            "treatment_decision",
            "Choose treatment based on diagnosis confidence",
            ["aggressive_treatment", "conservative_treatment", "no_treatment"]
        )

        # Add outcomes based on confidence level
        if result.confidence_level > 0.8:
            await tree.add_terminal_node(
                root_id, "treat_aggressively", 100.0, 0.9, "high_confidence", 1.0
            )
        else:
            await tree.add_terminal_node(
                root_id, "treat_conservatively", 50.0, 0.7, "medium_confidence", 1.0
            )

        # 6. Evaluate decision
        decision = await tree.evaluate_tree()

        # Validate workflow results
        assert result.posterior_probability > 0.1  # Should increase from prior
        assert result.confidence_level > 0.0
        assert decision is not None
        assert hasattr(decision, 'expected_value')

    @pytest.mark.asyncio
    async def test_uncertainty_quantification_workflow(self):
        """Test uncertainty quantification in decision-making workflow."""
        quantifier_config = UncertaintyQuantifierConfig(
            confidence_levels=[0.68, 0.95, 0.99]
        )
        quantifier = UncertaintyQuantifier(quantifier_config)

        # Simulate model predictions with uncertainty
        entity_id = "market_prediction"

        # Multiple prediction rounds
        for round_num in range(10):
            # Simulate varying prediction accuracy
            noise_level = 0.5 + round_num * 0.1
            predictions = np.random.normal(100, noise_level, 50)

            measure = await quantifier.quantify_uncertainty(
                entity_id=entity_id,
                data=predictions,
                method=QuantificationMethod.BAYESIAN
            )

            assert measure.total_uncertainty > 0
            assert len(measure.confidence_intervals) == 3

        # Analyze uncertainty evolution
        evolution = await quantifier.track_uncertainty_evolution(entity_id)

        assert evolution["status"] == "analyzed"
        assert "trend_slope" in evolution["analysis"]

    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self):
        """Test comparing different probabilistic models."""
        config = ProbabilisticModelConfig(max_training_iterations=20)

        # Generate synthetic data
        X = np.random.randn(100, 2)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

        # Test different models
        models = [
            GaussianProcess("gp_model", config),
            BayesianNetwork("bn_model", config)
        ]

        validation_scores = {}

        for model in models:
            if isinstance(model, GaussianProcess):
                # GP needs labels
                await model.train(X, y)

                # Validate
                X_val = X[:20]
                y_val = y[:20]
                validation_metrics = await model.validate_model(X_val, y_val)
                validation_scores[model.model_id] = validation_metrics["mse"]

            elif isinstance(model, BayesianNetwork):
                # BN works with discrete data
                X_discrete = (X > 0).astype(int)
                await model.train(X_discrete)

                # Validate
                X_val_discrete = X_discrete[:20]
                validation_metrics = await model.validate_model(X_val_discrete)
                validation_scores[model.model_id] = validation_metrics["log_likelihood"]

        # Compare models
        assert len(validation_scores) == 2
        assert "gp_model" in validation_scores
        assert "bn_model" in validation_scores


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_probability_values(self):
        """Test handling of invalid probability values."""
        config = BayesianConfig()
        engine = BayesianInferenceEngine(config)
        await engine.initialize()

        # Test invalid prior probability
        with pytest.raises(ValueError):
            await engine.add_hypothesis("test", "test", -0.1)

        with pytest.raises(ValueError):
            await engine.add_hypothesis("test", "test", 1.5)

        # Test invalid evidence likelihood
        with pytest.raises(ValueError):
            await engine.add_evidence("test", True, -0.1, 0.9)

    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test handling of insufficient data scenarios."""
        config = UncertaintyQuantifierConfig()
        quantifier = UncertaintyQuantifier(config)

        # Test with empty data
        with pytest.raises(ValidationError):
            await quantifier.quantify_uncertainty("test", [])

        # Test with single data point
        with pytest.raises(ValidationError):
            await quantifier.quantify_uncertainty("test", [1.0])

    @pytest.mark.asyncio
    async def test_model_training_errors(self):
        """Test model training error scenarios."""
        config = ProbabilisticModelConfig()
        model = GaussianProcess("test", config)

        # Test training without labels
        X = np.random.randn(10, 2)
        with pytest.raises(ValidationError):
            await model.train(X)

        # Test prediction before training
        with pytest.raises(ValidationError):
            await model.predict(X[0:1])

    @pytest.mark.asyncio
    async def test_tree_constraint_violations(self):
        """Test decision tree constraint violations."""
        config = DecisionTreeConfig(max_depth=2, max_branches_per_node=2)
        tree = DecisionTree(config)

        root_id = await tree.create_root_node("root", "Root")

        # Add nodes to reach depth limit
        child1_id = await tree.add_decision_node(root_id, "child1", ["action1"])
        child2_id = await tree.add_decision_node(child1_id, "child2", ["action2"])

        # This should fail due to depth limit
        with pytest.raises(ValidationError):
            await tree.add_decision_node(child2_id, "child3", ["action3"])


class TestPerformance:
    """Performance tests for Bayesian inference components."""

    @pytest.mark.asyncio
    async def test_engine_caching_performance(self):
        """Test caching improves performance."""
        config = BayesianConfig(enable_caching=True)
        engine = BayesianInferenceEngine(config)
        await engine.initialize()

        # Setup
        h_id = await engine.add_hypothesis("test", "test", 0.5)
        e_id = await engine.add_evidence("test", True, 0.8, 0.9)

        # First computation (cache miss)
        start_time = datetime.utcnow()
        result1 = await engine.compute_inference(h_id, [e_id])
        first_time = (datetime.utcnow() - start_time).total_seconds()

        # Second computation (cache hit)
        start_time = datetime.utcnow()
        result2 = await engine.compute_inference(h_id, [e_id])
        second_time = (datetime.utcnow() - start_time).total_seconds()

        # Verify results are identical
        assert result1.posterior_probability == result2.posterior_probability
        assert result1.confidence_level == result2.confidence_level

        # Performance should improve (cache hit should be faster)
        # Note: In practice, cache hits should be much faster
        metrics = await engine.get_engine_metrics()
        assert metrics["cache_hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test batch processing efficiency."""
        config = BeliefUpdaterConfig()
        updater = BeliefUpdater(config)

        # Initialize beliefs
        num_hypotheses = 20
        for i in range(num_hypotheses):
            await updater.initialize_belief(f"hypothesis_{i}", 0.5, 0.5)

        # Prepare batch updates
        batch_updates = [
            (f"hypothesis_{i}", f"evidence_{i}", EvidenceType.SUPPORTING, 0.7, 0.9)
            for i in range(num_hypotheses)
        ]

        # Batch processing
        start_time = datetime.utcnow()
        batch_results = await updater.batch_update(batch_updates)
        batch_time = (datetime.utcnow() - start_time).total_seconds()

        # Individual processing
        start_time = datetime.utcnow()
        individual_results = []
        for update in batch_updates:
            result = await updater.update_belief(*update)
            individual_results.append(result)
        individual_time = (datetime.utcnow() - start_time).total_seconds()

        # Verify results
        assert len(batch_results) == num_hypotheses
        assert len(individual_results) == num_hypotheses

        # Batch should complete successfully
        assert all(isinstance(r, BeliefUpdate) for r in batch_results)

    @pytest.mark.asyncio
    async def test_large_tree_evaluation(self):
        """Test decision tree performance with larger trees."""
        config = DecisionTreeConfig(max_depth=4, max_branches_per_node=3)
        tree = DecisionTree(config)

        # Build a moderately complex tree
        root_id = await tree.create_root_node("investment", "Investment decision")

        # Level 1: Market conditions
        market_high_id = await tree.add_decision_node(
            root_id, "market_high", ["buy", "hold", "sell"]
        )
        market_low_id = await tree.add_decision_node(
            root_id, "market_low", ["buy", "hold", "sell"]
        )

        # Level 2: Add terminal nodes for each market condition
        for market_id, market_name in [(market_high_id, "high"), (market_low_id, "low")]:
            for action, value in [("buy", 100), ("hold", 50), ("sell", 20)]:
                await tree.add_terminal_node(
                    market_id, f"{action}_{market_name}", value, 0.7
                )

        # Evaluate tree
        start_time = datetime.utcnow()
        result = await tree.evaluate_tree()
        evaluation_time = (datetime.utcnow() - start_time).total_seconds()

        # Verify evaluation completed
        assert result is not None
        assert evaluation_time < 1.0  # Should complete within 1 second

        # Check metrics
        metrics = await tree.get_tree_metrics()
        assert metrics.total_nodes >= 7  # Root + 2 decision + 6 terminal


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
