#!/usr/bin/env python3
"""
Integration Test for Tasks 5.2 and 5.3 - Complete Agent Reasoning System

This test validates the complete implementation of the Agent Reasoning System
including Bayesian inference, genetic algorithm optimization, and simulation engine.

Tests:
- Task 5.1: Bayesian Inference Integration (already completed)
- Task 5.2: Genetic Algorithm Optimization (new implementation)
- Task 5.3: Simulation Engine (new implementation)
- Full system integration and interoperability
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add the agentical directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class TestTask5Complete:
    """Comprehensive test suite for complete Agent Reasoning System."""

    def setup_method(self):
        """Setup test environment."""
        # Import after path setup
        from reasoning.bayesian_engine import (
            BayesianInferenceEngine, BayesianConfig, Hypothesis, Evidence
        )
        from reasoning.genetic_optimizer import (
            GeneticAlgorithmEngine, GeneticConfig, FitnessFunction, Individual,
            SelectionMethod, CrossoverMethod, MutationMethod, OptimizationObjective
        )
        from reasoning.simulation_engine import (
            SimulationEngine, SimulationConfig, Scenario, Parameter,
            SamplingMethod, DistributionType, SimulationType
        )

        # Store for use in tests
        self.bayesian_engine_class = BayesianInferenceEngine
        self.bayesian_config_class = BayesianConfig
        self.genetic_engine_class = GeneticAlgorithmEngine
        self.genetic_config_class = GeneticConfig
        self.simulation_engine_class = SimulationEngine
        self.simulation_config_class = SimulationConfig
        self.hypothesis_class = Hypothesis
        self.evidence_class = Evidence
        self.fitness_function_class = FitnessFunction
        self.individual_class = Individual
        self.scenario_class = Scenario
        self.parameter_class = Parameter

    @pytest.mark.asyncio
    async def test_task_5_2_genetic_optimizer_basic(self):
        """Test basic genetic algorithm optimization functionality."""

        # Create genetic algorithm configuration
        config = self.genetic_config_class(
            population_size=20,
            max_generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1,
            selection_method=SelectionMethod.TOURNAMENT,
            crossover_method=CrossoverMethod.UNIFORM,
            mutation_method=MutationMethod.GAUSSIAN
        )

        # Define a simple fitness function (maximize sum of squares)
        def fitness_evaluator(genotype, phenotype=None):
            if isinstance(genotype, (list, tuple)):
                return sum(x**2 for x in genotype if isinstance(x, (int, float)))
            return 0.0

        fitness_function = self.fitness_function_class(
            name="sum_of_squares",
            objective=OptimizationObjective.MAXIMIZE,
            evaluator=fitness_evaluator
        )

        # Create genetic algorithm engine
        ga_engine = self.genetic_engine_class(
            config=config,
            fitness_functions=[fitness_function]
        )

        # Define genotype generator (random list of floats)
        def genotype_generator():
            return [np.random.uniform(-10, 10) for _ in range(5)]

        # Run optimization
        result = await ga_engine.optimize(
            genotype_generator=genotype_generator,
            termination_criteria={"max_time_minutes": 0.5}
        )

        # Validate results
        assert result is not None
        assert result.success
        assert result.best_individual is not None
        assert len(result.best_individual.fitness_values) > 0
        assert result.generations_completed > 0
        assert result.total_evaluations > 0

        print(f"✅ Genetic Algorithm - Best fitness: {sum(result.best_individual.fitness_values):.2f}")
        print(f"✅ Genetic Algorithm - Generations: {result.generations_completed}")

    @pytest.mark.asyncio
    async def test_task_5_3_simulation_engine_basic(self):
        """Test basic Monte Carlo simulation functionality."""

        # Create simulation configuration
        config = self.simulation_config_class(
            simulation_type=SimulationType.MONTE_CARLO,
            num_simulations=1000,
            sampling_method=SamplingMethod.LATIN_HYPERCUBE,
            parallel_execution=False  # Disable for testing
        )

        # Create simulation engine
        sim_engine = self.simulation_engine_class(config=config)

        # Define parameters with uncertainty
        param1 = self.parameter_class(
            name="input_value",
            distribution_type=DistributionType.NORMAL,
            distribution_params={"mean": 10.0, "std": 2.0}
        )

        param2 = self.parameter_class(
            name="multiplier",
            distribution_type=DistributionType.UNIFORM,
            distribution_params={"low": 0.5, "high": 1.5}
        )

        # Create scenario
        scenario = self.scenario_class(
            name="test_scenario",
            description="Simple test scenario",
            parameters={"input_value": param1, "multiplier": param2}
        )

        # Define simulation model
        def simulation_model(parameters, initial_state):
            input_val = parameters.get("input_value", 0)
            multiplier = parameters.get("multiplier", 1)
            result = input_val * multiplier + np.random.normal(0, 0.5)
            return {"output": result}

        # Run simulation
        result = await sim_engine.run_simulation(
            scenarios=[scenario],
            model_function=simulation_model,
            output_variables=["output"]
        )

        # Validate results
        assert result is not None
        assert result.final_sample_size > 0
        assert "output" in result.summary_statistics
        assert "mean" in result.summary_statistics["output"]
        assert result.simulation_quality_score > 0

        print(f"✅ Simulation Engine - Sample size: {result.final_sample_size}")
        print(f"✅ Simulation Engine - Mean output: {result.summary_statistics['output']['mean']:.2f}")

    @pytest.mark.asyncio
    async def test_bayesian_genetic_integration(self):
        """Test integration between Bayesian inference and genetic optimization."""

        # Create Bayesian engine
        bayesian_config = self.bayesian_config_class(
            confidence_threshold=0.7,
            max_iterations=100
        )
        bayesian_engine = self.bayesian_engine_class(bayesian_config)

        # Create genetic algorithm with Bayesian fitness evaluation
        genetic_config = self.genetic_config_class(
            population_size=10,
            max_generations=5,
            use_bayesian_fitness=True
        )

        # Define Bayesian-enhanced fitness function
        async def bayesian_fitness_evaluator(genotype, phenotype=None):
            # Simple function: maximize negative distance from origin
            if isinstance(genotype, (list, tuple)):
                distance = sum(x**2 for x in genotype if isinstance(x, (int, float)))

                # Create evidence about fitness quality
                evidence = self.evidence_class(
                    name="fitness_evaluation",
                    value={"distance": distance},
                    likelihood=min(0.95, distance / 100.0),
                    reliability=0.9
                )

                # Create hypothesis about fitness quality
                hypothesis = self.hypothesis_class(
                    name="high_quality_fitness",
                    prior_probability=0.5
                )

                # Update belief using Bayesian inference
                belief_result = await bayesian_engine.update_belief(hypothesis, evidence)

                # Adjust fitness based on Bayesian confidence
                confidence_bonus = belief_result.confidence_level * 10
                return distance + confidence_bonus

            return 0.0

        fitness_function = self.fitness_function_class(
            name="bayesian_fitness",
            objective=OptimizationObjective.MAXIMIZE,
            evaluator=bayesian_fitness_evaluator,
            use_bayesian_evaluation=True
        )

        # Create genetic algorithm with Bayesian integration
        ga_engine = self.genetic_engine_class(
            config=genetic_config,
            fitness_functions=[fitness_function],
            bayesian_engine=bayesian_engine
        )

        # Define genotype generator
        def genotype_generator():
            return [np.random.uniform(-5, 5) for _ in range(3)]

        # Run optimization
        result = await ga_engine.optimize(genotype_generator=genotype_generator)

        # Validate integration
        assert result is not None
        assert result.success
        assert result.best_individual is not None
        assert result.best_individual.bayesian_confidence > 0

        print(f"✅ Bayesian-Genetic Integration - Best fitness: {sum(result.best_individual.fitness_values):.2f}")
        print(f"✅ Bayesian-Genetic Integration - Confidence: {result.best_individual.bayesian_confidence:.2f}")

    @pytest.mark.asyncio
    async def test_simulation_genetic_optimization(self):
        """Test using simulation engine to evaluate genetic algorithm fitness."""

        # Create simulation engine for fitness evaluation
        sim_config = self.simulation_config_class(
            num_simulations=100,
            sampling_method=SamplingMethod.RANDOM,
            parallel_execution=False
        )
        sim_engine = self.simulation_engine_class(sim_config)

        # Define simulation-based fitness function
        async def simulation_fitness_evaluator(genotype, phenotype=None):
            if not isinstance(genotype, (list, tuple)) or len(genotype) < 2:
                return 0.0

            # Use genotype as simulation parameters
            param1 = self.parameter_class(
                name="param1",
                distribution_type=DistributionType.NORMAL,
                distribution_params={"mean": genotype[0], "std": 1.0}
            )

            param2 = self.parameter_class(
                name="param2",
                distribution_type=DistributionType.NORMAL,
                distribution_params={"mean": genotype[1], "std": 1.0}
            )

            scenario = self.scenario_class(
                name="fitness_scenario",
                parameters={"param1": param1, "param2": param2}
            )

            # Simulation model: maximize param1 + param2
            def fitness_model(parameters, initial_state):
                p1 = parameters.get("param1", 0)
                p2 = parameters.get("param2", 0)
                return {"fitness": p1 + p2}

            # Run simulation
            sim_result = await sim_engine.run_simulation(
                scenarios=[scenario],
                model_function=fitness_model,
                output_variables=["fitness"]
            )

            # Return mean fitness as objective value
            if "fitness" in sim_result.summary_statistics:
                return sim_result.summary_statistics["fitness"].get("mean", 0.0)

            return 0.0

        # Create genetic algorithm configuration
        genetic_config = self.genetic_config_class(
            population_size=8,
            max_generations=3,
            parallel_evaluation=False  # Sequential for simulation compatibility
        )

        fitness_function = self.fitness_function_class(
            name="simulation_fitness",
            objective=OptimizationObjective.MAXIMIZE,
            evaluator=simulation_fitness_evaluator
        )

        ga_engine = self.genetic_engine_class(
            config=genetic_config,
            fitness_functions=[fitness_function]
        )

        # Define genotype generator (2D parameter space)
        def genotype_generator():
            return [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]

        # Run optimization
        result = await ga_engine.optimize(genotype_generator=genotype_generator)

        # Validate results
        assert result is not None
        assert result.success
        assert result.best_individual is not None

        print(f"✅ Simulation-Genetic Integration - Best genotype: {result.best_individual.genotype}")
        print(f"✅ Simulation-Genetic Integration - Best fitness: {sum(result.best_individual.fitness_values):.2f}")

    @pytest.mark.asyncio
    async def test_complete_reasoning_system_integration(self):
        """Test full integration of all three reasoning components."""

        # 1. Create all three engines
        bayesian_config = self.bayesian_config_class(confidence_threshold=0.75)
        bayesian_engine = self.bayesian_engine_class(bayesian_config)

        sim_config = self.simulation_config_class(num_simulations=50, parallel_execution=False)
        sim_engine = self.simulation_engine_class(sim_config)

        genetic_config = self.genetic_config_class(
            population_size=6,
            max_generations=3,
            use_bayesian_fitness=True
        )

        # 2. Define complex decision scenario
        # Agent needs to optimize resource allocation with uncertainty

        # Simulation parameters
        resource_param = self.parameter_class(
            name="resource_allocation",
            distribution_type=DistributionType.UNIFORM,
            distribution_params={"low": 0.1, "high": 0.9}
        )

        uncertainty_param = self.parameter_class(
            name="market_uncertainty",
            distribution_type=DistributionType.NORMAL,
            distribution_params={"mean": 0.0, "std": 0.2}
        )

        scenario = self.scenario_class(
            name="resource_optimization",
            parameters={
                "resource_allocation": resource_param,
                "market_uncertainty": uncertainty_param
            }
        )

        # 3. Define integrated fitness evaluation
        async def integrated_fitness_evaluator(genotype, phenotype=None):
            if not isinstance(genotype, (list, tuple)) or len(genotype) < 2:
                return 0.0

            strategy_weight = genotype[0]
            risk_tolerance = genotype[1]

            # Use simulation to evaluate strategy under uncertainty
            def strategy_model(parameters, initial_state):
                allocation = parameters.get("resource_allocation", 0.5)
                uncertainty = parameters.get("market_uncertainty", 0.0)

                # Simple strategy: balance allocation with risk tolerance
                base_return = allocation * strategy_weight
                risk_adjusted_return = base_return * (1 + uncertainty * risk_tolerance)

                return {"return": risk_adjusted_return, "risk": abs(uncertainty)}

            # Run simulation
            sim_result = await sim_engine.run_simulation(
                scenarios=[scenario],
                model_function=strategy_model,
                output_variables=["return", "risk"]
            )

            # Use Bayesian inference to assess strategy quality
            if "return" in sim_result.summary_statistics:
                mean_return = sim_result.summary_statistics["return"].get("mean", 0)
                return_std = sim_result.summary_statistics["return"].get("std", 1)

                # Create evidence about strategy performance
                evidence = self.evidence_class(
                    name="strategy_performance",
                    value={
                        "mean_return": mean_return,
                        "volatility": return_std,
                        "sharpe_ratio": mean_return / max(return_std, 0.001)
                    },
                    likelihood=min(0.95, max(0.1, mean_return + 1.0)),
                    reliability=0.85
                )

                # Create hypothesis about strategy effectiveness
                hypothesis = self.hypothesis_class(
                    name="effective_strategy",
                    prior_probability=0.6
                )

                # Update belief
                belief_result = await bayesian_engine.update_belief(hypothesis, evidence)

                # Combine simulation results with Bayesian confidence
                confidence_factor = belief_result.confidence_level
                fitness = mean_return * confidence_factor - return_std * 0.1

                return fitness

            return 0.0

        fitness_function = self.fitness_function_class(
            name="integrated_strategy_fitness",
            objective=OptimizationObjective.MAXIMIZE,
            evaluator=integrated_fitness_evaluator,
            use_bayesian_evaluation=True
        )

        # 4. Run integrated optimization
        ga_engine = self.genetic_engine_class(
            config=genetic_config,
            fitness_functions=[fitness_function],
            bayesian_engine=bayesian_engine
        )

        def genotype_generator():
            return [
                np.random.uniform(0.1, 2.0),  # strategy_weight
                np.random.uniform(-1.0, 1.0)  # risk_tolerance
            ]

        # Execute complete reasoning pipeline
        optimization_result = await ga_engine.optimize(genotype_generator=genotype_generator)

        # 5. Validate complete integration
        assert optimization_result is not None
        assert optimization_result.success
        assert optimization_result.best_individual is not None
        assert optimization_result.best_individual.fitness_values
        assert optimization_result.best_individual.bayesian_confidence > 0

        best_genotype = optimization_result.best_individual.genotype
        best_fitness = sum(optimization_result.best_individual.fitness_values)
        best_confidence = optimization_result.best_individual.bayesian_confidence

        print(f"✅ Complete Integration - Best strategy: {best_genotype}")
        print(f"✅ Complete Integration - Best fitness: {best_fitness:.3f}")
        print(f"✅ Complete Integration - Bayesian confidence: {best_confidence:.3f}")

        # Additional validation checks
        assert len(best_genotype) == 2
        assert best_fitness > -10  # Reasonable fitness range
        assert 0 <= best_confidence <= 1  # Valid confidence range

    def test_task_5_components_exist(self):
        """Test that all required Task 5 components are implemented."""

        # Task 5.1 Components (Bayesian)
        assert hasattr(self, 'bayesian_engine_class')
        assert hasattr(self, 'bayesian_config_class')
        assert hasattr(self, 'hypothesis_class')
        assert hasattr(self, 'evidence_class')

        # Task 5.2 Components (Genetic Algorithm)
        assert hasattr(self, 'genetic_engine_class')
        assert hasattr(self, 'genetic_config_class')
        assert hasattr(self, 'fitness_function_class')
        assert hasattr(self, 'individual_class')

        # Task 5.3 Components (Simulation)
        assert hasattr(self, 'simulation_engine_class')
        assert hasattr(self, 'simulation_config_class')
        assert hasattr(self, 'scenario_class')
        assert hasattr(self, 'parameter_class')

        print("✅ All Task 5 components are properly implemented")

    def test_reasoning_system_architecture(self):
        """Test that the reasoning system has proper architecture."""

        # Test component relationships
        bayesian_config = self.bayesian_config_class()
        genetic_config = self.genetic_config_class()
        simulation_config = self.simulation_config_class()

        # Test that engines can be instantiated
        bayesian_engine = self.bayesian_engine_class(bayesian_config)
        genetic_engine = self.genetic_engine_class(genetic_config, [])
        simulation_engine = self.simulation_engine_class(simulation_config)

        # Test that genetic engine can accept bayesian engine
        genetic_with_bayesian = self.genetic_engine_class(
            genetic_config, [], bayesian_engine
        )

        assert bayesian_engine is not None
        assert genetic_engine is not None
        assert simulation_engine is not None
        assert genetic_with_bayesian is not None

        print("✅ Reasoning system architecture is properly structured")


async def main():
    """Run all tests manually."""
    test_instance = TestTask5Complete()
    test_instance.setup_method()

    print("🚀 Running Task 5.2 and 5.3 Integration Tests")
    print("="*60)

    try:
        # Component existence tests
        test_instance.test_task_5_components_exist()
        test_instance.test_reasoning_system_architecture()

        # Basic functionality tests
        await test_instance.test_task_5_2_genetic_optimizer_basic()
        await test_instance.test_task_5_3_simulation_engine_basic()

        # Integration tests
        await test_instance.test_bayesian_genetic_integration()
        await test_instance.test_simulation_genetic_optimization()
        await test_instance.test_complete_reasoning_system_integration()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED - Tasks 5.2 and 5.3 Successfully Implemented!")
        print("✅ Genetic Algorithm Optimization - COMPLETE")
        print("✅ Simulation Engine - COMPLETE")
        print("✅ Full Reasoning System Integration - COMPLETE")
        print("📊 Task 5: Agent Reasoning System - 100% COMPLETE")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
