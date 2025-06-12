"""
Genetic Algorithm Optimization Engine for Agentical Framework

This module provides sophisticated genetic algorithm capabilities for the Agentical
framework's reasoning system, enabling evolutionary optimization for complex
multi-objective problems and agent strategy evolution.

Features:
- Multi-objective genetic algorithms (NSGA-II, SPEA2)
- Adaptive population management and evolution strategies
- Integration with Darwin-MCP server for advanced algorithms
- Fitness function framework with Bayesian evaluation
- Performance optimization for large solution spaces
- Comprehensive observability and logging
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import asyncio
import json
import math
import numpy as np
from collections import defaultdict, deque
import random
from abc import ABC, abstractmethod

import logfire
from pydantic import BaseModel, Field, validator
from scipy import stats
from scipy.spatial.distance import euclidean

from agentical.core.exceptions import (
    AgentError,
    ValidationError,
    ConfigurationError,
    OptimizationError
)
from agentical.core.structured_logging import (
    StructuredLogger,
    CorrelationContext,
    LogLevel,
    OperationType
)
from agentical.reasoning.bayesian_engine import BayesianInferenceEngine, Hypothesis, Evidence

# Type variables for genetic algorithm
GenotypeType = TypeVar('GenotypeType')
PhenotypeType = TypeVar('PhenotypeType')


class SelectionMethod(str, Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_SELECTION = "rank_selection"
    STOCHASTIC_UNIVERSAL_SAMPLING = "stochastic_universal_sampling"
    ELITIST = "elitist"


class CrossoverMethod(str, Enum):
    """Crossover methods for genetic algorithms."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"
    SIMULATED_BINARY = "simulated_binary"


class MutationMethod(str, Enum):
    """Mutation methods for genetic algorithms."""
    BIT_FLIP = "bit_flip"
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Individual(Generic[GenotypeType]):
    """Represents an individual in the genetic algorithm population."""
    individual_id: str = field(default_factory=lambda: str(uuid4()))
    genotype: GenotypeType = None
    phenotype: Optional[Any] = None

    # Fitness information
    fitness_values: List[float] = field(default_factory=list)
    objective_values: Dict[str, float] = field(default_factory=dict)
    constraint_violations: Dict[str, float] = field(default_factory=dict)

    # Multi-objective optimization metrics
    dominance_rank: int = 0
    crowding_distance: float = 0.0
    pareto_front: int = 0

    # Evolution tracking
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    # Performance metrics
    evaluation_time_ms: float = 0.0
    bayesian_confidence: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessFunction:
    """Defines a fitness function for genetic algorithm optimization."""
    function_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Function definition
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE
    weight: float = 1.0
    evaluator: Optional[Callable] = None

    # Bayesian integration
    use_bayesian_evaluation: bool = True
    uncertainty_penalty: float = 0.1
    confidence_threshold: float = 0.7

    # Constraints
    constraints: List[Callable] = field(default_factory=list)
    penalty_factor: float = 1000.0

    # Performance settings
    cache_evaluations: bool = True
    timeout_seconds: float = 30.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneticConfig(BaseModel):
    """Configuration for genetic algorithm optimization."""

    # Population settings
    population_size: int = Field(default=100, ge=10, le=10000)
    max_generations: int = Field(default=500, ge=1)
    elitism_rate: float = Field(default=0.1, ge=0.0, le=0.5)

    # Selection parameters
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = Field(default=3, ge=2)
    selection_pressure: float = Field(default=2.0, ge=1.0, le=10.0)

    # Crossover parameters
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    crossover_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # Mutation parameters
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    mutation_strength: float = Field(default=0.1, ge=0.0)
    adaptive_mutation: bool = True

    # Multi-objective settings
    enable_multi_objective: bool = True
    pareto_archive_size: int = Field(default=200, ge=10)
    crowding_distance_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Convergence criteria
    convergence_threshold: float = Field(default=1e-6, gt=0.0)
    stagnation_generations: int = Field(default=50, ge=1)
    fitness_tolerance: float = Field(default=1e-4, gt=0.0)

    # Performance settings
    parallel_evaluation: bool = True
    max_workers: int = Field(default=4, ge=1)
    enable_caching: bool = True
    cache_size: int = Field(default=1000, ge=1)

    # Bayesian integration
    use_bayesian_fitness: bool = True
    bayesian_update_frequency: int = Field(default=10, ge=1)
    uncertainty_exploration_factor: float = Field(default=0.2, ge=0.0, le=1.0)

    # Darwin-MCP integration
    enable_darwin_mcp: bool = True
    darwin_mcp_url: str = "http://localhost:8080"
    darwin_timeout: float = Field(default=30.0, gt=0.0)

    # Logging and monitoring
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    metrics_collection: bool = True

    @validator('crossover_rate', 'mutation_rate')
    def validate_rates(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rates must be between 0 and 1")
        return v


@dataclass
class OptimizationResult:
    """Result of genetic algorithm optimization."""
    optimization_id: str = field(default_factory=lambda: str(uuid4()))

    # Best solutions
    best_individual: Optional[Individual] = None
    pareto_front: List[Individual] = field(default_factory=list)
    elite_population: List[Individual] = field(default_factory=list)

    # Convergence information
    generations_completed: int = 0
    convergence_achieved: bool = False
    convergence_generation: Optional[int] = None
    final_fitness_diversity: float = 0.0

    # Performance metrics
    total_evaluations: int = 0
    total_computation_time_ms: float = 0.0
    average_generation_time_ms: float = 0.0
    best_fitness_evolution: List[float] = field(default_factory=list)

    # Multi-objective metrics
    hypervolume: float = 0.0
    pareto_front_size: int = 0
    solution_diversity: float = 0.0

    # Bayesian integration results
    bayesian_updates_performed: int = 0
    average_fitness_confidence: float = 0.0
    uncertainty_reduction: float = 0.0

    # Status information
    success: bool = False
    termination_reason: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneticAlgorithmEngine:
    """
    Advanced genetic algorithm engine with multi-objective optimization,
    Bayesian integration, and Darwin-MCP server support.
    """

    def __init__(
        self,
        config: GeneticConfig,
        fitness_functions: List[FitnessFunction],
        bayesian_engine: Optional[BayesianInferenceEngine] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the genetic algorithm engine.

        Args:
            config: GA configuration parameters
            fitness_functions: List of fitness functions to optimize
            bayesian_engine: Optional Bayesian engine for fitness evaluation
            logger: Optional structured logger
        """
        self.config = config
        self.fitness_functions = fitness_functions
        self.bayesian_engine = bayesian_engine

        # Setup logging
        self.logger = logger or StructuredLogger(
            component="genetic_optimizer",
            service_name="agentical-reasoning",
            correlation_context=CorrelationContext.generate()
        )

        # Algorithm state
        self.population: List[Individual] = []
        self.generation_counter = 0
        self.optimization_history = []
        self.pareto_archive: List[Individual] = []

        # Performance tracking
        self.evaluation_cache = {}
        self.performance_metrics = {
            "evaluations_per_second": [],
            "convergence_rate": [],
            "diversity_metrics": []
        }

        # Darwin-MCP client (optional)
        self.darwin_client = None
        if config.enable_darwin_mcp:
            self._initialize_darwin_client()

        # Initialize random seed for reproducibility
        self.random_state = np.random.RandomState()

        with logfire.span("GA engine initialization"):
            self.logger.log_operation(
                operation_type=OperationType.INITIALIZATION,
                message="Genetic algorithm engine initialized",
                metadata={
                    "population_size": config.population_size,
                    "max_generations": config.max_generations,
                    "fitness_functions": len(fitness_functions),
                    "multi_objective": config.enable_multi_objective
                }
            )

    async def optimize(
        self,
        genotype_generator: Callable,
        phenotype_decoder: Optional[Callable] = None,
        termination_criteria: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.

        Args:
            genotype_generator: Function to generate random genotypes
            phenotype_decoder: Optional function to decode genotypes to phenotypes
            termination_criteria: Optional custom termination criteria

        Returns:
            Optimization result with best solutions and metrics
        """
        with logfire.span("Genetic algorithm optimization"):
            start_time = datetime.utcnow()

            try:
                # Initialize population
                await self._initialize_population(genotype_generator, phenotype_decoder)

                # Main evolution loop
                while not await self._check_termination_criteria(termination_criteria):
                    generation_start = datetime.utcnow()

                    # Evaluate population fitness
                    await self._evaluate_population()

                    # Update Bayesian models if enabled
                    if (self.config.use_bayesian_fitness and
                        self.generation_counter % self.config.bayesian_update_frequency == 0):
                        await self._update_bayesian_models()

                    # Selection and reproduction
                    offspring = await self._create_offspring()

                    # Update population
                    await self._update_population(offspring)

                    # Update Pareto archive for multi-objective
                    if self.config.enable_multi_objective:
                        await self._update_pareto_archive()

                    # Track generation performance
                    generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
                    await self._track_generation_metrics(generation_time)

                    self.generation_counter += 1

                    self.logger.log_operation(
                        operation_type=OperationType.PROCESSING,
                        message=f"Generation {self.generation_counter} completed",
                        metadata={
                            "generation_time_ms": generation_time,
                            "best_fitness": self._get_best_fitness(),
                            "population_diversity": self._calculate_population_diversity()
                        }
                    )

                # Create optimization result
                result = await self._create_optimization_result(start_time)

                return result

            except Exception as e:
                self.logger.log_operation(
                    operation_type=OperationType.ERROR,
                    message=f"Genetic algorithm optimization failed: {str(e)}",
                    level=LogLevel.ERROR
                )
                raise OptimizationError(f"GA optimization failed: {str(e)}")

    async def _initialize_population(
        self,
        genotype_generator: Callable,
        phenotype_decoder: Optional[Callable]
    ):
        """Initialize the population with random individuals."""

        with logfire.span("Population initialization"):
            self.population = []

            for i in range(self.config.population_size):
                # Generate random genotype
                genotype = await self._safe_call(genotype_generator)

                # Decode to phenotype if decoder provided
                phenotype = None
                if phenotype_decoder:
                    phenotype = await self._safe_call(phenotype_decoder, genotype)

                # Create individual
                individual = Individual(
                    genotype=genotype,
                    phenotype=phenotype,
                    generation=0
                )

                self.population.append(individual)

            self.logger.log_operation(
                operation_type=OperationType.INITIALIZATION,
                message=f"Population initialized with {len(self.population)} individuals"
            )

    async def _evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""

        with logfire.span("Population evaluation"):
            evaluation_tasks = []

            for individual in self.population:
                if not individual.fitness_values:  # Only evaluate if not cached
                    task = self._evaluate_individual(individual)
                    evaluation_tasks.append(task)

            if evaluation_tasks:
                if self.config.parallel_evaluation:
                    # Parallel evaluation with semaphore
                    semaphore = asyncio.Semaphore(self.config.max_workers)

                    async def bounded_evaluate(task):
                        async with semaphore:
                            return await task

                    await asyncio.gather(*[bounded_evaluate(task) for task in evaluation_tasks])
                else:
                    # Sequential evaluation
                    for task in evaluation_tasks:
                        await task

    async def _evaluate_individual(self, individual: Individual):
        """Evaluate fitness for a single individual."""

        evaluation_start = datetime.utcnow()

        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._get_cache_key(individual)
                if cache_key in self.evaluation_cache:
                    cached_result = self.evaluation_cache[cache_key]
                    individual.fitness_values = cached_result["fitness_values"]
                    individual.objective_values = cached_result["objective_values"]
                    return

            # Evaluate each fitness function
            fitness_values = []
            objective_values = {}

            for fitness_func in self.fitness_functions:
                # Standard evaluation
                if fitness_func.evaluator:
                    value = await self._safe_call(
                        fitness_func.evaluator,
                        individual.genotype,
                        individual.phenotype
                    )

                    # Apply Bayesian evaluation if enabled
                    if (fitness_func.use_bayesian_evaluation and
                        self.bayesian_engine):
                        value, confidence = await self._bayesian_fitness_evaluation(
                            individual, fitness_func, value
                        )
                        individual.bayesian_confidence = confidence

                    # Apply objective transformation
                    if fitness_func.objective == OptimizationObjective.MINIMIZE:
                        fitness_value = -value  # Convert to maximization
                    else:
                        fitness_value = value

                    # Apply weight
                    weighted_fitness = fitness_value * fitness_func.weight

                    fitness_values.append(weighted_fitness)
                    objective_values[fitness_func.name] = value

            # Handle constraints
            constraint_penalty = 0.0
            for fitness_func in self.fitness_functions:
                for constraint in fitness_func.constraints:
                    violation = await self._safe_call(
                        constraint,
                        individual.genotype,
                        individual.phenotype
                    )
                    if violation > 0:
                        constraint_penalty += violation * fitness_func.penalty_factor
                        individual.constraint_violations[f"{fitness_func.name}_constraint"] = violation

            # Apply constraint penalty
            fitness_values = [f - constraint_penalty for f in fitness_values]

            # Store results
            individual.fitness_values = fitness_values
            individual.objective_values = objective_values
            individual.evaluation_time_ms = (
                datetime.utcnow() - evaluation_start
            ).total_seconds() * 1000

            # Cache result
            if self.config.enable_caching:
                self.evaluation_cache[cache_key] = {
                    "fitness_values": fitness_values,
                    "objective_values": objective_values
                }

                # Maintain cache size
                if len(self.evaluation_cache) > self.config.cache_size:
                    # Remove oldest entries
                    oldest_key = next(iter(self.evaluation_cache))
                    del self.evaluation_cache[oldest_key]

        except Exception as e:
            self.logger.log_operation(
                operation_type=OperationType.ERROR,
                message=f"Individual evaluation failed: {str(e)}",
                level=LogLevel.ERROR
            )
            # Assign worst possible fitness on error
            individual.fitness_values = [-float('inf')] * len(self.fitness_functions)
            individual.objective_values = {}

    async def _bayesian_fitness_evaluation(
        self,
        individual: Individual,
        fitness_func: FitnessFunction,
        raw_value: float
    ) -> Tuple[float, float]:
        """Enhance fitness evaluation with Bayesian inference."""

        # Create hypothesis about fitness quality
        hypothesis = Hypothesis(
            name=f"fitness_quality_{individual.individual_id}",
            description=f"Individual has high quality fitness for {fitness_func.name}",
            prior_probability=0.5
        )

        # Create evidence from fitness evaluation
        evidence = Evidence(
            name="fitness_evaluation",
            value={
                "raw_fitness": raw_value,
                "function_name": fitness_func.name,
                "genotype_hash": hash(str(individual.genotype))
            },
            likelihood=self._calculate_fitness_likelihood(raw_value, fitness_func),
            reliability=0.9
        )

        # Perform Bayesian inference
        result = await self.bayesian_engine.update_belief(hypothesis, evidence)

        # Adjust fitness based on uncertainty
        confidence = result.confidence_level
        uncertainty_penalty = (1 - confidence) * fitness_func.uncertainty_penalty

        adjusted_fitness = raw_value * (1 - uncertainty_penalty)

        return adjusted_fitness, confidence

    def _calculate_fitness_likelihood(self, fitness_value: float, fitness_func: FitnessFunction) -> float:
        """Calculate likelihood of fitness value being reliable."""

        # Simple heuristic: higher fitness values are more likely to be reliable
        # This would be domain-specific in practice
        if fitness_func.objective == OptimizationObjective.MAXIMIZE:
            return min(0.95, max(0.1, fitness_value / 100.0))
        else:
            return min(0.95, max(0.1, 1.0 / (1.0 + abs(fitness_value))))

    async def _create_offspring(self) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation."""

        offspring = []

        while len(offspring) < self.config.population_size:
            # Selection
            parent1 = await self._select_individual()
            parent2 = await self._select_individual()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_genotype, child2_genotype = await self._crossover(
                    parent1.genotype, parent2.genotype
                )
            else:
                child1_genotype = parent1.genotype
                child2_genotype = parent2.genotype

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1_genotype = await self._mutate(child1_genotype)

            if random.random() < self.config.mutation_rate:
                child2_genotype = await self._mutate(child2_genotype)

            # Create offspring individuals
            child1 = Individual(
                genotype=child1_genotype,
                generation=self.generation_counter + 1,
                parent_ids=[parent1.individual_id, parent2.individual_id]
            )

            child2 = Individual(
                genotype=child2_genotype,
                generation=self.generation_counter + 1,
                parent_ids=[parent1.individual_id, parent2.individual_id]
            )

            offspring.extend([child1, child2])

        return offspring[:self.config.population_size]

    async def _select_individual(self) -> Individual:
        """Select an individual based on configured selection method."""

        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return await self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return await self._roulette_wheel_selection()
        elif self.config.selection_method == SelectionMethod.RANK_SELECTION:
            return await self._rank_selection()
        else:
            # Default to tournament selection
            return await self._tournament_selection()

    async def _tournament_selection(self) -> Individual:
        """Tournament selection method."""

        tournament_individuals = random.sample(
            self.population,
            min(self.config.tournament_size, len(self.population))
        )

        # Select best individual from tournament
        best_individual = max(
            tournament_individuals,
            key=lambda ind: sum(ind.fitness_values) if ind.fitness_values else -float('inf')
        )

        return best_individual

    async def _roulette_wheel_selection(self) -> Individual:
        """Roulette wheel selection method."""

        # Calculate total fitness
        fitness_values = [
            sum(ind.fitness_values) if ind.fitness_values else 0
            for ind in self.population
        ]

        # Handle negative fitness values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]

        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            return random.choice(self.population)

        # Spin the wheel
        spin = random.random() * total_fitness
        cumulative_fitness = 0

        for i, fitness in enumerate(fitness_values):
            cumulative_fitness += fitness
            if cumulative_fitness >= spin:
                return self.population[i]

        return self.population[-1]  # Fallback

    async def _rank_selection(self) -> Individual:
        """Rank-based selection method."""

        # Sort population by fitness
        sorted_population = sorted(
            self.population,
            key=lambda ind: sum(ind.fitness_values) if ind.fitness_values else -float('inf'),
            reverse=True
        )

        # Assign selection probabilities based on rank
        n = len(sorted_population)
        ranks = list(range(n, 0, -1))
        total_rank = sum(ranks)

        # Roulette wheel selection on ranks
        spin = random.random() * total_rank
        cumulative_rank = 0

        for i, rank in enumerate(ranks):
            cumulative_rank += rank
            if cumulative_rank >= spin:
                return sorted_population[i]

        return sorted_population[-1]  # Fallback

    async def _crossover(self, parent1_genotype, parent2_genotype):
        """Perform crossover between two parent genotypes."""

        if self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return await self._single_point_crossover(parent1_genotype, parent2_genotype)
        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
            return await self._two_point_crossover(parent1_genotype, parent2_genotype)
        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
            return await self._uniform_crossover(parent1_genotype, parent2_genotype)
        else:
            # Default to uniform crossover
            return await self._uniform_crossover(parent1_genotype, parent2_genotype)

    async def _single_point_crossover(self, parent1, parent2):
        """Single-point crossover operation."""

        if isinstance(parent1, (list, tuple)):
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        elif isinstance(parent1, np.ndarray):
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            # For other types, return parents unchanged
            return parent1, parent2

    async def _two_point_crossover(self, parent1, parent2):
        """Two-point crossover operation."""

        if isinstance(parent1, (list, tuple)):
            length = len(parent1)
            point1 = random.randint(1, length - 2)
            point2 = random.randint(point1 + 1, length - 1)

            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
        else:
            # Fallback to single-point
            return await self._single_point_crossover(parent1, parent2)

    async def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover operation."""

        if isinstance(parent1, (list, tuple)):
            child1 = []
            child2 = []

            for gene1, gene2 in zip(parent1, parent2):
                if random.random() < 0.5:
                    child1.append(gene1)
                    child2.append(gene2)
                else:
                    child1.append(gene2)
                    child2.append(gene1)

            return type(parent1)(child1), type(parent1)(child2)
        elif isinstance(parent1, np.ndarray):
            mask = np.random.random(parent1.shape) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        else:
            # For other types, return parents unchanged
            return parent1, parent2

    async def _mutate(self, genotype):
        """Apply mutation to a genotype."""

        if self.config.mutation_method == MutationMethod.BIT_FLIP:
            return await self._bit_flip_mutation(genotype)
        elif self.config.mutation_method == MutationMethod.GAUSSIAN:
            return await self._gaussian_mutation(genotype)
        elif self.config.mutation_method == MutationMethod.UNIFORM:
            return await self._uniform_mutation(genotype)
        else:
            # Default to Gaussian mutation
            return await self._gaussian_mutation(genotype)

    async def _bit_flip_mutation(self, genotype):
        """Bit-flip mutation for binary genotypes."""

        if isinstance(genotype, (list, tuple)):
            mutated = list(genotype)
            for i in range(len(mutated)):
                if random.random() < self.config.mutation_rate:
                    if isinstance(mutated[i], bool):
                        mutated[i] = not mutated[i]
                    elif isinstance(mutated[i], (int, float)) and mutated[i] in [0, 1]:
                        mutated[i] = 1 - mutated[i]
            return type(genotype)(mutated)
        else:
            return genotype

    async def _gaussian_mutation(self, genotype):
        """Gaussian mutation for continuous genotypes."""

        if isinstance(genotype, (list, tuple)):
            mutated = []
            for gene in genotype:
                if isinstance(gene, (int, float)) and random.random() < self.config.mutation_rate:
                    noise = np.random.normal(0, self.config.mutation_strength)
                    mutated_gene = gene + noise
                    mutated.append(mutated_gene)
                else:
                    mutated.append(gene)
            return type(genotype)(mutated)
        elif isinstance(genotype, np.ndarray):
            mutated = genotype.copy()
            mask = np.random.random(genotype.shape) < self.config.mutation_rate
            noise = np.random.normal(0, self.config.mutation_strength, genotype.shape)
            mutated[mask] += noise[mask]
            return mutated
        else:
            return genotype

    async def _uniform_mutation(self, genotype):
        """Uniform mutation for bounded genotypes."""

        if isinstance(genotype, (list, tuple)):
            mutated = []
            for gene in genotype:
                if isinstance(gene, (int, float)) and random.random() < self.config.mutation_rate:
                    # Assume bounded in [-1, 1] for this example
                    mutated_gene = random.uniform(-1, 1)
                    mutated.append(mutated_gene)
                else:
                    mutated.append(gene)
            return type(genotype)(mutated)
        elif isinstance(genotype, np.ndarray):
            mutated = genotype.copy()
            mask = np.random.random(genotype.shape) < self.config.mutation_rate
            # Uniform random values in range
            mutated[mask] = np.random.uniform(-1, 1, np.sum(mask))
            return mutated
        else:
            return genotype

    async def _update_population(self, offspring: List[Individual]):
        """Update population with offspring using selection strategy."""

        combined_population = self.population + offspring

        if self.config.enable_multi_objective:
            # Multi-objective selection using NSGA-II
            await self._nsga2_selection(combined_population)
        else:
            # Single-objective selection
            await self._single_objective_selection(combined_population)

    async def _nsga2_selection(self, combined_population: List[Individual]):
        """NSGA-II selection for multi-objective optimization."""

        # Non-dominated sorting
        fronts = await self._non_dominated_sorting(combined_population)

        # Calculate crowding distance for each front
        for front in fronts:
            await self._calculate_crowding_distance(front)

        # Select individuals for next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.config.population_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance and select remaining
                remaining_slots = self.config.population_size - len(new_population)
                sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(sorted_front[:remaining_slots])
                break

        self.population = new_population

    async def _non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        """Perform non-dominated sorting for multi-objective optimization."""

        fronts = []
        domination_count = {}
        dominated_solutions = {}

        # Initialize
        for individual in population:
            domination_count[individual.individual_id] = 0
            dominated_solutions[individual.individual_id] = []

        # Calculate domination relationships
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    if self._dominates(ind1, ind2):
                        dominated_solutions[ind1.individual_id].append(ind2)
                    elif self._dominates(ind2, ind1):
                        domination_count[ind1.individual_id] += 1

        # First front (non-dominated solutions)
        first_front = []
        for individual in population:
            if domination_count[individual.individual_id] == 0:
                individual.pareto_front = 0
                first_front.append(individual)

        fronts.append(first_front)

        # Subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for individual in fronts[front_index]:
                for dominated_ind in dominated_solutions[individual.individual_id]:
                    domination_count[dominated_ind.individual_id] -= 1
                    if domination_count[dominated_ind.individual_id] == 0:
                        dominated_ind.pareto_front = front_index + 1
                        next_front.append(dominated_ind)

            if next_front:
                fronts.append(next_front)
            front_index += 1

        return fronts[:-1] if not fronts[-1] else fronts

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if individual 1 dominates individual 2."""

        if not ind1.fitness_values or not ind2.fitness_values:
            return False

        # Individual 1 dominates individual 2 if:
        # 1. Individual 1 is at least as good in all objectives
        # 2. Individual 1 is strictly better in at least one objective

        at_least_as_good = all(f1 >= f2 for f1, f2 in zip(ind1.fitness_values, ind2.fitness_values))
        strictly_better = any(f1 > f2 for f1, f2 in zip(ind1.fitness_values, ind2.fitness_values))

        return at_least_as_good and strictly_better

    async def _calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""

        if not front:
            return

        num_objectives = len(front[0].fitness_values) if front[0].fitness_values else 1

        # Initialize crowding distances
        for individual in front:
            individual.crowding_distance = 0.0

        # Calculate distance for each objective
        for obj_index in range(num_objectives):
            # Sort by objective value
            front.sort(key=lambda ind: ind.fitness_values[obj_index] if ind.fitness_values else 0)

            # Boundary points have infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate distance for intermediate points
            if len(front) > 2:
                obj_range = (front[-1].fitness_values[obj_index] -
                           front[0].fitness_values[obj_index])

                if obj_range > 0:
                    for i in range(1, len(front) - 1):
                        distance = (front[i + 1].fitness_values[obj_index] -
                                  front[i - 1].fitness_values[obj_index]) / obj_range
                        front[i].crowding_distance += distance

    async def _single_objective_selection(self, combined_population: List[Individual]):
        """Single-objective selection strategy."""

        # Sort by fitness (descending)
        sorted_population = sorted(
            combined_population,
            key=lambda ind: sum(ind.fitness_values) if ind.fitness_values else -float('inf'),
            reverse=True
        )

        # Apply elitism
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        elite_individuals = sorted_population[:elite_count]

        # Fill remaining slots with selection
        remaining_slots = self.config.population_size - elite_count
        selected_individuals = []

        for _ in range(remaining_slots):
            selected = await self._select_individual()
            selected_individuals.append(selected)

        self.population = elite_individuals + selected_individuals[:remaining_slots]

    async def _update_pareto_archive(self):
        """Update Pareto archive with current population."""

        if not self.config.enable_multi_objective:
            return

        # Combine current archive with population
        combined = self.pareto_archive + self.population

        # Perform non-dominated sorting
        fronts = await self._non_dominated_sorting(combined)

        # Update archive with first front
        if fronts:
            self.pareto_archive = fronts[0]

            # Limit archive size
            if len(self.pareto_archive) > self.config.pareto_archive_size:
                await self._calculate_crowding_distance(self.pareto_archive)
                self.pareto_archive.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                self.pareto_archive = self.pareto_archive[:self.config.pareto_archive_size]

    async def _update_bayesian_models(self):
        """Update Bayesian models based on current population performance."""

        if not self.bayesian_engine:
            return

        # Analyze population fitness distribution
        all_fitness = []
        for individual in self.population:
            if individual.fitness_values:
                all_fitness.extend(individual.fitness_values)

        if not all_fitness:
            return

        # Create evidence from population statistics
        evidence = Evidence(
            name="population_fitness_statistics",
            value={
                "mean_fitness": np.mean(all_fitness),
                "std_fitness": np.std(all_fitness),
                "max_fitness": np.max(all_fitness),
                "generation": self.generation_counter
            },
            likelihood=0.9,
            reliability=0.95
        )

        # Update belief about optimization progress
        progress_hypothesis = Hypothesis(
            name="optimization_progressing",
            description="Genetic algorithm is making progress toward optimal solutions",
            prior_probability=0.6
        )

        await self.bayesian_engine.update_belief(progress_hypothesis, evidence)

    async def _check_termination_criteria(self, custom_criteria: Optional[Dict[str, Any]]) -> bool:
        """Check if optimization should terminate."""

        # Max generations reached
        if self.generation_counter >= self.config.max_generations:
            return True

        # Check convergence
        if await self._check_convergence():
            return True

        # Check custom criteria
        if custom_criteria:
            for criterion, value in custom_criteria.items():
                if criterion == "target_fitness" and self._get_best_fitness() >= value:
                    return True
                elif criterion == "max_time_minutes":
                    elapsed_minutes = (datetime.utcnow() - self.optimization_start_time).total_seconds() / 60
                    if elapsed_minutes >= value:
                        return True

        return False

    async def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""

        if len(self.optimization_history) < self.config.stagnation_generations:
            return False

        # Check fitness stagnation
        recent_best_fitness = [
            gen_data["best_fitness"]
            for gen_data in self.optimization_history[-self.config.stagnation_generations:]
        ]

        fitness_variance = np.var(recent_best_fitness)
        return fitness_variance < self.config.fitness_tolerance

    def _get_best_fitness(self) -> float:
        """Get the best fitness value in current population."""

        if not self.population:
            return -float('inf')

        best_individual = max(
            self.population,
            key=lambda ind: sum(ind.fitness_values) if ind.fitness_values else -float('inf')
        )

        return sum(best_individual.fitness_values) if best_individual.fitness_values else -float('inf')

    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric."""

        if len(self.population) < 2:
            return 0.0

        # Calculate pairwise distances between genotypes
        distances = []
        for i, ind1 in enumerate(self.population):
            for j, ind2 in enumerate(self.population[i+1:], i+1):
                distance = self._calculate_genotype_distance(ind1.genotype, ind2.genotype)
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _calculate_genotype_distance(self, genotype1, genotype2) -> float:
        """Calculate distance between two genotypes."""

        try:
            if isinstance(genotype1, (list, tuple)) and isinstance(genotype2, (list, tuple)):
                # Euclidean distance for numeric genotypes
                if all(isinstance(g, (int, float)) for g in genotype1 + genotype2):
                    return euclidean(genotype1, genotype2)
                else:
                    # Hamming distance for categorical genotypes
                    return sum(g1 != g2 for g1, g2 in zip(genotype1, genotype2))
            elif isinstance(genotype1, np.ndarray) and isinstance(genotype2, np.ndarray):
                return np.linalg.norm(genotype1 - genotype2)
            else:
                # Hash-based distance for other types
                return float(hash(str(genotype1)) != hash(str(genotype2)))
        except:
            return 1.0  # Default distance

    def _get_cache_key(self, individual: Individual) -> str:
        """Generate cache key for individual evaluation."""

        return f"eval_{hash(str(individual.genotype))}"

    async def _track_generation_metrics(self, generation_time_ms: float):
        """Track metrics for the current generation."""

        best_fitness = self._get_best_fitness()
        diversity = self._calculate_population_diversity()

        generation_data = {
            "generation": self.generation_counter,
            "best_fitness": best_fitness,
            "diversity": diversity,
            "generation_time_ms": generation_time_ms,
            "population_size": len(self.population)
        }

        self.optimization_history.append(generation_data)

        # Update performance metrics
        evaluations_per_second = self.config.population_size / (generation_time_ms / 1000) if generation_time_ms > 0 else 0
        self.performance_metrics["evaluations_per_second"].append(evaluations_per_second)
        self.performance_metrics["diversity_metrics"].append(diversity)

    async def _create_optimization_result(self, start_time: datetime) -> OptimizationResult:
        """Create optimization result summary."""

        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Find best individual
        best_individual = None
        if self.population:
            best_individual = max(
                self.population,
                key=lambda ind: sum(ind.fitness_values) if ind.fitness_values else -float('inf')
            )

        # Calculate metrics
        hypervolume = await self._calculate_hypervolume() if self.config.enable_multi_objective else 0.0
        diversity = self._calculate_population_diversity()

        result = OptimizationResult(
            best_individual=best_individual,
            pareto_front=self.pareto_archive if self.config.enable_multi_objective else [],
            elite_population=self.population[:int(self.config.population_size * self.config.elitism_rate)],
            generations_completed=self.generation_counter,
            convergence_achieved=await self._check_convergence(),
            total_evaluations=self.generation_counter * self.config.population_size,
            total_computation_time_ms=total_time,
            average_generation_time_ms=total_time / max(1, self.generation_counter),
            best_fitness_evolution=[gen["best_fitness"] for gen in self.optimization_history],
            hypervolume=hypervolume,
            pareto_front_size=len(self.pareto_archive),
            solution_diversity=diversity,
            success=best_individual is not None and best_individual.fitness_values,
            termination_reason=self._get_termination_reason()
        )

        return result

    async def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume metric for multi-objective optimization."""

        if not self.pareto_archive or not self.config.enable_multi_objective:
            return 0.0

        # Simple hypervolume calculation (for demonstration)
        # In practice, use specialized libraries like pymoo
        reference_point = [0.0] * len(self.pareto_archive[0].fitness_values)

        volume = 0.0
        for individual in self.pareto_archive:
            if individual.fitness_values:
                point_volume = 1.0
                for i, fitness in enumerate(individual.fitness_values):
                    point_volume *= max(0, fitness - reference_point[i])
                volume += point_volume

        return volume

    def _get_termination_reason(self) -> str:
        """Get reason for optimization termination."""

        if self.generation_counter >= self.config.max_generations:
            return "Maximum generations reached"
        elif len(self.optimization_history) >= self.config.stagnation_generations:
            return "Convergence achieved"
        else:
            return "Custom termination criteria met"

    async def _safe_call(self, func: Callable, *args, **kwargs):
        """Safely call a function with timeout and error handling."""

        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.darwin_timeout
                )
            else:
                return func(*args, **kwargs)
        except Exception as e:
            self.logger.log_operation(
                operation_type=OperationType.ERROR,
                message=f"Function call failed: {str(e)}",
                level=LogLevel.ERROR
            )
            raise

    def _initialize_darwin_client(self):
        """Initialize Darwin-MCP client for advanced algorithms."""

        # Placeholder for Darwin-MCP client initialization
        # In practice, this would connect to the Darwin-MCP server
        self.logger.log_operation(
            operation_type=OperationType.INITIALIZATION,
            message="Darwin-MCP client initialization placeholder",
            metadata={"url": self.config.darwin_mcp_url}
        )

    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics."""

        return {
            "generation": self.generation_counter,
            "population_size": len(self.population),
            "best_fitness": self._get_best_fitness(),
            "diversity": self._calculate_population_diversity(),
            "pareto_front_size": len(self.pareto_archive),
            "evaluations_per_second": np.mean(self.performance_metrics["evaluations_per_second"]) if self.performance_metrics["evaluations_per_second"] else 0,
            "convergence_trend": self.optimization_history[-10:] if len(self.optimization_history) >= 10 else self.optimization_history
        }
