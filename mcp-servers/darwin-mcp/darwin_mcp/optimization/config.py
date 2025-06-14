"""
Darwin Optimization Configuration

This module provides configuration classes for Darwin genetic algorithm optimization.
The configuration system manages algorithm parameters, execution settings, and
optimization behavior.

Example usage:
    from darwin_mcp.optimization.config import OptimizationConfig
    
    config = OptimizationConfig(
        algorithm="nsga2",
        population_size=100,
        max_generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class AlgorithmType(Enum):
    """Supported genetic algorithm types."""
    GENETIC = "genetic"
    NSGA2 = "nsga2"
    NSGA3 = "nsga3"
    MOEAD = "moead"
    SPEA2 = "spea2"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class SelectionMethod(Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    RANDOM = "random"
    ELITIST = "elitist"


class CrossoverMethod(Enum):
    """Crossover methods for genetic algorithms."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    SIMULATED_BINARY = "simulated_binary"


class MutationMethod(Enum):
    """Mutation methods for genetic algorithms."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    BIT_FLIP = "bit_flip"
    SWAP = "swap"


@dataclass
class OptimizationConfig:
    """Configuration for genetic algorithm optimization.
    
    Attributes:
        algorithm: Algorithm type to use
        population_size: Number of individuals in population
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        selection_method: Selection method
        crossover_method: Crossover method
        mutation_method: Mutation method
        elitism: Number of elite individuals to preserve
        tournament_size: Size of tournament for tournament selection
        early_stopping: Enable early stopping
        convergence_threshold: Threshold for convergence detection
        convergence_generations: Generations to check for convergence
        parallel_execution: Enable parallel execution
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_seed: Random seed for reproducibility
        verbose: Enable verbose output
        callback_frequency: Frequency of progress callbacks
        save_history: Save evolution history
        adaptive_parameters: Enable adaptive parameter adjustment
    """
    
    # Core algorithm settings
    algorithm: Union[str, AlgorithmType] = AlgorithmType.GENETIC
    population_size: int = 100
    max_generations: int = 200
    
    # Genetic operator rates
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Selection and variation methods
    selection_method: Union[str, SelectionMethod] = SelectionMethod.TOURNAMENT
    crossover_method: Union[str, CrossoverMethod] = CrossoverMethod.SINGLE_POINT
    mutation_method: Union[str, MutationMethod] = MutationMethod.GAUSSIAN
    
    # Elite preservation
    elitism: int = 1
    tournament_size: int = 3
    
    # Convergence and stopping criteria
    early_stopping: bool = True
    convergence_threshold: float = 1e-6
    convergence_generations: int = 20
    max_evaluations: Optional[int] = None
    max_time_seconds: Optional[float] = None
    
    # Execution settings
    parallel_execution: bool = False
    n_jobs: int = 1
    random_seed: Optional[int] = None
    
    # Output and monitoring
    verbose: bool = True
    callback_frequency: int = 1
    save_history: bool = True
    
    # Advanced settings
    adaptive_parameters: bool = False
    constraint_handling: str = "penalty"
    penalty_weight: float = 1000.0
    
    # Multi-objective specific settings
    reference_point: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    
    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string enums to enum types
        if isinstance(self.algorithm, str):
            self.algorithm = AlgorithmType(self.algorithm.lower())
        
        if isinstance(self.selection_method, str):
            self.selection_method = SelectionMethod(self.selection_method.lower())
        
        if isinstance(self.crossover_method, str):
            self.crossover_method = CrossoverMethod(self.crossover_method.lower())
        
        if isinstance(self.mutation_method, str):
            self.mutation_method = MutationMethod(self.mutation_method.lower())
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        
        if self.max_generations <= 0:
            raise ValueError("Max generations must be positive")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        
        if self.elitism < 0:
            raise ValueError("Elitism must be non-negative")
        
        if self.elitism >= self.population_size:
            raise ValueError("Elitism must be less than population size")
        
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        
        if self.tournament_size > self.population_size:
            raise ValueError("Tournament size cannot exceed population size")
        
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        
        if self.convergence_generations <= 0:
            raise ValueError("Convergence generations must be positive")
        
        if self.callback_frequency <= 0:
            raise ValueError("Callback frequency must be positive")
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name as string.
        
        Returns:
            Algorithm name
        """
        return self.algorithm.value
    
    def is_multi_objective_algorithm(self) -> bool:
        """Check if algorithm supports multi-objective optimization.
        
        Returns:
            True if multi-objective algorithm
        """
        return self.algorithm in [
            AlgorithmType.NSGA2,
            AlgorithmType.NSGA3,
            AlgorithmType.MOEAD,
            AlgorithmType.SPEA2
        ]
    
    def supports_constraints(self) -> bool:
        """Check if algorithm supports constraints.
        
        Returns:
            True if constraints are supported
        """
        # Most algorithms support constraints through penalty methods
        return True
    
    def get_recommended_population_size(self, problem_dimension: int) -> int:
        """Get recommended population size for problem dimension.
        
        Args:
            problem_dimension: Number of decision variables
            
        Returns:
            Recommended population size
        """
        if self.is_multi_objective_algorithm():
            return max(100, problem_dimension * 10)
        else:
            return max(50, problem_dimension * 5)
    
    def adjust_for_problem(self, problem_dimension: int, is_multi_objective: bool = False):
        """Adjust configuration for specific problem characteristics.
        
        Args:
            problem_dimension: Number of decision variables
            is_multi_objective: Whether problem has multiple objectives
        """
        # Adjust population size if not explicitly set
        if self.population_size == 100:  # Default value
            self.population_size = self.get_recommended_population_size(problem_dimension)
        
        # Adjust algorithm for multi-objective problems
        if is_multi_objective and self.algorithm == AlgorithmType.GENETIC:
            self.algorithm = AlgorithmType.NSGA2
        
        # Adjust generations for problem complexity
        if problem_dimension > 50 and self.max_generations == 200:  # Default value
            self.max_generations = min(500, problem_dimension * 10)
    
    def enable_parallel(self, n_jobs: int = -1):
        """Enable parallel execution.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.parallel_execution = True
        self.n_jobs = n_jobs
    
    def enable_adaptive_parameters(self):
        """Enable adaptive parameter adjustment."""
        self.adaptive_parameters = True
    
    def set_convergence_criteria(self, threshold: float, generations: int):
        """Set convergence criteria.
        
        Args:
            threshold: Convergence threshold
            generations: Generations to check for convergence
        """
        self.convergence_threshold = threshold
        self.convergence_generations = generations
        self.early_stopping = True
    
    def set_time_limit(self, max_time_seconds: float):
        """Set maximum execution time.
        
        Args:
            max_time_seconds: Maximum time in seconds
        """
        self.max_time_seconds = max_time_seconds
    
    def set_evaluation_limit(self, max_evaluations: int):
        """Set maximum number of function evaluations.
        
        Args:
            max_evaluations: Maximum evaluations
        """
        self.max_evaluations = max_evaluations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "algorithm": self.algorithm.value,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "selection_method": self.selection_method.value,
            "crossover_method": self.crossover_method.value,
            "mutation_method": self.mutation_method.value,
            "elitism": self.elitism,
            "tournament_size": self.tournament_size,
            "early_stopping": self.early_stopping,
            "convergence_threshold": self.convergence_threshold,
            "convergence_generations": self.convergence_generations,
            "max_evaluations": self.max_evaluations,
            "max_time_seconds": self.max_time_seconds,
            "parallel_execution": self.parallel_execution,
            "n_jobs": self.n_jobs,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "callback_frequency": self.callback_frequency,
            "save_history": self.save_history,
            "adaptive_parameters": self.adaptive_parameters,
            "constraint_handling": self.constraint_handling,
            "penalty_weight": self.penalty_weight,
            "reference_point": self.reference_point,
            "weights": self.weights,
            "algorithm_params": self.algorithm_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            OptimizationConfig instance
        """
        # Remove None values
        filtered_data = {k: v for k, v in data.items() if v is not None}
        return cls(**filtered_data)
    
    def copy(self) -> "OptimizationConfig":
        """Create a copy of the configuration.
        
        Returns:
            New OptimizationConfig instance
        """
        return self.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (f"OptimizationConfig(algorithm={self.algorithm.value}, "
                f"population_size={self.population_size}, "
                f"max_generations={self.max_generations})")


# Predefined configurations for common scenarios
class PresetConfigs:
    """Predefined optimization configurations."""
    
    @staticmethod
    def fast_convergence() -> OptimizationConfig:
        """Configuration for fast convergence on simple problems."""
        return OptimizationConfig(
            population_size=50,
            max_generations=100,
            mutation_rate=0.2,
            crossover_rate=0.9,
            early_stopping=True,
            convergence_threshold=1e-4,
            convergence_generations=10
        )
    
    @staticmethod
    def thorough_search() -> OptimizationConfig:
        """Configuration for thorough search on complex problems."""
        return OptimizationConfig(
            population_size=200,
            max_generations=500,
            mutation_rate=0.05,
            crossover_rate=0.7,
            early_stopping=True,
            convergence_threshold=1e-8,
            convergence_generations=50
        )
    
    @staticmethod
    def multi_objective() -> OptimizationConfig:
        """Configuration for multi-objective optimization."""
        return OptimizationConfig(
            algorithm=AlgorithmType.NSGA2,
            population_size=100,
            max_generations=300,
            mutation_rate=0.1,
            crossover_rate=0.8,
            early_stopping=False
        )
    
    @staticmethod
    def high_dimensional() -> OptimizationConfig:
        """Configuration for high-dimensional problems."""
        return OptimizationConfig(
            population_size=300,
            max_generations=1000,
            mutation_rate=0.15,
            crossover_rate=0.8,
            parallel_execution=True,
            n_jobs=-1,
            adaptive_parameters=True
        )
    
    @staticmethod
    def constrained_optimization() -> OptimizationConfig:
        """Configuration for constrained optimization."""
        return OptimizationConfig(
            population_size=150,
            max_generations=400,
            mutation_rate=0.1,
            crossover_rate=0.8,
            constraint_handling="penalty",
            penalty_weight=10000.0
        )