"""
Darwin Optimization Results

This module provides classes for storing and analyzing optimization results
from Darwin genetic algorithm runs. It includes result data structures,
performance metrics, and analysis utilities.

Example usage:
    from darwin_mcp.optimization.results import OptimizationResult
    
    result = OptimizationResult(
        optimization_id="opt_123",
        problem_name="Portfolio Optimization",
        best_solution=[0.6, 0.4],
        best_fitness=0.123
    )
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete optimization result data structure.
    
    Attributes:
        optimization_id: Unique optimization identifier
        problem_name: Name of the optimization problem
        algorithm_name: Name of the algorithm used
        best_solution: Best solution found
        best_fitness: Best fitness value achieved
        generations_run: Number of generations executed
        execution_time: Total execution time in seconds
        convergence_generation: Generation where convergence occurred
        final_population: Final population of solutions
        evolution_history: Evolution history data
        algorithm_config: Algorithm configuration used
        metadata: Additional metadata
        pareto_frontier: Pareto frontier for multi-objective problems
        constraint_violations: Constraint violation summary
        performance_metrics: Performance analysis metrics
    """
    
    # Core result data
    optimization_id: Optional[str] = None
    problem_name: str = ""
    algorithm_name: str = ""
    best_solution: List[float] = field(default_factory=list)
    best_fitness: Union[float, List[float]] = 0.0
    
    # Execution metrics
    generations_run: int = 0
    execution_time: float = 0.0
    convergence_generation: Optional[int] = None
    total_evaluations: int = 0
    
    # Population data
    final_population: List[List[float]] = field(default_factory=list)
    final_fitness_values: List[Union[float, List[float]]] = field(default_factory=list)
    
    # Evolution tracking
    evolution_history: Dict[str, List[Any]] = field(default_factory=dict)
    convergence_curve: List[float] = field(default_factory=list)
    diversity_curve: List[float] = field(default_factory=list)
    
    # Multi-objective results
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    pareto_front_size: int = 0
    hypervolume: Optional[float] = None
    
    # Constraint handling
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    feasible_solutions_count: int = 0
    
    # Configuration and metadata
    algorithm_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance analysis
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def __post_init__(self):
        """Process results after initialization."""
        self._calculate_performance_metrics()
        self._process_evolution_history()
        
        # Set timestamps if not provided
        if not self.end_time:
            self.end_time = datetime.now(timezone.utc).isoformat()
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from results."""
        if not self.performance_metrics:
            self.performance_metrics = {}
        
        # Basic metrics
        if self.generations_run > 0:
            self.performance_metrics["evaluations_per_second"] = (
                self.total_evaluations / max(self.execution_time, 0.001)
            )
            self.performance_metrics["time_per_generation"] = (
                self.execution_time / self.generations_run
            )
        
        # Convergence metrics
        if self.convergence_generation is not None:
            self.performance_metrics["convergence_efficiency"] = (
                self.convergence_generation / max(self.generations_run, 1)
            )
        
        # Population diversity
        if self.final_population:
            self.performance_metrics["final_diversity"] = self._calculate_diversity(
                self.final_population
            )
        
        # Success metrics
        self.performance_metrics["optimization_success"] = (
            self.convergence_generation is not None or 
            len(self.constraint_violations) == 0
        )
    
    def _process_evolution_history(self):
        """Process evolution history data."""
        if not self.evolution_history:
            self.evolution_history = {
                "generations": list(range(self.generations_run)),
                "best_fitness": [],
                "average_fitness": [],
                "diversity": []
            }
    
    def _calculate_diversity(self, population: List[List[float]]) -> float:
        """Calculate population diversity.
        
        Args:
            population: Population of solutions
            
        Returns:
            Diversity metric (0-1, higher is more diverse)
        """
        if len(population) < 2:
            return 0.0
        
        try:
            pop_array = np.array(population)
            distances = []
            
            for i in range(len(pop_array)):
                for j in range(i + 1, len(pop_array)):
                    dist = np.linalg.norm(pop_array[i] - pop_array[j])
                    distances.append(dist)
            
            if distances:
                return float(np.mean(distances))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating diversity: {e}")
            return 0.0
    
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective result.
        
        Returns:
            True if multi-objective, False otherwise
        """
        return isinstance(self.best_fitness, list) or len(self.pareto_frontier) > 0
    
    def is_constrained(self) -> bool:
        """Check if this problem had constraints.
        
        Returns:
            True if constrained, False otherwise
        """
        return len(self.constraint_violations) > 0
    
    def is_converged(self) -> bool:
        """Check if optimization converged.
        
        Returns:
            True if converged, False otherwise
        """
        return self.convergence_generation is not None
    
    def is_feasible(self) -> bool:
        """Check if best solution is feasible.
        
        Returns:
            True if feasible, False otherwise
        """
        if not self.constraint_violations:
            return True
        
        return all(violation <= 1e-6 for violation in self.constraint_violations.values())
    
    def get_success_rate(self) -> float:
        """Get optimization success rate.
        
        Returns:
            Success rate (0-1)
        """
        if not self.is_constrained():
            return 1.0 if self.is_converged() else 0.8
        
        return float(self.feasible_solutions_count) / max(len(self.final_population), 1)
    
    def get_convergence_rate(self) -> float:
        """Get convergence rate.
        
        Returns:
            Convergence rate (generations per fitness improvement)
        """
        if not self.convergence_curve or len(self.convergence_curve) < 2:
            return 0.0
        
        # Calculate rate of fitness improvement
        improvements = 0
        for i in range(1, len(self.convergence_curve)):
            if abs(self.convergence_curve[i] - self.convergence_curve[i-1]) > 1e-8:
                improvements += 1
        
        return improvements / len(self.convergence_curve) if self.convergence_curve else 0.0
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get solution quality metrics.
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "best_fitness": self.best_fitness if isinstance(self.best_fitness, (int, float)) else 0.0,
            "convergence_rate": self.get_convergence_rate(),
            "success_rate": self.get_success_rate(),
            "execution_efficiency": self.total_evaluations / max(self.execution_time, 0.001)
        }
        
        if self.is_multi_objective():
            metrics["pareto_front_size"] = len(self.pareto_frontier)
            if self.hypervolume is not None:
                metrics["hypervolume"] = self.hypervolume
        
        if self.is_constrained():
            metrics["feasibility_rate"] = float(self.feasible_solutions_count) / max(len(self.final_population), 1)
            metrics["constraint_violation"] = sum(self.constraint_violations.values())
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "optimization_id": self.optimization_id,
            "problem_name": self.problem_name,
            "algorithm": self.algorithm_name,
            "status": "converged" if self.is_converged() else "completed",
            "best_fitness": self.best_fitness,
            "generations_run": self.generations_run,
            "execution_time": self.execution_time,
            "quality_metrics": self.get_quality_metrics(),
            "is_feasible": self.is_feasible(),
            "is_multi_objective": self.is_multi_objective()
        }
    
    def get_pareto_frontier_data(self) -> List[Dict[str, Any]]:
        """Get Pareto frontier data for multi-objective problems.
        
        Returns:
            List of Pareto optimal solutions
        """
        if not self.is_multi_objective():
            return []
        
        return self.pareto_frontier
    
    def add_evolution_data(self, generation: int, best_fitness: float, 
                          avg_fitness: float, diversity: float):
        """Add evolution data point.
        
        Args:
            generation: Generation number
            best_fitness: Best fitness in generation
            avg_fitness: Average fitness in generation
            diversity: Population diversity
        """
        if "generations" not in self.evolution_history:
            self.evolution_history["generations"] = []
        if "best_fitness" not in self.evolution_history:
            self.evolution_history["best_fitness"] = []
        if "average_fitness" not in self.evolution_history:
            self.evolution_history["average_fitness"] = []
        if "diversity" not in self.evolution_history:
            self.evolution_history["diversity"] = []
        
        self.evolution_history["generations"].append(generation)
        self.evolution_history["best_fitness"].append(best_fitness)
        self.evolution_history["average_fitness"].append(avg_fitness)
        self.evolution_history["diversity"].append(diversity)
        
        # Update convergence curve
        self.convergence_curve.append(best_fitness)
        self.diversity_curve.append(diversity)
    
    def set_pareto_frontier(self, solutions: List[List[float]], 
                           objectives: List[List[float]]):
        """Set Pareto frontier for multi-objective problems.
        
        Args:
            solutions: Pareto optimal solutions
            objectives: Corresponding objective values
        """
        self.pareto_frontier = []
        for sol, obj in zip(solutions, objectives):
            self.pareto_frontier.append({
                "solution": sol,
                "objectives": obj
            })
        
        self.pareto_front_size = len(self.pareto_frontier)
    
    def calculate_hypervolume(self, reference_point: List[float]) -> float:
        """Calculate hypervolume indicator for multi-objective results.
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not self.is_multi_objective() or not self.pareto_frontier:
            return 0.0
        
        # Simplified hypervolume calculation
        # In practice, would use proper hypervolume algorithm
        try:
            objectives = [pf["objectives"] for pf in self.pareto_frontier]
            
            # Calculate dominated volume (simplified)
            volume = 0.0
            for obj in objectives:
                point_volume = 1.0
                for i, (o, r) in enumerate(zip(obj, reference_point)):
                    point_volume *= max(0, r - o)
                volume += point_volume
            
            self.hypervolume = volume
            return volume
            
        except Exception as e:
            logger.warning(f"Error calculating hypervolume: {e}")
            return 0.0
    
    def export_data(self, format_type: str = "dict") -> Union[Dict[str, Any], str]:
        """Export result data in specified format.
        
        Args:
            format_type: Export format ("dict", "json")
            
        Returns:
            Exported data
        """
        data = {
            "optimization_id": self.optimization_id,
            "problem_name": self.problem_name,
            "algorithm_name": self.algorithm_name,
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "generations_run": self.generations_run,
            "execution_time": self.execution_time,
            "convergence_generation": self.convergence_generation,
            "total_evaluations": self.total_evaluations,
            "evolution_history": self.evolution_history,
            "pareto_frontier": self.pareto_frontier,
            "constraint_violations": self.constraint_violations,
            "performance_metrics": self.performance_metrics,
            "algorithm_config": self.algorithm_config,
            "metadata": self.metadata,
            "quality_metrics": self.get_quality_metrics(),
            "summary": self.get_summary()
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def save_to_file(self, filepath: str, format_type: str = "json"):
        """Save results to file.
        
        Args:
            filepath: Output file path
            format_type: File format ("json", "csv")
        """
        try:
            if format_type == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.export_data("dict"), f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "OptimizationResult":
        """Load results from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            OptimizationResult instance
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return cls(**{k: v for k, v in data.items() 
                         if k in cls.__dataclass_fields__})
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def compare_with(self, other: "OptimizationResult") -> Dict[str, Any]:
        """Compare this result with another result.
        
        Args:
            other: Other optimization result
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "algorithms": [self.algorithm_name, other.algorithm_name],
            "execution_times": [self.execution_time, other.execution_time],
            "generations": [self.generations_run, other.generations_run],
            "best_fitness": [self.best_fitness, other.best_fitness],
            "convergence": [self.is_converged(), other.is_converged()],
            "feasibility": [self.is_feasible(), other.is_feasible()]
        }
        
        # Determine winner
        if isinstance(self.best_fitness, (int, float)) and isinstance(other.best_fitness, (int, float)):
            comparison["fitness_winner"] = "self" if self.best_fitness < other.best_fitness else "other"
            comparison["time_winner"] = "self" if self.execution_time < other.execution_time else "other"
        
        return comparison
    
    def __str__(self) -> str:
        """String representation of the result."""
        return (f"OptimizationResult(problem='{self.problem_name}', "
                f"algorithm='{self.algorithm_name}', "
                f"fitness={self.best_fitness}, "
                f"generations={self.generations_run})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()