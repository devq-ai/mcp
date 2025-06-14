"""
Darwin Genetic Algorithm Implementation

This module provides a simplified genetic algorithm implementation for the Darwin
MCP server. It includes basic genetic operators and algorithm execution logic.

Example usage:
    from darwin_mcp.algorithms.genetic import GeneticAlgorithm
    from darwin_mcp.optimization.config import OptimizationConfig
    
    config = OptimizationConfig()
    algorithm = GeneticAlgorithm(config)
    result = algorithm.optimize(problem)
"""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Simplified genetic algorithm implementation."""
    
    def __init__(self, config):
        """Initialize genetic algorithm with configuration.
        
        Args:
            config: OptimizationConfig instance
        """
        self.config = config
        self.population = []
        self.fitness_values = []
        self.generation = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evolution_history = {
            "generations": [],
            "best_fitness": [],
            "average_fitness": [],
            "diversity": []
        }
        
        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    def optimize(self, problem: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute genetic algorithm optimization.
        
        Args:
            problem: Problem definition dictionary
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting genetic algorithm optimization")
        start_time = time.time()
        
        try:
            # Initialize population
            self._initialize_population(problem)
            
            # Evaluate initial population
            self._evaluate_population(problem)
            
            # Evolution loop
            for self.generation in range(self.config.max_generations):
                # Selection
                selected = self._selection()
                
                # Crossover
                offspring = self._crossover(selected)
                
                # Mutation
                mutated = self._mutation(offspring)
                
                # Replacement
                self._replacement(mutated, problem)
                
                # Update best solution
                self._update_best()
                
                # Record evolution data
                self._record_evolution_data()
                
                # Progress callback
                if progress_callback and self.generation % self.config.callback_frequency == 0:
                    progress_callback(self.generation, self.best_fitness, self.population)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged at generation {self.generation}")
                    break
            
            execution_time = time.time() - start_time
            
            # Prepare results
            result = {
                "best_solution": self.best_solution.copy() if self.best_solution else [],
                "best_fitness": self.best_fitness,
                "generations_run": self.generation + 1,
                "execution_time": execution_time,
                "convergence_generation": self.generation if self._check_convergence() else None,
                "final_population": [ind.copy() for ind in self.population],
                "evolution_history": self.evolution_history.copy(),
                "total_evaluations": (self.generation + 1) * self.config.population_size
            }
            
            logger.info(f"Optimization completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _initialize_population(self, problem: Dict[str, Any]):
        """Initialize random population."""
        variables = problem["variables"]
        self.population = []
        
        for _ in range(self.config.population_size):
            individual = []
            
            for var in variables:
                if var.get("type") == "continuous":
                    bounds = var.get("bounds", [0, 1])
                    value = random.uniform(bounds[0], bounds[1])
                elif var.get("type") == "discrete":
                    bounds = var.get("bounds", [0, 10])
                    value = random.randint(int(bounds[0]), int(bounds[1]))
                elif var.get("type") == "categorical":
                    values = var.get("values", [0, 1])
                    value = random.choice(values)
                else:
                    # Default to continuous [0, 1]
                    value = random.random()
                
                individual.append(value)
            
            self.population.append(individual)
    
    def _evaluate_population(self, problem: Dict[str, Any]):
        """Evaluate fitness for entire population."""
        fitness_function = problem.get("fitness_function")
        self.fitness_values = []
        
        if fitness_function:
            for individual in self.population:
                try:
                    fitness = fitness_function(individual)
                    self.fitness_values.append(fitness)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    self.fitness_values.append(float('inf'))
        else:
            # Default sphere function
            for individual in self.population:
                fitness = sum(x**2 for x in individual)
                self.fitness_values.append(fitness)
    
    def _selection(self) -> List[List[float]]:
        """Tournament selection."""
        selected = []
        tournament_size = getattr(self.config, 'tournament_size', 3)
        
        for _ in range(self.config.population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(self.population)), 
                                             min(tournament_size, len(self.population)))
            
            # Find best in tournament
            best_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
            selected.append(self.population[best_idx].copy())
        
        return selected
    
    def _crossover(self, selected: List[List[float]]) -> List[List[float]]:
        """Single-point crossover."""
        offspring = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            if random.random() < self.config.crossover_rate:
                # Perform crossover
                crossover_point = random.randint(1, len(parent1) - 1)
                
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                
                offspring.extend([child1, child2])
            else:
                # No crossover, keep parents
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring[:self.config.population_size]
    
    def _mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Gaussian mutation."""
        mutated = []
        
        for individual in offspring:
            mutated_individual = individual.copy()
            
            for i in range(len(mutated_individual)):
                if random.random() < self.config.mutation_rate:
                    # Gaussian mutation
                    mutation_strength = 0.1
                    noise = random.gauss(0, mutation_strength)
                    mutated_individual[i] += noise
                    
                    # Clip to reasonable bounds (simplified)
                    mutated_individual[i] = max(-10, min(10, mutated_individual[i]))
            
            mutated.append(mutated_individual)
        
        return mutated
    
    def _replacement(self, offspring: List[List[float]], problem: Dict[str, Any]):
        """Replace population with offspring."""
        # Evaluate offspring
        offspring_fitness = []
        fitness_function = problem.get("fitness_function")
        
        if fitness_function:
            for individual in offspring:
                try:
                    fitness = fitness_function(individual)
                    offspring_fitness.append(fitness)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    offspring_fitness.append(float('inf'))
        else:
            for individual in offspring:
                fitness = sum(x**2 for x in individual)
                offspring_fitness.append(fitness)
        
        # Elitism: keep best individuals
        elitism = getattr(self.config, 'elitism', 1)
        
        if elitism > 0:
            # Combine populations
            combined_pop = self.population + offspring
            combined_fitness = self.fitness_values + offspring_fitness
            
            # Sort by fitness
            sorted_indices = sorted(range(len(combined_fitness)), 
                                  key=lambda i: combined_fitness[i])
            
            # Keep best individuals
            self.population = [combined_pop[i] for i in sorted_indices[:self.config.population_size]]
            self.fitness_values = [combined_fitness[i] for i in sorted_indices[:self.config.population_size]]
        else:
            # Replace entire population
            self.population = offspring
            self.fitness_values = offspring_fitness
    
    def _update_best(self):
        """Update best solution found so far."""
        if self.fitness_values:
            current_best_idx = min(range(len(self.fitness_values)), 
                                 key=lambda i: self.fitness_values[i])
            current_best_fitness = self.fitness_values[current_best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[current_best_idx].copy()
    
    def _record_evolution_data(self):
        """Record evolution history data."""
        if self.fitness_values:
            avg_fitness = sum(self.fitness_values) / len(self.fitness_values)
            diversity = self._calculate_diversity()
            
            self.evolution_history["generations"].append(self.generation)
            self.evolution_history["best_fitness"].append(self.best_fitness)
            self.evolution_history["average_fitness"].append(avg_fitness)
            self.evolution_history["diversity"].append(diversity)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        try:
            pop_array = np.array(self.population)
            distances = []
            
            for i in range(len(pop_array)):
                for j in range(i + 1, len(pop_array)):
                    dist = np.linalg.norm(pop_array[i] - pop_array[j])
                    distances.append(dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except Exception:
            return 0.0
    
    def _check_convergence(self) -> bool:
        """Check convergence criteria."""
        if not self.config.early_stopping:
            return False
        
        convergence_generations = getattr(self.config, 'convergence_generations', 20)
        convergence_threshold = getattr(self.config, 'convergence_threshold', 1e-6)
        
        # Need enough history
        if len(self.evolution_history["best_fitness"]) < convergence_generations:
            return False
        
        # Check if fitness hasn't improved significantly
        recent_fitness = self.evolution_history["best_fitness"][-convergence_generations:]
        fitness_range = max(recent_fitness) - min(recent_fitness)
        
        return fitness_range < convergence_threshold
    
    def enable_parallel_evaluation(self, n_jobs: int = -1):
        """Enable parallel fitness evaluation (placeholder)."""
        # In a full implementation, this would set up parallel evaluation
        logger.info(f"Parallel evaluation requested with {n_jobs} jobs")
        # For now, just log the request
        pass
    
    def get_population_statistics(self) -> Dict[str, float]:
        """Get current population statistics."""
        if not self.fitness_values:
            return {}
        
        return {
            "best_fitness": min(self.fitness_values),
            "worst_fitness": max(self.fitness_values),
            "average_fitness": sum(self.fitness_values) / len(self.fitness_values),
            "diversity": self._calculate_diversity(),
            "generation": self.generation
        }