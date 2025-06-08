"""
Darwin Genetic Algorithm Operators

This module provides genetic algorithm operators for the Darwin MCP server.
It includes selection, crossover, and mutation operators used by genetic algorithms.

Example usage:
    from darwin_mcp.algorithms.operators import SelectionOperator, CrossoverOperator
    
    selection = SelectionOperator("tournament")
    crossover = CrossoverOperator("single_point")
"""

import logging
import random
from typing import Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SelectionOperator:
    """Selection operators for genetic algorithms."""
    
    def __init__(self, method: str = "tournament"):
        """Initialize selection operator.
        
        Args:
            method: Selection method (tournament, roulette, rank)
        """
        self.method = method
    
    def select(self, population: List[List[float]], fitness_values: List[float],
               num_parents: int = 2) -> List[List[float]]:
        """Select parents from population.
        
        Args:
            population: Current population
            fitness_values: Fitness values for population
            num_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        if self.method == "tournament":
            return self._tournament_selection(population, fitness_values, num_parents)
        elif self.method == "roulette":
            return self._roulette_selection(population, fitness_values, num_parents)
        else:
            return self._random_selection(population, num_parents)
    
    def _tournament_selection(self, population: List[List[float]], 
                             fitness_values: List[float], num_parents: int) -> List[List[float]]:
        """Tournament selection."""
        selected = []
        tournament_size = 3
        
        for _ in range(num_parents):
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
            selected.append(population[best_idx].copy())
        
        return selected
    
    def _roulette_selection(self, population: List[List[float]], 
                           fitness_values: List[float], num_parents: int) -> List[List[float]]:
        """Roulette wheel selection."""
        # Convert fitness to selection probabilities (assuming minimization)
        max_fitness = max(fitness_values)
        weights = [max_fitness - f + 1 for f in fitness_values]
        total_weight = sum(weights)
        
        selected = []
        for _ in range(num_parents):
            r = random.uniform(0, total_weight)
            cumsum = 0
            for i, weight in enumerate(weights):
                cumsum += weight
                if cumsum >= r:
                    selected.append(population[i].copy())
                    break
        
        return selected
    
    def _random_selection(self, population: List[List[float]], 
                         num_parents: int) -> List[List[float]]:
        """Random selection."""
        return [random.choice(population).copy() for _ in range(num_parents)]


class CrossoverOperator:
    """Crossover operators for genetic algorithms."""
    
    def __init__(self, method: str = "single_point"):
        """Initialize crossover operator.
        
        Args:
            method: Crossover method (single_point, two_point, uniform)
        """
        self.method = method
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        if self.method == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif self.method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif self.method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()
    
    def _single_point_crossover(self, parent1: List[float], 
                               parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover."""
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: List[float], 
                            parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Two-point crossover."""
        if len(parent1) <= 2:
            return self._single_point_crossover(parent1, parent2)
        
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: List[float], 
                          parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Uniform crossover."""
        child1 = []
        child2 = []
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2


class MutationOperator:
    """Mutation operators for genetic algorithms."""
    
    def __init__(self, method: str = "gaussian"):
        """Initialize mutation operator.
        
        Args:
            method: Mutation method (gaussian, uniform, polynomial)
        """
        self.method = method
    
    def mutate(self, individual: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Mutate an individual.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation per gene
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                if self.method == "gaussian":
                    mutated[i] = self._gaussian_mutation(mutated[i])
                elif self.method == "uniform":
                    mutated[i] = self._uniform_mutation(mutated[i])
                elif self.method == "polynomial":
                    mutated[i] = self._polynomial_mutation(mutated[i])
        
        return mutated
    
    def _gaussian_mutation(self, value: float, sigma: float = 0.1) -> float:
        """Gaussian mutation."""
        return value + random.gauss(0, sigma)
    
    def _uniform_mutation(self, value: float, bounds: List[float] = None) -> float:
        """Uniform mutation."""
        if bounds is None:
            bounds = [-10, 10]
        return random.uniform(bounds[0], bounds[1])
    
    def _polynomial_mutation(self, value: float, eta: float = 20.0) -> float:
        """Polynomial mutation."""
        u = random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1.0 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
        
        return value + delta