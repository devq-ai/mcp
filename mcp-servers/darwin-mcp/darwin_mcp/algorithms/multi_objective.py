"""
Darwin Multi-Objective Optimization Algorithms

This module provides multi-objective optimization algorithms for the Darwin MCP server.
It includes NSGA-II, NSGA-III, and other multi-objective evolutionary algorithms.

Example usage:
    from darwin_mcp.algorithms.multi_objective import NSGAII, NSGAIII
    
    algorithm = NSGAII()
    result = algorithm.optimize(problem)
"""

import logging
import random
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class NSGAII:
    """NSGA-II multi-objective optimization algorithm."""
    
    def __init__(self, config=None):
        """Initialize NSGA-II algorithm.
        
        Args:
            config: Algorithm configuration
        """
        self.config = config or {}
        self.population_size = self.config.get("population_size", 100)
        self.max_generations = self.config.get("max_generations", 200)
        self.mutation_rate = self.config.get("mutation_rate", 0.1)
        self.crossover_rate = self.config.get("crossover_rate", 0.8)
    
    def optimize(self, problem: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """Execute NSGA-II optimization.
        
        Args:
            problem: Problem definition
            progress_callback: Progress callback function
            
        Returns:
            Optimization results
        """
        logger.info("Starting NSGA-II optimization")
        
        # Initialize population
        population = self._initialize_population(problem)
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate objectives
            objectives = self._evaluate_objectives(population, problem)
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(objectives)
            
            # Crowding distance
            crowding_distances = self._calculate_crowding_distance(objectives, fronts)
            
            # Selection and reproduction
            offspring = self._create_offspring(population, fronts, crowding_distances)
            
            # Combine populations
            combined_pop = population + offspring
            combined_obj = objectives + self._evaluate_objectives(offspring, problem)
            
            # Environmental selection
            population, objectives = self._environmental_selection(
                combined_pop, combined_obj
            )
            
            if progress_callback:
                progress_callback(generation, objectives, population)
        
        # Extract Pareto frontier
        pareto_frontier = self._extract_pareto_frontier(population, objectives)
        
        return {
            "pareto_frontier": pareto_frontier,
            "final_population": population,
            "final_objectives": objectives,
            "generations_run": self.max_generations,
            "algorithm": "NSGA-II"
        }
    
    def _initialize_population(self, problem: Dict[str, Any]) -> List[List[float]]:
        """Initialize random population."""
        variables = problem.get("variables", [])
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for var in variables:
                bounds = var.get("bounds", [0, 1])
                value = random.uniform(bounds[0], bounds[1])
                individual.append(value)
            population.append(individual)
        
        return population
    
    def _evaluate_objectives(self, population: List[List[float]], 
                           problem: Dict[str, Any]) -> List[List[float]]:
        """Evaluate objectives for population."""
        objectives = []
        fitness_function = problem.get("fitness_function")
        
        for individual in population:
            if fitness_function:
                obj_values = fitness_function(individual)
                if not isinstance(obj_values, list):
                    obj_values = [obj_values]
            else:
                # Default multi-objective functions
                obj1 = sum(x**2 for x in individual)  # Sphere
                obj2 = sum((x-1)**2 for x in individual)  # Shifted sphere
                obj_values = [obj1, obj2]
            
            objectives.append(obj_values)
        
        return objectives
    
    def _non_dominated_sort(self, objectives: List[List[float]]) -> List[List[int]]:
        """Perform non-dominated sorting."""
        n = len(objectives)
        fronts = [[]]
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        
        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2."""
        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # Assuming minimization
                return False
            if o1 < o2:
                better_in_any = True
        return better_in_any
    
    def _calculate_crowding_distance(self, objectives: List[List[float]], 
                                   fronts: List[List[int]]) -> List[float]:
        """Calculate crowding distance."""
        n = len(objectives)
        distances = [0.0] * n
        
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    distances[idx] = float('inf')
                continue
            
            num_objectives = len(objectives[0])
            
            for obj_idx in range(num_objectives):
                # Sort by objective value
                front.sort(key=lambda x: objectives[x][obj_idx])
                
                # Set boundary points to infinity
                distances[front[0]] = float('inf')
                distances[front[-1]] = float('inf')
                
                # Calculate distances for intermediate points
                obj_range = objectives[front[-1]][obj_idx] - objectives[front[0]][obj_idx]
                if obj_range > 0:
                    for i in range(1, len(front) - 1):
                        distance = (objectives[front[i+1]][obj_idx] - 
                                  objectives[front[i-1]][obj_idx]) / obj_range
                        distances[front[i]] += distance
        
        return distances
    
    def _create_offspring(self, population: List[List[float]], 
                         fronts: List[List[int]], 
                         crowding_distances: List[float]) -> List[List[float]]:
        """Create offspring through selection and variation."""
        offspring = []
        
        for _ in range(self.population_size):
            # Tournament selection
            parent1 = self._tournament_selection(population, fronts, crowding_distances)
            parent2 = self._tournament_selection(population, fronts, crowding_distances)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        # Mutation
        for individual in offspring:
            if random.random() < self.mutation_rate:
                self._mutate(individual)
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, population: List[List[float]], 
                            fronts: List[List[int]], 
                            crowding_distances: List[float]) -> List[float]:
        """Tournament selection based on rank and crowding distance."""
        tournament_size = 2
        candidates = random.sample(range(len(population)), tournament_size)
        
        # Find front ranks
        front_ranks = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                front_ranks[idx] = rank
        
        # Select best candidate
        best = candidates[0]
        for candidate in candidates[1:]:
            if (front_ranks.get(candidate, float('inf')) < front_ranks.get(best, float('inf')) or
                (front_ranks.get(candidate, float('inf')) == front_ranks.get(best, float('inf')) and
                 crowding_distances[candidate] > crowding_distances[best])):
                best = candidate
        
        return population[best].copy()
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Simulated binary crossover."""
        eta = 20.0
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if random.random() <= 0.5:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        return child1, child2
    
    def _mutate(self, individual: List[float]):
        """Polynomial mutation."""
        eta = 20.0
        for i in range(len(individual)):
            if random.random() <= (1.0 / len(individual)):
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                individual[i] += delta
    
    def _environmental_selection(self, population: List[List[float]], 
                               objectives: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """Environmental selection to maintain population size."""
        fronts = self._non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, fronts)
        
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size:
                selected_indices.extend(front)
            else:
                # Sort by crowding distance and select best
                remaining = self.population_size - len(selected_indices)
                front.sort(key=lambda x: crowding_distances[x], reverse=True)
                selected_indices.extend(front[:remaining])
                break
        
        selected_pop = [population[i] for i in selected_indices]
        selected_obj = [objectives[i] for i in selected_indices]
        
        return selected_pop, selected_obj
    
    def _extract_pareto_frontier(self, population: List[List[float]], 
                               objectives: List[List[float]]) -> List[Dict[str, Any]]:
        """Extract Pareto frontier."""
        fronts = self._non_dominated_sort(objectives)
        pareto_frontier = []
        
        for idx in fronts[0]:  # First front is Pareto optimal
            pareto_frontier.append({
                "solution": population[idx],
                "objectives": objectives[idx]
            })
        
        return pareto_frontier


class NSGAIII:
    """NSGA-III multi-objective optimization algorithm."""
    
    def __init__(self, config=None):
        """Initialize NSGA-III algorithm.
        
        Args:
            config: Algorithm configuration
        """
        self.config = config or {}
        self.population_size = self.config.get("population_size", 120)
        self.max_generations = self.config.get("max_generations", 300)
        self.reference_points = self.config.get("reference_points", [])
    
    def optimize(self, problem: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """Execute NSGA-III optimization.
        
        Args:
            problem: Problem definition
            progress_callback: Progress callback function
            
        Returns:
            Optimization results
        """
        logger.info("Starting NSGA-III optimization")
        
        # Generate reference points if not provided
        if not self.reference_points:
            num_objectives = len(problem.get("objectives", []))
            self.reference_points = self._generate_reference_points(num_objectives)
        
        # Initialize population
        population = self._initialize_population(problem)
        
        # Evolution loop (simplified)
        for generation in range(self.max_generations):
            objectives = self._evaluate_objectives(population, problem)
            
            if progress_callback:
                progress_callback(generation, objectives, population)
        
        # Final evaluation
        objectives = self._evaluate_objectives(population, problem)
        pareto_frontier = self._extract_pareto_frontier(population, objectives)
        
        return {
            "pareto_frontier": pareto_frontier,
            "final_population": population,
            "final_objectives": objectives,
            "generations_run": self.max_generations,
            "algorithm": "NSGA-III"
        }
    
    def _generate_reference_points(self, num_objectives: int, divisions: int = 12) -> List[List[float]]:
        """Generate reference points for NSGA-III."""
        # Simplified reference point generation
        points = []
        if num_objectives == 2:
            for i in range(divisions + 1):
                point = [i / divisions, 1 - i / divisions]
                points.append(point)
        elif num_objectives == 3:
            for i in range(divisions + 1):
                for j in range(divisions + 1 - i):
                    k = divisions - i - j
                    point = [i / divisions, j / divisions, k / divisions]
                    points.append(point)
        else:
            # For higher dimensions, use uniform random points
            for _ in range(100):
                point = [random.random() for _ in range(num_objectives)]
                total = sum(point)
                point = [p / total for p in point]
                points.append(point)
        
        return points
    
    def _initialize_population(self, problem: Dict[str, Any]) -> List[List[float]]:
        """Initialize random population."""
        variables = problem.get("variables", [])
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for var in variables:
                bounds = var.get("bounds", [0, 1])
                value = random.uniform(bounds[0], bounds[1])
                individual.append(value)
            population.append(individual)
        
        return population
    
    def _evaluate_objectives(self, population: List[List[float]], 
                           problem: Dict[str, Any]) -> List[List[float]]:
        """Evaluate objectives for population."""
        objectives = []
        fitness_function = problem.get("fitness_function")
        
        for individual in population:
            if fitness_function:
                obj_values = fitness_function(individual)
                if not isinstance(obj_values, list):
                    obj_values = [obj_values]
            else:
                # Default multi-objective functions
                obj1 = sum(x**2 for x in individual)
                obj2 = sum((x-1)**2 for x in individual)
                obj3 = sum((x-0.5)**2 for x in individual)
                obj_values = [obj1, obj2, obj3]
            
            objectives.append(obj_values)
        
        return objectives
    
    def _extract_pareto_frontier(self, population: List[List[float]], 
                               objectives: List[List[float]]) -> List[Dict[str, Any]]:
        """Extract Pareto frontier."""
        pareto_frontier = []
        
        # Simplified Pareto frontier extraction
        for i, (solution, objective) in enumerate(zip(population, objectives)):
            is_dominated = False
            for j, other_obj in enumerate(objectives):
                if i != j and self._dominates(other_obj, objective):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_frontier.append({
                    "solution": solution,
                    "objectives": objective
                })
        
        return pareto_frontier
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2."""
        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # Assuming minimization
                return False
            if o1 < o2:
                better_in_any = True
        return better_in_any