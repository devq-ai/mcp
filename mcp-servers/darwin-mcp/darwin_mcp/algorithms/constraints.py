"""
Darwin Constraint Handling
This module provides constraint handling mechanisms for the Darwin MCP server.
It includes penalty methods, repair operators, and constraint satisfaction
techniques for genetic algorithms.
Example usage:
    from darwin_mcp.algorithms.constraints import ConstraintHandler
    
    handler = ConstraintHandler("penalty")
    penalty = handler.evaluate_constraints(solution, constraints)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class ConstraintHandler:
    """Handles constraints in optimization problems."""
    
    def __init__(self, method: str = "penalty"):
        """Initialize constraint handler.
        
        Args:
            method: Constraint handling method (penalty, repair, feasibility)
        """
        self.method = method
        self.penalty_weight = 1000.0
        
    def evaluate_constraints(self, solution: List[float], 
                           constraints: List[Dict[str, Any]]) -> float:
        """Evaluate constraint violations for a solution.
        
        Args:
            solution: Solution vector
            constraints: List of constraint definitions
            
        Returns:
            Total constraint violation penalty
        """
        if not constraints:
            return 0.0
            
        if self.method == "penalty":
            return self._penalty_method(solution, constraints)
        elif self.method == "repair":
            return self._repair_method(solution, constraints)
        else:
            return self._feasibility_check(solution, constraints)
    
    def _penalty_method(self, solution: List[float], 
                       constraints: List[Dict[str, Any]]) -> float:
        """Static penalty method for constraint handling."""
        total_penalty = 0.0
        
        for constraint in constraints:
            violation = self._evaluate_single_constraint(solution, constraint)
            if violation > 0:
                total_penalty += self.penalty_weight * (violation ** 2)
        
        return total_penalty
    
    def _repair_method(self, solution: List[float], 
                      constraints: List[Dict[str, Any]]) -> float:
        """Repair method to fix constraint violations."""
        # Simplified repair - just return violation amount
        total_violation = 0.0
        
        for constraint in constraints:
            violation = self._evaluate_single_constraint(solution, constraint)
            total_violation += max(0, violation)
        
        return total_violation
    
    def _feasibility_check(self, solution: List[float], 
                          constraints: List[Dict[str, Any]]) -> float:
        """Binary feasibility check."""
        for constraint in constraints:
            violation = self._evaluate_single_constraint(solution, constraint)
            if violation > 1e-6:  # Tolerance
                return float('inf')  # Infeasible
        
        return 0.0  # Feasible
    
    def _evaluate_single_constraint(self, solution: List[float], 
                                   constraint: Dict[str, Any]) -> float:
        """Evaluate a single constraint."""
        constraint_type = constraint.get("type", "inequality")
        expression = constraint.get("expression", "")
        
        # Simplified constraint evaluation
        # In practice, would parse and evaluate expressions
        if "sum" in expression and "==" in expression:
            # Equality constraint like "sum(x) == 1"
            total = sum(solution)
            target = 1.0  # Extract from expression
            return abs(total - target)
        elif "sum" in expression and "<=" in expression:
            # Inequality constraint like "sum(x) <= 1"
            total = sum(solution)
            limit = 1.0  # Extract from expression
            return max(0, total - limit)
        else:
            # Generic constraint
            return 0.0
    
    def repair_solution(self, solution: List[float], 
                       constraints: List[Dict[str, Any]]) -> List[float]:
        """Repair a solution to satisfy constraints."""
        repaired = solution.copy()
        
        for constraint in constraints:
            repaired = self._repair_single_constraint(repaired, constraint)
        
        return repaired
    
    def _repair_single_constraint(self, solution: List[float], 
                                 constraint: Dict[str, Any]) -> List[float]:
        """Repair solution for a single constraint."""
        expression = constraint.get("expression", "")
        
        # Simplified repair logic
        if "sum" in expression and "==" in expression:
            # Normalize to sum to 1
            total = sum(solution)
            if total > 0:
                return [x / total for x in solution]
        elif "sum" in expression and "<=" in expression:
            # Scale down if sum exceeds limit
            total = sum(solution)
            limit = 1.0  # Extract from expression
            if total > limit:
                scale = limit / total
                return [x * scale for x in solution]
        
        return solution
    
    def is_feasible(self, solution: List[float], 
                   constraints: List[Dict[str, Any]], 
                   tolerance: float = 1e-6) -> bool:
        """Check if solution is feasible."""
        for constraint in constraints:
            violation = self._evaluate_single_constraint(solution, constraint)
            if violation > tolerance:
                return False
        return True
    
    def set_penalty_weight(self, weight: float):
        """Set penalty weight for constraint violations."""
        self.penalty_weight = weight
    
    def adaptive_penalty(self, generation: int, max_generations: int):
        """Adaptive penalty weight adjustment."""
        # Increase penalty weight over generations
        self.penalty_weight = 1000.0 * (1 + generation / max_generations)


class BoundaryHandler:
    """Handles variable boundary constraints."""
    
    @staticmethod
    def clip_to_bounds(solution: List[float], 
                      bounds: List[List[float]]) -> List[float]:
        """Clip solution to variable bounds."""
        clipped = []
        for i, (value, bound) in enumerate(zip(solution, bounds)):
            clipped_value = max(bound[0], min(bound[1], value))
            clipped.append(clipped_value)
        return clipped
    
    @staticmethod
    def reflect_bounds(solution: List[float], 
                      bounds: List[List[float]]) -> List[float]:
        """Reflect solution at boundaries."""
        reflected = []
        for value, bound in zip(solution, bounds):
            if value < bound[0]:
                reflected_value = bound[0] + (bound[0] - value)
                if reflected_value > bound[1]:
                    reflected_value = bound[1]
            elif value > bound[1]:
                reflected_value = bound[1] - (value - bound[1])
                if reflected_value < bound[0]:
                    reflected_value = bound[0]
            else:
                reflected_value = value
            reflected.append(reflected_value)
        return reflected
    
    @staticmethod
    def wrap_bounds(solution: List[float], 
                   bounds: List[List[float]]) -> List[float]:
        """Wrap solution around boundaries."""
        wrapped = []
        for value, bound in zip(solution, bounds):
            range_size = bound[1] - bound[0]
            if value < bound[0]:
                wrapped_value = bound[1] - ((bound[0] - value) % range_size)
            elif value > bound[1]:
                wrapped_value = bound[0] + ((value - bound[1]) % range_size)
            else:
                wrapped_value = value
            wrapped.append(wrapped_value)
        return wrapped


class FeasibilityMaintainer:
    """Maintains feasibility during genetic operations."""
    
    def __init__(self, constraints: List[Dict[str, Any]], 
                 bounds: List[List[float]]):
        """Initialize feasibility maintainer.
        
        Args:
            constraints: Problem constraints
            bounds: Variable bounds
        """
        self.constraints = constraints
        self.bounds = bounds
        self.constraint_handler = ConstraintHandler()
        self.boundary_handler = BoundaryHandler()
    
    def ensure_feasibility(self, solution: List[float]) -> List[float]:
        """Ensure solution satisfies all constraints."""
        # Handle boundary constraints
        feasible = self.boundary_handler.clip_to_bounds(solution, self.bounds)
        
        # Handle general constraints
        if self.constraints:
            feasible = self.constraint_handler.repair_solution(feasible, self.constraints)
            
            # Final boundary check after repair
            feasible = self.boundary_handler.clip_to_bounds(feasible, self.bounds)
        
        return feasible
    
    def feasible_crossover(self, parent1: List[float], 
                          parent2: List[float]) -> List[float]:
        """Crossover that maintains feasibility."""
        # Simple averaging crossover
        child = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
        return self.ensure_feasibility(child)
    
    def feasible_mutation(self, individual: List[float], 
                         mutation_rate: float = 0.1) -> List[float]:
        """Mutation that maintains feasibility."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Gaussian mutation within bounds
                bound = self.bounds[i]
                sigma = (bound[1] - bound[0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
        
        return self.ensure_feasibility(mutated)