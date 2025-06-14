"""
Darwin Optimization Problem Definition

This module provides the core classes for defining optimization problems
in the Darwin genetic algorithm platform. It includes variable definitions,
objectives, constraints, and problem validation.

The problem definition system supports:
- Multiple variable types (continuous, discrete, categorical, permutation)
- Single and multi-objective optimization
- Equality and inequality constraints
- Custom fitness functions
- Problem validation and analysis

Example usage:
    from darwin_mcp.optimization.problem import OptimizationProblem, Variable, Objective
    
    problem = OptimizationProblem(
        name="Portfolio Optimization",
        variables=[
            Variable("stocks", "continuous", bounds=[0, 0.8]),
            Variable("bonds", "continuous", bounds=[0, 0.5])
        ],
        objectives=[
            Objective("maximize_return", "maximize"),
            Objective("minimize_risk", "minimize")
        ]
    )
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class VariableType(Enum):
    """Supported variable types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    PERMUTATION = "permutation"
    BINARY = "binary"


class ObjectiveType(Enum):
    """Optimization objective types."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintType(Enum):
    """Constraint types."""
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOUND = "bound"


@dataclass
class Variable:
    """Optimization variable definition.
    
    Attributes:
        name: Variable name
        type: Variable type (continuous, discrete, etc.)
        bounds: Lower and upper bounds for continuous/discrete variables
        values: Possible values for categorical variables
        description: Optional description
        initial_value: Optional initial value
        precision: Precision for discrete variables
    """
    name: str
    type: Union[str, VariableType]
    bounds: Optional[List[Union[int, float]]] = None
    values: Optional[List[Any]] = None
    description: Optional[str] = None
    initial_value: Optional[Any] = None
    precision: Optional[int] = None
    
    def __post_init__(self):
        """Validate variable definition after initialization."""
        # Convert string type to enum
        if isinstance(self.type, str):
            try:
                self.type = VariableType(self.type.lower())
            except ValueError:
                raise ValueError(f"Invalid variable type: {self.type}")
        
        # Validate bounds for continuous/discrete variables
        if self.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
            if not self.bounds or len(self.bounds) != 2:
                raise ValueError(f"Variable {self.name} of type {self.type.value} must have bounds [min, max]")
            
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Variable {self.name}: lower bound must be less than upper bound")
        
        # Validate values for categorical variables
        elif self.type == VariableType.CATEGORICAL:
            if not self.values:
                raise ValueError(f"Categorical variable {self.name} must have values")
            
            if len(self.values) < 2:
                raise ValueError(f"Categorical variable {self.name} must have at least 2 values")
        
        # Validate permutation variables
        elif self.type == VariableType.PERMUTATION:
            if not self.bounds or len(self.bounds) != 1:
                raise ValueError(f"Permutation variable {self.name} must have size [n]")
            
            if not isinstance(self.bounds[0], int) or self.bounds[0] < 2:
                raise ValueError(f"Permutation variable {self.name} size must be integer >= 2")
    
    def get_domain_size(self) -> Union[int, float]:
        """Get the size of the variable domain.
        
        Returns:
            Domain size (finite for discrete, infinite for continuous)
        """
        if self.type == VariableType.CONTINUOUS:
            return float('inf')
        elif self.type == VariableType.DISCRETE:
            return int(self.bounds[1] - self.bounds[0] + 1)
        elif self.type == VariableType.CATEGORICAL:
            return len(self.values)
        elif self.type == VariableType.PERMUTATION:
            import math
            return math.factorial(self.bounds[0])
        elif self.type == VariableType.BINARY:
            return 2
        else:
            return 1
    
    def is_valid_value(self, value: Any) -> bool:
        """Check if a value is valid for this variable.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if self.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
                return self.bounds[0] <= value <= self.bounds[1]
            elif self.type == VariableType.CATEGORICAL:
                return value in self.values
            elif self.type == VariableType.BINARY:
                return value in [0, 1]
            elif self.type == VariableType.PERMUTATION:
                if not isinstance(value, (list, tuple)):
                    return False
                return (len(value) == self.bounds[0] and 
                       set(value) == set(range(self.bounds[0])))
            return True
        except (TypeError, IndexError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "bounds": self.bounds,
            "values": self.values,
            "description": self.description,
            "initial_value": self.initial_value,
            "precision": self.precision
        }


@dataclass
class Objective:
    """Optimization objective definition.
    
    Attributes:
        name: Objective name
        type: Objective type (minimize or maximize)
        function: Function name or callable
        weight: Weight for multi-objective optimization
        description: Optional description
        target_value: Optional target value for goal programming
    """
    name: str
    type: Union[str, ObjectiveType]
    function: Optional[Union[str, Callable]] = None
    weight: float = 1.0
    description: Optional[str] = None
    target_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate objective definition after initialization."""
        # Convert string type to enum
        if isinstance(self.type, str):
            try:
                self.type = ObjectiveType(self.type.lower())
            except ValueError:
                raise ValueError(f"Invalid objective type: {self.type}")
        
        # Validate weight
        if self.weight <= 0:
            raise ValueError(f"Objective {self.name}: weight must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "function": self.function if isinstance(self.function, str) else str(self.function),
            "weight": self.weight,
            "description": self.description,
            "target_value": self.target_value
        }


@dataclass
class Constraint:
    """Optimization constraint definition.
    
    Attributes:
        name: Constraint name
        type: Constraint type (equality or inequality)
        expression: Constraint expression or function
        tolerance: Tolerance for equality constraints
        description: Optional description
        penalty_weight: Penalty weight for constraint violations
    """
    name: str
    type: Union[str, ConstraintType]
    expression: Union[str, Callable]
    tolerance: float = 1e-6
    description: Optional[str] = None
    penalty_weight: float = 1000.0
    
    def __post_init__(self):
        """Validate constraint definition after initialization."""
        # Convert string type to enum
        if isinstance(self.type, str):
            try:
                self.type = ConstraintType(self.type.lower())
            except ValueError:
                raise ValueError(f"Invalid constraint type: {self.type}")
        
        # Validate tolerance
        if self.tolerance <= 0:
            raise ValueError(f"Constraint {self.name}: tolerance must be positive")
        
        # Validate penalty weight
        if self.penalty_weight <= 0:
            raise ValueError(f"Constraint {self.name}: penalty weight must be positive")
    
    def evaluate(self, solution: List[float]) -> float:
        """Evaluate constraint for a solution.
        
        Args:
            solution: Solution vector
            
        Returns:
            Constraint value (0 if satisfied, positive if violated)
        """
        try:
            if callable(self.expression):
                value = self.expression(solution)
            else:
                # For string expressions, would need expression parser
                # Simplified implementation for demo
                value = 0.0
            
            if self.type == ConstraintType.EQUALITY:
                return abs(value) if abs(value) > self.tolerance else 0.0
            elif self.type == ConstraintType.INEQUALITY:
                return max(0.0, value)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating constraint {self.name}: {e}")
            return float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "expression": self.expression if isinstance(self.expression, str) else str(self.expression),
            "tolerance": self.tolerance,
            "description": self.description,
            "penalty_weight": self.penalty_weight
        }


@dataclass
class OptimizationProblem:
    """Complete optimization problem definition.
    
    Attributes:
        name: Problem name
        variables: List of optimization variables
        objectives: List of optimization objectives
        constraints: List of constraints
        description: Optional problem description
        metadata: Additional metadata
    """
    name: str
    variables: List[Variable] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate problem definition after initialization."""
        # Convert dict inputs to proper objects if needed
        self.variables = [
            var if isinstance(var, Variable) else Variable(**var)
            for var in self.variables
        ]
        
        self.objectives = [
            obj if isinstance(obj, Objective) else Objective(**obj)
            for obj in self.objectives
        ]
        
        self.constraints = [
            const if isinstance(const, Constraint) else Constraint(**const)
            for const in self.constraints
        ]
        
        # Validate problem
        self._validate_problem()
    
    def _validate_problem(self):
        """Validate the complete problem definition."""
        # Check basic requirements
        if not self.variables:
            raise ValueError("Problem must have at least one variable")
        
        if not self.objectives:
            raise ValueError("Problem must have at least one objective")
        
        # Check for duplicate names
        var_names = [var.name for var in self.variables]
        if len(var_names) != len(set(var_names)):
            raise ValueError("Variable names must be unique")
        
        obj_names = [obj.name for obj in self.objectives]
        if len(obj_names) != len(set(obj_names)):
            raise ValueError("Objective names must be unique")
        
        const_names = [const.name for const in self.constraints]
        if len(const_names) != len(set(const_names)):
            raise ValueError("Constraint names must be unique")
    
    def get_dimension(self) -> int:
        """Get the total dimension of the problem.
        
        Returns:
            Total number of decision variables
        """
        total_dim = 0
        for var in self.variables:
            if var.type == VariableType.PERMUTATION:
                total_dim += var.bounds[0]
            else:
                total_dim += 1
        return total_dim
    
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective problem.
        
        Returns:
            True if multiple objectives, False otherwise
        """
        return len(self.objectives) > 1
    
    def is_constrained(self) -> bool:
        """Check if this is a constrained problem.
        
        Returns:
            True if constraints exist, False otherwise
        """
        return len(self.constraints) > 0
    
    def get_bounds(self) -> List[List[float]]:
        """Get variable bounds for the problem.
        
        Returns:
            List of [min, max] bounds for each variable
        """
        bounds = []
        for var in self.variables:
            if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
                bounds.append(var.bounds)
            elif var.type == VariableType.BINARY:
                bounds.append([0, 1])
            elif var.type == VariableType.CATEGORICAL:
                bounds.append([0, len(var.values) - 1])
            elif var.type == VariableType.PERMUTATION:
                # For permutation, each position can be any value
                for _ in range(var.bounds[0]):
                    bounds.append([0, var.bounds[0] - 1])
        return bounds
    
    def evaluate_constraints(self, solution: List[float]) -> Dict[str, float]:
        """Evaluate all constraints for a solution.
        
        Args:
            solution: Solution vector
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        for constraint in self.constraints:
            violations[constraint.name] = constraint.evaluate(solution)
        return violations
    
    def get_constraint_violation(self, solution: List[float]) -> float:
        """Get total constraint violation for a solution.
        
        Args:
            solution: Solution vector
            
        Returns:
            Total weighted constraint violation
        """
        total_violation = 0.0
        for constraint in self.constraints:
            violation = constraint.evaluate(solution)
            total_violation += violation * constraint.penalty_weight
        return total_violation
    
    def is_feasible(self, solution: List[float], tolerance: float = 1e-6) -> bool:
        """Check if a solution is feasible.
        
        Args:
            solution: Solution vector
            tolerance: Tolerance for constraint satisfaction
            
        Returns:
            True if feasible, False otherwise
        """
        for constraint in self.constraints:
            violation = constraint.evaluate(solution)
            if violation > tolerance:
                return False
        return True
    
    def get_complexity_estimate(self) -> Dict[str, Any]:
        """Estimate problem complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Calculate domain sizes
        total_domain_size = 1
        finite_domains = 0
        
        for var in self.variables:
            domain_size = var.get_domain_size()
            if domain_size != float('inf'):
                total_domain_size *= domain_size
                finite_domains += 1
        
        # Classify complexity
        dimension = self.get_dimension()
        
        if dimension <= 5 and len(self.objectives) == 1 and len(self.constraints) <= 2:
            complexity = "low"
        elif dimension <= 20 and len(self.objectives) <= 3 and len(self.constraints) <= 10:
            complexity = "medium"
        elif dimension <= 100 and len(self.objectives) <= 5 and len(self.constraints) <= 50:
            complexity = "high"
        else:
            complexity = "very_high"
        
        return {
            "complexity": complexity,
            "dimension": dimension,
            "objectives_count": len(self.objectives),
            "constraints_count": len(self.constraints),
            "total_domain_size": total_domain_size if finite_domains > 0 else float('inf'),
            "continuous_variables": len([v for v in self.variables if v.type == VariableType.CONTINUOUS]),
            "discrete_variables": len([v for v in self.variables if v.type == VariableType.DISCRETE]),
            "categorical_variables": len([v for v in self.variables if v.type == VariableType.CATEGORICAL]),
            "is_multi_objective": self.is_multi_objective(),
            "is_constrained": self.is_constrained()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "variables": [var.to_dict() for var in self.variables],
            "objectives": [obj.to_dict() for obj in self.objectives],
            "constraints": [const.to_dict() for const in self.constraints],
            "metadata": self.metadata,
            "complexity": self.get_complexity_estimate()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string representation.
        
        Args:
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationProblem":
        """Create OptimizationProblem from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            OptimizationProblem instance
        """
        return cls(
            name=data["name"],
            description=data.get("description"),
            variables=data.get("variables", []),
            objectives=data.get("objectives", []),
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "OptimizationProblem":
        """Create OptimizationProblem from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            OptimizationProblem instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def copy(self) -> "OptimizationProblem":
        """Create a copy of the problem.
        
        Returns:
            New OptimizationProblem instance
        """
        return self.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """String representation of the problem."""
        return (f"OptimizationProblem(name='{self.name}', "
                f"variables={len(self.variables)}, "
                f"objectives={len(self.objectives)}, "
                f"constraints={len(self.constraints)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()