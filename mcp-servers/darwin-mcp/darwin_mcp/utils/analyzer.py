"""
Darwin Problem Analyzer

This module provides problem analysis capabilities for the Darwin MCP server.
It analyzes optimization problem complexity, provides recommendations, and
estimates computational requirements.

Example usage:
    from darwin_mcp.utils.analyzer import ProblemAnalyzer
    
    analyzer = ProblemAnalyzer()
    analysis = await analyzer.analyze(variables, objectives, constraints)
"""

import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ProblemAnalyzer:
    """Analyzes optimization problem complexity and provides recommendations."""
    
    def __init__(self):
        """Initialize problem analyzer."""
        self.complexity_thresholds = {
            "variables": {"low": 5, "medium": 20, "high": 100},
            "objectives": {"low": 1, "medium": 3, "high": 10},
            "constraints": {"low": 2, "medium": 10, "high": 50}
        }
    
    async def analyze(self, variables: List[Dict], objectives: List[Dict], 
                     constraints: List[Dict] = None) -> Dict[str, Any]:
        """Analyze optimization problem.
        
        Args:
            variables: Problem variables
            objectives: Problem objectives
            constraints: Problem constraints
            
        Returns:
            Analysis results and recommendations
        """
        if constraints is None:
            constraints = []
        
        analysis = {
            "complexity": self._analyze_complexity(variables, objectives, constraints),
            "problem_type": self._classify_problem_type(variables, objectives, constraints),
            "computational_requirements": self._estimate_computational_requirements(variables, objectives),
            "algorithm_recommendations": self._recommend_algorithms(variables, objectives, constraints),
            "parameter_recommendations": self._recommend_parameters(variables, objectives, constraints),
            "potential_issues": self._identify_potential_issues(variables, objectives, constraints),
            "estimated_runtime": self._estimate_runtime(variables, objectives, constraints)
        }
        
        return analysis
    
    def _analyze_complexity(self, variables: List[Dict], objectives: List[Dict], 
                           constraints: List[Dict]) -> Dict[str, Any]:
        """Analyze problem complexity."""
        var_count = len(variables)
        obj_count = len(objectives)
        const_count = len(constraints)
        
        # Classify each dimension
        var_complexity = self._classify_dimension(var_count, "variables")
        obj_complexity = self._classify_dimension(obj_count, "objectives")
        const_complexity = self._classify_dimension(const_count, "constraints")
        
        # Calculate search space size
        search_space_size = self._calculate_search_space_size(variables)
        
        # Determine overall complexity
        complexity_scores = {
            "low": 1, "medium": 2, "high": 3, "very_high": 4
        }
        
        score_map = {"low": 1, "medium": 2, "high": 3}
        total_score = (score_map.get(var_complexity, 3) + 
                      score_map.get(obj_complexity, 3) + 
                      score_map.get(const_complexity, 1))
        
        if total_score <= 3:
            overall = "low"
        elif total_score <= 5:
            overall = "medium"
        elif total_score <= 7:
            overall = "high"
        else:
            overall = "very_high"
        
        return {
            "overall": overall,
            "variables": {"count": var_count, "complexity": var_complexity},
            "objectives": {"count": obj_count, "complexity": obj_complexity},
            "constraints": {"count": const_count, "complexity": const_complexity},
            "search_space_size": search_space_size,
            "is_multi_objective": obj_count > 1,
            "is_constrained": const_count > 0
        }
    
    def _classify_dimension(self, count: int, dimension: str) -> str:
        """Classify dimension complexity."""
        thresholds = self.complexity_thresholds[dimension]
        
        if count <= thresholds["low"]:
            return "low"
        elif count <= thresholds["medium"]:
            return "medium"
        else:
            return "high"
    
    def _calculate_search_space_size(self, variables: List[Dict]) -> float:
        """Calculate search space size."""
        total_size = 1.0
        
        for var in variables:
            var_type = var.get("type", "continuous")
            
            if var_type == "continuous":
                total_size *= float('inf')
                break
            elif var_type == "discrete":
                bounds = var.get("bounds", [0, 10])
                size = bounds[1] - bounds[0] + 1
                total_size *= size
            elif var_type == "categorical":
                values = var.get("values", [])
                total_size *= len(values)
            elif var_type == "binary":
                total_size *= 2
            elif var_type == "permutation":
                bounds = var.get("bounds", [5])
                total_size *= math.factorial(bounds[0])
        
        return total_size
    
    def _classify_problem_type(self, variables: List[Dict], objectives: List[Dict], 
                              constraints: List[Dict]) -> Dict[str, Any]:
        """Classify problem type."""
        # Variable type analysis
        var_types = [var.get("type", "continuous") for var in variables]
        continuous_count = var_types.count("continuous")
        discrete_count = var_types.count("discrete")
        categorical_count = var_types.count("categorical")
        binary_count = var_types.count("binary")
        
        # Problem classification
        if len(objectives) == 1:
            if len(constraints) == 0:
                problem_class = "single_objective_unconstrained"
            else:
                problem_class = "single_objective_constrained"
        else:
            if len(constraints) == 0:
                problem_class = "multi_objective_unconstrained"
            else:
                problem_class = "multi_objective_constrained"
        
        # Variable type classification
        if continuous_count == len(variables):
            variable_class = "continuous"
        elif discrete_count + categorical_count + binary_count == len(variables):
            variable_class = "discrete"
        else:
            variable_class = "mixed"
        
        return {
            "problem_class": problem_class,
            "variable_class": variable_class,
            "variable_distribution": {
                "continuous": continuous_count,
                "discrete": discrete_count,
                "categorical": categorical_count,
                "binary": binary_count
            }
        }
    
    def _estimate_computational_requirements(self, variables: List[Dict], 
                                           objectives: List[Dict]) -> Dict[str, Any]:
        """Estimate computational requirements."""
        var_count = len(variables)
        obj_count = len(objectives)
        
        # Base computational complexity
        base_complexity = var_count * obj_count
        
        # Estimate evaluations needed
        if obj_count == 1:
            if var_count <= 10:
                evaluations = var_count * 1000
            elif var_count <= 50:
                evaluations = var_count * 2000
            else:
                evaluations = var_count * 5000
        else:
            # Multi-objective typically needs more evaluations
            evaluations = max(10000, var_count * obj_count * 1000)
        
        # Memory requirements (rough estimate)
        population_size = max(50, var_count * 10)
        memory_mb = (population_size * var_count * 8) / (1024 * 1024)  # 8 bytes per float
        
        return {
            "estimated_evaluations": evaluations,
            "estimated_memory_mb": memory_mb,
            "recommended_population_size": population_size,
            "computational_complexity": "O(n²)" if obj_count > 1 else "O(n)",
            "parallel_efficiency": "high" if var_count > 10 else "medium"
        }
    
    def _recommend_algorithms(self, variables: List[Dict], objectives: List[Dict], 
                             constraints: List[Dict]) -> List[Dict[str, Any]]:
        """Recommend suitable algorithms."""
        recommendations = []
        
        obj_count = len(objectives)
        var_count = len(variables)
        const_count = len(constraints)
        
        if obj_count == 1:
            # Single-objective algorithms
            recommendations.append({
                "algorithm": "genetic",
                "suitability": "high",
                "reason": "Good general-purpose algorithm for single-objective problems",
                "config_suggestions": {
                    "population_size": max(50, var_count * 5),
                    "max_generations": max(100, var_count * 10)
                }
            })
            
            if const_count == 0:
                recommendations.append({
                    "algorithm": "differential_evolution",
                    "suitability": "high",
                    "reason": "Excellent for unconstrained continuous optimization",
                    "config_suggestions": {
                        "population_size": var_count * 10,
                        "max_generations": 200
                    }
                })
        else:
            # Multi-objective algorithms
            recommendations.append({
                "algorithm": "nsga2",
                "suitability": "high",
                "reason": "Proven multi-objective algorithm with good convergence",
                "config_suggestions": {
                    "population_size": max(100, var_count * 10),
                    "max_generations": max(200, var_count * 20)
                }
            })
            
            if obj_count <= 3:
                recommendations.append({
                    "algorithm": "nsga3",
                    "suitability": "medium",
                    "reason": "Good for 3+ objectives with reference points",
                    "config_suggestions": {
                        "population_size": max(120, var_count * 12),
                        "max_generations": max(300, var_count * 25)
                    }
                })
        
        return recommendations
    
    def _recommend_parameters(self, variables: List[Dict], objectives: List[Dict], 
                             constraints: List[Dict]) -> Dict[str, Any]:
        """Recommend algorithm parameters."""
        var_count = len(variables)
        obj_count = len(objectives)
        const_count = len(constraints)
        
        # Base recommendations
        if obj_count == 1:
            population_size = max(50, var_count * 5)
            max_generations = max(100, var_count * 10)
        else:
            population_size = max(100, var_count * 10)
            max_generations = max(200, var_count * 20)
        
        # Adjust for constraints
        if const_count > 0:
            population_size = int(population_size * 1.5)
            max_generations = int(max_generations * 1.2)
        
        # Mutation and crossover rates
        if var_count <= 10:
            mutation_rate = 0.1
            crossover_rate = 0.8
        elif var_count <= 50:
            mutation_rate = 0.15
            crossover_rate = 0.7
        else:
            mutation_rate = 0.2
            crossover_rate = 0.6
        
        return {
            "population_size": population_size,
            "max_generations": max_generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "tournament_size": min(5, max(3, population_size // 20)),
            "elitism": max(1, population_size // 50),
            "early_stopping": True,
            "convergence_threshold": 1e-6 if var_count <= 20 else 1e-4
        }
    
    def _identify_potential_issues(self, variables: List[Dict], objectives: List[Dict], 
                                  constraints: List[Dict]) -> List[Dict[str, str]]:
        """Identify potential optimization issues."""
        issues = []
        
        var_count = len(variables)
        obj_count = len(objectives)
        const_count = len(constraints)
        
        # High dimensionality
        if var_count > 100:
            issues.append({
                "type": "warning",
                "message": f"High dimensionality ({var_count} variables) may lead to slow convergence",
                "recommendation": "Consider dimensionality reduction or parallel execution"
            })
        
        # Many objectives
        if obj_count > 5:
            issues.append({
                "type": "warning",
                "message": f"Many objectives ({obj_count}) can make convergence difficult",
                "recommendation": "Consider objective reduction or preference-based approaches"
            })
        
        # Many constraints
        if const_count > 20:
            issues.append({
                "type": "warning",
                "message": f"Many constraints ({const_count}) may create small feasible regions",
                "recommendation": "Use constraint handling techniques and larger populations"
            })
        
        # Mixed variable types
        var_types = set(var.get("type", "continuous") for var in variables)
        if len(var_types) > 2:
            issues.append({
                "type": "info",
                "message": "Mixed variable types require specialized operators",
                "recommendation": "Ensure algorithm supports mixed variable types"
            })
        
        # Unbounded variables
        unbounded = [var for var in variables 
                    if var.get("type") == "continuous" and not var.get("bounds")]
        if unbounded:
            issues.append({
                "type": "error",
                "message": f"{len(unbounded)} continuous variables lack bounds",
                "recommendation": "Define reasonable bounds for all continuous variables"
            })
        
        return issues
    
    def _estimate_runtime(self, variables: List[Dict], objectives: List[Dict], 
                         constraints: List[Dict]) -> Dict[str, Any]:
        """Estimate optimization runtime."""
        var_count = len(variables)
        obj_count = len(objectives)
        const_count = len(constraints)
        
        # Base time per evaluation (seconds)
        base_eval_time = 0.001
        
        # Adjust for problem complexity
        eval_time = base_eval_time * (1 + var_count * 0.1) * (1 + obj_count * 0.5)
        if const_count > 0:
            eval_time *= (1 + const_count * 0.1)
        
        # Estimate total evaluations
        if obj_count == 1:
            total_evaluations = max(5000, var_count * 1000)
        else:
            total_evaluations = max(10000, var_count * obj_count * 1000)
        
        # Total runtime estimate
        total_time = eval_time * total_evaluations
        
        # Convert to readable format
        if total_time < 60:
            runtime_str = f"{total_time:.1f} seconds"
        elif total_time < 3600:
            runtime_str = f"{total_time/60:.1f} minutes"
        else:
            runtime_str = f"{total_time/3600:.1f} hours"
        
        return {
            "estimated_total_time_seconds": total_time,
            "estimated_time_readable": runtime_str,
            "time_per_evaluation_seconds": eval_time,
            "estimated_total_evaluations": total_evaluations,
            "confidence": "medium"  # Estimates are rough
        }