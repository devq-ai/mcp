"""
Darwin Result Visualizer

This module provides visualization utilities for the Darwin MCP server.
It creates charts and graphs for optimization results, evolution history,
and performance analysis.

Example usage:
    from darwin_mcp.utils.visualizer import ResultVisualizer
    
    visualizer = ResultVisualizer()
    chart = await visualizer.create_visualization(optimization, "convergence", "png")
"""

import logging
import base64
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Creates visualizations for optimization results."""
    
    def __init__(self):
        """Initialize result visualizer."""
        self.chart_types = {
            "convergence": self._create_convergence_chart,
            "pareto_frontier": self._create_pareto_chart,
            "variable_distribution": self._create_variable_distribution,
            "objective_space": self._create_objective_space,
            "diversity": self._create_diversity_chart
        }
    
    async def create_visualization(self, optimization: Dict[str, Any], 
                                 chart_type: str, format_type: str = "png") -> Dict[str, Any]:
        """Create visualization for optimization results.
        
        Args:
            optimization: Optimization data
            chart_type: Type of chart to create
            format_type: Output format (png, svg, html)
            
        Returns:
            Visualization data
        """
        if chart_type not in self.chart_types:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        chart_data = self.chart_types[chart_type](optimization, format_type)
        
        return {
            "chart_type": chart_type,
            "format": format_type,
            "data": chart_data,
            "title": self._get_chart_title(chart_type, optimization),
            "description": self._get_chart_description(chart_type)
        }
    
    def _create_convergence_chart(self, optimization: Dict[str, Any], 
                                format_type: str) -> str:
        """Create convergence chart."""
        results = optimization.get("results", {})
        history = results.get("evolution_history", {})
        
        generations = history.get("generations", [])
        best_fitness = history.get("best_fitness", [])
        avg_fitness = history.get("average_fitness", [])
        
        # Mock chart creation (in real implementation, would use matplotlib/bokeh)
        chart_data = {
            "type": "line",
            "data": {
                "labels": generations,
                "datasets": [
                    {
                        "label": "Best Fitness",
                        "data": best_fitness,
                        "color": "blue"
                    },
                    {
                        "label": "Average Fitness", 
                        "data": avg_fitness,
                        "color": "red"
                    }
                ]
            },
            "options": {
                "title": "Fitness Convergence",
                "xlabel": "Generation",
                "ylabel": "Fitness Value"
            }
        }
        
        if format_type == "png":
            # Mock base64 encoded image
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        elif format_type == "svg":
            return "<svg><!-- Mock SVG chart --></svg>"
        else:
            return str(chart_data)
    
    def _create_pareto_chart(self, optimization: Dict[str, Any], 
                           format_type: str) -> str:
        """Create Pareto frontier chart."""
        results = optimization.get("results", {})
        pareto_frontier = results.get("pareto_frontier", [])
        
        if not pareto_frontier:
            return "No Pareto frontier data available"
        
        # Extract objective values
        objectives = [pf.get("objectives", []) for pf in pareto_frontier]
        
        chart_data = {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": "Pareto Frontier",
                    "data": [{"x": obj[0], "y": obj[1]} for obj in objectives if len(obj) >= 2],
                    "color": "green"
                }]
            },
            "options": {
                "title": "Pareto Frontier",
                "xlabel": "Objective 1",
                "ylabel": "Objective 2"
            }
        }
        
        if format_type == "png":
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        elif format_type == "svg":
            return "<svg><!-- Mock Pareto SVG chart --></svg>"
        else:
            return str(chart_data)
    
    def _create_variable_distribution(self, optimization: Dict[str, Any], 
                                    format_type: str) -> str:
        """Create variable distribution chart."""
        results = optimization.get("results", {})
        final_population = results.get("final_population", [])
        
        if not final_population:
            return "No population data available"
        
        # Calculate variable statistics
        pop_array = np.array(final_population)
        var_means = np.mean(pop_array, axis=0).tolist()
        var_stds = np.std(pop_array, axis=0).tolist()
        
        chart_data = {
            "type": "bar",
            "data": {
                "labels": [f"Var {i+1}" for i in range(len(var_means))],
                "datasets": [{
                    "label": "Mean Values",
                    "data": var_means,
                    "color": "blue"
                }]
            },
            "options": {
                "title": "Variable Distributions",
                "xlabel": "Variables",
                "ylabel": "Mean Value"
            }
        }
        
        if format_type == "png":
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        elif format_type == "svg":
            return "<svg><!-- Mock variable distribution SVG --></svg>"
        else:
            return str(chart_data)
    
    def _create_objective_space(self, optimization: Dict[str, Any], 
                              format_type: str) -> str:
        """Create objective space visualization."""
        results = optimization.get("results", {})
        final_population = results.get("final_population", [])
        fitness_values = results.get("final_fitness_values", [])
        
        chart_data = {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": "Population",
                    "data": [{"x": i, "y": fit} for i, fit in enumerate(fitness_values)],
                    "color": "purple"
                }]
            },
            "options": {
                "title": "Objective Space",
                "xlabel": "Individual Index",
                "ylabel": "Fitness Value"
            }
        }
        
        if format_type == "png":
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        elif format_type == "svg":
            return "<svg><!-- Mock objective space SVG --></svg>"
        else:
            return str(chart_data)
    
    def _create_diversity_chart(self, optimization: Dict[str, Any], 
                              format_type: str) -> str:
        """Create population diversity chart."""
        results = optimization.get("results", {})
        history = results.get("evolution_history", {})
        
        generations = history.get("generations", [])
        diversity = history.get("diversity", [])
        
        chart_data = {
            "type": "line",
            "data": {
                "labels": generations,
                "datasets": [{
                    "label": "Population Diversity",
                    "data": diversity,
                    "color": "orange"
                }]
            },
            "options": {
                "title": "Population Diversity Over Time",
                "xlabel": "Generation",
                "ylabel": "Diversity Index"
            }
        }
        
        if format_type == "png":
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        elif format_type == "svg":
            return "<svg><!-- Mock diversity SVG chart --></svg>"
        else:
            return str(chart_data)
    
    def _get_chart_title(self, chart_type: str, optimization: Dict[str, Any]) -> str:
        """Get chart title."""
        problem_name = optimization.get("name", "Optimization")
        
        titles = {
            "convergence": f"{problem_name} - Convergence Analysis",
            "pareto_frontier": f"{problem_name} - Pareto Frontier",
            "variable_distribution": f"{problem_name} - Variable Distribution",
            "objective_space": f"{problem_name} - Objective Space",
            "diversity": f"{problem_name} - Population Diversity"
        }
        
        return titles.get(chart_type, f"{problem_name} - Visualization")
    
    def _get_chart_description(self, chart_type: str) -> str:
        """Get chart description."""
        descriptions = {
            "convergence": "Shows how fitness values evolve over generations",
            "pareto_frontier": "Displays Pareto-optimal solutions for multi-objective problems",
            "variable_distribution": "Shows distribution of variable values in final population",
            "objective_space": "Visualizes the objective space and solution quality",
            "diversity": "Tracks population diversity throughout evolution"
        }
        
        return descriptions.get(chart_type, "Optimization visualization")


class ValidationHelper:
    """Helper for data validation."""
    
    @staticmethod
    def validate_problem_definition(problem: Dict[str, Any]) -> List[str]:
        """Validate problem definition."""
        errors = []
        
        if "variables" not in problem or not problem["variables"]:
            errors.append("Problem must have at least one variable")
        
        if "objectives" not in problem or not problem["objectives"]:
            errors.append("Problem must have at least one objective")
        
        # Validate variables
        for i, var in enumerate(problem.get("variables", [])):
            if "name" not in var:
                errors.append(f"Variable {i+1} missing name")
            
            if "type" not in var:
                errors.append(f"Variable {i+1} missing type")
            
            var_type = var.get("type")
            if var_type in ["continuous", "discrete"] and "bounds" not in var:
                errors.append(f"Variable {i+1} of type {var_type} missing bounds")
            
            if var_type == "categorical" and "values" not in var:
                errors.append(f"Variable {i+1} of type categorical missing values")
        
        # Validate objectives
        for i, obj in enumerate(problem.get("objectives", [])):
            if "name" not in obj:
                errors.append(f"Objective {i+1} missing name")
            
            if "type" not in obj:
                errors.append(f"Objective {i+1} missing type")
            
            if obj.get("type") not in ["minimize", "maximize"]:
                errors.append(f"Objective {i+1} type must be 'minimize' or 'maximize'")
        
        return errors
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate algorithm configuration."""
        errors = []
        
        if "population_size" in config:
            if not isinstance(config["population_size"], int) or config["population_size"] <= 0:
                errors.append("Population size must be a positive integer")
        
        if "max_generations" in config:
            if not isinstance(config["max_generations"], int) or config["max_generations"] <= 0:
                errors.append("Max generations must be a positive integer")
        
        if "mutation_rate" in config:
            rate = config["mutation_rate"]
            if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                errors.append("Mutation rate must be between 0 and 1")
        
        if "crossover_rate" in config:
            rate = config["crossover_rate"]
            if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                errors.append("Crossover rate must be between 0 and 1")
        
        return errors


class ConversionHelper:
    """Helper for data conversion."""
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ConversionHelper.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ConversionHelper.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def ensure_serializable(data: Any) -> Any:
        """Ensure data is JSON serializable."""
        try:
            import json
            json.dumps(data)
            return data
        except TypeError:
            return ConversionHelper.convert_numpy_types(data)
    
    @staticmethod
    def normalize_bounds(bounds: List[Any]) -> List[float]:
        """Normalize bounds to float values."""
        try:
            return [float(bounds[0]), float(bounds[1])]
        except (ValueError, IndexError, TypeError):
            return [0.0, 1.0]
    
    @staticmethod
    def format_time_duration(seconds: float) -> str:
        """Format time duration for display."""
        if seconds < 1:
            return f"{seconds*1000:.1f} ms"
        elif seconds < 60:
            return f"{seconds:.1f} s"
        elif seconds < 3600:
            return f"{seconds/60:.1f} min"
        else:
            return f"{seconds/3600:.1f} h"