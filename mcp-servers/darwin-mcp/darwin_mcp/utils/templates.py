"""
Darwin Template Manager

This module provides template management for common optimization problems
in the Darwin MCP server. Templates simplify problem setup by providing
pre-configured optimization scenarios.

Example usage:
    from darwin_mcp.utils.templates import TemplateManager
    
    manager = TemplateManager()
    template = await manager.create_template("portfolio_optimization", {
        "num_assets": 5,
        "risk_tolerance": 0.1
    })
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages optimization problem templates."""
    
    def __init__(self):
        """Initialize template manager."""
        self.templates = {
            "function_optimization": self._function_optimization_template,
            "portfolio_optimization": self._portfolio_optimization_template,
            "neural_network_tuning": self._neural_network_tuning_template,
            "scheduling": self._scheduling_template,
            "design_optimization": self._design_optimization_template
        }
    
    async def create_template(self, template_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization template.
        
        Args:
            template_type: Type of template to create
            parameters: Template-specific parameters
            
        Returns:
            Complete optimization problem template
        """
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return self.templates[template_type](parameters)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates.
        
        Returns:
            List of template descriptions
        """
        return [
            {
                "name": "function_optimization",
                "description": "Mathematical function optimization",
                "parameters": ["function_name", "dimensions", "bounds"]
            },
            {
                "name": "portfolio_optimization", 
                "description": "Financial portfolio optimization",
                "parameters": ["num_assets", "risk_tolerance", "target_return"]
            },
            {
                "name": "neural_network_tuning",
                "description": "Neural network hyperparameter tuning",
                "parameters": ["model_type", "dataset_size", "max_layers"]
            },
            {
                "name": "scheduling",
                "description": "Task scheduling optimization",
                "parameters": ["num_tasks", "num_resources", "time_horizon"]
            },
            {
                "name": "design_optimization",
                "description": "Engineering design optimization",
                "parameters": ["design_type", "constraints", "objectives"]
            }
        ]
    
    def _function_optimization_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create function optimization template."""
        function_name = params.get("function_name", "sphere")
        dimensions = params.get("dimensions", 2)
        bounds = params.get("bounds", [-5, 5])
        
        return {
            "name": f"{function_name.title()} Function Optimization",
            "variables": [
                {
                    "name": f"x{i+1}",
                    "type": "continuous",
                    "bounds": bounds,
                    "description": f"Decision variable {i+1}"
                }
                for i in range(dimensions)
            ],
            "objectives": [
                {
                    "name": "minimize_function",
                    "type": "minimize",
                    "function": function_name,
                    "description": f"Minimize {function_name} function"
                }
            ],
            "constraints": [],
            "config": {
                "algorithm": "genetic",
                "population_size": max(50, dimensions * 10),
                "max_generations": max(100, dimensions * 20),
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        }
    
    def _portfolio_optimization_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio optimization template."""
        num_assets = params.get("num_assets", 5)
        risk_tolerance = params.get("risk_tolerance", 0.1)
        target_return = params.get("target_return", 0.08)
        
        return {
            "name": "Portfolio Optimization",
            "variables": [
                {
                    "name": f"weight_asset_{i+1}",
                    "type": "continuous", 
                    "bounds": [0.0, 1.0],
                    "description": f"Weight allocation for asset {i+1}"
                }
                for i in range(num_assets)
            ],
            "objectives": [
                {
                    "name": "maximize_return",
                    "type": "maximize",
                    "function": "portfolio_return",
                    "description": "Maximize expected portfolio return"
                },
                {
                    "name": "minimize_risk",
                    "type": "minimize", 
                    "function": "portfolio_risk",
                    "description": "Minimize portfolio risk (variance)"
                }
            ],
            "constraints": [
                {
                    "name": "weight_sum",
                    "type": "equality",
                    "expression": "sum(weights) == 1.0",
                    "description": "Weights must sum to 1"
                },
                {
                    "name": "risk_limit",
                    "type": "inequality",
                    "expression": f"portfolio_risk <= {risk_tolerance}",
                    "description": f"Risk must not exceed {risk_tolerance}"
                }
            ],
            "config": {
                "algorithm": "nsga2",
                "population_size": 100,
                "max_generations": 200,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        }
    
    def _neural_network_tuning_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create neural network tuning template."""
        model_type = params.get("model_type", "feedforward")
        dataset_size = params.get("dataset_size", "medium")
        max_layers = params.get("max_layers", 5)
        
        return {
            "name": "Neural Network Hyperparameter Tuning",
            "variables": [
                {
                    "name": "learning_rate",
                    "type": "continuous",
                    "bounds": [0.001, 0.1],
                    "description": "Learning rate for training"
                },
                {
                    "name": "batch_size",
                    "type": "categorical",
                    "values": [16, 32, 64, 128, 256],
                    "description": "Training batch size"
                },
                {
                    "name": "hidden_layers",
                    "type": "discrete",
                    "bounds": [1, max_layers],
                    "description": "Number of hidden layers"
                },
                {
                    "name": "neurons_per_layer",
                    "type": "discrete", 
                    "bounds": [10, 500],
                    "description": "Neurons per hidden layer"
                },
                {
                    "name": "dropout_rate",
                    "type": "continuous",
                    "bounds": [0.0, 0.5],
                    "description": "Dropout rate for regularization"
                }
            ],
            "objectives": [
                {
                    "name": "maximize_accuracy",
                    "type": "maximize",
                    "function": "validation_accuracy",
                    "description": "Maximize validation accuracy"
                },
                {
                    "name": "minimize_training_time",
                    "type": "minimize",
                    "function": "training_time",
                    "description": "Minimize training time"
                }
            ],
            "constraints": [
                {
                    "name": "memory_limit",
                    "type": "inequality",
                    "expression": "estimated_memory <= 8000",
                    "description": "Memory usage must not exceed 8GB"
                }
            ],
            "config": {
                "algorithm": "nsga2",
                "population_size": 50,
                "max_generations": 100,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8
            }
        }
    
    def _scheduling_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create scheduling optimization template."""
        num_tasks = params.get("num_tasks", 10)
        num_resources = params.get("num_resources", 3)
        time_horizon = params.get("time_horizon", 24)
        
        return {
            "name": "Task Scheduling Optimization",
            "variables": [
                {
                    "name": f"task_{i+1}_start_time",
                    "type": "continuous",
                    "bounds": [0, time_horizon],
                    "description": f"Start time for task {i+1}"
                }
                for i in range(num_tasks)
            ] + [
                {
                    "name": f"task_{i+1}_resource",
                    "type": "discrete",
                    "bounds": [1, num_resources],
                    "description": f"Assigned resource for task {i+1}"
                }
                for i in range(num_tasks)
            ],
            "objectives": [
                {
                    "name": "minimize_makespan",
                    "type": "minimize",
                    "function": "calculate_makespan",
                    "description": "Minimize total completion time"
                },
                {
                    "name": "minimize_resource_conflicts",
                    "type": "minimize",
                    "function": "resource_conflicts",
                    "description": "Minimize resource conflicts"
                }
            ],
            "constraints": [
                {
                    "name": "task_precedence",
                    "type": "inequality",
                    "expression": "precedence_constraints",
                    "description": "Respect task precedence relationships"
                },
                {
                    "name": "resource_capacity",
                    "type": "inequality",
                    "expression": "resource_capacity_constraints",
                    "description": "Respect resource capacity limits"
                }
            ],
            "config": {
                "algorithm": "genetic",
                "population_size": 100,
                "max_generations": 300,
                "mutation_rate": 0.2,
                "crossover_rate": 0.8
            }
        }
    
    def _design_optimization_template(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create design optimization template."""
        design_type = params.get("design_type", "structural")
        constraints = params.get("constraints", ["stress", "deflection"])
        objectives = params.get("objectives", ["weight", "cost"])
        
        variables = []
        if design_type == "structural":
            variables = [
                {
                    "name": "beam_width",
                    "type": "continuous",
                    "bounds": [10, 100],
                    "description": "Beam width in mm"
                },
                {
                    "name": "beam_height", 
                    "type": "continuous",
                    "bounds": [20, 200],
                    "description": "Beam height in mm"
                },
                {
                    "name": "material_grade",
                    "type": "categorical",
                    "values": ["steel_a36", "steel_a572", "aluminum_6061"],
                    "description": "Material grade selection"
                }
            ]
        
        return {
            "name": f"{design_type.title()} Design Optimization",
            "variables": variables,
            "objectives": [
                {
                    "name": f"minimize_{obj}",
                    "type": "minimize",
                    "function": f"calculate_{obj}",
                    "description": f"Minimize {obj}"
                }
                for obj in objectives
            ],
            "constraints": [
                {
                    "name": f"{const}_constraint",
                    "type": "inequality", 
                    "expression": f"{const}_limit",
                    "description": f"Respect {const} limits"
                }
                for const in constraints
            ],
            "config": {
                "algorithm": "genetic",
                "population_size": 80,
                "max_generations": 200,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        }