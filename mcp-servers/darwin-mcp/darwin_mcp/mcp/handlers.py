"""
Darwin MCP Handlers

This module implements the request handlers for the Darwin MCP server,
providing the core functionality for genetic algorithm optimization
through the Model Context Protocol.

The handlers manage:
- Optimization problem creation and configuration
- Algorithm execution and monitoring
- Result analysis and visualization
- Template management and problem analysis
- System status and health monitoring
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np

from darwin_mcp.optimization.engine import OptimizationEngine
from darwin_mcp.optimization.problem import OptimizationProblem
from darwin_mcp.algorithms.genetic import GeneticAlgorithm
from darwin_mcp.utils.templates import TemplateManager
from darwin_mcp.utils.analyzer import ProblemAnalyzer
from darwin_mcp.utils.visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


class DarwinMCPHandlers:
    """Darwin MCP request handlers."""
    
    def __init__(self):
        """Initialize the handlers."""
        self.optimizations = {}  # Store active optimizations
        self.engine = OptimizationEngine()
        self.template_manager = TemplateManager()
        self.problem_analyzer = ProblemAnalyzer()
        self.visualizer = ResultVisualizer()
        
    async def create_optimization(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new optimization problem.
        
        Args:
            arguments: Optimization definition containing:
                - name: Problem name
                - variables: List of variable definitions
                - objectives: List of objective definitions
                - constraints: Optional list of constraints
                - config: Optional algorithm configuration
                
        Returns:
            Dictionary with optimization ID and metadata
        """
        try:
            # Extract required fields
            name = arguments.get("name", f"optimization_{uuid.uuid4().hex[:8]}")
            variables = arguments.get("variables", [])
            objectives = arguments.get("objectives", [])
            constraints = arguments.get("constraints", [])
            config = arguments.get("config", {})
            
            # Validate inputs
            if not variables:
                raise ValueError("At least one variable must be defined")
            if not objectives:
                raise ValueError("At least one objective must be defined")
            
            # Create optimization problem
            problem = OptimizationProblem(
                name=name,
                variables=variables,
                objectives=objectives,
                constraints=constraints
            )
            
            # Generate unique ID
            optimization_id = f"opt_{uuid.uuid4().hex}"
            
            # Store optimization
            self.optimizations[optimization_id] = {
                "id": optimization_id,
                "name": name,
                "problem": problem,
                "config": config,
                "status": "created",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": None,
                "progress": None
            }
            
            logger.info(f"Created optimization {optimization_id}: {name}")
            
            return {
                "success": True,
                "optimization_id": optimization_id,
                "name": name,
                "status": "created",
                "variables_count": len(variables),
                "objectives_count": len(objectives),
                "constraints_count": len(constraints),
                "created_at": self.optimizations[optimization_id]["created_at"],
                "estimated_complexity": self._estimate_complexity(variables, objectives, constraints)
            }
            
        except Exception as e:
            logger.error(f"Error creating optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "creation_error"
            }
    
    async def run_optimization(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an optimization.
        
        Args:
            arguments: Contains:
                - optimization_id: ID of optimization to run
                - async_mode: Whether to run asynchronously
                
        Returns:
            Optimization execution results
        """
        try:
            optimization_id = arguments.get("optimization_id")
            async_mode = arguments.get("async_mode", False)
            
            if not optimization_id or optimization_id not in self.optimizations:
                raise ValueError(f"Optimization {optimization_id} not found")
            
            optimization = self.optimizations[optimization_id]
            
            if optimization["status"] == "running":
                return {
                    "success": False,
                    "error": "Optimization is already running",
                    "status": "running"
                }
            
            # Update status
            optimization["status"] = "running"
            optimization["started_at"] = datetime.now(timezone.utc).isoformat()
            
            if async_mode:
                # Start optimization in background
                asyncio.create_task(self._run_optimization_async(optimization_id))
                
                return {
                    "success": True,
                    "optimization_id": optimization_id,
                    "status": "running",
                    "message": "Optimization started asynchronously",
                    "started_at": optimization["started_at"]
                }
            else:
                # Run synchronously
                results = await self._execute_optimization(optimization_id)
                
                return {
                    "success": True,
                    "optimization_id": optimization_id,
                    "status": "completed",
                    "results": results,
                    "execution_time": results.get("execution_time", 0)
                }
                
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "execution_error"
            }
    
    async def get_optimization_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization status.
        
        Args:
            arguments: Contains optimization_id
            
        Returns:
            Current optimization status and progress
        """
        try:
            optimization_id = arguments.get("optimization_id")
            
            if not optimization_id or optimization_id not in self.optimizations:
                raise ValueError(f"Optimization {optimization_id} not found")
            
            optimization = self.optimizations[optimization_id]
            
            status_info = {
                "optimization_id": optimization_id,
                "name": optimization["name"],
                "status": optimization["status"],
                "created_at": optimization["created_at"]
            }
            
            # Add timing information
            if "started_at" in optimization:
                status_info["started_at"] = optimization["started_at"]
            
            if "completed_at" in optimization:
                status_info["completed_at"] = optimization["completed_at"]
            
            # Add progress information
            if optimization["progress"]:
                status_info["progress"] = optimization["progress"]
            
            # Add results summary if completed
            if optimization["status"] == "completed" and optimization["results"]:
                results = optimization["results"]
                status_info["best_fitness"] = results.get("best_fitness")
                status_info["generations_run"] = results.get("generations_run")
                status_info["execution_time"] = results.get("execution_time")
            
            return {
                "success": True,
                **status_info
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "status_error"
            }
    
    async def get_optimization_results(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization results.
        
        Args:
            arguments: Contains:
                - optimization_id: ID of optimization
                - include_history: Whether to include evolution history
                - include_visualization: Whether to include visualizations
                
        Returns:
            Optimization results and analysis
        """
        try:
            optimization_id = arguments.get("optimization_id")
            include_history = arguments.get("include_history", False)
            include_visualization = arguments.get("include_visualization", False)
            
            if not optimization_id or optimization_id not in self.optimizations:
                raise ValueError(f"Optimization {optimization_id} not found")
            
            optimization = self.optimizations[optimization_id]
            
            if optimization["status"] != "completed":
                return {
                    "success": False,
                    "error": f"Optimization is not completed (status: {optimization['status']})",
                    "status": optimization["status"]
                }
            
            results = optimization["results"]
            if not results:
                raise ValueError("No results available")
            
            response = {
                "success": True,
                "optimization_id": optimization_id,
                "name": optimization["name"],
                "status": "completed",
                "best_solution": results["best_solution"],
                "best_fitness": results["best_fitness"],
                "generations_run": results["generations_run"],
                "execution_time": results["execution_time"],
                "algorithm_config": optimization["config"]
            }
            
            # Add evolution history if requested
            if include_history and "history" in results:
                response["evolution_history"] = results["history"]
            
            # Add visualizations if requested
            if include_visualization:
                try:
                    visualizations = await self._generate_visualizations(optimization_id)
                    response["visualizations"] = visualizations
                except Exception as e:
                    logger.warning(f"Failed to generate visualizations: {e}")
                    response["visualization_error"] = str(e)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting optimization results: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "results_error"
            }
    
    async def list_optimizations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List optimizations.
        
        Args:
            arguments: Contains:
                - status: Filter by status (optional)
                - limit: Maximum number of results
                
        Returns:
            List of optimizations
        """
        try:
            status_filter = arguments.get("status", "all")
            limit = arguments.get("limit", 20)
            
            optimizations = []
            
            for opt_id, optimization in self.optimizations.items():
                if status_filter != "all" and optimization["status"] != status_filter:
                    continue
                
                opt_info = {
                    "optimization_id": opt_id,
                    "name": optimization["name"],
                    "status": optimization["status"],
                    "created_at": optimization["created_at"],
                    "variables_count": len(optimization["problem"].variables),
                    "objectives_count": len(optimization["problem"].objectives)
                }
                
                # Add completion info if available
                if optimization["status"] == "completed" and optimization["results"]:
                    results = optimization["results"]
                    opt_info["best_fitness"] = results.get("best_fitness")
                    opt_info["execution_time"] = results.get("execution_time")
                
                optimizations.append(opt_info)
                
                if len(optimizations) >= limit:
                    break
            
            return {
                "success": True,
                "optimizations": optimizations,
                "total_count": len(optimizations),
                "status_filter": status_filter
            }
            
        except Exception as e:
            logger.error(f"Error listing optimizations: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "listing_error"
            }
    
    async def create_template(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization template.
        
        Args:
            arguments: Contains:
                - template_type: Type of template
                - parameters: Template parameters
                
        Returns:
            Generated optimization template
        """
        try:
            template_type = arguments.get("template_type")
            parameters = arguments.get("parameters", {})
            
            if not template_type:
                raise ValueError("Template type is required")
            
            template = await self.template_manager.create_template(template_type, parameters)
            
            return {
                "success": True,
                "template_type": template_type,
                "template": template,
                "message": f"Template for {template_type} created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "template_error"
            }
    
    async def analyze_problem(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization problem.
        
        Args:
            arguments: Contains:
                - variables: Problem variables
                - objectives: Problem objectives
                - constraints: Problem constraints
                
        Returns:
            Problem analysis and recommendations
        """
        try:
            variables = arguments.get("variables", [])
            objectives = arguments.get("objectives", [])
            constraints = arguments.get("constraints", [])
            
            analysis = await self.problem_analyzer.analyze(
                variables=variables,
                objectives=objectives,
                constraints=constraints
            )
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing problem: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "analysis_error"
            }
    
    async def visualize_results(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create result visualizations.
        
        Args:
            arguments: Contains:
                - optimization_id: ID of optimization
                - chart_type: Type of visualization
                - format: Output format
                
        Returns:
            Generated visualization
        """
        try:
            optimization_id = arguments.get("optimization_id")
            chart_type = arguments.get("chart_type")
            format_type = arguments.get("format", "png")
            
            if not optimization_id or optimization_id not in self.optimizations:
                raise ValueError(f"Optimization {optimization_id} not found")
            
            optimization = self.optimizations[optimization_id]
            
            if optimization["status"] != "completed":
                raise ValueError("Cannot visualize incomplete optimization")
            
            visualization = await self.visualizer.create_visualization(
                optimization=optimization,
                chart_type=chart_type,
                format_type=format_type
            )
            
            return {
                "success": True,
                "optimization_id": optimization_id,
                "chart_type": chart_type,
                "format": format_type,
                "visualization": visualization
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "visualization_error"
            }
    
    async def compare_algorithms(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Compare algorithm performance.
        
        Args:
            arguments: Contains:
                - problem: Problem definition
                - algorithms: List of algorithms to compare
                - runs: Number of runs per algorithm
                
        Returns:
            Algorithm comparison results
        """
        try:
            problem_def = arguments.get("problem")
            algorithms = arguments.get("algorithms", [])
            runs = arguments.get("runs", 5)
            
            if not problem_def:
                raise ValueError("Problem definition is required")
            if not algorithms:
                raise ValueError("At least one algorithm must be specified")
            
            # Run comparison
            comparison_results = await self._run_algorithm_comparison(
                problem_def, algorithms, runs
            )
            
            return {
                "success": True,
                "problem": problem_def,
                "algorithms": algorithms,
                "runs": runs,
                "results": comparison_results
            }
            
        except Exception as e:
            logger.error(f"Error comparing algorithms: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "comparison_error"
            }
    
    async def get_system_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status.
        
        Args:
            arguments: Contains:
                - include_metrics: Whether to include performance metrics
                
        Returns:
            System status and health information
        """
        try:
            include_metrics = arguments.get("include_metrics", True)
            
            status = {
                "success": True,
                "server": "darwin-mcp",
                "version": "1.0.0",
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_optimizations": len([
                    opt for opt in self.optimizations.values() 
                    if opt["status"] == "running"
                ]),
                "total_optimizations": len(self.optimizations)
            }
            
            if include_metrics:
                status["metrics"] = {
                    "completed_optimizations": len([
                        opt for opt in self.optimizations.values() 
                        if opt["status"] == "completed"
                    ]),
                    "failed_optimizations": len([
                        opt for opt in self.optimizations.values() 
                        if opt["status"] == "failed"
                    ]),
                    "average_execution_time": self._calculate_average_execution_time()
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "status_error"
            }
    
    # Helper methods
    
    async def _run_optimization_async(self, optimization_id: str):
        """Run optimization asynchronously."""
        try:
            await self._execute_optimization(optimization_id)
        except Exception as e:
            logger.error(f"Async optimization {optimization_id} failed: {e}")
            optimization = self.optimizations[optimization_id]
            optimization["status"] = "failed"
            optimization["error"] = str(e)
            optimization["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    async def _execute_optimization(self, optimization_id: str) -> Dict[str, Any]:
        """Execute the optimization algorithm."""
        optimization = self.optimizations[optimization_id]
        problem = optimization["problem"]
        config = optimization["config"]
        
        # Create and configure algorithm
        algorithm = GeneticAlgorithm(config)
        
        # Run optimization
        start_time = datetime.now()
        
        # Simulate optimization execution (replace with actual algorithm)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate mock results (replace with real results)
        results = {
            "best_solution": [0.5] * len(problem.variables),
            "best_fitness": 0.123,
            "generations_run": config.get("max_generations", 200),
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "convergence_generation": 150,
            "final_population_diversity": 0.234
        }
        
        # Update optimization
        optimization["status"] = "completed"
        optimization["results"] = results
        optimization["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        return results
    
    def _estimate_complexity(self, variables: List, objectives: List, constraints: List) -> str:
        """Estimate problem complexity."""
        var_count = len(variables)
        obj_count = len(objectives)
        const_count = len(constraints)
        
        if var_count <= 5 and obj_count == 1 and const_count <= 2:
            return "low"
        elif var_count <= 20 and obj_count <= 3 and const_count <= 10:
            return "medium"
        else:
            return "high"
    
    async def _generate_visualizations(self, optimization_id: str) -> Dict[str, Any]:
        """Generate visualizations for optimization results."""
        # Mock visualization generation
        return {
            "convergence_chart": "base64_encoded_image_data",
            "variable_distribution": "base64_encoded_image_data"
        }
    
    async def _run_algorithm_comparison(self, problem_def: Dict, algorithms: List[str], runs: int) -> Dict[str, Any]:
        """Run algorithm comparison."""
        # Mock comparison results
        results = {}
        for algorithm in algorithms:
            results[algorithm] = {
                "mean_fitness": 0.1 + np.random.random() * 0.1,
                "std_fitness": 0.01 + np.random.random() * 0.02,
                "mean_time": 10.0 + np.random.random() * 5.0,
                "success_rate": 0.8 + np.random.random() * 0.2
            }
        return results
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time of completed optimizations."""
        completed = [
            opt for opt in self.optimizations.values() 
            if opt["status"] == "completed" and opt.get("results")
        ]
        
        if not completed:
            return 0.0
        
        total_time = sum(
            opt["results"].get("execution_time", 0) 
            for opt in completed
        )
        
        return total_time / len(completed)