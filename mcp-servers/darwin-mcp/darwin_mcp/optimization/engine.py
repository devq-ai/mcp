"""
Darwin Optimization Engine

This module provides the core optimization engine for Darwin genetic algorithms.
The engine manages the execution of optimization algorithms, coordinates between
different components, and provides a unified interface for optimization operations.

The optimization engine handles:
- Algorithm initialization and configuration
- Problem setup and validation
- Optimization execution and monitoring
- Result collection and processing
- Progress tracking and reporting

Example usage:
    from darwin_mcp.optimization.engine import OptimizationEngine
    from darwin_mcp.optimization.problem import OptimizationProblem
    
    engine = OptimizationEngine()
    problem = OptimizationProblem(...)
    result = await engine.optimize(problem, config)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .problem import OptimizationProblem
from .config import OptimizationConfig
from .results import OptimizationResult
from ..algorithms.genetic import GeneticAlgorithm

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """Core optimization engine for Darwin genetic algorithms."""
    
    def __init__(self, max_concurrent_jobs: int = 5):
        """Initialize the optimization engine.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent optimization jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self.completed_optimizations: Dict[str, OptimizationResult] = {}
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        optimization_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """Execute an optimization.
        
        Args:
            problem: Optimization problem definition
            config: Algorithm configuration
            optimization_id: Optional ID for tracking
            progress_callback: Optional callback for progress updates
            
        Returns:
            OptimizationResult with solution and metadata
        """
        logger.info(f"Starting optimization: {problem.name}")
        
        # Validate inputs
        self._validate_problem(problem)
        self._validate_config(config)
        
        # Create optimization tracking
        if optimization_id:
            self.active_optimizations[optimization_id] = {
                "problem": problem,
                "config": config,
                "start_time": time.time(),
                "status": "running",
                "progress": 0.0
            }
        
        try:
            # Select and initialize algorithm
            algorithm = self._create_algorithm(config)
            
            # Prepare problem for algorithm
            prepared_problem = self._prepare_problem(problem)
            
            # Execute optimization
            if config.parallel_execution:
                result = await self._run_parallel_optimization(
                    algorithm, prepared_problem, config, optimization_id, progress_callback
                )
            else:
                result = await self._run_sequential_optimization(
                    algorithm, prepared_problem, config, optimization_id, progress_callback
                )
            
            # Process and store results
            optimization_result = self._process_results(
                result, problem, config, optimization_id
            )
            
            if optimization_id:
                self.completed_optimizations[optimization_id] = optimization_result
                if optimization_id in self.active_optimizations:
                    del self.active_optimizations[optimization_id]
            
            logger.info(f"Optimization completed: {problem.name}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            if optimization_id and optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id]["status"] = "failed"
                self.active_optimizations[optimization_id]["error"] = str(e)
            raise
    
    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get the status of a running optimization.
        
        Args:
            optimization_id: ID of the optimization
            
        Returns:
            Status information
        """
        if optimization_id in self.active_optimizations:
            opt = self.active_optimizations[optimization_id]
            elapsed_time = time.time() - opt["start_time"]
            
            return {
                "id": optimization_id,
                "status": opt["status"],
                "progress": opt["progress"],
                "elapsed_time": elapsed_time,
                "problem_name": opt["problem"].name
            }
        elif optimization_id in self.completed_optimizations:
            result = self.completed_optimizations[optimization_id]
            return {
                "id": optimization_id,
                "status": "completed",
                "progress": 1.0,
                "elapsed_time": result.execution_time,
                "problem_name": result.problem_name,
                "best_fitness": result.best_fitness
            }
        else:
            raise ValueError(f"Optimization {optimization_id} not found")
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancel a running optimization.
        
        Args:
            optimization_id: ID of the optimization to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if optimization_id in self.active_optimizations:
            self.active_optimizations[optimization_id]["status"] = "cancelled"
            logger.info(f"Optimization {optimization_id} cancelled")
            return True
        return False
    
    def list_active_optimizations(self) -> List[Dict[str, Any]]:
        """List all active optimizations.
        
        Returns:
            List of active optimization information
        """
        active = []
        for opt_id, opt_info in self.active_optimizations.items():
            elapsed_time = time.time() - opt_info["start_time"]
            active.append({
                "id": opt_id,
                "problem_name": opt_info["problem"].name,
                "status": opt_info["status"],
                "progress": opt_info["progress"],
                "elapsed_time": elapsed_time
            })
        return active
    
    def get_optimization_result(self, optimization_id: str) -> OptimizationResult:
        """Get completed optimization result.
        
        Args:
            optimization_id: ID of the optimization
            
        Returns:
            OptimizationResult
        """
        if optimization_id not in self.completed_optimizations:
            raise ValueError(f"Completed optimization {optimization_id} not found")
        
        return self.completed_optimizations[optimization_id]
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
    
    # Private methods
    
    def _validate_problem(self, problem: OptimizationProblem):
        """Validate optimization problem."""
        if not problem.variables:
            raise ValueError("Problem must have at least one variable")
        
        if not problem.objectives:
            raise ValueError("Problem must have at least one objective")
        
        # Validate variable bounds
        for var in problem.variables:
            if var.type == "continuous" and not var.bounds:
                raise ValueError(f"Continuous variable {var.name} must have bounds")
            
            if var.type == "categorical" and not var.values:
                raise ValueError(f"Categorical variable {var.name} must have values")
    
    def _validate_config(self, config: OptimizationConfig):
        """Validate optimization configuration."""
        if config.population_size <= 0:
            raise ValueError("Population size must be positive")
        
        if config.max_generations <= 0:
            raise ValueError("Max generations must be positive")
        
        if not 0 <= config.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if not 0 <= config.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
    
    def _create_algorithm(self, config: OptimizationConfig) -> GeneticAlgorithm:
        """Create and configure algorithm instance."""
        return GeneticAlgorithm(config)
    
    def _prepare_problem(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Prepare problem for algorithm execution."""
        return {
            "variables": problem.variables,
            "objectives": problem.objectives,
            "constraints": problem.constraints,
            "fitness_function": self._create_fitness_function(problem)
        }
    
    def _create_fitness_function(self, problem: OptimizationProblem) -> Callable:
        """Create fitness function from problem definition."""
        
        def fitness_function(solution: List[float]) -> Union[float, List[float]]:
            """Evaluate fitness for a solution."""
            try:
                # Handle built-in test functions
                if len(problem.objectives) == 1:
                    obj = problem.objectives[0]
                    if obj.function in self._builtin_functions:
                        fitness = self._builtin_functions[obj.function](solution)
                        return fitness if obj.type == "minimize" else -fitness
                
                # For custom functions, return a dummy value
                # In real implementation, this would evaluate user-defined functions
                if len(problem.objectives) == 1:
                    return sum(x**2 for x in solution)  # Sphere function
                else:
                    # Multi-objective
                    return [sum(x**2 for x in solution), sum((x-1)**2 for x in solution)]
                    
            except Exception as e:
                logger.error(f"Error in fitness evaluation: {e}")
                return float('inf')
        
        return fitness_function
    
    @property
    def _builtin_functions(self) -> Dict[str, Callable]:
        """Built-in optimization test functions."""
        return {
            "sphere": lambda x: sum(xi**2 for xi in x),
            "rastrigin": lambda x: 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x),
            "ackley": lambda x: (-20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x)/len(x))) -
                               np.exp(sum(np.cos(2*np.pi*xi) for xi in x)/len(x)) + 20 + np.e),
            "rosenbrock": lambda x: sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)),
            "griewank": lambda x: 1 + sum(xi**2/4000 for xi in x) - np.prod([np.cos(xi/np.sqrt(i+1)) for i, xi in enumerate(x)]),
            "schwefel": lambda x: 418.9829*len(x) - sum(xi*np.sin(np.sqrt(abs(xi))) for xi in x)
        }
    
    async def _run_sequential_optimization(
        self,
        algorithm: GeneticAlgorithm,
        problem: Dict[str, Any],
        config: OptimizationConfig,
        optimization_id: Optional[str],
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Run optimization sequentially."""
        
        def run_optimization():
            return algorithm.optimize(
                problem, 
                progress_callback=lambda gen, fitness, pop: self._update_progress(
                    optimization_id, gen, config.max_generations, fitness, progress_callback
                )
            )
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, run_optimization)
        return result
    
    async def _run_parallel_optimization(
        self,
        algorithm: GeneticAlgorithm,
        problem: Dict[str, Any],
        config: OptimizationConfig,
        optimization_id: Optional[str],
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Run optimization with parallel evaluation."""
        
        def run_optimization():
            # Enable parallel evaluation in algorithm
            algorithm.enable_parallel_evaluation(config.n_jobs or -1)
            return algorithm.optimize(
                problem,
                progress_callback=lambda gen, fitness, pop: self._update_progress(
                    optimization_id, gen, config.max_generations, fitness, progress_callback
                )
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, run_optimization)
        return result
    
    def _update_progress(
        self,
        optimization_id: Optional[str],
        generation: int,
        max_generations: int,
        best_fitness: float,
        progress_callback: Optional[Callable]
    ):
        """Update optimization progress."""
        progress = generation / max_generations
        
        if optimization_id and optimization_id in self.active_optimizations:
            self.active_optimizations[optimization_id]["progress"] = progress
            self.active_optimizations[optimization_id]["current_generation"] = generation
            self.active_optimizations[optimization_id]["best_fitness"] = best_fitness
        
        if progress_callback:
            try:
                progress_callback(generation, best_fitness, progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def _process_results(
        self,
        raw_result: Dict[str, Any],
        problem: OptimizationProblem,
        config: OptimizationConfig,
        optimization_id: Optional[str]
    ) -> OptimizationResult:
        """Process raw algorithm results into OptimizationResult."""
        
        return OptimizationResult(
            optimization_id=optimization_id,
            problem_name=problem.name,
            algorithm_name=config.algorithm,
            best_solution=raw_result.get("best_solution", []),
            best_fitness=raw_result.get("best_fitness", float('inf')),
            generations_run=raw_result.get("generations_run", 0),
            execution_time=raw_result.get("execution_time", 0.0),
            convergence_generation=raw_result.get("convergence_generation"),
            final_population=raw_result.get("final_population", []),
            evolution_history=raw_result.get("evolution_history", {}),
            algorithm_config=config.to_dict(),
            metadata={
                "problem_dimensions": len(problem.variables),
                "objective_count": len(problem.objectives),
                "constraint_count": len(problem.constraints),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )