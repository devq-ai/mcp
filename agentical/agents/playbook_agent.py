"""
Playbook Agent Implementation for Agentical Framework

This module provides the PlaybookAgent implementation for strategic execution,
playbook management, and orchestrated workflow automation.

Features:
- Playbook creation and management
- Strategic execution planning and coordination
- Workflow orchestration and automation
- Step-by-step execution with validation
- Dynamic playbook adaptation
- Execution history and analytics
- Template management and customization
- Multi-agent orchestration through playbooks
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from pathlib import Path
from enum import Enum

import logfire
from pydantic import BaseModel, Field, validator

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.db.models.playbook import (
    Playbook, PlaybookStep, PlaybookExecution, PlaybookStatus,
    ExecutionStatus, StepType, StepStatus, PlaybookCategory
)
from agentical.core.exceptions import (
    AgentExecutionError, ValidationError, PlaybookError,
    PlaybookNotFoundError, PlaybookExecutionError
)
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class ExecutionMode(Enum):
    """Playbook execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    INTERACTIVE = "interactive"
    AUTOMATED = "automated"


class ValidationLevel(Enum):
    """Validation levels for playbook execution."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    NONE = "none"


class PlaybookExecutionRequest(BaseModel):
    """Request model for playbook execution tasks."""
    playbook_id: str = Field(..., description="Unique playbook identifier")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL, description="Execution mode")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    validation_level: ValidationLevel = Field(default=ValidationLevel.STANDARD, description="Validation level")
    dry_run: bool = Field(default=False, description="Perform dry run without actual execution")
    timeout_minutes: int = Field(default=30, description="Execution timeout in minutes")
    checkpoint_interval: int = Field(default=5, description="Checkpoint interval in steps")
    continue_on_error: bool = Field(default=False, description="Continue execution on non-critical errors")


class PlaybookCreationRequest(BaseModel):
    """Request model for playbook creation tasks."""
    name: str = Field(..., description="Playbook name")
    description: str = Field(..., description="Playbook description")
    category: PlaybookCategory = Field(..., description="Playbook category")
    steps: List[Dict[str, Any]] = Field(..., description="Playbook steps definition")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Playbook variables")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    template: Optional[str] = Field(default=None, description="Base template to use")
    validation_rules: Optional[List[str]] = Field(default=None, description="Validation rules")


class PlaybookAnalysisRequest(BaseModel):
    """Request model for playbook analysis tasks."""
    playbook_id: Optional[str] = Field(default=None, description="Specific playbook to analyze")
    execution_id: Optional[str] = Field(default=None, description="Specific execution to analyze")
    analysis_type: str = Field(default="performance", description="Type of analysis")
    time_range: Optional[Tuple[datetime, datetime]] = Field(default=None, description="Analysis time range")
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")


class PlaybookAgent(EnhancedBaseAgent[PlaybookExecutionRequest, Dict[str, Any]]):
    """
    Specialized agent for strategic execution and playbook management.

    Capabilities:
    - Create and manage playbooks for strategic execution
    - Execute complex multi-step workflows with validation
    - Orchestrate multi-agent coordination through playbooks
    - Provide execution analytics and optimization recommendations
    - Manage playbook templates and customization
    - Handle dynamic playbook adaptation during execution
    - Support various execution modes and validation levels
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "PlaybookAgent",
        description: str = "Specialized agent for strategic execution and playbook management",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.PLAYBOOK_AGENT,
            **kwargs
        )

        # Playbook execution configuration
        self.execution_modes = {
            ExecutionMode.SEQUENTIAL: self._execute_sequential,
            ExecutionMode.PARALLEL: self._execute_parallel,
            ExecutionMode.CONDITIONAL: self._execute_conditional,
            ExecutionMode.INTERACTIVE: self._execute_interactive,
            ExecutionMode.AUTOMATED: self._execute_automated
        }

        # Step type handlers
        self.step_handlers = {
            "action": self._execute_action_step,
            "validation": self._execute_validation_step,
            "decision": self._execute_decision_step,
            "loop": self._execute_loop_step,
            "parallel": self._execute_parallel_step,
            "wait": self._execute_wait_step,
            "notification": self._execute_notification_step,
            "api_call": self._execute_api_call_step,
            "agent_call": self._execute_agent_call_step
        }

        # Validation rules
        self.validation_rules = {
            ValidationLevel.STRICT: ["all_steps_required", "strict_types", "mandatory_outputs"],
            ValidationLevel.STANDARD: ["required_steps", "type_validation"],
            ValidationLevel.PERMISSIVE: ["basic_validation"],
            ValidationLevel.NONE: []
        }

        # Execution state tracking
        self.active_executions = {}
        self.execution_history = []

        # Template library
        self.playbook_templates = {
            "incident_response": self._get_incident_response_template(),
            "deployment": self._get_deployment_template(),
            "troubleshooting": self._get_troubleshooting_template(),
            "maintenance": self._get_maintenance_template(),
            "testing": self._get_testing_template()
        }

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "playbook_execution",
            "playbook_creation",
            "playbook_management",
            "strategic_planning",
            "workflow_orchestration",
            "multi_agent_coordination",
            "execution_analytics",
            "template_management",
            "dynamic_adaptation",
            "checkpoint_management",
            "error_recovery",
            "performance_optimization",
            "compliance_validation",
            "audit_trail_management",
            "interactive_execution"
        ]

    async def _execute_core_logic(
        self,
        request: Union[PlaybookExecutionRequest, PlaybookCreationRequest, PlaybookAnalysisRequest],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the core playbook logic."""

        with logfire.span("PlaybookAgent execution", request_type=type(request).__name__):
            try:
                if isinstance(request, PlaybookExecutionRequest):
                    return await self._handle_execution_request(request, context)
                elif isinstance(request, PlaybookCreationRequest):
                    return await self._handle_creation_request(request, context)
                elif isinstance(request, PlaybookAnalysisRequest):
                    return await self._handle_analysis_request(request, context)
                else:
                    # Handle generic playbook requests
                    return await self._handle_generic_request(request, context)

            except Exception as e:
                logfire.error("PlaybookAgent execution failed", error=str(e))
                raise AgentExecutionError(f"Playbook operation failed: {str(e)}")

    async def _handle_execution_request(
        self,
        request: PlaybookExecutionRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle playbook execution requests."""

        with logfire.span("Playbook execution", playbook_id=request.playbook_id):
            execution_id = str(uuid.uuid4())

            result = {
                "execution_id": execution_id,
                "playbook_id": request.playbook_id,
                "execution_mode": request.execution_mode.value,
                "started_at": datetime.utcnow().isoformat(),
                "status": ExecutionStatus.RUNNING.value,
                "steps_executed": [],
                "checkpoints": [],
                "errors": [],
                "metrics": {}
            }

            try:
                # Load playbook
                playbook = await self._load_playbook(request.playbook_id)
                if not playbook:
                    raise PlaybookNotFoundError(f"Playbook {request.playbook_id} not found")

                # Validate execution parameters
                if request.validation_level != ValidationLevel.NONE:
                    await self._validate_execution_parameters(playbook, request)

                # Initialize execution context
                execution_context = {
                    "playbook": playbook,
                    "parameters": request.parameters,
                    "variables": {},
                    "execution_id": execution_id,
                    "dry_run": request.dry_run,
                    "agent_context": context
                }

                # Track active execution
                self.active_executions[execution_id] = {
                    "request": request,
                    "context": execution_context,
                    "started_at": datetime.utcnow(),
                    "current_step": 0
                }

                # Execute playbook based on mode
                execution_handler = self.execution_modes[request.execution_mode]
                execution_result = await execution_handler(playbook, execution_context, request)

                result.update(execution_result)
                result["status"] = ExecutionStatus.COMPLETED.value
                result["completed_at"] = datetime.utcnow().isoformat()

                # Calculate execution metrics
                result["metrics"] = await self._calculate_execution_metrics(result)

                logfire.info("Playbook execution completed",
                           execution_id=execution_id,
                           steps_executed=len(result["steps_executed"]))

            except Exception as e:
                result["status"] = ExecutionStatus.FAILED.value
                result["error"] = str(e)
                result["failed_at"] = datetime.utcnow().isoformat()
                logfire.error("Playbook execution failed", execution_id=execution_id, error=str(e))

            finally:
                # Clean up active execution tracking
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]

                # Add to execution history
                self.execution_history.append(result)

            return result

    async def _handle_creation_request(
        self,
        request: PlaybookCreationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle playbook creation requests."""

        with logfire.span("Playbook creation", playbook_name=request.name):
            playbook_id = str(uuid.uuid4())

            result = {
                "playbook_id": playbook_id,
                "name": request.name,
                "category": request.category.value,
                "created_at": datetime.utcnow().isoformat(),
                "status": "created",
                "validation_results": {}
            }

            try:
                # Apply template if specified
                if request.template and request.template in self.playbook_templates:
                    template_steps = self.playbook_templates[request.template]
                    request.steps.extend(template_steps)

                # Validate playbook structure
                validation_results = await self._validate_playbook_structure(request)
                result["validation_results"] = validation_results

                if not validation_results.get("valid", False):
                    result["status"] = "validation_failed"
                    return result

                # Create playbook object
                playbook_data = {
                    "id": playbook_id,
                    "name": request.name,
                    "description": request.description,
                    "category": request.category,
                    "steps": request.steps,
                    "variables": request.variables or {},
                    "metadata": request.metadata or {},
                    "status": PlaybookStatus.ACTIVE,
                    "created_at": datetime.utcnow(),
                    "version": "1.0"
                }

                # Store playbook (placeholder - would integrate with actual database)
                await self._store_playbook(playbook_data)

                result["step_count"] = len(request.steps)
                result["variable_count"] = len(request.variables or {})

                logfire.info("Playbook created", playbook_id=playbook_id, name=request.name)

            except Exception as e:
                result["status"] = "creation_failed"
                result["error"] = str(e)
                logfire.error("Playbook creation failed", error=str(e))

            return result

    async def _handle_analysis_request(
        self,
        request: PlaybookAnalysisRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle playbook analysis requests."""

        with logfire.span("Playbook analysis", analysis_type=request.analysis_type):
            result = {
                "analysis_type": request.analysis_type,
                "analyzed_at": datetime.utcnow().isoformat(),
                "metrics": {},
                "insights": [],
                "recommendations": []
            }

            try:
                if request.playbook_id:
                    # Analyze specific playbook
                    result.update(await self._analyze_playbook(request.playbook_id, request))
                elif request.execution_id:
                    # Analyze specific execution
                    result.update(await self._analyze_execution(request.execution_id, request))
                else:
                    # Analyze all playbooks/executions
                    result.update(await self._analyze_global_performance(request))

                logfire.info("Playbook analysis completed", analysis_type=request.analysis_type)

            except Exception as e:
                result["status"] = "analysis_failed"
                result["error"] = str(e)
                logfire.error("Playbook analysis failed", error=str(e))

            return result

    async def _execute_sequential(
        self,
        playbook: Dict[str, Any],
        context: Dict[str, Any],
        request: PlaybookExecutionRequest
    ) -> Dict[str, Any]:
        """Execute playbook steps sequentially."""

        result = {
            "execution_mode": "sequential",
            "steps_executed": [],
            "total_steps": len(playbook["steps"])
        }

        for i, step in enumerate(playbook["steps"]):
            step_result = await self._execute_step(step, context, i)
            result["steps_executed"].append(step_result)

            # Handle step failure
            if step_result["status"] == StepStatus.FAILED.value:
                if not request.continue_on_error:
                    break

            # Create checkpoint if needed
            if (i + 1) % request.checkpoint_interval == 0:
                checkpoint = await self._create_checkpoint(context, i + 1)
                result.setdefault("checkpoints", []).append(checkpoint)

        return result

    async def _execute_parallel(
        self,
        playbook: Dict[str, Any],
        context: Dict[str, Any],
        request: PlaybookExecutionRequest
    ) -> Dict[str, Any]:
        """Execute playbook steps in parallel."""

        result = {
            "execution_mode": "parallel",
            "steps_executed": [],
            "total_steps": len(playbook["steps"])
        }

        # Create tasks for all steps
        tasks = []
        for i, step in enumerate(playbook["steps"]):
            task = asyncio.create_task(self._execute_step(step, context, i))
            tasks.append(task)

        # Wait for all tasks to complete
        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, step_result in enumerate(step_results):
            if isinstance(step_result, Exception):
                result["steps_executed"].append({
                    "step_index": i,
                    "status": StepStatus.FAILED.value,
                    "error": str(step_result)
                })
            else:
                result["steps_executed"].append(step_result)

        return result

    async def _execute_conditional(
        self,
        playbook: Dict[str, Any],
        context: Dict[str, Any],
        request: PlaybookExecutionRequest
    ) -> Dict[str, Any]:
        """Execute playbook steps with conditional logic."""

        result = {
            "execution_mode": "conditional",
            "steps_executed": [],
            "conditions_evaluated": []
        }

        for i, step in enumerate(playbook["steps"]):
            # Evaluate step conditions
            should_execute = await self._evaluate_step_conditions(step, context)

            result["conditions_evaluated"].append({
                "step_index": i,
                "condition": step.get("condition", "always"),
                "result": should_execute
            })

            if should_execute:
                step_result = await self._execute_step(step, context, i)
                result["steps_executed"].append(step_result)

        return result

    async def _execute_interactive(
        self,
        playbook: Dict[str, Any],
        context: Dict[str, Any],
        request: PlaybookExecutionRequest
    ) -> Dict[str, Any]:
        """Execute playbook steps interactively with user input."""

        result = {
            "execution_mode": "interactive",
            "steps_executed": [],
            "user_interactions": []
        }

        for i, step in enumerate(playbook["steps"]):
            # Check if step requires user confirmation
            if step.get("interactive", False):
                confirmation = await self._request_user_confirmation(step, context)
                result["user_interactions"].append({
                    "step_index": i,
                    "confirmation": confirmation,
                    "timestamp": datetime.utcnow().isoformat()
                })

                if not confirmation:
                    continue

            step_result = await self._execute_step(step, context, i)
            result["steps_executed"].append(step_result)

        return result

    async def _execute_automated(
        self,
        playbook: Dict[str, Any],
        context: Dict[str, Any],
        request: PlaybookExecutionRequest
    ) -> Dict[str, Any]:
        """Execute playbook steps in fully automated mode."""

        result = {
            "execution_mode": "automated",
            "steps_executed": [],
            "automation_level": "full"
        }

        for i, step in enumerate(playbook["steps"]):
            # Override any interactive flags for automation
            step["interactive"] = False
            step["auto_confirm"] = True

            step_result = await self._execute_step(step, context, i)
            result["steps_executed"].append(step_result)

        return result

    async def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        step_index: int
    ) -> Dict[str, Any]:
        """Execute a single playbook step."""

        step_result = {
            "step_index": step_index,
            "step_name": step.get("name", f"Step {step_index}"),
            "step_type": step.get("type", "action"),
            "started_at": datetime.utcnow().isoformat(),
            "status": StepStatus.RUNNING.value,
            "output": {},
            "duration_seconds": 0
        }

        start_time = datetime.utcnow()

        try:
            step_type = step.get("type", "action")

            if step_type in self.step_handlers:
                handler = self.step_handlers[step_type]
                step_output = await handler(step, context)
                step_result["output"] = step_output
                step_result["status"] = StepStatus.COMPLETED.value
            else:
                raise ValueError(f"Unknown step type: {step_type}")

        except Exception as e:
            step_result["status"] = StepStatus.FAILED.value
            step_result["error"] = str(e)
            logfire.error("Step execution failed", step_index=step_index, error=str(e))

        finally:
            end_time = datetime.utcnow()
            step_result["completed_at"] = end_time.isoformat()
            step_result["duration_seconds"] = (end_time - start_time).total_seconds()

        return step_result

    async def _execute_action_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action step."""
        action = step.get("action", "default_action")
        parameters = step.get("parameters", {})

        # Placeholder implementation
        return {
            "action": action,
            "parameters": parameters,
            "result": f"Action '{action}' executed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _execute_validation_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validation step."""
        validation_type = step.get("validation_type", "basic")
        criteria = step.get("criteria", {})

        # Placeholder validation logic
        validation_result = True  # Would implement actual validation

        return {
            "validation_type": validation_type,
            "criteria": criteria,
            "result": validation_result,
            "message": "Validation passed" if validation_result else "Validation failed"
        }

    async def _execute_decision_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a decision step."""
        decision_logic = step.get("decision_logic", {})
        options = step.get("options", [])

        # Placeholder decision logic
        selected_option = options[0] if options else "default"

        return {
            "decision_logic": decision_logic,
            "available_options": options,
            "selected_option": selected_option,
            "reasoning": "Selected based on current context"
        }

    async def _execute_loop_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a loop step."""
        loop_condition = step.get("condition", "count")
        iterations = step.get("iterations", 1)
        loop_steps = step.get("steps", [])

        results = []
        for i in range(iterations):
            iteration_result = {"iteration": i, "steps": []}
            for loop_step in loop_steps:
                step_result = await self._execute_step(loop_step, context, i)
                iteration_result["steps"].append(step_result)
            results.append(iteration_result)

        return {
            "loop_type": loop_condition,
            "total_iterations": iterations,
            "results": results
        }

    async def _execute_parallel_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a parallel step."""
        parallel_steps = step.get("steps", [])

        tasks = []
        for i, parallel_step in enumerate(parallel_steps):
            task = asyncio.create_task(self._execute_step(parallel_step, context, i))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "parallel_steps": len(parallel_steps),
            "results": [str(r) if isinstance(r, Exception) else r for r in results]
        }

    async def _execute_wait_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait step."""
        wait_duration = step.get("duration", 1)
        wait_condition = step.get("condition", None)

        if wait_condition:
            # Wait for condition (placeholder)
            await asyncio.sleep(1)  # Simulate condition check
        else:
            await asyncio.sleep(wait_duration)

        return {
            "wait_type": "condition" if wait_condition else "duration",
            "duration": wait_duration,
            "condition": wait_condition
        }

    async def _execute_notification_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a notification step."""
        message = step.get("message", "Notification from playbook")
        channels = step.get("channels", ["log"])

        # Placeholder notification logic
        for channel in channels:
            logfire.info("Playbook notification", message=message, channel=channel)

        return {
            "message": message,
            "channels": channels,
            "sent_at": datetime.utcnow().isoformat()
        }

    async def _execute_api_call_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call step."""
        url = step.get("url", "")
        method = step.get("method", "GET")
        headers = step.get("headers", {})
        data = step.get("data", {})

        # Placeholder API call logic
        return {
            "url": url,
            "method": method,
            "status_code": 200,
            "response": {"result": "API call simulated"},
            "headers": headers
        }

    async def _execute_agent_call_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent call step."""
        agent_type = step.get("agent_type", "generic")
        agent_request = step.get("request", {})

        # Placeholder agent call logic
        return {
            "agent_type": agent_type,
            "request": agent_request,
            "result": f"Agent '{agent_type}' called successfully",
            "response_time": 0.5
        }

    async def _load_playbook(self, playbook_id: str) -> Optional[Dict[str, Any]]:
        """Load playbook by ID."""
        # Placeholder implementation - would integrate with actual database
        return {
            "id": playbook_id,
            "name": "Sample Playbook",
            "description": "A sample playbook for testing",
            "steps": [
                {"name": "Initialize", "type": "action", "action": "initialize"},
                {"name": "Validate", "type": "validation", "validation_type": "basic"},
                {"name": "Execute", "type": "action", "action": "execute"},
                {"name": "Finalize", "type": "action", "action": "finalize"}
            ],
            "variables": {},
            "metadata": {}
        }

    async def _store_playbook(self, playbook_data: Dict[str, Any]) -> bool:
        """Store playbook data."""
        # Placeholder implementation - would integrate with actual database
        return True

    async def _validate_execution_parameters(self, playbook: Dict[str, Any], request: PlaybookExecutionRequest) -> bool:
        """Validate execution parameters against playbook requirements."""
        # Placeholder validation logic
        return True

    async def _validate_playbook_structure(self, request: PlaybookCreationRequest) -> Dict[str, Any]:
        """Validate playbook structure."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Basic structure validation
        if not request.steps:
            validation_result["valid"] = False
            validation_result["errors"].append("Playbook must have at least one step")

        for i, step in enumerate(request.steps):
            if "type" not in step:
                validation_result["warnings"].append(f"Step {i} missing type, defaulting to 'action'")

        return validation_result

    async def _evaluate_step_conditions(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate step execution conditions."""
        condition = step.get("condition", "always")

        if condition == "always":
            return True
        elif condition == "never":
            return False
        else:
            # Placeholder condition evaluation
            return True

    async def _request_user_confirmation(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Request user confirmation for interactive steps."""
        # Placeholder implementation - would integrate with actual UI
        return True

    async def _create_checkpoint(self, context: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """Create execution checkpoint."""
        return {
            "checkpoint_id": str(uuid.uuid4()),
            "step_number": step_number,
            "timestamp": datetime.utcnow().isoformat(),
            "context_snapshot": {
                "variables": context.get("variables", {}),
                "execution_id": context.get("execution_id")
            }
        }

    async def _calculate_execution_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate execution performance metrics."""
        total_steps = len(result["steps_executed"])
        successful_steps = len([s for s in result["steps_executed"]
                               if s.get("status") == StepStatus.COMPLETED.value])

        total_duration = sum(s.get("duration_seconds", 0) for s in result["steps_executed"])

        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "success_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_step_duration": total_duration / total_steps if total_steps > 0 else 0
        }

    async def _analyze_playbook(self, playbook_id: str, request: PlaybookAnalysisRequest) -> Dict[str, Any]:
        """Analyze specific playbook performance."""
        # Placeholder analysis
        return {
            "playbook_id": playbook_id,
            "metrics": {
                "total_executions": 10,
                "success_rate": 95.0,
                "average_duration": 300,
                "performance_trend": "improving"
            },
            "insights": ["High success rate", "Consistent performance"],
            "recommendations": ["Consider parallel execution for better performance"]
        }

    async def _analyze_execution(self, execution_id: str, request: PlaybookAnalysisRequest) -> Dict[str, Any]:
        """Analyze specific execution."""
        # Placeholder analysis
        return {
            "execution_id": execution_id,
            "metrics": {
                "duration": 245,
                "steps_completed": 8,
                "success_rate": 100.0
            },
            "insights": ["Execution completed successfully"],
            "recommendations": ["No optimization needed"]
        }

    async def _analyze_global_performance(self, request: PlaybookAnalysisRequest) -> Dict[str, Any]:
        """Analyze global playbook performance."""
        # Placeholder analysis
        return {
            "global_metrics": {
                "total_playbooks": 5,
                "total_executions": 50,
                "overall_success_rate": 92.0,
                "average_execution_time": 280,
                "most_used_category": "deployment"
            },
            "insights": [
                "High overall success rate",
                "Deployment playbooks most frequently used",
                "Performance trending upward"
            ],
            "recommendations": [
                "Consider creating more incident response templates",
                "Optimize long-running maintenance playbooks",
                "Implement more parallel execution patterns"
            ]
        }

    def _get_incident_response_template(self) -> List[Dict[str, Any]]:
        """Get incident response playbook template."""
        return [
            {
                "name": "Assess Incident",
                "type": "action",
                "action": "assess_incident",
                "parameters": {"severity_check": True}
            },
            {
                "name": "Notify Stakeholders",
                "type": "notification",
                "message": "Incident detected and assessment initiated",
                "channels": ["email", "slack"]
            },
            {
                "name": "Implement Immediate Fix",
                "type": "action",
                "action": "immediate_fix",
                "condition": "severity_high"
            },
            {
                "name": "Monitor Resolution",
                "type": "validation",
                "validation_type": "service_health",
                "criteria": {"uptime": ">95%"}
            },
            {
                "name": "Post-Incident Review",
                "type": "action",
                "action": "create_review",
                "interactive": True
            }
        ]

    def _get_deployment_template(self) -> List[Dict[str, Any]]:
        """Get deployment playbook template."""
        return [
            {
                "name": "Pre-deployment Validation",
                "type": "validation",
                "validation_type": "code_quality",
                "criteria": {"test_coverage": ">90%", "build_status": "passed"}
            },
            {
                "name": "Create Deployment Package",
                "type": "action",
                "action": "create_package",
                "parameters": {"environment": "production"}
            },
            {
                "name": "Deploy to Staging",
                "type": "action",
                "action": "deploy",
                "parameters": {"target": "staging"}
            },
            {
                "name": "Run Smoke Tests",
                "type": "validation",
                "validation_type": "smoke_test",
                "criteria": {"api_health": "ok", "database_connection": "ok"}
            },
            {
                "name": "Deploy to Production",
                "type": "action",
                "action": "deploy",
                "parameters": {"target": "production"},
                "interactive": True
            },
            {
                "name": "Post-deployment Monitoring",
                "type": "wait",
                "duration": 300,
                "condition": "monitoring_stable"
            }
        ]

    def _get_troubleshooting_template(self) -> List[Dict[str, Any]]:
        """Get troubleshooting playbook template."""
        return [
            {
                "name": "Gather System Information",
                "type": "action",
                "action": "collect_system_info",
                "parameters": {"include_logs": True, "include_metrics": True}
            },
            {
                "name": "Identify Root Cause",
                "type": "decision",
                "decision_logic": {"type": "diagnostic_tree"},
                "options": ["performance", "connectivity", "configuration", "data"]
            },
            {
                "name": "Apply Diagnostic Tests",
                "type": "loop",
                "condition": "until_found",
                "iterations": 5,
                "steps": [
                    {
                        "name": "Run Diagnostic",
                        "type": "action",
                        "action": "run_diagnostic"
                    }
                ]
            },
            {
                "name": "Implement Solution",
                "type": "action",
                "action": "apply_solution",
                "condition": "root_cause_identified"
            },
            {
                "name": "Verify Resolution",
                "type": "validation",
                "validation_type": "system_health",
                "criteria": {"errors": "none", "performance": "normal"}
            }
        ]

    def _get_maintenance_template(self) -> List[Dict[str, Any]]:
        """Get maintenance playbook template."""
        return [
            {
                "name": "Schedule Maintenance Window",
                "type": "notification",
                "message": "Maintenance window scheduled",
                "channels": ["email", "status_page"]
            },
            {
                "name": "Create System Backup",
                "type": "action",
                "action": "create_backup",
                "parameters": {"type": "full", "verify": True}
            },
            {
                "name": "Put System in Maintenance Mode",
                "type": "action",
                "action": "maintenance_mode",
                "parameters": {"enabled": True}
            },
            {
                "name": "Perform Maintenance Tasks",
                "type": "parallel",
                "steps": [
                    {
                        "name": "Update Software",
                        "type": "action",
                        "action": "update_software"
                    },
                    {
                        "name": "Clean Database",
                        "type": "action",
                        "action": "clean_database"
                    },
                    {
                        "name": "Optimize Performance",
                        "type": "action",
                        "action": "optimize_performance"
                    }
                ]
            },
            {
                "name": "Restore System Operation",
                "type": "action",
                "action": "maintenance_mode",
                "parameters": {"enabled": False}
            },
            {
                "name": "Verify System Health",
                "type": "validation",
                "validation_type": "full_health_check",
                "criteria": {"all_services": "operational"}
            }
        ]

    def _get_testing_template(self) -> List[Dict[str, Any]]:
        """Get testing playbook template."""
        return [
            {
                "name": "Setup Test Environment",
                "type": "action",
                "action": "setup_test_env",
                "parameters": {"clean_state": True}
            },
            {
                "name": "Load Test Data",
                "type": "action",
                "action": "load_test_data",
                "parameters": {"dataset": "integration_tests"}
            },
            {
                "name": "Run Unit Tests",
                "type": "action",
                "action": "run_tests",
                "parameters": {"type": "unit", "coverage": True}
            },
            {
                "name": "Run Integration Tests",
                "type": "action",
                "action": "run_tests",
                "parameters": {"type": "integration", "parallel": True}
            },
            {
                "name": "Run Performance Tests",
                "type": "action",
                "action": "run_tests",
                "parameters": {"type": "performance", "baseline": True}
            },
            {
                "name": "Generate Test Report",
                "type": "action",
                "action": "generate_report",
                "parameters": {"format": "html", "include_coverage": True}
            },
            {
                "name": "Cleanup Test Environment",
                "type": "action",
                "action": "cleanup_test_env"
            }
        ]

    async def _handle_generic_request(self, request: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic playbook requests."""
        return {
            "result": "Generic playbook operation completed",
            "request_type": type(request).__name__,
            "processed_at": datetime.utcnow().isoformat(),
            "context_keys": list(context.keys()) if context else []
        }

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for the PlaybookAgent."""
        return {
            "max_concurrent_executions": 5,
            "default_timeout_minutes": 30,
            "checkpoint_interval": 5,
            "enable_interactive_mode": True,
            "default_execution_mode": ExecutionMode.SEQUENTIAL.value,
            "default_validation_level": ValidationLevel.STANDARD.value,
            "auto_create_checkpoints": True,
            "execution_history_limit": 100,
            "template_library_enabled": True,
            "parallel_step_limit": 10,
            "retry_failed_steps": False,
            "step_timeout_seconds": 300
        }

    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_keys = [
            "max_concurrent_executions",
            "default_timeout_minutes",
            "checkpoint_interval"
        ]

        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required configuration key: {key}")

        if config["max_concurrent_executions"] <= 0:
            raise ValidationError("max_concurrent_executions must be positive")

        if config["default_timeout_minutes"] <= 0:
            raise ValidationError("default_timeout_minutes must be positive")

        if config["checkpoint_interval"] <= 0:
            raise ValidationError("checkpoint_interval must be positive")

        # Validate execution mode if specified
        if "default_execution_mode" in config:
            valid_modes = [mode.value for mode in ExecutionMode]
            if config["default_execution_mode"] not in valid_modes:
                raise ValidationError(f"Invalid execution mode: {config['default_execution_mode']}")

        # Validate validation level if specified
        if "default_validation_level" in config:
            valid_levels = [level.value for level in ValidationLevel]
            if config["default_validation_level"] not in valid_levels:
                raise ValidationError(f"Invalid validation level: {config['default_validation_level']}")

        return True

    async def validate_playbook(self, playbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a playbook and return validation results."""
        try:
            errors = []
            warnings = []
            suggestions = []
            
            # Basic validation
            if not playbook_data.get("name"):
                errors.append("Playbook name is required")
            
            if not playbook_data.get("steps"):
                errors.append("Playbook must have at least one step")
            
            # Validate steps
            steps = playbook_data.get("steps", [])
            for i, step in enumerate(steps):
                if not step.get("name"):
                    errors.append(f"Step {i+1}: name is required")
                if not step.get("type"):
                    errors.append(f"Step {i+1}: type is required")
            
            # Calculate complexity score
            complexity_score = min(len(steps) * 2, 10)
            
            # Estimate duration (in minutes)
            estimated_duration = len(steps) * 5
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "estimated_duration": estimated_duration,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "suggestions": [],
                "estimated_duration": None,
                "complexity_score": 0
            }

    async def get_available_templates(self, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available playbook templates."""
        try:
            templates = []
            
            for template_name, template_func in self.playbook_templates.items():
                template_data = {
                    "name": template_name,
                    "description": f"{template_name.replace(\"_\", \" \").title()} playbook template",
                    "category": self._get_template_category(template_name),
                    "steps": template_func(),
                    "variables": self._get_template_variables(template_name),
                    "metadata": {"template": True, "source": "builtin"},
                    "tags": self._get_template_tags(template_name)
                }
                
                if not category_filter or template_data["category"] == category_filter:
                    templates.append(template_data)
            
            return templates
            
        except Exception as e:
            logfire.error(f"Error getting templates: {str(e)}")
            return []
    
    def _get_template_category(self, template_name: str) -> str:
        """Get category for a template."""
        category_map = {
            "incident_response": "security",
            "deployment": "deployment", 
            "troubleshooting": "troubleshooting",
            "maintenance": "maintenance",
            "testing": "testing"
        }
        return category_map.get(template_name, "general")
    
    def _get_template_variables(self, template_name: str) -> Dict[str, Any]:
        """Get default variables for a template."""
        return {
            "timeout": 30,
            "retry_count": 3,
            "environment": "production"
        }
    
    def _get_template_tags(self, template_name: str) -> List[str]:
        """Get tags for a template."""
        tag_map = {
            "incident_response": ["security", "emergency", "automation"],
            "deployment": ["deployment", "automation", "ci-cd"],
            "troubleshooting": ["debugging", "diagnostics", "automation"],
            "maintenance": ["maintenance", "scheduled", "automation"],
            "testing": ["testing", "qa", "automation"]
        }
        return tag_map.get(template_name, ["automation"])

    async def create_from_template(self, template_name: str, playbook_name: str, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a playbook from a template."""
        try:
            if template_name not in self.playbook_templates:
                raise ValueError(f"Template {template_name} not found")
            
            # Get template steps
            template_func = self.playbook_templates[template_name]
            steps = template_func()
            
            # Apply parameters to customize the playbook
            customized_steps = self._customize_template_steps(steps, parameters)
            
            return {
                "name": playbook_name,
                "description": f"Playbook created from {template_name} template",
                "category": self._get_template_category(template_name),
                "steps": customized_steps,
                "variables": {**self._get_template_variables(template_name), **parameters},
                "metadata": {
                    "template": template_name,
                    "created_from_template": True,
                    "creation_date": datetime.utcnow().isoformat()
                },
                "validation_rules": [],
                "tags": self._get_template_tags(template_name)
            }
            
        except Exception as e:
            logfire.error(f"Error creating from template: {str(e)}")
            raise ValueError(f"Failed to create from template: {str(e)}")
    
    def _customize_template_steps(self, steps: List[Dict[str, Any]], 
                                parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Customize template steps with parameters."""
        customized_steps = []
        
        for step in steps:
            customized_step = step.copy()
            
            # Apply parameter substitutions
            if "parameters" in customized_step:
                step_params = customized_step["parameters"].copy()
                for key, value in parameters.items():
                    if key in step_params:
                        step_params[key] = value
                customized_step["parameters"] = step_params
            
            customized_steps.append(customized_step)
        
        return customized_steps

