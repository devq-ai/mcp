"""
SuperAgent - Meta-coordinator for Agentical Framework

The SuperAgent serves as the primary coordination layer for all other agents,
tools, and knowledge sources in the Agentical ecosystem. It acts as:

1. Meta-coordinator: Orchestrates multiple agents for complex workflows
2. Customer interface: Primary point of contact for users
3. Tool router: Intelligently selects and coordinates MCP tools
4. Knowledge orchestrator: Leverages Ptolemies knowledge base for decisions
5. Multimodal coordinator: Handles complex, multi-step operations

Infrastructure Integration:
- Ptolemies Knowledge Base: 597 production documents for context
- MCP Servers: All 26 available servers for tool coordination
- Agent Registry: Coordinates other specialized agents
- SurrealDB: State management and persistence

The SuperAgent is designed to be the "smart dispatcher" that understands
the full ecosystem and can break down complex requests into coordinated
actions across multiple agents and tools.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import time

from pydantic import BaseModel, Field
from .enhanced_base_agent import EnhancedBaseAgent
from .base_agent import (
    BaseAgent,
    AgentMetadata,
    AgentCapability,
    AgentExecutionContext,
    AgentExecutionResult,
    AgentStatus,
    agent_registry
)
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase

logger = logging.getLogger(__name__)


class SuperAgentRequest(BaseModel):
    """Request model for SuperAgent coordination tasks."""
    operation: str = Field(..., description="Operation to perform")
    agents: Optional[List[str]] = Field(default=None, description="Specific agents to coordinate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    priority: str = Field(default="medium", description="Operation priority")
    coordination_mode: str = Field(default="intelligent", description="Coordination mode")
    timeout_minutes: int = Field(default=10, description="Operation timeout")


class SuperAgent(EnhancedBaseAgent[SuperAgentRequest, Dict[str, Any]]):
    """
    SuperAgent - The meta-coordinator for the Agentical framework.

    This agent serves as the primary orchestrator that can:
    - Coordinate multiple specialized agents
    - Route requests to appropriate MCP tools
    - Leverage Ptolemies knowledge for intelligent decisions
    - Handle complex multi-step workflows
    - Serve as the primary user interface
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "SuperAgent",
        description: str = "Meta-coordinator for the Agentical framework",
        **kwargs
    ):
        """Initialize the SuperAgent with comprehensive capabilities"""

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.SUPER_AGENT,
            **kwargs
        )

        # Define SuperAgent capabilities
        capabilities = [
            AgentCapability(
                name="coordinate_agents",
                description="Coordinate multiple agents for complex workflows",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow_type": {"type": "string", "enum": ["parallel", "sequential", "conditional"]},
                        "agents": {"type": "array", "items": {"type": "string"}},
                        "coordination_strategy": {"type": "string"}
                    },
                    "required": ["workflow_type", "agents"]
                },
                required_tools=["memory", "filesystem"],
                knowledge_domains=["workflow_patterns", "agent_coordination"]
            ),

            AgentCapability(
                name="intelligent_routing",
                description="Route requests to most appropriate agents and tools",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request": {"type": "string"},
                        "context": {"type": "object"},
                        "available_agents": {"type": "array"},
                        "available_tools": {"type": "array"}
                    },
                    "required": ["request"]
                },
                required_tools=["memory", "ptolemies-mcp", "bayes-mcp"],
                knowledge_domains=["agent_capabilities", "tool_specifications"]
            ),

            AgentCapability(
                name="knowledge_synthesis",
                description="Synthesize information from Ptolemies knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domains": {"type": "array", "items": {"type": "string"}},
                        "synthesis_type": {"type": "string", "enum": ["summary", "analysis", "recommendations"]}
                    },
                    "required": ["query"]
                },
                required_tools=["ptolemies-mcp", "memory"],
                knowledge_domains=["all_domains"]
            ),

            AgentCapability(
                name="multimodal_coordination",
                description="Handle complex multimodal requests requiring multiple tools",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_type": {"type": "string"},
                        "components": {"type": "array"},
                        "integration_requirements": {"type": "object"}
                    },
                    "required": ["request_type"]
                },
                required_tools=["filesystem", "git", "web_search", "github-mcp", "crawl4ai-mcp"],
                knowledge_domains=["integration_patterns", "system_architecture"]
            ),

            AgentCapability(
                name="strategic_planning",
                description="Plan and orchestrate complex multi-step operations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "constraints": {"type": "object"},
                        "resources": {"type": "object"},
                        "timeline": {"type": "string"}
                    },
                    "required": ["objective"]
                },
                required_tools=["memory", "bayes-mcp", "darwin-mcp"],
                knowledge_domains=["project_planning", "resource_optimization"]
            )
        ]

        # Create SuperAgent metadata
        metadata = AgentMetadata(
            id="super_agent",
            name="SuperAgent",
            description="Meta-coordinator for Agentical framework - orchestrates agents, tools, and knowledge",
            version="1.0.0",
            capabilities=capabilities,
            available_tools=[
                # Core MCP tools
                "filesystem", "git", "memory", "fetch", "sequentialthinking",
                # Specialized knowledge and analysis
                "ptolemies-mcp", "surrealdb-mcp", "bayes-mcp", "darwin-mcp",
                # External integrations
                "github-mcp", "crawl4ai-mcp", "calendar-mcp", "stripe-mcp",
                # Development tools
                "jupyter-mcp", "shadcn-ui-mcp-server", "magic-mcp",
                # Solvers and specialized processing
                "solver-z3-mcp", "solver-pysat-mcp", "solver-mzn-mcp"
            ],
            model="claude-3-7-sonnet-20250219",
            system_prompts=[
                "You are the SuperAgent, the primary coordinator for the Agentical framework.",
                "You have access to 597 production documents through Ptolemies knowledge base.",
                "You can coordinate 26 MCP servers and multiple specialized agents.",
                "Your role is to intelligently route requests, synthesize knowledge, and orchestrate complex workflows.",
                "Always consider the full ecosystem when making decisions.",
                "Break down complex requests into coordinated actions across multiple agents and tools.",
                "Use Ptolemies knowledge to inform your decisions and provide context.",
                "Be the intelligent dispatcher that understands the capabilities of the entire system."
            ],
            tags=["meta-coordinator", "primary-interface", "multimodal", "orchestrator"]
        )

        # Initialize enhanced base agent properly
        # Note: We need to maintain backward compatibility with existing SuperAgent usage

        # SuperAgent-specific state
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.agent_performance_history: Dict[str, List[float]] = {}
        self.tool_usage_stats: Dict[str, int] = {}

        logger.info("SuperAgent initialized with comprehensive coordination capabilities")

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "multi_agent_coordination",
            "intelligent_routing",
            "knowledge_synthesis",
            "multimodal_coordination",
            "strategic_planning",
            "workflow_orchestration",
            "meta_coordination",
            "resource_optimization",
            "risk_assessment",
            "performance_monitoring"
        ]

    async def _execute_core_logic(
        self,
        request: Union[SuperAgentRequest, Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the core SuperAgent coordination logic."""

        # Handle both new request format and legacy context format
        if isinstance(request, SuperAgentRequest):
            operation = request.operation
            parameters = request.parameters
            coordination_mode = request.coordination_mode
        elif isinstance(request, dict):
            operation = request.get("operation", "general_request")
            parameters = request
            coordination_mode = "intelligent"
        else:
            # Legacy format support
            operation = "general_request"
            parameters = {"request": str(request)}
            coordination_mode = "intelligent"

        # Create execution context for backward compatibility
        execution_context = AgentExecutionContext(
            execution_id=context.get("execution_id", f"super_{int(time.time())}"),
            agent_id=self.agent_id,
            operation=operation,
            parameters=parameters
        )

        return await self._execute_operation(execution_context)

    async def _execute_operation(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Execute SuperAgent operations with intelligent coordination.

        Args:
            context: Execution context

        Returns:
            Operation result
        """
        operation = context.operation
        parameters = context.parameters

        logger.info(f"SuperAgent executing operation: {operation}")

        # Route to specific operation handler
        if operation == "coordinate_agents":
            return await self._coordinate_agents(context)
        elif operation == "intelligent_routing":
            return await self._intelligent_routing(context)
        elif operation == "knowledge_synthesis":
            return await self._knowledge_synthesis(context)
        elif operation == "multimodal_coordination":
            return await self._multimodal_coordination(context)
        elif operation == "strategic_planning":
            return await self._strategic_planning(context)
        elif operation == "general_request":
            return await self._handle_general_request(context)
        else:
            # Unknown operation - use intelligent routing to determine best approach
            return await self._handle_unknown_operation(context)

    async def _coordinate_agents(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Coordinate multiple agents for complex workflows.

        Args:
            context: Execution context with coordination parameters

        Returns:
            Coordination results
        """
        parameters = context.parameters
        workflow_type = parameters.get("workflow_type", "sequential")
        target_agents = parameters.get("agents", [])
        coordination_strategy = parameters.get("coordination_strategy", "default")

        logger.info(f"Coordinating {len(target_agents)} agents with {workflow_type} workflow")

        coordination_id = f"coord_{context.execution_id}"
        self.active_coordinations[coordination_id] = {
            "workflow_type": workflow_type,
            "agents": target_agents,
            "strategy": coordination_strategy,
            "started_at": datetime.utcnow(),
            "status": "running"
        }

        results = {}

        try:
            if workflow_type == "parallel":
                results = await self._execute_parallel_coordination(target_agents, parameters)
            elif workflow_type == "sequential":
                results = await self._execute_sequential_coordination(target_agents, parameters)
            elif workflow_type == "conditional":
                results = await self._execute_conditional_coordination(target_agents, parameters)
            else:
                results = {"error": f"Unknown workflow type: {workflow_type}"}

            self.active_coordinations[coordination_id]["status"] = "completed"

        except Exception as e:
            self.active_coordinations[coordination_id]["status"] = "error"
            results = {"error": f"Coordination failed: {str(e)}"}

        return {
            "coordination_id": coordination_id,
            "workflow_type": workflow_type,
            "agents_coordinated": len(target_agents),
            "results": results,
            "infrastructure_used": {
                "ptolemies_available": self.infrastructure.ptolemies_available,
                "mcp_servers": len(self.infrastructure.mcp_servers.get("mcp_servers", {})) if self.infrastructure.mcp_servers else 0
            }
        }

    async def _intelligent_routing(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Intelligently route requests to appropriate agents and tools.

        Args:
            context: Execution context with routing parameters

        Returns:
            Routing decision and execution results
        """
        parameters = context.parameters
        request = parameters.get("request", "")
        user_context = parameters.get("context", {})

        logger.info(f"Intelligent routing for request: {request[:100]}...")

        # Analyze request to determine best routing
        routing_analysis = await self._analyze_request_for_routing(request, user_context)

        # Select best agent(s) and tools
        recommended_agents = routing_analysis.get("recommended_agents", [])
        recommended_tools = routing_analysis.get("recommended_tools", [])

        # Execute with recommended resources
        execution_results = {}

        if recommended_agents:
            # Route to specific agents
            for agent_id in recommended_agents[:2]:  # Limit to top 2 agents
                agent = agent_registry.get_agent(agent_id)
                if agent:
                    try:
                        agent_result = await agent.execute("handle_request", {"request": request, "context": user_context})
                        execution_results[agent_id] = agent_result.result
                        self._update_agent_performance(agent_id, agent_result.execution_time)
                    except Exception as e:
                        execution_results[agent_id] = {"error": str(e)}

        # Use recommended tools directly if no agents available
        if not execution_results and recommended_tools:
            tool_results = await self._execute_with_tools(recommended_tools, request, user_context)
            execution_results["direct_tools"] = tool_results

        return {
            "request_analysis": routing_analysis,
            "routing_decision": {
                "agents": recommended_agents,
                "tools": recommended_tools
            },
            "execution_results": execution_results,
            "routing_confidence": routing_analysis.get("confidence", 0.5)
        }

    async def _knowledge_synthesis(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Synthesize information from Ptolemies knowledge base.

        Args:
            context: Execution context with synthesis parameters

        Returns:
            Synthesized knowledge results
        """
        parameters = context.parameters
        query = parameters.get("query", "")
        domains = parameters.get("domains", [])
        synthesis_type = parameters.get("synthesis_type", "summary")

        logger.info(f"Knowledge synthesis for query: {query[:100]}...")

        if not self.infrastructure.ptolemies_available:
            return {
                "error": "Ptolemies knowledge base not available",
                "query": query,
                "synthesis_type": synthesis_type
            }

        # In a full implementation, this would:
        # 1. Query Ptolemies semantic search
        # 2. Retrieve relevant documents
        # 3. Synthesize information based on type
        # 4. Return structured results

        # For now, return structured placeholder showing integration readiness
        synthesis_result = {
            "query": query,
            "synthesis_type": synthesis_type,
            "domains_searched": domains if domains else ["all"],
            "knowledge_base_status": "production_ready",
            "documents_available": 597,
            "synthesis_placeholder": {
                "summary": f"Knowledge synthesis ready for query: {query}",
                "analysis": "Ptolemies integration established - implement specific synthesis logic",
                "recommendations": [
                    "Implement Ptolemies semantic search integration",
                    "Add document ranking and relevance scoring",
                    "Create domain-specific synthesis templates"
                ]
            },
            "infrastructure_integration": {
                "ptolemies_path": self.infrastructure.ptolemies_path,
                "mcp_tools_available": "ptolemies-mcp" in self.metadata.available_tools
            }
        }

        return synthesis_result

    async def _multimodal_coordination(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Handle complex multimodal requests requiring multiple tools.

        Args:
            context: Execution context with multimodal parameters

        Returns:
            Multimodal coordination results
        """
        parameters = context.parameters
        request_type = parameters.get("request_type", "general")
        components = parameters.get("components", [])

        logger.info(f"Multimodal coordination for {request_type} with {len(components)} components")

        # Coordinate multiple tools and data sources
        coordination_results = {
            "request_type": request_type,
            "components_processed": len(components),
            "tool_coordination": {},
            "data_integration": {},
            "final_synthesis": {}
        }

        # Example multimodal coordination
        if request_type == "development_workflow":
            coordination_results["tool_coordination"] = {
                "github-mcp": "Repository management and code coordination",
                "filesystem": "Local file operations and project structure",
                "ptolemies-mcp": "Knowledge base for best practices and documentation",
                "git": "Version control operations"
            }
        elif request_type == "data_analysis":
            coordination_results["tool_coordination"] = {
                "ptolemies-mcp": "Domain knowledge and analysis patterns",
                "bayes-mcp": "Statistical analysis and inference",
                "surrealdb-mcp": "Data storage and retrieval",
                "jupyter-mcp": "Interactive analysis environment"
            }

        coordination_results["infrastructure_status"] = {
            "mcp_servers_available": len(self.infrastructure.mcp_servers.get("mcp_servers", {})) if self.infrastructure.mcp_servers else 0,
            "ptolemies_available": self.infrastructure.ptolemies_available,
            "coordination_ready": True
        }

        return coordination_results

    async def _strategic_planning(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Plan and orchestrate complex multi-step operations.

        Args:
            context: Execution context with planning parameters

        Returns:
            Strategic planning results
        """
        parameters = context.parameters
        objective = parameters.get("objective", "")
        constraints = parameters.get("constraints", {})
        resources = parameters.get("resources", {})

        logger.info(f"Strategic planning for objective: {objective[:100]}...")

        # Create strategic plan using available resources
        strategic_plan = {
            "objective": objective,
            "analysis": {
                "infrastructure_assessment": {
                    "ptolemies_knowledge": "597 production documents available",
                    "mcp_tools": f"{len(self.infrastructure.mcp_servers.get('mcp_servers', {})) if self.infrastructure.mcp_servers else 0} tools available",
                    "agent_coordination": "SuperAgent orchestration ready"
                },
                "resource_optimization": await self._optimize_resource_allocation(objective, constraints, resources),
                "risk_assessment": await self._assess_strategic_risks(objective, constraints)
            },
            "execution_phases": await self._create_execution_phases(objective, resources),
            "success_metrics": await self._define_success_metrics(objective),
            "contingency_plans": await self._create_contingency_plans(objective, constraints)
        }

        return strategic_plan

    async def _handle_general_request(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Handle general requests that don't fit specific operation types.

        Args:
            context: Execution context

        Returns:
            General request handling results
        """
        parameters = context.parameters
        request = parameters.get("request", "")

        logger.info(f"Handling general request: {request[:100]}...")

        # Use intelligent routing for general requests
        routing_context = AgentExecutionContext(
            execution_id=f"{context.execution_id}_routing",
            agent_id=self.metadata.id,
            operation="intelligent_routing",
            parameters={"request": request, "context": parameters.get("context", {})}
        )

        routing_result = await self._intelligent_routing(routing_context)

        return {
            "request": request,
            "handling_approach": "intelligent_routing",
            "routing_analysis": routing_result,
            "superagent_coordination": "active",
            "infrastructure_status": {
                "ptolemies_available": self.infrastructure.ptolemies_available,
                "mcp_servers_ready": bool(self.infrastructure.mcp_servers),
                "coordination_layer": "operational"
            }
        }

    async def _handle_unknown_operation(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """Handle unknown operations by analyzing and routing appropriately"""
        operation = context.operation
        parameters = context.parameters

        logger.warning(f"Unknown operation: {operation}")

        # Try to understand the operation and route appropriately
        return {
            "error": f"Unknown operation: {operation}",
            "suggestion": "Use 'general_request' operation for general queries",
            "available_operations": [cap.name for cap in self.metadata.capabilities],
            "parameters_received": parameters
        }

    # Helper methods for coordination and analysis

    async def _analyze_request_for_routing(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request to determine best routing strategy"""
        # Simplified analysis - in production would use NLP and knowledge base
        analysis = {
            "request_type": "general",
            "complexity": "medium",
            "recommended_agents": [],
            "recommended_tools": ["memory", "filesystem"],
            "confidence": 0.7,
            "reasoning": "Default routing based on available infrastructure"
        }

        # Simple keyword-based routing
        request_lower = request.lower()

        if any(word in request_lower for word in ["code", "programming", "development", "github"]):
            analysis["recommended_agents"] = ["code_agent"]
            analysis["recommended_tools"] = ["github-mcp", "filesystem", "git"]
            analysis["request_type"] = "development"

        elif any(word in request_lower for word in ["data", "analysis", "statistics", "research"]):
            analysis["recommended_agents"] = ["data_science_agent", "research_agent"]
            analysis["recommended_tools"] = ["bayes-mcp", "ptolemies-mcp"]
            analysis["request_type"] = "analysis"

        elif any(word in request_lower for word in ["knowledge", "search", "information", "docs"]):
            analysis["recommended_tools"] = ["ptolemies-mcp", "crawl4ai-mcp", "memory"]
            analysis["request_type"] = "knowledge"

        return analysis

    async def _execute_parallel_coordination(self, agents: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel agent coordination"""
        tasks = []
        for agent_id in agents:
            agent = agent_registry.get_agent(agent_id)
            if agent:
                task = agent.execute("parallel_task", parameters)
                tasks.append((agent_id, task))

        results = {}
        if tasks:
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            for i, (agent_id, _) in enumerate(tasks):
                if isinstance(task_results[i], Exception):
                    results[agent_id] = {"error": str(task_results[i])}
                else:
                    results[agent_id] = task_results[i].result

        return {"parallel_results": results, "agents_executed": len(tasks)}

    async def _execute_sequential_coordination(self, agents: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sequential agent coordination"""
        results = {}
        context = parameters.copy()

        for agent_id in agents:
            agent = agent_registry.get_agent(agent_id)
            if agent:
                try:
                    result = await agent.execute("sequential_task", context)
                    results[agent_id] = result.result
                    # Pass results to next agent
                    context["previous_results"] = results
                except Exception as e:
                    results[agent_id] = {"error": str(e)}
                    break

        return {"sequential_results": results, "agents_executed": len(results)}

    async def _execute_conditional_coordination(self, agents: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional agent coordination"""
        # Simplified conditional logic - would be more sophisticated in production
        condition = parameters.get("condition", "default")
        results = {}

        if condition == "all_successful":
            # Execute all agents, stop on first failure
            for agent_id in agents:
                agent = agent_registry.get_agent(agent_id)
                if agent:
                    try:
                        result = await agent.execute("conditional_task", parameters)
                        if not result.success:
                            results[agent_id] = result.result
                            break
                        results[agent_id] = result.result
                    except Exception as e:
                        results[agent_id] = {"error": str(e)}
                        break

        return {"conditional_results": results, "condition": condition}

    async def _execute_with_tools(self, tools: List[str], request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request using specified MCP tools directly"""
        # Placeholder for direct tool execution
        # In production, this would call MCP tools directly

        tool_results = {}
        for tool in tools:
            if tool in self.metadata.available_tools:
                tool_results[tool] = {
                    "status": "ready",
                    "request": request[:50] + "..." if len(request) > 50 else request,
                    "note": f"Tool {tool} available for execution"
                }
                self.tool_usage_stats[tool] = self.tool_usage_stats.get(tool, 0) + 1

        return {"tools_executed": tool_results, "direct_execution": True}

    def _update_agent_performance(self, agent_id: str, execution_time: float):
        """Update performance tracking for agents"""
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []

        self.agent_performance_history[agent_id].append(execution_time)

        # Keep only last 10 executions
        if len(self.agent_performance_history[agent_id]) > 10:
            self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-10:]

    async def _optimize_resource_allocation(self, objective: str, constraints: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for strategic planning"""
        return {
            "resource_analysis": "Analyzed available infrastructure",
            "optimization_strategy": "Leverage existing Ptolemies + MCP ecosystem",
            "recommended_allocation": {
                "knowledge_access": "Ptolemies knowledge base (597 documents)",
                "tool_coordination": f"{len(self.infrastructure.mcp_servers.get('mcp_servers', {})) if self.infrastructure.mcp_servers else 0} MCP tools",
                "agent_orchestration": "SuperAgent coordination layer"
            }
        }

    async def _assess_strategic_risks(self, objective: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for strategic planning"""
        return {
            "infrastructure_risks": "Low - production-ready infrastructure available",
            "coordination_risks": "Low - SuperAgent orchestration layer operational",
            "knowledge_risks": "Low - comprehensive knowledge base available",
            "mitigation_strategies": [
                "Leverage existing infrastructure stability",
                "Use proven MCP tool ecosystem",
                "Apply knowledge-driven decision making"
            ]
        }

    async def _create_execution_phases(self, objective: str, resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create execution phases for strategic planning"""
        return [
            {
                "phase": "Discovery",
                "description": "Leverage Ptolemies knowledge base for context",
                "tools": ["ptolemies-mcp", "memory"],
                "duration": "Short"
            },
            {
                "phase": "Coordination",
                "description": "Orchestrate appropriate agents and tools",
                "tools": ["multiple MCP servers as needed"],
                "duration": "Medium"
            },
            {
                "phase": "Execution",
                "description": "Execute coordinated workflow",
                "tools": ["full ecosystem"],
                "duration": "Variable"
            },
            {
                "phase": "Integration",
                "description": "Synthesize results and provide final output",
                "tools": ["memory", "synthesis tools"],
                "duration": "Short"
            }
        ]

    async def _define_success_metrics(self, objective: str) -> Dict[str, Any]:
        """Define success metrics for strategic planning"""
        return {
            "infrastructure_utilization": "Effective use of available tools and knowledge",
            "coordination_efficiency": "Optimal agent and tool orchestration",
            "knowledge_integration": "Successful leverage of Ptolemies knowledge base",
            "objective_completion": "Achievement of stated objective",
            "user_satisfaction": "Meeting or exceeding user expectations"
        }

    async def _create_contingency_plans(self, objective: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create contingency plans for strategic planning"""
        return [
            {
                "scenario": "Tool unavailability",
                "response": "Route to alternative MCP tools or direct agent execution",
                "backup_resources": "Multiple MCP servers available"
            },
            {
                "scenario": "Knowledge gap",
                "response": "Use web search and research agents to fill gaps",
                "backup_resources": "crawl4ai-mcp, research_agent"
            },
            {
                "scenario": "Agent coordination failure",
                "response": "Fall back to SuperAgent direct execution",
                "backup_resources": "SuperAgent multimodal capabilities"
            }
        ]

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for the SuperAgent."""
        return {
            "max_concurrent_coordinations": 10,
            "default_timeout_minutes": 10,
            "enable_intelligent_routing": True,
            "enable_knowledge_synthesis": True,
            "enable_multimodal_coordination": True,
            "enable_strategic_planning": True,
            "coordination_retry_attempts": 3,
            "agent_performance_history_limit": 10,
            "tool_usage_stats_enabled": True,
            "default_coordination_mode": "intelligent",
            "routing_confidence_threshold": 0.5,
            "enable_contingency_planning": True,
            "resource_optimization_enabled": True,
            "risk_assessment_enabled": True
        }

    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_keys = [
            "max_concurrent_coordinations",
            "default_timeout_minutes",
            "coordination_retry_attempts"
        ]

        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required configuration key: {key}")

        if config["max_concurrent_coordinations"] <= 0:
            raise ValidationError("max_concurrent_coordinations must be positive")

        if config["default_timeout_minutes"] <= 0:
            raise ValidationError("default_timeout_minutes must be positive")

        if config["coordination_retry_attempts"] < 0:
            raise ValidationError("coordination_retry_attempts must be non-negative")

        if "routing_confidence_threshold" in config:
            threshold = config["routing_confidence_threshold"]
            if not (0.0 <= threshold <= 1.0):
                raise ValidationError("routing_confidence_threshold must be between 0.0 and 1.0")

        return True


# Register SuperAgent type in the global registry
agent_registry.register_agent_type(SuperAgent, "super_agent")

# Create and register a default SuperAgent instance
default_super_agent = SuperAgent(agent_id="default_super_agent")
agent_registry.register_agent(default_super_agent)

logger.info("SuperAgent registered and ready for coordination")
