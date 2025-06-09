"""
Agents Package for Agentical Framework

This package provides the agent infrastructure for the Agentical framework,
including base agent classes, concrete agent implementations, and agent
management utilities.

The agent system is based on a perception-decision-action model, where agents:
1. Perceive inputs and context (perception phase)
2. Make decisions based on their understanding (decision phase)
3. Take actions through tool usage (action phase)
4. Reflect on their performance (reflection phase)
"""

from agentical.agents.base_agent import (
    BaseAgent,
    AgentMetadata,
    AgentCapability,
    AgentStatus,
    AgentExecutionContext,
    AgentExecutionResult,
    InfrastructureConnections
)

from agentical.agents.generic_agent import (
    GenericAgent,
    PerceptionResult,
    DecisionResult,
    ActionResult,
    ToolExecutionResult
)

from agentical.agents.agent_registry import (
    AgentRegistry,
    get_agent_registry
)

# Re-export for convenience
agent_registry = get_agent_registry()

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentMetadata",
    "AgentCapability",
    "AgentStatus",
    "AgentExecutionContext",
    "AgentExecutionResult",
    "InfrastructureConnections",
    
    # Concrete implementations
    "GenericAgent",
    "PerceptionResult",
    "DecisionResult",
    "ActionResult", 
    "ToolExecutionResult",
    
    # Registry
    "AgentRegistry",
    "get_agent_registry",
    "agent_registry"
]