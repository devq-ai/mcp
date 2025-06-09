"""
Agent Registry for Agentical Framework

This module provides a centralized registry for creating, storing, and accessing
agents within the Agentical framework. It handles agent instantiation, lifecycle
management, and execution routing.

Features:
- Agent type registration and instantiation
- Agent instance management
- Agent discovery and listing
- Execution routing and coordination
- Caching and optimization
"""

from typing import Dict, Any, List, Optional, Type, Callable, Union
import os
import json
import logging
import asyncio
from datetime import datetime
import uuid
from pathlib import Path

import logfire

from agentical.agents.base_agent import BaseAgent, AgentMetadata, AgentExecutionResult
from agentical.agents.generic_agent import GenericAgent
from agentical.agents.super_agent import SuperAgent
from agentical.core.exceptions import (
    AgenticalError, 
    AgentError,
    AgentExecutionError, 
    AgentNotFoundError
)

# Set up logging
logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing agents in the Agentical framework.
    
    The registry stores agent types and instances, handles agent creation,
    and routes execution requests to the appropriate agent.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the agent registry."""
        # Only initialize once (singleton pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.agent_factories: Dict[str, Callable[..., BaseAgent]] = {}
        
        # Register built-in agent types
        self.register_agent_type("generic", GenericAgent)
        self.register_agent_type("super", SuperAgent)
        
        # Pre-initialize some standard agents
        self.get_or_create_agent("generic_agent", "generic")
        self.get_or_create_agent("super_agent", "super")
        
        self.initialized = True
        logger.info("Agent registry initialized with built-in agent types: generic, super")
    
    def register_agent_type(self, type_name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent type with the registry.
        
        Args:
            type_name: Unique type name
            agent_class: Agent class (must inherit from BaseAgent)
        
        Raises:
            ValueError: If agent_class does not inherit from BaseAgent
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class {agent_class.__name__} must inherit from BaseAgent")
            
        self.agent_types[type_name] = agent_class
        logger.info(f"Registered agent type: {type_name}")
    
    def register_agent_factory(self, type_name: str, factory: Callable[..., BaseAgent]) -> None:
        """
        Register a factory function for creating agents of a specific type.
        
        Args:
            type_name: Unique type name
            factory: Factory function that returns a BaseAgent instance
        """
        self.agent_factories[type_name] = factory
        logger.info(f"Registered agent factory for type: {type_name}")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an existing agent instance with the registry.
        
        Args:
            agent: Agent instance to register
        """
        self.agent_instances[agent.metadata.id] = agent
        logger.info(f"Registered agent instance: {agent.metadata.id}")
    
    def create_agent(self, agent_id: str, agent_type: str, **kwargs) -> BaseAgent:
        """
        Create a new agent of the specified type.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent to create
            **kwargs: Additional arguments to pass to the agent constructor
        
        Returns:
            Newly created agent instance
            
        Raises:
            ValueError: If agent_type is not registered
        """
        with logfire.span("Create agent", agent_id=agent_id, agent_type=agent_type):
            # Check if agent type exists
            if agent_type not in self.agent_types and agent_type not in self.agent_factories:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Check if agent already exists
            if agent_id in self.agent_instances:
                logger.warning(f"Agent {agent_id} already exists, returning existing instance")
                return self.agent_instances[agent_id]
            
            # Create the agent
            if agent_type in self.agent_factories:
                # Use factory function
                agent = self.agent_factories[agent_type](agent_id=agent_id, **kwargs)
            else:
                # Use agent class constructor
                agent_class = self.agent_types[agent_type]
                agent = agent_class(agent_id=agent_id, **kwargs)
            
            # Register the agent
            self.agent_instances[agent_id] = agent
            
            logfire.info(f"Created agent {agent_id} of type {agent_type}")
            return agent
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent ID to retrieve
        
        Returns:
            Agent instance
            
        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_id not in self.agent_instances:
            raise AgentNotFoundError(f"Agent not found: {agent_id}")
        
        return self.agent_instances[agent_id]
    
    def get_or_create_agent(self, agent_id: str, agent_type: str, **kwargs) -> BaseAgent:
        """
        Get an existing agent or create it if it doesn't exist.
        
        Args:
            agent_id: Agent ID to retrieve or create
            agent_type: Type of agent to create if it doesn't exist
            **kwargs: Additional arguments for agent creation
        
        Returns:
            Agent instance
        """
        try:
            return self.get_agent(agent_id)
        except AgentNotFoundError:
            return self.create_agent(agent_id, agent_type, **kwargs)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents with their metadata.
        
        Returns:
            List of agent metadata
        """
        agents_info = []
        
        for agent_id, agent in self.agent_instances.items():
            agents_info.append({
                "id": agent.metadata.id,
                "name": agent.metadata.name,
                "type": agent.__class__.__name__,
                "status": agent.status.value,
                "description": agent.metadata.description,
                "capabilities_count": len(agent.metadata.capabilities)
            })
        
        return agents_info
    
    def get_available_agent_types(self) -> List[str]:
        """
        Get a list of available agent types.
        
        Returns:
            List of agent type names
        """
        return list(self.agent_types.keys()) + list(self.agent_factories.keys())
    
    async def execute_agent(self, agent_id: str, operation: str, parameters: Dict[str, Any] = None) -> AgentExecutionResult:
        """
        Execute an operation on an agent.
        
        Args:
            agent_id: ID of the agent to execute
            operation: Operation to execute
            parameters: Operation parameters
        
        Returns:
            Execution result
            
        Raises:
            AgentNotFoundError: If agent is not found
            AgentExecutionError: If execution fails
        """
        with logfire.span("Agent registry execution", agent_id=agent_id, operation=operation):
            try:
                # Get the agent
                agent = self.get_agent(agent_id)
                
                # Execute the operation
                result = await agent.execute(operation, parameters or {})
                
                logfire.info(f"Executed {operation} on agent {agent_id}", 
                           success=result.success,
                           execution_time=result.execution_time)
                
                return result
                
            except AgentNotFoundError:
                logfire.error(f"Agent not found: {agent_id}")
                raise
                
            except Exception as e:
                logfire.error(f"Error executing {operation} on agent {agent_id}: {str(e)}")
                raise AgentExecutionError(f"Error executing {operation} on agent {agent_id}: {str(e)}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_id not in self.agent_instances:
            raise AgentNotFoundError(f"Agent not found: {agent_id}")
        
        del self.agent_instances[agent_id]
        logger.info(f"Unregistered agent: {agent_id}")
    
    def clear_registry(self) -> None:
        """Clear all registered agents."""
        self.agent_instances.clear()
        logger.info("Cleared agent registry")
    
    def get_agent_count(self) -> int:
        """
        Get the number of registered agents.
        
        Returns:
            Number of registered agents
        """
        return len(self.agent_instances)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about the registry.
        
        Returns:
            Registry information
        """
        return {
            "agent_count": len(self.agent_instances),
            "agent_type_count": len(self.agent_types) + len(self.agent_factories),
            "available_agent_types": self.get_available_agent_types(),
            "agents": self.list_agents()
        }


# Create a global instance of the registry
agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.
    
    Returns:
        Global agent registry
    """
    return agent_registry