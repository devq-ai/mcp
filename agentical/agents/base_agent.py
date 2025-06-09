"""
Agentical Base Agent

Foundation class for all agents in the Agentical framework.
Integrates with existing DevQ.ai infrastructure:
- Ptolemies Knowledge Base (597 production documents)
- MCP Server Ecosystem (26 operational servers)
- SurrealDB for state management
- Logfire for observability

This base class provides:
- Pydantic AI integration for type-safe agent operations
- Knowledge base access through Ptolemies
- Tool coordination via MCP servers
- State persistence and recovery
- Comprehensive logging and error handling
"""

import os
import sys
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Type
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class AgentCapability(BaseModel):
    """Definition of an agent capability"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input validation schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output format schema")
    required_tools: List[str] = Field(default_factory=list, description="Required MCP tools")
    knowledge_domains: List[str] = Field(default_factory=list, description="Relevant knowledge domains")


class AgentMetadata(BaseModel):
    """Agent metadata and configuration"""
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(..., description="Agent purpose and functionality")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    available_tools: List[str] = Field(default_factory=list, description="Available MCP tools")
    model: str = Field(default="claude-3-7-sonnet-20250219", description="LLM model to use")
    system_prompts: List[str] = Field(default_factory=list, description="System prompts for the agent")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Agent classification tags")


class AgentExecutionContext(BaseModel):
    """Context for agent execution"""
    execution_id: str = Field(..., description="Unique execution identifier")
    agent_id: str = Field(..., description="Agent identifier")
    operation: str = Field(..., description="Operation being performed")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    knowledge_context: Dict[str, Any] = Field(default_factory=dict, description="Knowledge from Ptolemies")
    tool_results: Dict[str, Any] = Field(default_factory=dict, description="Results from MCP tools")
    state: Dict[str, Any] = Field(default_factory=dict, description="Agent state")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class AgentExecutionResult(BaseModel):
    """Result of agent execution"""
    success: bool = Field(..., description="Execution success status")
    execution_id: str = Field(..., description="Execution identifier")
    agent_id: str = Field(..., description="Agent identifier")
    operation: str = Field(..., description="Operation performed")
    result: Dict[str, Any] = Field(default_factory=dict, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    tools_used: List[str] = Field(default_factory=list, description="MCP tools that were used")
    knowledge_queries: int = Field(default=0, description="Number of Ptolemies queries")
    tokens_used: Optional[int] = Field(None, description="LLM tokens consumed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


@dataclass
class InfrastructureConnections:
    """Infrastructure connection details"""
    ptolemies_available: bool = False
    ptolemies_path: Optional[str] = None
    mcp_servers: Dict[str, Any] = None
    surrealdb_config: Dict[str, str] = None


class BaseAgent(ABC):
    """
    Base class for all Agentical agents.
    
    Provides infrastructure integration, state management, and execution framework.
    All agents should inherit from this class and implement required methods.
    """
    
    def __init__(self, metadata: AgentMetadata):
        """
        Initialize the base agent.
        
        Args:
            metadata: Agent metadata and configuration
        """
        self.metadata = metadata
        self.status = AgentStatus.IDLE
        self.infrastructure = self._initialize_infrastructure()
        self.execution_history: List[AgentExecutionResult] = []
        self.current_context: Optional[AgentExecutionContext] = None
        
        logger.info(f"Initialized agent {self.metadata.id} ({self.metadata.name})")
    
    def _initialize_infrastructure(self) -> InfrastructureConnections:
        """Initialize connections to existing infrastructure"""
        infrastructure = InfrastructureConnections()
        
        # Find project root and Ptolemies
        project_root = self._find_project_root()
        ptolemies_path = project_root / "ptolemies"
        
        if ptolemies_path.exists():
            infrastructure.ptolemies_available = True
            infrastructure.ptolemies_path = str(ptolemies_path)
            logger.info(f"Ptolemies knowledge base found at {ptolemies_path}")
        
        # Load MCP server configuration
        mcp_config_path = project_root / "mcp" / "mcp-servers.json"
        if mcp_config_path.exists():
            try:
                with open(mcp_config_path, 'r') as f:
                    infrastructure.mcp_servers = json.load(f)
                logger.info(f"Loaded {len(infrastructure.mcp_servers.get('mcp_servers', {}))} MCP servers")
            except Exception as e:
                logger.warning(f"Failed to load MCP config: {e}")
        
        # SurrealDB configuration
        infrastructure.surrealdb_config = {
            "url": os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc"),
            "username": os.getenv("SURREALDB_USERNAME", "root"),
            "password": os.getenv("SURREALDB_PASSWORD", "root"),
            "namespace": os.getenv("SURREALDB_NAMESPACE", "devq"),
            "database": os.getenv("SURREALDB_DATABASE", "main")
        }
        
        return infrastructure
    
    def _find_project_root(self) -> Path:
        """Find the DevQ.ai project root directory"""
        current = Path(__file__).parent
        
        while current.parent != current:
            if (current / "ptolemies").exists() and (current / "mcp").exists():
                return current
            if current.name == "devqai":
                return current
            current = current.parent
            
        return Path(__file__).parent.parent.parent
    
    async def execute(self, operation: str, parameters: Dict[str, Any] = None) -> AgentExecutionResult:
        """
        Execute an agent operation with full infrastructure integration.
        
        Args:
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            Execution result
        """
        execution_id = f"{self.metadata.id}_{operation}_{datetime.utcnow().isoformat()}"
        start_time = datetime.utcnow()
        
        # Create execution context
        context = AgentExecutionContext(
            execution_id=execution_id,
            agent_id=self.metadata.id,
            operation=operation,
            parameters=parameters or {}
        )
        
        self.current_context = context
        self.status = AgentStatus.RUNNING
        
        try:
            logger.info(f"Starting execution {execution_id} for operation {operation}")
            
            # Pre-execution: Gather knowledge context
            if context.parameters.get("use_knowledge", True):
                context.knowledge_context = await self._gather_knowledge_context(operation, parameters)
                context.knowledge_queries = len(context.knowledge_context)
            
            # Pre-execution: Validate tools
            required_tools = await self._get_required_tools(operation)
            available_tools = await self._validate_tool_availability(required_tools)
            
            # Execute the specific agent operation
            result_data = await self._execute_operation(context)
            
            # Post-execution: Update state if needed
            if hasattr(self, '_update_agent_state'):
                await self._update_agent_state(context, result_data)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create successful result
            result = AgentExecutionResult(
                success=True,
                execution_id=execution_id,
                agent_id=self.metadata.id,
                operation=operation,
                result=result_data,
                execution_time=execution_time,
                tools_used=available_tools,
                knowledge_queries=context.knowledge_queries
            )
            
            self.status = AgentStatus.COMPLETED
            logger.info(f"Completed execution {execution_id} successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            # Create error result
            result = AgentExecutionResult(
                success=False,
                execution_id=execution_id,
                agent_id=self.metadata.id,
                operation=operation,
                error=error_msg,
                execution_time=execution_time,
                knowledge_queries=context.knowledge_queries if context else 0
            )
            
            self.status = AgentStatus.ERROR
            logger.error(f"Execution {execution_id} failed: {error_msg}")
        
        finally:
            self.current_context = None
            self.execution_history.append(result)
        
        return result
    
    async def _gather_knowledge_context(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather relevant context from Ptolemies knowledge base.
        
        Args:
            operation: Operation being performed
            parameters: Operation parameters
            
        Returns:
            Knowledge context from Ptolemies
        """
        if not self.infrastructure.ptolemies_available:
            return {"error": "Ptolemies knowledge base not available"}
        
        try:
            # In a full implementation, this would:
            # 1. Query Ptolemies semantic search
            # 2. Get relevant documents based on operation
            # 3. Extract key information for agent context
            
            # For now, return placeholder indicating Ptolemies integration
            return {
                "status": "ptolemies_available",
                "documents_available": 597,
                "operation": operation,
                "note": "Ptolemies integration ready - implement specific queries based on operation"
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather knowledge context: {e}")
            return {"error": f"Knowledge gathering failed: {e}"}
    
    async def _get_required_tools(self, operation: str) -> List[str]:
        """
        Get required MCP tools for an operation.
        
        Args:
            operation: Operation being performed
            
        Returns:
            List of required tool names
        """
        # Check if operation has specific tool requirements
        for capability in self.metadata.capabilities:
            if capability.name == operation:
                return capability.required_tools
        
        # Default tools for most operations
        return ["filesystem", "memory"]
    
    async def _validate_tool_availability(self, required_tools: List[str]) -> List[str]:
        """
        Validate that required MCP tools are available.
        
        Args:
            required_tools: List of required tool names
            
        Returns:
            List of available tools
        """
        if not self.infrastructure.mcp_servers:
            return []
        
        available_servers = self.infrastructure.mcp_servers.get("mcp_servers", {})
        available_tools = []
        
        for tool in required_tools:
            if tool in available_servers:
                available_tools.append(tool)
            else:
                logger.warning(f"Required tool {tool} not available in MCP servers")
        
        return available_tools
    
    @abstractmethod
    async def _execute_operation(self, context: AgentExecutionContext) -> Dict[str, Any]:
        """
        Execute the specific agent operation.
        
        This method must be implemented by all concrete agents.
        
        Args:
            context: Execution context with all necessary information
            
        Returns:
            Operation result data
        """
        pass
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return self.metadata.capabilities
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.metadata.id,
            "name": self.metadata.name,
            "status": self.status.value,
            "current_operation": self.current_context.operation if self.current_context else None,
            "execution_history_count": len(self.execution_history),
            "infrastructure": {
                "ptolemies_available": self.infrastructure.ptolemies_available,
                "mcp_servers_count": len(self.infrastructure.mcp_servers.get("mcp_servers", {})) if self.infrastructure.mcp_servers else 0,
                "surrealdb_configured": bool(self.infrastructure.surrealdb_config.get("url"))
            }
        }
    
    async def get_execution_history(self, limit: Optional[int] = None) -> List[AgentExecutionResult]:
        """Get agent execution history"""
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.metadata.id}, status={self.status.value})>"


class AgentRegistry:
    """
    Registry for managing agent instances and discovery.
    
    Integrates with existing infrastructure to provide agent coordination.
    """
    
    def __init__(self):
        """Initialize the agent registry"""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        logger.info("Initialized agent registry")
    
    def register_agent_type(self, agent_type: Type[BaseAgent], type_name: str):
        """
        Register an agent type for dynamic creation.
        
        Args:
            agent_type: Agent class
            type_name: Type identifier
        """
        self.agent_types[type_name] = agent_type
        logger.info(f"Registered agent type: {type_name}")
    
    def register_agent(self, agent: BaseAgent):
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance
        """
        self.agents[agent.metadata.id] = agent
        logger.info(f"Registered agent: {agent.metadata.id}")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "id": agent.metadata.id,
                "name": agent.metadata.name,
                "type": agent.__class__.__name__,
                "status": agent.status.value,
                "capabilities": len(agent.metadata.capabilities)
            }
            for agent in self.agents.values()
        ]
    
    async def execute_agent(self, agent_id: str, operation: str, parameters: Dict[str, Any] = None) -> AgentExecutionResult:
        """
        Execute an operation on a specific agent.
        
        Args:
            agent_id: Agent identifier
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            Execution result
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        return await agent.execute(operation, parameters)


# Global agent registry instance
agent_registry = AgentRegistry()