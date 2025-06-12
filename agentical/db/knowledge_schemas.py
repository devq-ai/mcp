"""
Agent Knowledge Schemas for Agentical

This module defines comprehensive knowledge schemas for different types of agents,
including their capabilities, tool usage patterns, and domain-specific knowledge
structures for the SurrealDB graph database.

Features:
- Agent-specific knowledge schema definitions
- Capability and skill modeling
- Tool usage pattern schemas
- Domain expertise representation
- Knowledge graph relationship patterns
- Performance and learning metrics schemas
"""

from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class AgentDomain(str, Enum):
    """Domains of agent expertise."""
    CODE_DEVELOPMENT = "code_development"
    DATA_SCIENCE = "data_science"
    DATABASE_ADMIN = "database_admin"
    DEVOPS = "devops"
    CLOUD_PLATFORM = "cloud_platform"
    VERSION_CONTROL = "version_control"
    LEGAL_ANALYSIS = "legal_analysis"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    RESEARCH = "research"
    TESTING = "testing"
    BLOCKCHAIN = "blockchain"
    USER_EXPERIENCE = "user_experience"
    DOCUMENTATION = "documentation"
    INSPECTION = "inspection"
    PLAYBOOK_MANAGEMENT = "playbook_management"
    META_COORDINATION = "meta_coordination"


class KnowledgeType(str, Enum):
    """Types of knowledge an agent can possess."""
    PROCEDURAL = "procedural"  # How to do things
    DECLARATIVE = "declarative"  # Facts and information
    CONDITIONAL = "conditional"  # When to do things
    CONTEXTUAL = "contextual"  # Situational knowledge
    EXPERIENTIAL = "experiential"  # Learned from experience
    DOMAIN_SPECIFIC = "domain_specific"  # Specialized domain knowledge


class CapabilityLevel(str, Enum):
    """Levels of capability proficiency."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


@dataclass
class KnowledgeEntity:
    """Represents a piece of knowledge in the agent's knowledge base."""
    id: str
    title: str
    description: str
    knowledge_type: KnowledgeType
    domain: AgentDomain
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    success_rate: float = 0.0


@dataclass
class AgentCapability:
    """Represents a specific capability of an agent."""
    name: str
    description: str
    domain: AgentDomain
    level: CapabilityLevel
    required_tools: List[str] = field(default_factory=list)
    knowledge_requirements: List[str] = field(default_factory=list)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    learning_resources: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolUsagePattern:
    """Represents how an agent typically uses a tool."""
    tool_name: str
    usage_frequency: float
    success_rate: float
    common_parameters: Dict[str, Any] = field(default_factory=dict)
    typical_contexts: List[str] = field(default_factory=list)
    error_patterns: List[Dict[str, Any]] = field(default_factory=list)
    optimization_tips: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None


@dataclass
class LearningRecord:
    """Records learning events and knowledge acquisition."""
    id: str
    agent_id: str
    knowledge_entity_id: str
    learning_event: str
    outcome: str
    confidence_change: float
    performance_impact: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)


class AgentKnowledgeSchema:
    """
    Base schema for agent knowledge representation.

    This class provides the foundation for modeling agent knowledge,
    capabilities, and learning patterns in the graph database.
    """

    def __init__(self, agent_id: str, domain: AgentDomain):
        self.agent_id = agent_id
        self.domain = domain
        self.knowledge_entities: Dict[str, KnowledgeEntity] = {}
        self.capabilities: Dict[str, AgentCapability] = {}
        self.tool_patterns: Dict[str, ToolUsagePattern] = {}
        self.learning_history: List[LearningRecord] = []
        self.schema_version = "1.0"
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def add_knowledge_entity(self, entity: KnowledgeEntity) -> str:
        """Add a knowledge entity to the agent's knowledge base."""
        self.knowledge_entities[entity.id] = entity
        self.updated_at = datetime.utcnow()
        return entity.id

    def add_capability(self, capability: AgentCapability) -> str:
        """Add a capability to the agent's capability set."""
        self.capabilities[capability.name] = capability
        self.updated_at = datetime.utcnow()
        return capability.name

    def record_tool_usage(self, tool_name: str, success: bool,
                         parameters: Dict[str, Any] = None,
                         context: str = None) -> None:
        """Record tool usage for pattern learning."""
        if tool_name not in self.tool_patterns:
            self.tool_patterns[tool_name] = ToolUsagePattern(
                tool_name=tool_name,
                usage_frequency=0.0,
                success_rate=0.0
            )

        pattern = self.tool_patterns[tool_name]
        pattern.usage_frequency += 1

        # Update success rate using running average
        if pattern.success_rate == 0.0:
            pattern.success_rate = 1.0 if success else 0.0
        else:
            pattern.success_rate = (pattern.success_rate + (1.0 if success else 0.0)) / 2

        if parameters:
            # Update common parameters
            for key, value in parameters.items():
                if key not in pattern.common_parameters:
                    pattern.common_parameters[key] = []
                pattern.common_parameters[key].append(value)

        if context and context not in pattern.typical_contexts:
            pattern.typical_contexts.append(context)

        pattern.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def get_relevant_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeEntity]:
        """Get knowledge entities relevant to a query."""
        # Simple relevance scoring based on title and tags
        relevant = []
        query_lower = query.lower()

        for entity in self.knowledge_entities.values():
            score = 0.0

            # Title match
            if query_lower in entity.title.lower():
                score += 2.0

            # Description match
            if query_lower in entity.description.lower():
                score += 1.0

            # Tag match
            for tag in entity.tags:
                if query_lower in tag.lower():
                    score += 0.5

            # Confidence and success rate boost
            score *= entity.confidence * (entity.success_rate if entity.success_rate > 0 else 0.5)

            if score > 0:
                relevant.append((entity, score))

        # Sort by relevance score and return top results
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, score in relevant[:limit]]

    def to_graph_representation(self) -> Dict[str, Any]:
        """Convert schema to graph database representation."""
        return {
            "agent_id": self.agent_id,
            "domain": self.domain.value,
            "schema_version": self.schema_version,
            "knowledge_entities": {
                eid: {
                    "id": entity.id,
                    "title": entity.title,
                    "description": entity.description,
                    "knowledge_type": entity.knowledge_type.value,
                    "domain": entity.domain.value,
                    "content": entity.content,
                    "tags": entity.tags,
                    "confidence": entity.confidence,
                    "source": entity.source,
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat(),
                    "access_count": entity.access_count,
                    "success_rate": entity.success_rate
                }
                for eid, entity in self.knowledge_entities.items()
            },
            "capabilities": {
                name: {
                    "name": cap.name,
                    "description": cap.description,
                    "domain": cap.domain.value,
                    "level": cap.level.value,
                    "required_tools": cap.required_tools,
                    "knowledge_requirements": cap.knowledge_requirements,
                    "success_metrics": cap.success_metrics,
                    "learning_resources": cap.learning_resources,
                    "prerequisites": cap.prerequisites,
                    "performance_history": cap.performance_history
                }
                for name, cap in self.capabilities.items()
            },
            "tool_patterns": {
                tool: {
                    "tool_name": pattern.tool_name,
                    "usage_frequency": pattern.usage_frequency,
                    "success_rate": pattern.success_rate,
                    "common_parameters": pattern.common_parameters,
                    "typical_contexts": pattern.typical_contexts,
                    "error_patterns": pattern.error_patterns,
                    "optimization_tips": pattern.optimization_tips,
                    "last_used": pattern.last_used.isoformat() if pattern.last_used else None
                }
                for tool, pattern in self.tool_patterns.items()
            },
            "learning_history": [
                {
                    "id": record.id,
                    "agent_id": record.agent_id,
                    "knowledge_entity_id": record.knowledge_entity_id,
                    "learning_event": record.learning_event,
                    "outcome": record.outcome,
                    "confidence_change": record.confidence_change,
                    "performance_impact": record.performance_impact,
                    "timestamp": record.timestamp.isoformat(),
                    "context": record.context
                }
                for record in self.learning_history
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class CodeAgentSchema(AgentKnowledgeSchema):
    """Knowledge schema for code development agents."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentDomain.CODE_DEVELOPMENT)
        self._initialize_code_capabilities()
        self._initialize_code_knowledge()

    def _initialize_code_capabilities(self):
        """Initialize standard code development capabilities."""
        capabilities = [
            AgentCapability(
                name="code_generation",
                description="Generate code in various programming languages",
                domain=AgentDomain.CODE_DEVELOPMENT,
                level=CapabilityLevel.ADVANCED,
                required_tools=["editor", "compiler", "linter"],
                knowledge_requirements=["programming_languages", "design_patterns", "best_practices"]
            ),
            AgentCapability(
                name="code_review",
                description="Review code for quality, security, and best practices",
                domain=AgentDomain.CODE_DEVELOPMENT,
                level=CapabilityLevel.EXPERT,
                required_tools=["static_analyzer", "security_scanner"],
                knowledge_requirements=["code_quality_standards", "security_vulnerabilities"]
            ),
            AgentCapability(
                name="testing",
                description="Write and execute tests for code validation",
                domain=AgentDomain.CODE_DEVELOPMENT,
                level=CapabilityLevel.ADVANCED,
                required_tools=["test_framework", "coverage_tool"],
                knowledge_requirements=["testing_strategies", "test_automation"]
            ),
            AgentCapability(
                name="debugging",
                description="Identify and fix bugs in code",
                domain=AgentDomain.CODE_DEVELOPMENT,
                level=CapabilityLevel.EXPERT,
                required_tools=["debugger", "profiler", "logger"],
                knowledge_requirements=["debugging_techniques", "error_patterns"]
            )
        ]

        for cap in capabilities:
            self.add_capability(cap)

    def _initialize_code_knowledge(self):
        """Initialize standard code development knowledge."""
        knowledge_entities = [
            KnowledgeEntity(
                id="python_best_practices",
                title="Python Best Practices",
                description="Best practices for Python development",
                knowledge_type=KnowledgeType.PROCEDURAL,
                domain=AgentDomain.CODE_DEVELOPMENT,
                content={
                    "practices": [
                        "Use meaningful variable names",
                        "Follow PEP 8 style guide",
                        "Write docstrings for functions",
                        "Handle exceptions properly",
                        "Use list comprehensions appropriately"
                    ],
                    "anti_patterns": [
                        "Using global variables excessively",
                        "Not using context managers for resources",
                        "Ignoring exceptions silently"
                    ]
                },
                tags=["python", "best_practices", "coding_standards"]
            ),
            KnowledgeEntity(
                id="design_patterns",
                title="Software Design Patterns",
                description="Common software design patterns and their applications",
                knowledge_type=KnowledgeType.DECLARATIVE,
                domain=AgentDomain.CODE_DEVELOPMENT,
                content={
                    "patterns": {
                        "singleton": "Ensure a class has only one instance",
                        "factory": "Create objects without specifying exact classes",
                        "observer": "Define subscription mechanism for object events",
                        "strategy": "Define family of algorithms and make them interchangeable"
                    }
                },
                tags=["design_patterns", "architecture", "oop"]
            )
        ]

        for entity in knowledge_entities:
            self.add_knowledge_entity(entity)


class DataScienceAgentSchema(AgentKnowledgeSchema):
    """Knowledge schema for data science agents."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentDomain.DATA_SCIENCE)
        self._initialize_ds_capabilities()
        self._initialize_ds_knowledge()

    def _initialize_ds_capabilities(self):
        """Initialize data science capabilities."""
        capabilities = [
            AgentCapability(
                name="data_analysis",
                description="Analyze and explore datasets",
                domain=AgentDomain.DATA_SCIENCE,
                level=CapabilityLevel.EXPERT,
                required_tools=["pandas", "numpy", "matplotlib", "seaborn"],
                knowledge_requirements=["statistics", "data_visualization"]
            ),
            AgentCapability(
                name="machine_learning",
                description="Build and deploy ML models",
                domain=AgentDomain.DATA_SCIENCE,
                level=CapabilityLevel.ADVANCED,
                required_tools=["scikit-learn", "tensorflow", "pytorch"],
                knowledge_requirements=["ml_algorithms", "model_evaluation"]
            ),
            AgentCapability(
                name="feature_engineering",
                description="Create and select features for ML models",
                domain=AgentDomain.DATA_SCIENCE,
                level=CapabilityLevel.ADVANCED,
                required_tools=["feature_tools", "sklearn_preprocessing"],
                knowledge_requirements=["feature_selection", "dimensionality_reduction"]
            )
        ]

        for cap in capabilities:
            self.add_capability(cap)

    def _initialize_ds_knowledge(self):
        """Initialize data science knowledge."""
        knowledge_entities = [
            KnowledgeEntity(
                id="ml_model_selection",
                title="Machine Learning Model Selection",
                description="Guidelines for selecting appropriate ML models",
                knowledge_type=KnowledgeType.CONDITIONAL,
                domain=AgentDomain.DATA_SCIENCE,
                content={
                    "classification": {
                        "small_dataset": ["logistic_regression", "svm", "random_forest"],
                        "large_dataset": ["gradient_boosting", "neural_networks"],
                        "interpretability_required": ["decision_tree", "logistic_regression"]
                    },
                    "regression": {
                        "linear_relationship": ["linear_regression", "ridge", "lasso"],
                        "non_linear": ["random_forest", "gradient_boosting", "neural_networks"]
                    }
                },
                tags=["machine_learning", "model_selection", "algorithms"]
            )
        ]

        for entity in knowledge_entities:
            self.add_knowledge_entity(entity)


class SuperAgentSchema(AgentKnowledgeSchema):
    """Knowledge schema for super/meta-coordination agents."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentDomain.META_COORDINATION)
        self._initialize_coordination_capabilities()
        self._initialize_coordination_knowledge()

    def _initialize_coordination_capabilities(self):
        """Initialize meta-coordination capabilities."""
        capabilities = [
            AgentCapability(
                name="agent_orchestration",
                description="Coordinate multiple agents for complex tasks",
                domain=AgentDomain.META_COORDINATION,
                level=CapabilityLevel.MASTER,
                required_tools=["workflow_engine", "message_queue", "monitoring"],
                knowledge_requirements=["agent_capabilities", "task_decomposition", "coordination_patterns"]
            ),
            AgentCapability(
                name="resource_allocation",
                description="Allocate computational and tool resources efficiently",
                domain=AgentDomain.META_COORDINATION,
                level=CapabilityLevel.EXPERT,
                required_tools=["resource_monitor", "scheduler"],
                knowledge_requirements=["resource_optimization", "performance_metrics"]
            ),
            AgentCapability(
                name="conflict_resolution",
                description="Resolve conflicts between agents or tasks",
                domain=AgentDomain.META_COORDINATION,
                level=CapabilityLevel.ADVANCED,
                required_tools=["decision_engine", "priority_queue"],
                knowledge_requirements=["conflict_patterns", "resolution_strategies"]
            )
        ]

        for cap in capabilities:
            self.add_capability(cap)

    def _initialize_coordination_knowledge(self):
        """Initialize coordination knowledge."""
        knowledge_entities = [
            KnowledgeEntity(
                id="coordination_patterns",
                title="Multi-Agent Coordination Patterns",
                description="Patterns for coordinating multiple agents",
                knowledge_type=KnowledgeType.PROCEDURAL,
                domain=AgentDomain.META_COORDINATION,
                content={
                    "patterns": {
                        "pipeline": "Sequential processing through specialized agents",
                        "broadcast": "Distribute task to multiple agents simultaneously",
                        "auction": "Agents bid for tasks based on capability and availability",
                        "hierarchy": "Tree-like delegation of tasks to specialized sub-agents"
                    },
                    "selection_criteria": {
                        "task_complexity": "Use hierarchy for complex, decomposable tasks",
                        "time_constraints": "Use broadcast for parallel processing",
                        "specialization": "Use auction for highly specialized tasks"
                    }
                },
                tags=["coordination", "multi_agent", "patterns", "orchestration"]
            )
        ]

        for entity in knowledge_entities:
            self.add_knowledge_entity(entity)


# Factory function for creating agent schemas
def create_agent_schema(agent_id: str, agent_type: str) -> AgentKnowledgeSchema:
    """Create an appropriate knowledge schema based on agent type."""
    schema_map = {
        "code_agent": CodeAgentSchema,
        "data_science_agent": DataScienceAgentSchema,
        "super_agent": SuperAgentSchema,
        # Add more mappings as needed
    }

    schema_class = schema_map.get(agent_type, AgentKnowledgeSchema)

    if schema_class == AgentKnowledgeSchema:
        # For generic agents, infer domain from agent type
        domain_map = {
            "dba_agent": AgentDomain.DATABASE_ADMIN,
            "devops_agent": AgentDomain.DEVOPS,
            "gcp_agent": AgentDomain.CLOUD_PLATFORM,
            "github_agent": AgentDomain.VERSION_CONTROL,
            "legal_agent": AgentDomain.LEGAL_ANALYSIS,
            "infosec_agent": AgentDomain.SECURITY,
            "pulumi_agent": AgentDomain.INFRASTRUCTURE,
            "research_agent": AgentDomain.RESEARCH,
            "tester_agent": AgentDomain.TESTING,
            "token_agent": AgentDomain.BLOCKCHAIN,
            "ux_agent": AgentDomain.USER_EXPERIENCE,
            "codifier_agent": AgentDomain.DOCUMENTATION,
            "io_agent": AgentDomain.INSPECTION,
            "playbook_agent": AgentDomain.PLAYBOOK_MANAGEMENT
        }

        domain = domain_map.get(agent_type, AgentDomain.CODE_DEVELOPMENT)
        return AgentKnowledgeSchema(agent_id, domain)

    return schema_class(agent_id)


# Utility functions for knowledge management
def merge_knowledge_schemas(schemas: List[AgentKnowledgeSchema]) -> Dict[str, Any]:
    """Merge multiple knowledge schemas for collaborative learning."""
    merged = {
        "agents": [],
        "combined_knowledge": {},
        "shared_capabilities": {},
        "cross_domain_patterns": [],
        "collaboration_opportunities": []
    }

    for schema in schemas:
        merged["agents"].append({
            "agent_id": schema.agent_id,
            "domain": schema.domain.value,
            "knowledge_count": len(schema.knowledge_entities),
            "capability_count": len(schema.capabilities)
        })

        # Merge knowledge entities
        for eid, entity in schema.knowledge_entities.items():
            if eid not in merged["combined_knowledge"]:
                merged["combined_knowledge"][eid] = []
            merged["combined_knowledge"][eid].append({
                "agent_id": schema.agent_id,
                "entity": entity
            })

        # Identify shared capabilities
        for cap_name, capability in schema.capabilities.items():
            if cap_name not in merged["shared_capabilities"]:
                merged["shared_capabilities"][cap_name] = []
            merged["shared_capabilities"][cap_name].append({
                "agent_id": schema.agent_id,
                "capability": capability
            })

    return merged


def extract_learning_insights(schemas: List[AgentKnowledgeSchema]) -> Dict[str, Any]:
    """Extract learning insights from agent knowledge schemas."""
    insights = {
        "top_performing_tools": {},
        "common_failure_patterns": [],
        "knowledge_gaps": [],
        "learning_recommendations": [],
        "performance_trends": {}
    }

    # Aggregate tool performance across agents
    for schema in schemas:
        for tool_name, pattern in schema.tool_patterns.items():
            if tool_name not in insights["top_performing_tools"]:
                insights["top_performing_tools"][tool_name] = {
                    "total_usage": 0,
                    "average_success_rate": 0.0,
                    "agents_using": []
                }

            insights["top_performing_tools"][tool_name]["total_usage"] += pattern.usage_frequency
            insights["top_performing_tools"][tool_name]["average_success_rate"] += pattern.success_rate
            insights["top_performing_tools"][tool_name]["agents_using"].append(schema.agent_id)

    # Calculate averages
    for tool_data in insights["top_performing_tools"].values():
        if tool_data["agents_using"]:
            tool_data["average_success_rate"] /= len(tool_data["agents_using"])

    return insights
