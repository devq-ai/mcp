"""
Agent Types Module for Agentical Framework

This module provides all specialized agent type implementations built on the
enhanced base agent architecture. Each agent type is designed for specific
domain expertise while maintaining consistency with the framework.

Available Agent Types:
- CODE_AGENT: Software development and programming tasks
- DATA_SCIENCE_AGENT: Data analysis, ML, and statistical tasks
- DBA_AGENT: Database administration and optimization
- DEVOPS_AGENT: Infrastructure, deployment, and operations
- GCP_AGENT: Google Cloud Platform services and management
- GITHUB_AGENT: GitHub operations and repository management
- LEGAL_AGENT: Legal document analysis and compliance
- INFOSEC_AGENT: Security analysis and threat assessment
- PULUMI_AGENT: Infrastructure as code with Pulumi
- RESEARCH_AGENT: Research, analysis, and knowledge synthesis
- TESTER_AGENT: Testing, QA, and validation tasks
- TOKEN_AGENT: Token management and analysis
- UAT_AGENT: User acceptance testing coordination
- UX_AGENT: User experience design and analysis

Features:
- Standardized capabilities and tool integration
- Domain-specific validation and error handling
- Performance optimization for specialized tasks
- Comprehensive observability and monitoring
- Registry integration for discovery and coordination
"""

from .code_agent import CodeAgent
from .data_science_agent import DataScienceAgent
from .dba_agent import DbaAgent
from .devops_agent import DevOpsAgent
from .gcp_agent import GcpAgent
from .github_agent import GitHubAgent
from .legal_agent import LegalAgent
from .infosec_agent import InfoSecAgent
from .pulumi_agent import PulumiAgent
from .research_agent import ResearchAgent
from .tester_agent import TesterAgent
from .token_agent import TokenAgent
from .uat_agent import UatAgent
from .ux_agent import UxAgent

__all__ = [
    "CodeAgent",
    "DataScienceAgent",
    "DbaAgent",
    "DevOpsAgent",
    "GcpAgent",
    "GitHubAgent",
    "LegalAgent",
    "InfoSecAgent",
    "PulumiAgent",
    "ResearchAgent",
    "TesterAgent",
    "TokenAgent",
    "UatAgent",
    "UxAgent",
]

# Agent type registry mapping for dynamic instantiation
AGENT_TYPE_REGISTRY = {
    "code_agent": CodeAgent,
    "data_science_agent": DataScienceAgent,
    "dba_agent": DbaAgent,
    "devops_agent": DevOpsAgent,
    "gcp_agent": GcpAgent,
    "github_agent": GitHubAgent,
    "legal_agent": LegalAgent,
    "infosec_agent": InfoSecAgent,
    "pulumi_agent": PulumiAgent,
    "research_agent": ResearchAgent,
    "tester_agent": TesterAgent,
    "token_agent": TokenAgent,
    "uat_agent": UatAgent,
    "ux_agent": UxAgent,
}

def get_agent_class(agent_type: str):
    """
    Get agent class by type name for dynamic instantiation.

    Args:
        agent_type: String identifier for the agent type

    Returns:
        Agent class for the specified type

    Raises:
        ValueError: If agent type is not recognized
    """
    if agent_type not in AGENT_TYPE_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return AGENT_TYPE_REGISTRY[agent_type]

def list_available_agents():
    """
    List all available agent types with their capabilities.

    Returns:
        Dict mapping agent types to their key capabilities
    """
    agent_capabilities = {}

    for agent_type, agent_class in AGENT_TYPE_REGISTRY.items():
        if hasattr(agent_class, 'get_capabilities'):
            capabilities = agent_class.get_capabilities()
        else:
            capabilities = getattr(agent_class, 'default_capabilities', [])

        agent_capabilities[agent_type] = {
            'class': agent_class.__name__,
            'capabilities': capabilities,
            'description': getattr(agent_class, '__doc__', '').split('\n')[0] if agent_class.__doc__ else ''
        }

    return agent_capabilities
