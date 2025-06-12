"""
Agentical API v1 Module

This module contains version 1 of the Agentical API endpoints,
providing comprehensive REST APIs for agent management, system workflow orchestration,
playbook execution, and system monitoring.

WORKFLOW TYPE CLARIFICATION:
- System Workflows (/workflows/) - Multi-agent orchestration, business processes, external integrations
- Agent Workflows - Internal agent logic patterns (managed via agent configuration APIs)
See docs/workflow_quick_reference.md for detailed differences.

Features:
- Agent Management APIs (CRUD, monitoring, analytics, internal workflow configuration)
- Playbook Management APIs (creation, execution, monitoring)
- System Workflow Management APIs (multi-agent orchestration, business process control)
- System Analytics and Monitoring APIs (workflows, agents, infrastructure)
- Real-time WebSocket connections
- Comprehensive error handling and validation
"""

from fastapi import APIRouter
from .endpoints.agents import router as agents_router
from .endpoints.auth import router as auth_router
from .endpoints.playbooks import router as playbooks_router
from .endpoints.workflows import router as workflows_router
from .endpoints.analytics import router as analytics_router

# Initialize v1 API router
api_v1_router = APIRouter(prefix="/v1")

# Include endpoint routers
api_v1_router.include_router(auth_router)
api_v1_router.include_router(agents_router)
api_v1_router.include_router(playbooks_router)
api_v1_router.include_router(workflows_router)
api_v1_router.include_router(analytics_router)

__all__ = ["api_v1_router"]
