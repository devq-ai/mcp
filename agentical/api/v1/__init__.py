"""
Agentical API v1 Module

This module contains version 1 of the Agentical API endpoints,
providing comprehensive REST APIs for agent management, workflow orchestration,
playbook execution, and system monitoring.

Features:
- Agent Management APIs (CRUD, monitoring, analytics)
- Playbook Management APIs (creation, execution, monitoring)
- Workflow Management APIs (orchestration, control)
- System Analytics and Monitoring APIs
- Real-time WebSocket connections
- Comprehensive error handling and validation
"""

from fastapi import APIRouter
from .endpoints.agents import router as agents_router
from .endpoints.playbooks import router as playbooks_router

# Initialize v1 API router
api_v1_router = APIRouter(prefix="/v1")

# Include endpoint routers
api_v1_router.include_router(agents_router)
api_v1_router.include_router(playbooks_router)

__all__ = ["api_v1_router"]
