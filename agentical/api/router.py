"""
API Router Configuration for Agentical

This module configures the main API router and includes all endpoint routers
from the different API modules, following FastAPI best practices.
"""

from fastapi import APIRouter

from .health import router as health_router
from .agents import router as agents_router
from .v1 import api_v1_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(health_router)
api_router.include_router(agents_router)

# Include v1 API with comprehensive agent management
api_router.include_router(api_v1_router, prefix="/v1")

# Additional routers can be included here as they are developed
# api_router.include_router(workflows_router)
# api_router.include_router(playbooks_router)

__all__ = ["api_router"]
