"""
Agentical API Module

This module contains all API endpoints for the Agentical framework,
organized by functionality and following FastAPI best practices.
"""

from .router import api_router

__all__ = ["api_router"]