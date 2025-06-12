"""
Agents API Router

Basic agents router for the Agentical framework.
This module will be expanded with full agent functionality in future tasks.
"""

from fastapi import APIRouter

# Create agents router
router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/")
async def list_agents():
    """List available agents (placeholder)"""
    return {
        "message": "Agents endpoint - implementation coming in future tasks",
        "status": "placeholder"
    }

@router.post("/execute")
async def execute_agent():
    """Execute agent (placeholder)"""
    return {
        "message": "Agent execution endpoint - implementation coming in future tasks",
        "status": "placeholder"
    }