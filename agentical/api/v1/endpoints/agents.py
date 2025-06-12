"""
Agent Management API Endpoints for Agentical

This module provides comprehensive REST API endpoints for managing all agent types
in the Agentical framework, including real-time monitoring, configuration,
execution control, and analytics.

SCOPE CLARIFICATION:
This module manages individual AGENTS and their configurations, which is distinct from:
- System Workflows (managed via /workflows/ endpoints) - multi-agent orchestration
- Agent Internal Workflows - cognitive patterns within individual agents

Agent Management Features:
- CRUD operations for all 18 specialized agent types
- Real-time status monitoring and health checks
- Agent configuration and parameter management
- Execution history and performance analytics
- WebSocket support for real-time updates
- Comprehensive error handling and validation

Technical Features:
- RESTful API design with OpenAPI documentation
- Async operations with comprehensive error handling
- Logfire integration for observability
- Database persistence with relationship management
- Real-time capabilities via WebSocket
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import logfire
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from agentical.core.exceptions import ValidationError
from agentical.db.database import get_async_db
from agentical.db.models.agent import Agent, AgentType, AgentStatus, AgentExecution
from agentical.db.repositories.agent import AsyncAgentRepository

# Initialize router
router = APIRouter(prefix="/agents", tags=["agents"])

# Agent type mapping for 18 specialized agent types
AGENT_TYPE_MAPPING = {
    # Core development agents
    "code_agent": {
        "name": "Code Agent",
        "description": "Code analysis, review, generation, and debugging",
        "capabilities": ["code_analysis", "debugging", "optimization", "code_generation", "refactoring"],
        "default_config": {"max_files": 100, "languages": ["python", "javascript", "typescript"], "analysis_depth": "comprehensive"}
    },
    "data_science_agent": {
        "name": "Data Science Agent",
        "description": "Data analysis, machine learning, and statistical modeling",
        "capabilities": ["data_processing", "modeling", "visualization", "statistical_analysis", "ml_training"],
        "default_config": {"max_dataset_size_mb": 1000, "supported_formats": ["csv", "json", "parquet"], "ml_frameworks": ["sklearn", "pytorch"]}
    },
    "dba_agent": {
        "name": "Database Administrator Agent",
        "description": "Database administration, optimization, and query analysis",
        "capabilities": ["query_optimization", "schema_design", "performance_tuning", "backup_management", "monitoring"],
        "default_config": {"supported_dbs": ["postgresql", "mysql", "mongodb"], "max_connections": 50, "query_timeout": 300}
    },
    "devops_agent": {
        "name": "DevOps Agent",
        "description": "DevOps automation, CI/CD, and infrastructure management",
        "capabilities": ["ci_cd", "deployment", "monitoring", "infrastructure_automation", "container_management"],
        "default_config": {"platforms": ["docker", "kubernetes", "jenkins"], "max_deployments": 10, "rollback_enabled": True}
    },
    "gcp_agent": {
        "name": "Google Cloud Platform Agent",
        "description": "GCP resource management and cloud operations",
        "capabilities": ["gcp_resources", "cloud_deployment", "cost_optimization", "monitoring", "security_management"],
        "default_config": {"regions": ["us-central1", "us-east1"], "services": ["compute", "storage", "bigquery"], "budget_alerts": True}
    },
    "github_agent": {
        "name": "GitHub Agent",
        "description": "GitHub repository and workflow management",
        "capabilities": ["repo_management", "pr_review", "issue_tracking", "workflow_automation", "release_management"],
        "default_config": {"max_repos": 50, "auto_review": True, "webhook_enabled": True, "branch_protection": True}
    },
    "legal_agent": {
        "name": "Legal Agent",
        "description": "Legal document analysis and compliance checking",
        "capabilities": ["contract_review", "compliance_check", "risk_assessment", "document_analysis", "regulatory_monitoring"],
        "default_config": {"jurisdictions": ["US", "EU"], "document_types": ["contracts", "policies"], "compliance_frameworks": ["GDPR", "CCPA"]}
    },
    "infosec_agent": {
        "name": "Information Security Agent",
        "description": "Security analysis, threat detection, and compliance",
        "capabilities": ["security_audit", "vulnerability_scan", "compliance", "threat_detection", "incident_response"],
        "default_config": {"scan_depth": "comprehensive", "frameworks": ["NIST", "ISO27001"], "alert_threshold": "medium"}
    },
    "pulumi_agent": {
        "name": "Pulumi Agent",
        "description": "Infrastructure as Code management with Pulumi",
        "capabilities": ["iac_deployment", "resource_management", "cloud_provisioning", "stack_management", "policy_enforcement"],
        "default_config": {"providers": ["aws", "gcp", "azure"], "stack_limit": 20, "policy_enforcement": True}
    },
    "research_agent": {
        "name": "Research Agent",
        "description": "Research, information gathering, and analysis",
        "capabilities": ["web_research", "data_collection", "analysis", "report_generation", "fact_checking"],
        "default_config": {"search_engines": ["google", "bing"], "max_sources": 100, "fact_check_enabled": True}
    },
    "analyst_agent": {
        "name": "Business Analyst Agent",
        "description": "Business analysis, reporting, and insights",
        "capabilities": ["business_analysis", "reporting", "insights", "kpi_tracking", "trend_analysis"],
        "default_config": {"reporting_frequency": "weekly", "kpi_threshold": 0.05, "visualization_enabled": True}
    },
    "content_agent": {
        "name": "Content Agent",
        "description": "Content creation, editing, and management",
        "capabilities": ["content_generation", "editing", "seo_optimization", "plagiarism_check", "multi_language"],
        "default_config": {"content_types": ["blog", "social", "technical"], "seo_enabled": True, "languages": ["en", "es", "fr"]}
    },
    "marketing_agent": {
        "name": "Marketing Agent",
        "description": "Marketing strategy, campaigns, and analytics",
        "capabilities": ["campaign_management", "analytics", "optimization", "lead_generation", "social_media"],
        "default_config": {"platforms": ["google", "facebook", "linkedin"], "budget_management": True, "auto_optimization": True}
    },
    "sales_agent": {
        "name": "Sales Agent",
        "description": "Sales process automation and CRM management",
        "capabilities": ["lead_management", "sales_automation", "crm_integration", "pipeline_analysis", "forecasting"],
        "default_config": {"crm_systems": ["salesforce", "hubspot"], "pipeline_stages": 5, "auto_follow_up": True}
    },
    "support_agent": {
        "name": "Customer Support Agent",
        "description": "Customer support automation and ticket management",
        "capabilities": ["ticket_management", "knowledge_base", "chat_automation", "escalation_management", "sentiment_analysis"],
        "default_config": {"channels": ["email", "chat", "phone"], "auto_response": True, "escalation_threshold": 2}
    },
    "tester_agent": {
        "name": "QA Tester Agent",
        "description": "Automated testing and quality assurance",
        "capabilities": ["test_automation", "bug_detection", "performance_testing", "regression_testing", "test_reporting"],
        "default_config": {"test_types": ["unit", "integration", "e2e"], "parallel_execution": True, "coverage_threshold": 80}
    },
    "nlp_agent": {
        "name": "Natural Language Processing Agent",
        "description": "Text analysis, language processing, and NLP tasks",
        "capabilities": ["text_analysis", "sentiment_analysis", "entity_extraction", "language_translation", "summarization"],
        "default_config": {"models": ["bert", "gpt"], "languages": ["en", "es", "fr"], "confidence_threshold": 0.8}
    },
    "monitoring_agent": {
        "name": "System Monitoring Agent",
        "description": "System monitoring, alerting, and performance tracking",
        "capabilities": ["system_monitoring", "alerting", "performance_tracking", "log_analysis", "anomaly_detection"],
        "default_config": {"metrics": ["cpu", "memory", "disk"], "alert_threshold": 80, "retention_days": 30}
    }
}

# Pydantic Models
class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent."""
    agent_type: str = Field(..., description="Type of agent to create")
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Custom agent name")
    description: Optional[str] = Field(None, max_length=1000, description="Agent description")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configuration parameters")
    capabilities: Optional[List[str]] = Field(None, description="Specific capabilities to enable")
    resource_limits: Optional[Dict[str, Any]] = Field(None, description="Resource constraints")
    tags: List[str] = Field(default_factory=list, description="Agent tags for organization")

    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v not in AGENT_TYPE_MAPPING:
            raise ValueError(f"Invalid agent type. Must be one of: {list(AGENT_TYPE_MAPPING.keys())}")
        return v

    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return [tag.strip().lower() for tag in v if tag.strip()]

    class Config:
        schema_extra = {
            "example": {
                "agent_type": "code_agent",
                "name": "Code Review Assistant",
                "description": "Specialized agent for code analysis and review",
                "config": {
                    "max_files": 100,
                    "languages": ["python", "javascript", "typescript"],
                    "analysis_depth": "comprehensive"
                },
                "capabilities": ["code_analysis", "debugging", "optimization"],
                "resource_limits": {
                    "max_memory_mb": 512,
                    "max_cpu_percent": 80
                },
                "tags": ["development", "code-quality"]
            }
        }


class AgentUpdateRequest(BaseModel):
    """Request model for updating an existing agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    resource_limits: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None


class AgentExecutionRequest(BaseModel):
    """Request model for executing agent operations."""
    operation: str = Field(..., min_length=1, description="Operation to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    timeout: Optional[int] = Field(None, gt=0, le=3600, description="Execution timeout in seconds")
    priority: int = Field(1, ge=1, le=10, description="Execution priority (1-10)")
    async_execution: bool = Field(False, description="Execute asynchronously")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL for async execution")

    class Config:
        schema_extra = {
            "example": {
                "operation": "analyze_code",
                "parameters": {
                    "code": "def hello(): return 'world'",
                    "language": "python",
                    "analysis_type": "quality"
                },
                "timeout": 30,
                "priority": 5,
                "async_execution": False
            }
        }


class AgentResponse(BaseModel):
    """Response model for agent data."""
    id: str
    agent_type: str
    name: str
    description: Optional[str]
    status: AgentStatus
    config: Dict[str, Any]
    capabilities: List[str]
    resource_limits: Optional[Dict[str, Any]]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    last_active: Optional[datetime]
    execution_count: int
    success_rate: float
    average_execution_time: float
    performance_metrics: Dict[str, Any]

    class Config:
        from_attributes = True


class AgentExecutionResponse(BaseModel):
    """Response model for agent execution data."""
    id: str
    agent_id: str
    operation: str
    status: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    priority: int

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """Response model for agent list with pagination."""
    agents: List[AgentResponse]
    total: int
    page: int
    size: int
    pages: int
    filters_applied: Dict[str, Any]


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics."""
    total_agents: int
    active_agents: int
    agent_types_in_use: Dict[str, int]
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_success_rate: float
    most_active_agents: List[Dict[str, Any]]


# WebSocket connection manager
class AgentWebSocketManager:
    """Manages WebSocket connections for real-time agent updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: str):
        """Accept a WebSocket connection for a specific agent."""
        await websocket.accept()
        if agent_id not in self.active_connections:
            self.active_connections[agent_id] = []
        self.active_connections[agent_id].append(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: str):
        """Remove a WebSocket connection."""
        if agent_id in self.active_connections:
            if websocket in self.active_connections[agent_id]:
                self.active_connections[agent_id].remove(websocket)
            if not self.active_connections[agent_id]:
                del self.active_connections[agent_id]

    async def broadcast_agent_update(self, agent_id: str, data: Dict[str, Any]):
        """Broadcast agent updates to all connected clients."""
        if agent_id in self.active_connections:
            message = json.dumps(data)
            disconnected = []
            for websocket in self.active_connections[agent_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket, agent_id)


ws_manager = AgentWebSocketManager()


# Dependency injection
async def get_agent_repository(db: AsyncSession = Depends(get_async_db)) -> AsyncAgentRepository:
    """Get agent repository instance."""
    return AsyncAgentRepository(db)


# API Endpoints
@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    background_tasks: BackgroundTasks,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Create a new agent instance.

    Creates a new agent of the specified type with configuration and capabilities.
    Supports all 18 specialized agent types with custom configurations.
    """
    with logfire.span("Create agent", agent_type=request.agent_type):
        try:
            # Get agent type info
            type_info = AGENT_TYPE_MAPPING[request.agent_type]

            # Merge default config with provided config
            merged_config = type_info["default_config"].copy()
            if request.config:
                merged_config.update(request.config)

            # Set capabilities if not provided
            capabilities = request.capabilities or type_info["capabilities"]

            # Create agent record
            agent_data = {
                "id": str(uuid.uuid4()),
                "agent_type": AgentType(request.agent_type),
                "name": request.name or f"{type_info['name']} {datetime.utcnow().strftime('%Y%m%d%H%M')}",
                "description": request.description or type_info["description"],
                "status": AgentStatus.INACTIVE,
                "config": merged_config,
                "capabilities": capabilities,
                "resource_limits": request.resource_limits,
                "tags": request.tags,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            agent = await agent_repo.create(agent_data)

            # Initialize performance metrics
            performance_metrics = {
                "execution_count": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "last_active": None,
                "health_score": 100.0
            }

            logfire.info("Agent created successfully",
                        agent_id=agent.id,
                        agent_type=request.agent_type,
                        name=agent.name)

            # Schedule background initialization
            background_tasks.add_task(_initialize_agent_background, agent.id)

            return AgentResponse(
                id=agent.id,
                agent_type=agent.agent_type.value,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                config=agent.config,
                capabilities=capabilities,
                resource_limits=agent.resource_limits,
                tags=agent.tags or [],
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                last_active=None,
                execution_count=0,
                success_rate=0.0,
                average_execution_time=0.0,
                performance_metrics=performance_metrics
            )

        except ValidationError as e:
            logfire.error("Agent creation validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Agent validation failed: {str(e)}"
            )
        except Exception as e:
            logfire.error("Failed to create agent", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create agent"
            )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    sort_by: str = Query("created_at", regex="^(name|created_at|updated_at|last_active|execution_count)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    List agents with filtering, searching, and pagination.

    Supports filtering by type, status, tags, and text search.
    Results are paginated and can be sorted by various fields.
    """
    with logfire.span("List agents"):
        try:
            # Build filters
            filters = {}
            if agent_type and agent_type in AGENT_TYPE_MAPPING:
                filters['agent_type'] = agent_type
            if status:
                filters['status'] = status
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                filters['tags'] = tag_list
            if search:
                filters['search'] = search

            # Get agents with pagination
            agents, total = await agent_repo.list_agents(
                page=page,
                size=size,
                filters=filters,
                sort_by=sort_by,
                sort_order=sort_order
            )

            # Build response
            agent_responses = []
            for agent in agents:
                # Get performance metrics
                metrics = await agent_repo.get_agent_metrics(agent.id)

                response = AgentResponse(
                    id=agent.id,
                    agent_type=agent.agent_type.value,
                    name=agent.name,
                    description=agent.description,
                    status=agent.status,
                    config=agent.config or {},
                    capabilities=agent.capabilities or [],
                    resource_limits=agent.resource_limits,
                    tags=agent.tags or [],
                    created_at=agent.created_at,
                    updated_at=agent.updated_at,
                    last_active=metrics.get('last_active'),
                    execution_count=metrics.get('execution_count', 0),
                    success_rate=metrics.get('success_rate', 0.0),
                    average_execution_time=metrics.get('average_execution_time', 0.0),
                    performance_metrics=metrics
                )
                agent_responses.append(response)

            pages = (total + size - 1) // size

            logfire.info("Agents listed",
                        total=total,
                        page=page,
                        size=size,
                        filters=filters)

            return AgentListResponse(
                agents=agent_responses,
                total=total,
                page=page,
                size=size,
                pages=pages,
                filters_applied=filters
            )

        except Exception as e:
            logfire.error("Failed to list agents", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list agents"
            )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Get a specific agent by ID.

    Returns detailed agent information including configuration, capabilities,
    and performance metrics.
    """
    with logfire.span("Get agent", agent_id=agent_id):
        try:
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            # Get performance metrics
            metrics = await agent_repo.get_agent_metrics(agent_id)

            response = AgentResponse(
                id=agent.id,
                agent_type=agent.agent_type.value,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                config=agent.config or {},
                capabilities=agent.capabilities or [],
                resource_limits=agent.resource_limits,
                tags=agent.tags or [],
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                last_active=metrics.get('last_active'),
                execution_count=metrics.get('execution_count', 0),
                success_rate=metrics.get('success_rate', 0.0),
                average_execution_time=metrics.get('average_execution_time', 0.0),
                performance_metrics=metrics
            )

            logfire.info("Agent retrieved", agent_id=agent_id)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to get agent", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get agent"
            )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Update an existing agent.

    Updates agent configuration, capabilities, or metadata.
    Validates changes before applying.
    """
    with logfire.span("Update agent", agent_id=agent_id):
        try:
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            # Build update data
            update_data = {}
            for field, value in request.dict(exclude_unset=True).items():
                if value is not None:
                    update_data[field] = value

            if update_data:
                update_data['updated_at'] = datetime.utcnow()
                agent = await agent_repo.update(agent_id, update_data)

                # Broadcast update via WebSocket
                await ws_manager.broadcast_agent_update(agent_id, {
                    "type": "agent_updated",
                    "agent_id": agent_id,
                    "changes": list(update_data.keys()),
                    "timestamp": datetime.utcnow().isoformat()
                })

            # Get updated metrics
            metrics = await agent_repo.get_agent_metrics(agent_id)

            response = AgentResponse(
                id=agent.id,
                agent_type=agent.agent_type.value,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                config=agent.config or {},
                capabilities=agent.capabilities or [],
                resource_limits=agent.resource_limits,
                tags=agent.tags or [],
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                last_active=metrics.get('last_active'),
                execution_count=metrics.get('execution_count', 0),
                success_rate=metrics.get('success_rate', 0.0),
                average_execution_time=metrics.get('average_execution_time', 0.0),
                performance_metrics=metrics
            )

            logfire.info("Agent updated", agent_id=agent_id, changes=list(update_data.keys()))

            return response

        except HTTPException:
            raise
        except ValidationError as e:
            logfire.error("Agent update validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Agent validation failed: {str(e)}"
            )
        except Exception as e:
            logfire.error("Failed to update agent", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update agent"
            )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    force: bool = Query(False, description="Force delete even if agent is running"),
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Delete an agent.

    Soft deletes by default. Use force=true to delete even if agent is active.
    """
    with logfire.span("Delete agent", agent_id=agent_id):
        try:
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            # Check if agent is active
            if not force and agent.status in [AgentStatus.ACTIVE, AgentStatus.BUSY]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Cannot delete active agent. Use force=true to override."
                )

            await agent_repo.delete(agent_id)

            # Broadcast deletion via WebSocket
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_deleted",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            logfire.info("Agent deleted", agent_id=agent_id, force=force)

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to delete agent", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete agent"
            )


@router.post("/{agent_id}/execute", response_model=AgentExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def execute_agent_operation(
    agent_id: str,
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Execute an operation on a specific agent.

    Starts agent operation execution with the provided parameters.
    Returns execution details and runs asynchronously if requested.
    """
    with logfire.span("Execute agent operation", agent_id=agent_id, operation=request.operation):
        try:
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            if agent.status not in [AgentStatus.ACTIVE, AgentStatus.INACTIVE]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Agent is in {agent.status.value} state and cannot execute operations"
                )

            # Create execution record
            execution_data = {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "operation": request.operation,
                "parameters": request.parameters,
                "status": "pending",
                "priority": request.priority,
                "timeout": request.timeout,
                "callback_url": request.callback_url,
                "created_at": datetime.utcnow()
            }

            execution = await agent_repo.create_execution(execution_data)

            # Update agent status to busy
            await agent_repo.update(agent_id, {
                "status": AgentStatus.BUSY,
                "updated_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            })

            # Start execution in background
            if request.async_execution:
                background_tasks.add_task(
                    _execute_agent_operation_background,
                    execution.id,
                    agent_id,
                    request.operation,
                    request.parameters,
                    request.timeout
                )
            else:
                # Execute synchronously with timeout
                try:
                    result = await _execute_agent_operation_sync(
                        execution.id,
                        agent_id,
                        request.operation,
                        request.parameters,
                        request.timeout
                    )
                    execution = await agent_repo.get_execution_by_id(execution.id)
                except Exception as e:
                    logfire.error("Synchronous execution failed", error=str(e))
                    # Update execution with error
                    await agent_repo.update_execution(execution.id, {
                        "status": "failed",
                        "error_message": str(e),
                        "completed_at": datetime.utcnow()
                    })
                    execution = await agent_repo.get_execution_by_id(execution.id)

            response = AgentExecutionResponse(
                id=execution.id,
                agent_id=execution.agent_id,
                operation=execution.operation,
                status=execution.status,
                parameters=execution.parameters or {},
                result=execution.result,
                error_message=execution.error_message,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_seconds=execution.duration_seconds,
                priority=execution.priority or 1
            )

            logfire.info("Agent operation started",
                        agent_id=agent_id,
                        execution_id=execution.id,
                        operation=request.operation,
                        async_execution=request.async_execution)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to execute agent operation", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to execute agent operation"
            )


@router.get("/{agent_id}/executions")
async def list_agent_executions(
    agent_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by execution status"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    List executions for a specific agent.

    Returns paginated list of agent executions with filtering options.
    """
    with logfire.span("List agent executions", agent_id=agent_id):
        try:
            # Verify agent exists
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            # Build filters
            filters = {"agent_id": agent_id}
            if status:
                filters['status'] = status
            if from_date:
                filters['from_date'] = from_date
            if to_date:
                filters['to_date'] = to_date

            # Get executions
            executions, total = await agent_repo.list_executions(
                page=page,
                size=size,
                filters=filters
            )

            # Build responses
            execution_responses = []
            for execution in executions:
                response = AgentExecutionResponse(
                    id=execution.id,
                    agent_id=execution.agent_id,
                    operation=execution.operation,
                    status=execution.status,
                    parameters=execution.parameters or {},
                    result=execution.result,
                    error_message=execution.error_message,
                    started_at=execution.started_at,
                    completed_at=execution.completed_at,
                    duration_seconds=execution.duration_seconds,
                    priority=execution.priority or 1
                )
                execution_responses.append(response)

            pages = (total + size - 1) // size

            logfire.info("Agent executions listed",
                        agent_id=agent_id,
                        total=total,
                        page=page,
                        size=size)

            return {
                "executions": execution_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": pages
            }

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to list agent executions", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list agent executions"
            )


@router.get("/{agent_id}/executions/{execution_id}", response_model=AgentExecutionResponse)
async def get_agent_execution(
    agent_id: str,
    execution_id: str,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Get a specific agent execution.

    Returns detailed execution information including results and metrics.
    """
    with logfire.span("Get agent execution", agent_id=agent_id, execution_id=execution_id):
        try:
            execution = await agent_repo.get_execution_by_id(execution_id)
            if not execution or execution.agent_id != agent_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent execution not found"
                )

            response = AgentExecutionResponse(
                id=execution.id,
                agent_id=execution.agent_id,
                operation=execution.operation,
                status=execution.status,
                parameters=execution.parameters or {},
                result=execution.result,
                error_message=execution.error_message,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_seconds=execution.duration_seconds,
                priority=execution.priority or 1
            )

            logfire.info("Agent execution retrieved",
                        agent_id=agent_id,
                        execution_id=execution_id)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to get agent execution",
                         agent_id=agent_id,
                         execution_id=execution_id,
                         error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get agent execution"
            )


@router.post("/{agent_id}/stop", status_code=status.HTTP_200_OK)
async def stop_agent(
    agent_id: str,
    reason: Optional[str] = Query(None, description="Reason for stopping the agent"),
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Stop a running agent.

    Changes agent status to inactive and cancels running operations.
    """
    with logfire.span("Stop agent", agent_id=agent_id):
        try:
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )

            if agent.status == AgentStatus.INACTIVE:
                return {"message": f"Agent {agent_id} is already inactive"}

            # Update agent status
            await agent_repo.update(agent_id, {
                "status": AgentStatus.INACTIVE,
                "updated_at": datetime.utcnow()
            })

            # Cancel any running executions
            await agent_repo.cancel_agent_executions(agent_id, reason or "Agent stopped by user")

            # Broadcast update via WebSocket
            await ws_manager.broadcast_agent_update(agent_id, {
                "type": "agent_stopped",
                "agent_id": agent_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })

            logfire.info("Agent stopped", agent_id=agent_id, reason=reason)

            return {"message": f"Agent {agent_id} stopped successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Failed to stop agent", agent_id=agent_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop agent"
            )


@router.get("/stats/summary", response_model=AgentStatsResponse)
async def get_agent_stats(
    from_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    to_date: Optional[datetime] = Query(None, description="End date for statistics"),
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    Get agent system statistics.

    Returns comprehensive statistics about agents and executions.
    """
    with logfire.span("Get agent stats"):
        try:
            stats = await agent_repo.get_agent_stats(from_date, to_date)

            logfire.info("Agent stats retrieved",
                        from_date=from_date,
                        to_date=to_date)

            return AgentStatsResponse(**stats)

        except Exception as e:
            logfire.error("Failed to get agent stats", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get agent statistics"
            )


@router.get("/types", response_model=Dict[str, Any])
async def get_available_agent_types():
    """
    Get list of available agent types.

    Returns all 18 supported agent types with descriptions and capabilities.
    """
    with logfire.span("Get available agent types"):
        try:
            agent_types = []
            for agent_type, info in AGENT_TYPE_MAPPING.items():
                agent_types.append({
                    "type": agent_type,
                    "name": info["name"],
                    "description": info["description"],
                    "capabilities": info["capabilities"],
                    "default_config": info["default_config"]
                })

            logfire.info("Available agent types retrieved", count=len(agent_types))

            return {
                "agent_types": agent_types,
                "total": len(agent_types)
            }

        except Exception as e:
            logfire.error("Failed to get available agent types", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get available agent types"
            )


@router.websocket("/{agent_id}/ws")
async def agent_websocket(
    websocket: WebSocket,
    agent_id: str,
    agent_repo: AsyncAgentRepository = Depends(get_agent_repository)
):
    """
    WebSocket endpoint for real-time agent updates.

    Provides real-time updates for agent status changes and execution progress.
    """
    try:
        # Verify agent exists
        agent = await agent_repo.get_by_id(agent_id)
        if not agent:
            await websocket.close(code=4004, reason="Agent not found")
            return

        await ws_manager.connect(websocket, agent_id)

        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                # Echo back for keepalive
                await websocket.send_text(f"ack: {data}")

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, agent_id)

    except Exception as e:
        logfire.error("WebSocket error", agent_id=agent_id, error=str(e))
        await websocket.close(code=1011, reason="Internal error")


# Helper Functions
async def _initialize_agent_background(agent_id: str) -> None:
    """Background task to initialize agent after creation."""
    try:
        # This would perform agent-specific initialization
        logfire.info("Background agent initialization completed", agent_id=agent_id)
    except Exception as e:
        logfire.error("Background agent initialization failed", agent_id=agent_id, error=str(e))


async def _execute_agent_operation_background(
    execution_id: str,
    agent_id: str,
    operation: str,
    parameters: Dict[str, Any],
    timeout: Optional[int]
) -> None:
    """Background task to execute agent operation."""
    try:
        # This would execute the actual agent operation
        # For now, simulate execution
        await asyncio.sleep(2)  # Simulate work

        # Mock successful result
        result = {
            "operation": operation,
            "status": "completed",
            "result": f"Operation {operation} completed successfully",
            "processed_parameters": parameters
        }

        # Update execution record
        from ....db.database import get_async_db
        async with get_async_db() as db:
            repo = AsyncAgentRepository(db)
            await repo.update_execution(execution_id, {
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow(),
                "duration_seconds": 2.0
            })

            # Update agent status back to active
            await repo.update(agent_id, {
                "status": AgentStatus.ACTIVE,
                "updated_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            })

        # Broadcast completion
        await ws_manager.broadcast_agent_update(agent_id, {
            "type": "execution_completed",
            "execution_id": execution_id,
            "agent_id": agent_id,
            "operation": operation,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })

        logfire.info("Background agent execution completed",
                    execution_id=execution_id, agent_id=agent_id)
    except Exception as e:
        logfire.error("Background agent execution failed",
                     execution_id=execution_id, agent_id=agent_id, error=str(e))


async def _execute_agent_operation_sync(
    execution_id: str,
    agent_id: str,
    operation: str,
    parameters: Dict[str, Any],
    timeout: Optional[int]
) -> Dict[str, Any]:
    """Execute agent operation synchronously."""
    try:
        # This would execute the actual agent operation
        # For now, simulate execution
        await asyncio.sleep(1)  # Simulate work

        # Mock successful result
        result = {
            "operation": operation,
            "status": "completed",
            "result": f"Operation {operation} completed successfully",
            "processed_parameters": parameters
        }

        # Update execution record
        from ....db.database import get_async_db
        async with get_async_db() as db:
            repo = AsyncAgentRepository(db)
            await repo.update_execution(execution_id, {
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow(),
                "duration_seconds": 1.0
            })

            # Update agent status back to active
            await repo.update(agent_id, {
                "status": AgentStatus.ACTIVE,
                "updated_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            })

        # Broadcast completion
        await ws_manager.broadcast_agent_update(agent_id, {
            "type": "execution_completed",
            "execution_id": execution_id,
            "agent_id": agent_id,
            "operation": operation,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })

        return result

    except Exception as e:
        logfire.error("Synchronous agent execution failed",
                     execution_id=execution_id, agent_id=agent_id, error=str(e))
        raise
