"""
Tests for Workflow Management API Endpoints

This module contains comprehensive tests for the workflow management endpoints,
including CRUD operations, execution control, status monitoring, and WebSocket functionality.

Features:
- Workflow CRUD operation tests
- Execution control tests (start/pause/resume/stop)
- Status monitoring and progress tracking tests
- WebSocket real-time update tests
- Error handling and edge case tests
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from agentical.main import app
from agentical.db.models.workflow import (
    WorkflowType, WorkflowStatus, ExecutionStatus, StepType, StepStatus
)
from agentical.api.v1.endpoints.workflows import ws_manager


class TestWorkflowCRUD:
    """Test workflow CRUD operations."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def sample_workflow_data(self):
        """Sample workflow data for testing."""
        return {
            "name": "Test Workflow",
            "description": "A test workflow for unit testing",
            "workflow_type": WorkflowType.SEQUENTIAL.value,
            "config": {"timeout": 300, "retry_count": 3},
            "steps": [
                {
                    "id": "step1",
                    "name": "Initial Step",
                    "type": StepType.AGENT_TASK.value,
                    "config": {"agent_type": "code_agent", "task": "analyze_code"},
                    "order": 0
                },
                {
                    "id": "step2",
                    "name": "Processing Step",
                    "type": StepType.TOOL_EXECUTION.value,
                    "config": {"tool": "file_processor", "input": "analysis_result"},
                    "dependencies": ["step1"],
                    "order": 1
                }
            ],
            "tags": ["test", "automation"],
            "timeout_seconds": 1800,
            "retry_config": {"max_retries": 3, "backoff": "exponential"}
        }

    @pytest.fixture
    def mock_workflow_repo(self):
        """Mock workflow repository."""
        repo = AsyncMock()
        repo.create.return_value = MagicMock(
            id="workflow_123",
            name="Test Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            status=WorkflowStatus.DRAFT,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        return repo

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_create_workflow_success(self, mock_get_repo, client, sample_workflow_data, mock_workflow_repo):
        """Test successful workflow creation."""
        mock_get_repo.return_value = mock_workflow_repo

        response = client.post("/api/v1/workflows/", json=sample_workflow_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_workflow_data["name"]
        assert data["workflow_type"] == sample_workflow_data["workflow_type"]
        assert len(data["steps"]) == 2

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_create_workflow_validation_error(self, mock_get_repo, client, mock_workflow_repo):
        """Test workflow creation with validation errors."""
        mock_get_repo.return_value = mock_workflow_repo

        invalid_data = {
            "name": "",  # Empty name should fail validation
            "workflow_type": "invalid_type",
            "steps": []
        }

        response = client.post("/api/v1/workflows/", json=invalid_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_create_workflow_circular_dependency(self, mock_get_repo, client, mock_workflow_repo):
        """Test workflow creation with circular step dependencies."""
        mock_get_repo.return_value = mock_workflow_repo

        circular_data = {
            "name": "Circular Workflow",
            "workflow_type": WorkflowType.SEQUENTIAL.value,
            "steps": [
                {
                    "id": "step1",
                    "type": StepType.AGENT_TASK.value,
                    "config": {},
                    "dependencies": ["step2"]
                },
                {
                    "id": "step2",
                    "type": StepType.AGENT_TASK.value,
                    "config": {},
                    "dependencies": ["step1"]
                }
            ]
        }

        response = client.post("/api/v1/workflows/", json=circular_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_list_workflows(self, mock_get_repo, client, mock_workflow_repo):
        """Test workflow listing with pagination and filtering."""
        mock_workflows = [
            MagicMock(
                id=f"workflow_{i}",
                name=f"Workflow {i}",
                workflow_type=WorkflowType.SEQUENTIAL,
                status=WorkflowStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            for i in range(5)
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 5)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/?page=1&size=10")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["size"] == 10
        assert len(data["workflows"]) == 5

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_get_workflow_by_id(self, mock_get_repo, client, mock_workflow_repo):
        """Test getting a specific workflow by ID."""
        mock_workflow = MagicMock(
            id="workflow_123",
            name="Test Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            status=WorkflowStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.get_workflow_steps.return_value = []
        mock_workflow_repo.get_execution_count.return_value = 0
        mock_workflow_repo.get_last_execution.return_value = None
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "workflow_123"
        assert data["name"] == "Test Workflow"

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_get_workflow_not_found(self, mock_get_repo, client, mock_workflow_repo):
        """Test getting a non-existent workflow."""
        mock_workflow_repo.get_by_id.return_value = None
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_update_workflow(self, mock_get_repo, client, mock_workflow_repo):
        """Test workflow updates."""
        mock_workflow = MagicMock(
            id="workflow_123",
            name="Test Workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            status=WorkflowStatus.DRAFT
        )
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.update.return_value = mock_workflow
        mock_get_repo.return_value = mock_workflow_repo

        update_data = {
            "name": "Updated Workflow",
            "status": WorkflowStatus.ACTIVE.value
        }

        response = client.put("/api/v1/workflows/workflow_123", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        mock_workflow_repo.update.assert_called_once()

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_delete_workflow_success(self, mock_get_repo, client, mock_workflow_repo):
        """Test successful workflow deletion."""
        mock_workflow = MagicMock(id="workflow_123")
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.get_active_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo

        response = client.delete("/api/v1/workflows/workflow_123")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_workflow_repo.delete.assert_called_once_with("workflow_123", soft_delete=True)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_delete_workflow_with_active_executions(self, mock_get_repo, client, mock_workflow_repo):
        """Test workflow deletion with active executions."""
        mock_workflow = MagicMock(id="workflow_123")
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.get_active_executions.return_value = [MagicMock()]  # Active execution exists
        mock_get_repo.return_value = mock_workflow_repo

        response = client.delete("/api/v1/workflows/workflow_123")

        assert response.status_code == status.HTTP_409_CONFLICT

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_delete_workflow_force(self, mock_get_repo, client, mock_workflow_repo):
        """Test forced workflow deletion."""
        mock_workflow = MagicMock(id="workflow_123")
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.get_active_executions.return_value = [MagicMock()]
        mock_get_repo.return_value = mock_workflow_repo

        response = client.delete("/api/v1/workflows/workflow_123?force=true")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_workflow_repo.delete.assert_called_once_with("workflow_123", soft_delete=False)


class TestWorkflowExecution:
    """Test workflow execution operations."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_workflow_manager(self):
        """Mock workflow manager."""
        manager = AsyncMock()
        manager.execute_workflow.return_value = None
        manager.schedule_workflow_execution.return_value = None
        manager.pause_execution.return_value = None
        manager.resume_execution.return_value = None
        manager.cancel_execution.return_value = None
        return manager

    @pytest.fixture
    def mock_workflow_repo(self):
        """Mock workflow repository."""
        repo = AsyncMock()
        return repo

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_execute_workflow_success(self, mock_get_repo, mock_get_manager,
                                    client, mock_workflow_repo, mock_workflow_manager):
        """Test successful workflow execution."""
        mock_workflow = MagicMock(
            id="workflow_123",
            status=WorkflowStatus.ACTIVE
        )
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123",
            status=ExecutionStatus.PENDING,
            created_at=datetime.utcnow()
        )

        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.create_execution.return_value = mock_execution
        mock_workflow_repo.get_step_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        execution_data = {
            "input_data": {"key": "value"},
            "priority": 5
        }

        response = client.post("/api/v1/workflows/workflow_123/execute", json=execution_data)

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["workflow_id"] == "workflow_123"
        assert data["status"] == ExecutionStatus.PENDING.value

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_execute_inactive_workflow(self, mock_get_repo, mock_get_manager,
                                     client, mock_workflow_repo, mock_workflow_manager):
        """Test executing an inactive workflow."""
        mock_workflow = MagicMock(
            id="workflow_123",
            status=WorkflowStatus.INACTIVE
        )
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        execution_data = {"input_data": {"key": "value"}}

        response = client.post("/api/v1/workflows/workflow_123/execute", json=execution_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_schedule_workflow_execution(self, mock_get_repo, mock_get_manager,
                                       client, mock_workflow_repo, mock_workflow_manager):
        """Test scheduling workflow execution for future time."""
        mock_workflow = MagicMock(
            id="workflow_123",
            status=WorkflowStatus.ACTIVE
        )
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123",
            status=ExecutionStatus.PENDING,
            created_at=datetime.utcnow()
        )

        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.create_execution.return_value = mock_execution
        mock_workflow_repo.get_step_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        future_time = datetime.utcnow() + timedelta(hours=1)
        execution_data = {
            "input_data": {"key": "value"},
            "scheduled_at": future_time.isoformat()
        }

        response = client.post("/api/v1/workflows/workflow_123/execute", json=execution_data)

        assert response.status_code == status.HTTP_202_ACCEPTED
        mock_workflow_manager.schedule_workflow_execution.assert_called_once()

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_list_workflow_executions(self, mock_get_repo, client, mock_workflow_repo):
        """Test listing workflow executions."""
        mock_workflow = MagicMock(id="workflow_123")
        mock_executions = [
            MagicMock(
                id=f"execution_{i}",
                workflow_id="workflow_123",
                status=ExecutionStatus.COMPLETED,
                created_at=datetime.utcnow()
            )
            for i in range(3)
        ]

        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.list_executions.return_value = (mock_executions, 3)
        mock_workflow_repo.get_step_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert len(data["executions"]) == 3

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_get_workflow_execution(self, mock_get_repo, client, mock_workflow_repo):
        """Test getting a specific workflow execution."""
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123",
            status=ExecutionStatus.COMPLETED,
            created_at=datetime.utcnow()
        )

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_workflow_repo.get_step_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions/execution_123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "execution_123"
        assert data["workflow_id"] == "workflow_123"

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_control_workflow_execution_pause(self, mock_get_repo, mock_get_manager,
                                            client, mock_workflow_repo, mock_workflow_manager):
        """Test pausing workflow execution."""
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123"
        )

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        control_data = {
            "action": "pause",
            "reason": "Manual pause for testing"
        }

        response = client.post(
            "/api/v1/workflows/workflow_123/executions/execution_123/control",
            json=control_data
        )

        assert response.status_code == status.HTTP_200_OK
        mock_workflow_manager.pause_execution.assert_called_once_with("execution_123")

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_control_workflow_execution_resume(self, mock_get_repo, mock_get_manager,
                                             client, mock_workflow_repo, mock_workflow_manager):
        """Test resuming workflow execution."""
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123"
        )

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        control_data = {"action": "resume"}

        response = client.post(
            "/api/v1/workflows/workflow_123/executions/execution_123/control",
            json=control_data
        )

        assert response.status_code == status.HTTP_200_OK
        mock_workflow_manager.resume_execution.assert_called_once_with("execution_123")

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_manager')
    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_control_workflow_execution_stop(self, mock_get_repo, mock_get_manager,
                                           client, mock_workflow_repo, mock_workflow_manager):
        """Test stopping workflow execution."""
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123"
        )

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_get_repo.return_value = mock_workflow_repo
        mock_get_manager.return_value = mock_workflow_manager

        control_data = {"action": "stop"}

        response = client.post(
            "/api/v1/workflows/workflow_123/executions/execution_123/control",
            json=control_data
        )

        assert response.status_code == status.HTTP_200_OK
        mock_workflow_manager.cancel_execution.assert_called_once_with("execution_123")

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_get_workflow_stats(self, mock_get_repo, client, mock_workflow_repo):
        """Test getting workflow statistics."""
        mock_stats = {
            "total_workflows": 10,
            "active_workflows": 8,
            "total_executions": 150,
            "running_executions": 3,
            "completed_executions": 120,
            "failed_executions": 27,
            "success_rate": 80.0,
            "average_duration_seconds": 300.5,
            "executions_last_24h": 25,
            "most_used_workflows": []
        }

        mock_workflow_repo.get_workflow_stats.return_value = mock_stats
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/stats/summary")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_workflows"] == 10
        assert data["success_rate"] == 80.0


class TestWorkflowWebSocket:
    """Test workflow WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_websocket_manager_connect(self):
        """Test WebSocket manager connection handling."""
        mock_websocket = MagicMock(spec=WebSocket)
        workflow_id = "workflow_123"

        # Test connection
        assert workflow_id not in ws_manager.active_connections

        # In a real test, you'd need to mock the websocket.accept() call
        # ws_manager.connect(mock_websocket, workflow_id)

        # Test disconnect
        ws_manager.disconnect(mock_websocket, workflow_id)
        assert workflow_id not in ws_manager.active_connections

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Test WebSocket broadcast functionality."""
        workflow_id = "workflow_123"
        test_data = {
            "type": "execution_update",
            "execution_id": "execution_123",
            "status": "running"
        }

        # Test broadcast to empty connections (should not raise error)
        await ws_manager.broadcast_workflow_update(workflow_id, test_data)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_websocket_endpoint_workflow_not_found(self, mock_get_repo, client):
        """Test WebSocket endpoint with non-existent workflow."""
        mock_workflow_repo = AsyncMock()
        mock_workflow_repo.get_by_id.return_value = None
        mock_get_repo.return_value = mock_workflow_repo

        with pytest.raises(Exception):  # WebSocket would close with error
            with client.websocket_connect("/api/v1/workflows/nonexistent/ws"):
                pass


class TestWorkflowValidation:
    """Test workflow validation functions."""

    @pytest.mark.asyncio
    async def test_validate_workflow_steps_success(self):
        """Test successful workflow step validation."""
        from agentical.api.v1.endpoints.workflows import _validate_workflow_steps

        valid_steps = [
            {
                "id": "step1",
                "type": StepType.AGENT_TASK.value,
                "config": {"agent_type": "code_agent"}
            },
            {
                "id": "step2",
                "type": StepType.TOOL_EXECUTION.value,
                "config": {"tool": "processor"},
                "dependencies": ["step1"]
            }
        ]

        # Should not raise exception
        await _validate_workflow_steps(valid_steps)

    @pytest.mark.asyncio
    async def test_validate_workflow_steps_missing_fields(self):
        """Test workflow step validation with missing required fields."""
        from agentical.api.v1.endpoints.workflows import _validate_workflow_steps
        from agentical.core.exceptions import ValidationError

        invalid_steps = [
            {
                "id": "step1",
                # Missing 'type' and 'config'
            }
        ]

        with pytest.raises(ValidationError):
            await _validate_workflow_steps(invalid_steps)

    @pytest.mark.asyncio
    async def test_validate_workflow_steps_duplicate_ids(self):
        """Test workflow step validation with duplicate step IDs."""
        from agentical.api.v1.endpoints.workflows import _validate_workflow_steps
        from agentical.core.exceptions import ValidationError

        invalid_steps = [
            {
                "id": "step1",
                "type": StepType.AGENT_TASK.value,
                "config": {}
            },
            {
                "id": "step1",  # Duplicate ID
                "type": StepType.TOOL_EXECUTION.value,
                "config": {}
            }
        ]

        with pytest.raises(ValidationError):
            await _validate_workflow_steps(invalid_steps)

    @pytest.mark.asyncio
    async def test_validate_workflow_steps_invalid_dependencies(self):
        """Test workflow step validation with invalid dependencies."""
        from agentical.api.v1.endpoints.workflows import _validate_workflow_steps
        from agentical.core.exceptions import ValidationError

        invalid_steps = [
            {
                "id": "step1",
                "type": StepType.AGENT_TASK.value,
                "config": {},
                "dependencies": ["nonexistent_step"]  # Invalid dependency
            }
        ]

        with pytest.raises(ValidationError):
            await _validate_workflow_steps(invalid_steps)

    @pytest.mark.asyncio
    async def test_validate_workflow_steps_invalid_type(self):
        """Test workflow step validation with invalid step type."""
        from agentical.api.v1.endpoints.workflows import _validate_workflow_steps
        from agentical.core.exceptions import ValidationError

        invalid_steps = [
            {
                "id": "step1",
                "type": "invalid_type",
                "config": {}
            }
        ]

        with pytest.raises(ValidationError):
            await _validate_workflow_steps(invalid_steps)


class TestWorkflowErrorHandling:
    """Test error handling in workflow endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_database_error_handling(self, mock_get_repo, client):
        """Test handling of database errors."""
        mock_workflow_repo = AsyncMock()
        mock_workflow_repo.create.side_effect = Exception("Database connection failed")
        mock_get_repo.return_value = mock_workflow_repo

        workflow_data = {
            "name": "Test Workflow",
            "workflow_type": WorkflowType.SEQUENTIAL.value,
            "steps": []
        }

        response = client.post("/api/v1/workflows/", json=workflow_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_manager_error_handling(self, mock_get_repo, client):
        """Test handling of workflow manager errors."""
        mock_workflow_repo = AsyncMock()
        mock_workflow = MagicMock(
            id="workflow_123",
            status=WorkflowStatus.ACTIVE
        )
        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_get_repo.return_value = mock_workflow_repo

        with patch('agentical.api.v1.endpoints.workflows.get_workflow_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.execute_workflow.side_effect = Exception("Execution failed")
            mock_get_manager.return_value = mock_manager

            execution_data = {"input_data": {"key": "value"}}
            response = client.post("/api/v1/workflows/workflow_123/execute", json=execution_data)

            # Should still return 202 since execution starts in background
            assert response.status_code == status.HTTP_202_ACCEPTED


class TestWorkflowPagination:
    """Test pagination functionality in workflow endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_pagination(self, mock_get_repo, client):
        """Test workflow list pagination."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [MagicMock(id=f"workflow_{i}") for i in range(50)]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows[:20], 50)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/?page=1&size=20")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 50
        assert data["page"] == 1
        assert data["size"] == 20
        assert data["pages"] == 3

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_execution_list_pagination(self, mock_get_repo, client):
        """Test execution list pagination."""
        mock_workflow_repo = AsyncMock()
        mock_workflow = MagicMock(id="workflow_123")
        mock_executions = [MagicMock(id=f"execution_{i}") for i in range(30)]

        mock_workflow_repo.get_by_id.return_value = mock_workflow
        mock_workflow_repo.list_executions.return_value = (mock_executions[:10], 30)
        mock_workflow_repo.get_step_executions.return_value = []
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions?page=1&size=10")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 30
        assert data["page"] == 1
        assert data["size"] == 10
        assert data["pages"] == 3


class TestWorkflowFiltering:
    """Test filtering functionality in workflow endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_filtering_by_type(self, mock_get_repo, client):
        """Test workflow list filtering by workflow type."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [
            MagicMock(
                id="workflow_1",
                workflow_type=WorkflowType.SEQUENTIAL,
                status=WorkflowStatus.ACTIVE
            )
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 1)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get(f"/api/v1/workflows/?workflow_type={WorkflowType.SEQUENTIAL.value}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_filtering_by_status(self, mock_get_repo, client):
        """Test workflow list filtering by status."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [
            MagicMock(
                id="workflow_1",
                workflow_type=WorkflowType.SEQUENTIAL,
                status=WorkflowStatus.ACTIVE
            )
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 1)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get(f"/api/v1/workflows/?status={WorkflowStatus.ACTIVE.value}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_filtering_by_tags(self, mock_get_repo, client):
        """Test workflow list filtering by tags."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [
            MagicMock(
                id="workflow_1",
                tags=["automation", "test"]
            )
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 1)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/?tags=automation,test")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_search(self, mock_get_repo, client):
        """Test workflow list text search."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [
            MagicMock(
                id="workflow_1",
                name="Test Automation Workflow"
            )
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 1)
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/?search=automation")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1


class TestWorkflowSorting:
    """Test sorting functionality in workflow endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_sorting(self, mock_get_repo, client):
        """Test workflow list sorting options."""
        mock_workflow_repo = AsyncMock()
        mock_workflows = [
            MagicMock(id="workflow_1", name="A Workflow"),
            MagicMock(id="workflow_2", name="B Workflow")
        ]
        mock_workflow_repo.list_workflows.return_value = (mock_workflows, 2)
        mock_get_repo.return_value = mock_workflow_repo

        # Test sorting by name ascending
        response = client.get("/api/v1/workflows/?sort_by=name&sort_order=asc")

        assert response.status_code == status.HTTP_200_OK
        mock_workflow_repo.list_workflows.assert_called_once()
        call_args = mock_workflow_repo.list_workflows.call_args
        assert call_args[1]["sort_by"] == "name"
        assert call_args[1]["sort_order"] == "asc"

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_workflow_list_sorting_invalid_field(self, mock_get_repo, client):
        """Test workflow list sorting with invalid field."""
        response = client.get("/api/v1/workflows/?sort_by=invalid_field")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestWorkflowLogs:
    """Test workflow execution log streaming."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_stream_execution_logs(self, mock_get_repo, client):
        """Test streaming workflow execution logs."""
        mock_workflow_repo = AsyncMock()
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123",
            status=ExecutionStatus.COMPLETED
        )
        mock_logs = [
            {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "Step started"},
            {"timestamp": "2024-01-01T00:01:00Z", "level": "INFO", "message": "Step completed"}
        ]

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_workflow_repo.get_execution_logs.return_value = mock_logs
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions/execution_123/logs")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_stream_execution_logs_not_found(self, mock_get_repo, client):
        """Test streaming logs for non-existent execution."""
        mock_workflow_repo = AsyncMock()
        mock_workflow_repo.get_execution_by_id.return_value = None
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions/nonexistent/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('agentical.api.v1.endpoints.workflows.get_workflow_repository')
    def test_stream_execution_logs_follow(self, mock_get_repo, client):
        """Test streaming logs with follow option."""
        mock_workflow_repo = AsyncMock()
        mock_execution = MagicMock(
            id="execution_123",
            workflow_id="workflow_123",
            status=ExecutionStatus.RUNNING
        )
        mock_logs = []

        mock_workflow_repo.get_execution_by_id.return_value = mock_execution
        mock_workflow_repo.get_execution_logs.return_value = mock_logs
        mock_get_repo.return_value = mock_workflow_repo

        response = client.get("/api/v1/workflows/workflow_123/executions/execution_123/logs?follow=true")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


# Integration tests that would run with actual database
@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for workflow endpoints with real database."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_full_workflow_lifecycle(self, client):
        """Test complete workflow lifecycle from creation to execution."""
        # This would be an integration test that:
        # 1. Creates a workflow
        # 2. Updates the workflow
        # 3. Executes the workflow
        # 4. Monitors execution
        # 5. Controls execution (pause/resume)
        # 6. Gets execution results
        # 7. Deletes the workflow

        # For now, this is a placeholder
        pass

    def test_concurrent_workflow_executions(self, client):
        """Test handling of concurrent workflow executions."""
        # This would test the system's ability to handle
        # multiple simultaneous workflow executions
        pass

    def test_workflow_performance_under_load(self, client):
        """Test workflow system performance under load."""
        # This would test the system's performance
        # with many workflows and executions
        pass


# Performance tests
@pytest.mark.performance
class TestWorkflowPerformance:
    """Performance tests for workflow endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_workflow_list_performance(self, client):
        """Test performance of workflow listing with large datasets."""
        # This would test pagination performance
        # with large numbers of workflows
        pass

    def test_execution_monitoring_performance(self, client):
        """Test performance of execution monitoring."""
        # This would test the performance of
        # getting execution status and progress
        pass


if __name__ == "__main__":
    pytest.main([__file__])
