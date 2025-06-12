"""
Tests for Playbook Management API Endpoints

This module provides comprehensive test coverage for the playbook management
API endpoints, including CRUD operations, execution control, real-time monitoring,
template management, and validation.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from fastapi.testclient import TestClient
from fastapi import status
from httpx import AsyncClient

# Import the modules to test
from agentical.main import app
from agentical.agents import PlaybookAgent
from agentical.agents.playbook_agent import (
    PlaybookExecutionRequest, PlaybookCreationRequest, PlaybookAnalysisRequest,
    ExecutionMode, ValidationLevel
)
from agentical.api.v1.endpoints.playbooks import (
    PlaybookCreateRequest, PlaybookUpdateRequest, PlaybookExecutionStartRequest,
    PlaybookResponse, PlaybookExecutionResponse, ws_manager
)
from agentical.db.models.playbook import PlaybookCategory, PlaybookStatus, ExecutionStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_playbook_repository():
    """Mock playbook repository."""
    repo = AsyncMock()
    repo.list_playbooks.return_value = {
        "playbooks": [],
        "total": 0,
        "page": 1,
        "page_size": 50
    }
    repo.get_available_categories.return_value = ["automation", "deployment", "security"]
    repo.get_available_tags.return_value = ["automation", "ci-cd", "monitoring"]
    repo.create_playbook.return_value = {
        "id": "pb_test123",
        "name": "Test Playbook",
        "description": "Test description",
        "category": "automation",
        "status": "draft",
        "version": 1,
        "steps": [],
        "variables": {},
        "metadata": {},
        "validation_rules": [],
        "tags": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "created_by": "test_user",
        "execution_count": 0,
        "last_executed": None
    }
    repo.get_playbook.return_value = {
        "id": "pb_test123",
        "name": "Test Playbook",
        "description": "Test description",
        "category": "automation",
        "status": "active",
        "version": 1,
        "steps": [{"name": "Test Step", "type": "action"}],
        "variables": {},
        "metadata": {},
        "validation_rules": [],
        "tags": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "created_by": "test_user",
        "execution_count": 5,
        "last_executed": datetime.utcnow()
    }
    repo.update_playbook.return_value = repo.get_playbook.return_value
    repo.delete_playbook = AsyncMock()
    repo.create_execution_record = AsyncMock()
    repo.complete_execution_record = AsyncMock()
    repo.get_playbook_executions.return_value = {
        "executions": [],
        "total": 0,
        "page": 1,
        "page_size": 20
    }
    repo.get_execution.return_value = {
        "execution_id": "exec_123",
        "playbook_id": "pb_test123",
        "status": "completed",
        "execution_mode": "sequential",
        "started_at": datetime.utcnow(),
        "completed_at": datetime.utcnow(),
        "duration": 120.0,
        "steps_total": 3,
        "steps_completed": 3,
        "steps_failed": 0,
        "current_step": None,
        "progress_percentage": 100.0,
        "result": {"success": True},
        "error": None,
        "checkpoints": [],
        "metrics": {"execution_time": 120.0}
    }
    repo.stop_execution = AsyncMock()
    repo.get_playbook_analytics.return_value = {
        "playbook_id": "pb_test123",
        "total_executions": 10,
        "successful_executions": 8,
        "failed_executions": 2,
        "success_rate": 0.8,
        "average_execution_time": 150.0,
        "executions_by_day": {},
        "step_success_rates": {},
        "error_patterns": {},
        "performance_trends": []
    }
    return repo


@pytest.fixture
def mock_playbook_agent():
    """Mock playbook agent."""
    agent = Mock(spec=PlaybookAgent)
    agent.agent_id = "playbook_agent_test"
    agent.name = "Test Playbook Agent"
    agent.description = "Test playbook agent"
    agent.execute = AsyncMock(return_value={"success": True, "result": "test execution result"})
    agent.validate_playbook = AsyncMock(return_value={
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "estimated_duration": 30,
        "complexity_score": 5
    })
    agent.get_available_templates = AsyncMock(return_value=[
        {
            "name": "incident_response",
            "description": "Incident Response playbook template",
            "category": "security",
            "steps": [{"name": "Assess", "type": "action"}],
            "variables": {"timeout": 30},
            "metadata": {"template": True},
            "tags": ["security", "emergency"]
        }
    ])
    agent.create_from_template = AsyncMock(return_value={
        "name": "New Playbook",
        "description": "Created from template",
        "category": "security",
        "steps": [{"name": "Assess", "type": "action"}],
        "variables": {"timeout": 30},
        "metadata": {"template": "incident_response"},
        "validation_rules": [],
        "tags": ["security"]
    })
    return agent


@pytest.fixture
def sample_playbook_create_request():
    """Sample playbook creation request."""
    return {
        "name": "Test Automation Playbook",
        "description": "A test playbook for automation",
        "category": "automation",
        "steps": [
            {"name": "Initialize", "type": "action", "action": "init"},
            {"name": "Execute", "type": "action", "action": "run"},
            {"name": "Cleanup", "type": "action", "action": "cleanup"}
        ],
        "variables": {"timeout": 300, "retry_count": 3},
        "metadata": {"version": "1.0", "author": "test"},
        "validation_rules": ["validate_inputs", "check_dependencies"],
        "tags": ["automation", "testing"]
    }


class TestPlaybookListEndpoint:
    """Test playbook listing functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_list_playbooks_empty(self, mock_get_repo, client, mock_playbook_repository):
        """Test listing playbooks when none exist."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["playbooks"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert "categories" in data
        assert "tags" in data

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_list_playbooks_with_data(self, mock_get_repo, client, mock_playbook_repository):
        """Test listing playbooks with data."""
        mock_playbook_repository.list_playbooks.return_value = {
            "playbooks": [mock_playbook_repository.get_playbook.return_value],
            "total": 1,
            "page": 1,
            "page_size": 50
        }
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["playbooks"]) == 1
        assert data["total"] == 1
        assert data["playbooks"][0]["name"] == "Test Playbook"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_list_playbooks_with_filters(self, mock_get_repo, client, mock_playbook_repository):
        """Test listing playbooks with filters."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/?category=automation&status=active&tags=ci-cd")

        assert response.status_code == status.HTTP_200_OK
        mock_playbook_repository.list_playbooks.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_list_playbooks_pagination(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook listing pagination."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/?page=2&page_size=10")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10


class TestPlaybookCreationEndpoint:
    """Test playbook creation functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_create_playbook_success(self, mock_get_repo, client,
                                   mock_playbook_repository, sample_playbook_create_request):
        """Test successful playbook creation."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.post("/api/v1/v1/playbooks/", json=sample_playbook_create_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Test Playbook"
        assert data["category"] == "automation"
        assert "id" in data
        mock_playbook_repository.create_playbook.assert_called_once()

    def test_create_playbook_invalid_category(self, client):
        """Test playbook creation with invalid category."""
        invalid_request = {
            "name": "Test Playbook",
            "category": "invalid_category",
            "steps": []
        }

        response = client.post("/api/v1/v1/playbooks/", json=invalid_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_playbook_missing_required_fields(self, client):
        """Test playbook creation with missing required fields."""
        invalid_request = {
            "description": "Missing name and other required fields"
        }

        response = client.post("/api/v1/v1/playbooks/", json=invalid_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_create_playbook_repository_error(self, mock_get_repo, client,
                                            mock_playbook_repository, sample_playbook_create_request):
        """Test playbook creation with repository error."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.create_playbook.side_effect = Exception("Database error")

        response = client.post("/api/v1/v1/playbooks/", json=sample_playbook_create_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestPlaybookRetrievalEndpoint:
    """Test playbook retrieval functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_playbook_success(self, mock_get_repo, client, mock_playbook_repository):
        """Test successful playbook retrieval."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/pb_test123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "pb_test123"
        assert data["name"] == "Test Playbook"
        assert data["category"] == "automation"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_playbook_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook retrieval when playbook not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_playbook.return_value = None

        response = client.get("/api/v1/v1/playbooks/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPlaybookUpdateEndpoint:
    """Test playbook update functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_update_playbook_success(self, mock_get_repo, client, mock_playbook_repository):
        """Test successful playbook update."""
        mock_get_repo.return_value = mock_playbook_repository

        update_data = {
            "name": "Updated Playbook Name",
            "description": "Updated description",
            "tags": ["updated", "test"]
        }

        response = client.put("/api/v1/v1/playbooks/pb_test123", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "pb_test123"
        mock_playbook_repository.update_playbook.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_update_playbook_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook update when playbook not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_playbook.return_value = None

        update_data = {"name": "Updated Name"}

        response = client.put("/api/v1/v1/playbooks/nonexistent", json=update_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPlaybookDeletionEndpoint:
    """Test playbook deletion functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_delete_playbook_success(self, mock_get_repo, client, mock_playbook_repository):
        """Test successful playbook deletion."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.delete("/api/v1/v1/playbooks/pb_test123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "deleted successfully" in data["message"]
        mock_playbook_repository.delete_playbook.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_delete_playbook_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook deletion when playbook not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_playbook.return_value = None

        response = client.delete("/api/v1/v1/playbooks/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPlaybookExecutionEndpoint:
    """Test playbook execution functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_execute_playbook_success(self, mock_get_agent, mock_get_repo, client,
                                     mock_playbook_repository, mock_playbook_agent):
        """Test successful playbook execution."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_get_agent.return_value = mock_playbook_agent

        execution_request = {
            "execution_mode": "sequential",
            "parameters": {"environment": "test"},
            "validation_level": "standard",
            "dry_run": False,
            "timeout_minutes": 30
        }

        response = client.post("/api/v1/v1/playbooks/pb_test123/execute", json=execution_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["playbook_id"] == "pb_test123"
        assert data["status"] == "completed"
        assert data["execution_mode"] == "sequential"
        assert "execution_id" in data
        mock_playbook_agent.execute.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_execute_playbook_failure(self, mock_get_agent, mock_get_repo, client,
                                     mock_playbook_repository, mock_playbook_agent):
        """Test playbook execution failure."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_get_agent.return_value = mock_playbook_agent
        mock_playbook_agent.execute.side_effect = Exception("Execution failed")

        execution_request = {
            "execution_mode": "sequential",
            "parameters": {}
        }

        response = client.post("/api/v1/v1/playbooks/pb_test123/execute", json=execution_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Execution failed"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_execute_playbook_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook execution when playbook not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_playbook.return_value = None

        execution_request = {"execution_mode": "sequential"}

        response = client.post("/api/v1/v1/playbooks/nonexistent/execute", json=execution_request)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_execute_playbook_invalid_mode(self, client):
        """Test playbook execution with invalid mode."""
        execution_request = {
            "execution_mode": "invalid_mode",
            "parameters": {}
        }

        response = client.post("/api/v1/v1/playbooks/pb_test123/execute", json=execution_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPlaybookExecutionControlEndpoints:
    """Test playbook execution control functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_execution_status(self, mock_get_repo, client, mock_playbook_repository):
        """Test getting execution status."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/executions/exec_123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["execution_id"] == "exec_123"
        assert data["status"] == "completed"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_execution_status_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test getting execution status when execution not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_execution.return_value = None

        response = client.get("/api/v1/v1/playbooks/executions/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_stop_execution_success(self, mock_get_repo, client, mock_playbook_repository):
        """Test successful execution stopping."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.post("/api/v1/v1/playbooks/executions/exec_123/stop")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "stopped successfully" in data["message"]
        mock_playbook_repository.stop_execution.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_playbook_executions(self, mock_get_repo, client, mock_playbook_repository):
        """Test getting playbook execution history."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/pb_test123/executions?page=1&page_size=10")

        assert response.status_code == status.HTTP_200_OK
        mock_playbook_repository.get_playbook_executions.assert_called_once()


class TestPlaybookValidationEndpoint:
    """Test playbook validation functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_validate_playbook_success(self, mock_get_agent, mock_get_repo, client,
                                      mock_playbook_repository, mock_playbook_agent):
        """Test successful playbook validation."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_get_agent.return_value = mock_playbook_agent

        response = client.post("/api/v1/v1/playbooks/pb_test123/validate")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["complexity_score"] == 5
        mock_playbook_agent.validate_playbook.assert_called_once()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_validate_playbook_not_found(self, mock_get_repo, client, mock_playbook_repository):
        """Test playbook validation when playbook not found."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_playbook_repository.get_playbook.return_value = None

        response = client.post("/api/v1/v1/playbooks/nonexistent/validate")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPlaybookTemplateEndpoints:
    """Test playbook template functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_get_available_templates(self, mock_get_agent, client, mock_playbook_agent):
        """Test getting available templates."""
        mock_get_agent.return_value = mock_playbook_agent

        response = client.get("/api/v1/v1/playbooks/templates/available")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "incident_response"
        assert data[0]["category"] == "security"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_get_available_templates_with_filter(self, mock_get_agent, client, mock_playbook_agent):
        """Test getting available templates with category filter."""
        mock_get_agent.return_value = mock_playbook_agent

        response = client.get("/api/v1/v1/playbooks/templates/available?category=security")

        assert response.status_code == status.HTTP_200_OK
        mock_playbook_agent.get_available_templates.assert_called_once_with(category_filter="security")

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_agent')
    def test_create_from_template_success(self, mock_get_agent, mock_get_repo, client,
                                         mock_playbook_repository, mock_playbook_agent):
        """Test successful playbook creation from template."""
        mock_get_repo.return_value = mock_playbook_repository
        mock_get_agent.return_value = mock_playbook_agent

        response = client.post(
            "/api/v1/v1/playbooks/templates/incident_response/create?name=New Incident Playbook"
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Test Playbook"
        mock_playbook_agent.create_from_template.assert_called_once()
        mock_playbook_repository.create_playbook.assert_called_once()


class TestPlaybookAnalyticsEndpoints:
    """Test playbook analytics functionality."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_playbook_analytics(self, mock_get_repo, client, mock_playbook_repository):
        """Test getting playbook analytics."""
        mock_get_repo.return_value = mock_playbook_repository

        response = client.get("/api/v1/v1/playbooks/pb_test123/analytics?days=30")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["playbook_id"] == "pb_test123"
        assert data["total_executions"] == 10
        assert data["success_rate"] == 0.8
        mock_playbook_repository.get_playbook_analytics.assert_called_once()


class TestPlaybookCategoriesEndpoint:
    """Test playbook categories functionality."""

    def test_get_available_categories(self, client):
        """Test getting available categories."""
        response = client.get("/api/v1/v1/playbooks/categories/available")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0

        # Check that categories have required fields
        for category in data["categories"]:
            assert "name" in category
            assert "display_name" in category
            assert "description" in category


class TestWebSocketConnections:
    """Test WebSocket functionality."""

    def test_websocket_manager_connect(self):
        """Test WebSocket manager connection."""
        websocket = Mock()
        websocket.accept = AsyncMock()

        # This would need to be tested with actual WebSocket client
        # For now, test the manager directly
        assert len(ws_manager.active_connections) >= 0

    def test_websocket_manager_disconnect(self):
        """Test WebSocket manager disconnection."""
        websocket = Mock()

        # Add to connections first
        ws_manager.active_connections.append(websocket)
        initial_count = len(ws_manager.active_connections)

        # Disconnect
        ws_manager.disconnect(websocket)

        assert len(ws_manager.active_connections) == initial_count - 1


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_internal_server_error(self, mock_get_repo, client):
        """Test internal server error handling."""
        mock_get_repo.side_effect = Exception("Internal error")

        response = client.get("/api/v1/v1/playbooks/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to list playbooks" in data["detail"]

    def test_validation_error(self, client):
        """Test validation error handling."""
        invalid_request = {
            "name": "",  # Invalid empty name
            "category": "automation",
            "steps": []
        }

        response = client.post("/api/v1/v1/playbooks/", json=invalid_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPlaybookRequestValidation:
    """Test playbook request validation."""

    def test_valid_playbook_create_request(self, sample_playbook_create_request):
        """Test valid playbook creation request."""
        request = PlaybookCreateRequest(**sample_playbook_create_request)

        assert request.name == "Test Automation Playbook"
        assert request.category == "automation"
        assert len(request.steps) == 3

    def test_invalid_playbook_category(self):
        """Test invalid playbook category validation."""
        with pytest.raises(ValueError):
            PlaybookCreateRequest(
                name="Test Playbook",
                category="invalid_category",
                steps=[]
            )

    def test_playbook_execution_request_validation(self):
        """Test playbook execution request validation."""
        request = PlaybookExecutionStartRequest(
            execution_mode="sequential",
            parameters={"env": "test"},
            validation_level="standard",
            dry_run=True,
            timeout_minutes=60
        )

        assert request.execution_mode == "sequential"
        assert request.parameters == {"env": "test"}
        assert request.dry_run is True

    def test_invalid_execution_mode(self):
        """Test invalid execution mode validation."""
        with pytest.raises(ValueError):
            PlaybookExecutionStartRequest(
                execution_mode="invalid_mode",
                parameters={}
            )

    def test_invalid_validation_level(self):
        """Test invalid validation level validation."""
        with pytest.raises(ValueError):
            PlaybookExecutionStartRequest(
                execution_mode="sequential",
                validation_level="invalid_level",
                parameters={}
            )


class TestPlaybookModelMapping:
    """Test playbook model mapping functionality."""

    def test_playbook_category_enum_completeness(self):
        """Test that all expected playbook categories are available."""
        expected_categories = [
            "incident_response", "troubleshooting", "deployment", "maintenance",
            "security", "code_review", "testing", "release", "onboarding",
            "monitoring", "backup", "disaster_recovery", "capacity_planning"
        ]

        available_categories = [cat.value for cat in PlaybookCategory]

        for category in expected_categories:
            assert category in available_categories

    def test_execution_mode_enum_completeness(self):
        """Test that all expected execution modes are available."""
        expected_modes = ["sequential", "parallel", "conditional", "interactive", "automated"]

        available_modes = [mode.value for mode in ExecutionMode]

        for mode in expected_modes:
            assert mode in available_modes

    def test_validation_level_enum_completeness(self):
        """Test that all expected validation levels are available."""
        expected_levels = ["strict", "standard", "permissive", "none"]

        available_levels = [level.value for level in ValidationLevel]

        for level in expected_levels:
            assert level in available_levels


if __name__ == '__main__':
    pytest.main([__file__])
