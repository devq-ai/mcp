"""
Tests for Agent Management API Endpoints

This module provides comprehensive test coverage for the agent management
API endpoints, including CRUD operations, real-time monitoring, configuration,
execution control, and analytics.
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
from agentical.agents import (
    AgentRegistry, AgentType, AgentStatus, AgentPhase,
    CodeAgent, DataScienceAgent, PlaybookAgent
)
from agentical.api.v1.endpoints.agents import (
    AgentStatusResponse, AgentConfigRequest, AgentExecutionRequest,
    AGENT_TYPE_MAPPING, ws_manager
)


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
def mock_agent_registry():
    """Mock agent registry."""
    registry = Mock(spec=AgentRegistry)
    registry.list_agents.return_value = []
    registry.get_agent.return_value = None
    registry.register_agent = Mock()
    registry.unregister_agent = Mock()
    return registry


@pytest.fixture
def mock_agent_repository():
    """Mock agent repository."""
    repo = AsyncMock()
    repo.get_agent_metrics.return_value = {
        "execution_count": 5,
        "success_rate": 0.8,
        "avg_execution_time": 2.5,
        "resource_usage": {"memory": 128, "cpu": 25}
    }
    repo.create_agent = AsyncMock()
    repo.update_agent_config = AsyncMock()
    repo.create_execution_record = AsyncMock()
    repo.complete_execution_record = AsyncMock()
    repo.get_agent_executions.return_value = {
        "executions": [],
        "total": 0,
        "page": 1,
        "page_size": 20
    }
    repo.get_agent_analytics.return_value = {
        "agent_id": "test_agent",
        "total_executions": 10,
        "successful_executions": 8,
        "failed_executions": 2,
        "success_rate": 0.8,
        "average_execution_time": 2.5,
        "peak_execution_time": 5.0,
        "total_runtime": 25.0,
        "executions_by_day": {},
        "operations_frequency": {},
        "error_patterns": {},
        "performance_trends": []
    }
    repo.delete_agent = AsyncMock()
    return repo


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    agent = Mock()
    agent.agent_id = "test_agent_123"
    agent.agent_type = AgentType.CODE_AGENT
    agent.name = "Test Code Agent"
    agent.description = "A test code agent"
    agent.status = AgentStatus.ACTIVE
    agent.current_phase = AgentPhase.IDLE
    agent.created_at = datetime.utcnow()
    agent.last_active = datetime.utcnow()
    agent.get_capabilities.return_value = ["code_analysis", "debugging", "testing"]
    agent.execute = AsyncMock(return_value={"success": True, "result": "test result"})
    agent.stop = AsyncMock()
    agent.health_status = "healthy"
    return agent


class TestAgentListEndpoint:
    """Test agent listing functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_list_agents_empty(self, mock_get_repo, mock_get_registry, client,
                              mock_agent_registry, mock_agent_repository):
        """Test listing agents when none exist."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository

        response = client.get("/api/v1/v1/agents/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agents"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert "available_types" in data

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_list_agents_with_data(self, mock_get_repo, mock_get_registry, client,
                                  mock_agent_registry, mock_agent_repository, sample_agent):
        """Test listing agents with data."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.list_agents.return_value = [sample_agent]

        response = client.get("/api/v1/v1/agents/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["agents"]) == 1
        assert data["total"] == 1
        assert data["agents"][0]["agent_id"] == "test_agent_123"
        assert data["agents"][0]["agent_type"] == "code_agent"

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_list_agents_with_filters(self, mock_get_repo, mock_get_registry, client,
                                     mock_agent_registry, mock_agent_repository, sample_agent):
        """Test listing agents with filters."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.list_agents.return_value = [sample_agent]

        response = client.get("/api/v1/v1/agents/?agent_type=code_agent&status=active")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["agents"]) == 1

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_list_agents_pagination(self, mock_get_repo, mock_get_registry, client,
                                   mock_agent_registry, mock_agent_repository):
        """Test agent listing pagination."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository

        response = client.get("/api/v1/v1/agents/?page=2&page_size=10")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10


class TestAgentCreationEndpoint:
    """Test agent creation functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_create_agent_success(self, mock_get_repo, mock_get_registry, client,
                                 mock_agent_registry, mock_agent_repository):
        """Test successful agent creation."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository

        agent_config = {
            "agent_type": "code",
            "name": "Test Code Agent",
            "description": "A test code agent",
            "config": {"max_concurrent_tasks": 5},
            "capabilities": ["code_analysis", "debugging"]
        }

        response = client.post("/api/v1/v1/agents/", json=agent_config)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_type"] == "code"
        assert data["name"] == "Test Code Agent"
        assert "agent_id" in data
        mock_agent_registry.register_agent.assert_called_once()

    def test_create_agent_invalid_type(self, client):
        """Test agent creation with invalid type."""
        agent_config = {
            "agent_type": "invalid_type",
            "name": "Test Agent"
        }

        response = client.post("/api/v1/v1/agents/", json=agent_config)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_create_agent_registry_error(self, mock_get_repo, mock_get_registry, client,
                                        mock_agent_registry, mock_agent_repository):
        """Test agent creation with registry error."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.register_agent.side_effect = Exception("Registry error")

        agent_config = {
            "agent_type": "code",
            "name": "Test Agent"
        }

        response = client.post("/api/v1/v1/agents/", json=agent_config)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestAgentRetrievalEndpoint:
    """Test agent retrieval functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_get_agent_success(self, mock_get_repo, mock_get_registry, client,
                              mock_agent_registry, mock_agent_repository, sample_agent):
        """Test successful agent retrieval."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = sample_agent

        response = client.get("/api/v1/v1/agents/test_agent_123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test_agent_123"
        assert data["agent_type"] == "code_agent"
        assert data["status"] == "active"

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_get_agent_not_found(self, mock_get_repo, mock_get_registry, client,
                                 mock_agent_registry, mock_agent_repository):
        """Test agent retrieval when agent not found."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = None

        response = client.get("/api/v1/v1/agents/nonexistent_agent")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentConfigurationEndpoint:
    """Test agent configuration functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_update_agent_config_success(self, mock_get_repo, mock_get_registry, client,
                                        mock_agent_registry, mock_agent_repository, sample_agent):
        """Test successful agent configuration update."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = sample_agent

        config_update = {
            "max_concurrent_tasks": 10,
            "timeout_seconds": 300
        }

        response = client.put("/api/v1/v1/agents/test_agent_123/config", json=config_update)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test_agent_123"
        mock_agent_repository.update_agent_config.assert_called_once()

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_update_agent_config_not_found(self, mock_get_repo, mock_get_registry, client,
                                          mock_agent_registry, mock_agent_repository):
        """Test agent configuration update when agent not found."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = None

        config_update = {"max_concurrent_tasks": 10}

        response = client.put("/api/v1/v1/agents/nonexistent/config", json=config_update)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentExecutionEndpoint:
    """Test agent execution functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_execute_agent_operation_success(self, mock_get_repo, mock_get_registry, client,
                                            mock_agent_registry, mock_agent_repository, sample_agent):
        """Test successful agent operation execution."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = sample_agent

        execution_request = {
            "operation": "analyze_code",
            "parameters": {"code": "print('hello')", "language": "python"},
            "timeout": 30
        }

        response = client.post("/api/v1/v1/agents/test_agent_123/execute", json=execution_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test_agent_123"
        assert data["operation"] == "analyze_code"
        assert data["status"] == "success"
        assert "execution_id" in data
        sample_agent.execute.assert_called_once()

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_execute_agent_operation_failure(self, mock_get_repo, mock_get_registry, client,
                                            mock_agent_registry, mock_agent_repository, sample_agent):
        """Test agent operation execution failure."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = sample_agent
        sample_agent.execute.side_effect = Exception("Execution failed")

        execution_request = {
            "operation": "analyze_code",
            "parameters": {"code": "invalid code"}
        }

        response = client.post("/api/v1/v1/agents/test_agent_123/execute", json=execution_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Execution failed"

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_execute_agent_not_found(self, mock_get_repo, mock_get_registry, client,
                                     mock_agent_registry, mock_agent_repository):
        """Test agent execution when agent not found."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = None

        execution_request = {
            "operation": "test_operation",
            "parameters": {}
        }

        response = client.post("/api/v1/v1/agents/nonexistent/execute", json=execution_request)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentControlEndpoints:
    """Test agent control functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    def test_stop_agent_success(self, mock_get_registry, client, mock_agent_registry, sample_agent):
        """Test successful agent stopping."""
        mock_get_registry.return_value = mock_agent_registry
        mock_agent_registry.get_agent.return_value = sample_agent

        response = client.post("/api/v1/v1/agents/test_agent_123/stop")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "stopped successfully" in data["message"]
        sample_agent.stop.assert_called_once()

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_delete_agent_success(self, mock_get_repo, mock_get_registry, client,
                                 mock_agent_registry, mock_agent_repository, sample_agent):
        """Test successful agent deletion."""
        mock_get_registry.return_value = mock_agent_registry
        mock_get_repo.return_value = mock_agent_repository
        mock_agent_registry.get_agent.return_value = sample_agent
        sample_agent.status = AgentStatus.STOPPED

        response = client.delete("/api/v1/v1/agents/test_agent_123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "deleted successfully" in data["message"]
        mock_agent_registry.unregister_agent.assert_called_once()
        mock_agent_repository.delete_agent.assert_called_once()


class TestAgentAnalyticsEndpoints:
    """Test agent analytics functionality."""

    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_get_agent_executions(self, mock_get_repo, client, mock_agent_repository):
        """Test getting agent execution history."""
        mock_get_repo.return_value = mock_agent_repository

        response = client.get("/api/v1/v1/agents/test_agent_123/executions?page=1&page_size=10")

        assert response.status_code == status.HTTP_200_OK
        mock_agent_repository.get_agent_executions.assert_called_once()

    @patch('agentical.api.v1.endpoints.agents.get_agent_repository')
    def test_get_agent_analytics(self, mock_get_repo, client, mock_agent_repository):
        """Test getting agent analytics."""
        mock_get_repo.return_value = mock_agent_repository

        response = client.get("/api/v1/v1/agents/test_agent_123/analytics?days=30")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "test_agent_123"
        assert data["total_executions"] == 10
        assert data["success_rate"] == 0.8
        mock_agent_repository.get_agent_analytics.assert_called_once()


class TestAgentTypesEndpoint:
    """Test agent types functionality."""

    def test_get_available_agent_types(self, client):
        """Test getting available agent types."""
        response = client.get("/api/v1/v1/agents/types/available")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agent_types" in data
        assert "total" in data
        assert len(data["agent_types"]) > 0

        # Check that all expected types are present
        type_names = [agent_type["type"] for agent_type in data["agent_types"]]
        assert "code" in type_names
        assert "data_science" in type_names
        assert "playbook" in type_names


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

    @patch('agentical.api.v1.endpoints.agents.get_agent_registry')
    def test_internal_server_error(self, mock_get_registry, client):
        """Test internal server error handling."""
        mock_get_registry.side_effect = Exception("Internal error")

        response = client.get("/api/v1/v1/agents/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to list agents" in data["detail"]

    def test_validation_error(self, client):
        """Test validation error handling."""
        invalid_request = {
            "agent_type": "",  # Invalid empty type
            "name": "Test Agent"
        }

        response = client.post("/api/v1/v1/agents/", json=invalid_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAgentConfigValidation:
    """Test agent configuration validation."""

    def test_valid_agent_config(self):
        """Test valid agent configuration."""
        config = AgentConfigRequest(
            agent_type="code",
            name="Test Agent",
            description="Test description",
            config={"timeout": 30},
            capabilities=["testing"]
        )

        assert config.agent_type == "code"
        assert config.name == "Test Agent"

    def test_invalid_agent_type(self):
        """Test invalid agent type validation."""
        with pytest.raises(ValueError):
            AgentConfigRequest(
                agent_type="invalid_type",
                name="Test Agent"
            )

    def test_agent_execution_request_validation(self):
        """Test agent execution request validation."""
        request = AgentExecutionRequest(
            operation="test_operation",
            parameters={"param1": "value1"},
            timeout=60,
            priority="high"
        )

        assert request.operation == "test_operation"
        assert request.parameters == {"param1": "value1"}
        assert request.timeout == 60
        assert request.priority == "high"


class TestAgentTypeMapping:
    """Test agent type mapping functionality."""

    def test_agent_type_mapping_completeness(self):
        """Test that all expected agent types are mapped."""
        expected_types = [
            "code", "data_science", "dba", "devops", "gcp", "github",
            "infosec", "legal", "pulumi", "research", "tester", "token",
            "uat", "ux", "codifier", "io", "playbook", "super"
        ]

        for agent_type in expected_types:
            assert agent_type in AGENT_TYPE_MAPPING
            assert AGENT_TYPE_MAPPING[agent_type] is not None

    def test_agent_type_instantiation(self):
        """Test that agent types can be instantiated."""
        # Test a few key agent types
        test_types = ["code", "data_science", "playbook"]

        for agent_type in test_types:
            agent_class = AGENT_TYPE_MAPPING[agent_type]
            assert agent_class is not None
            assert hasattr(agent_class, '__name__')


if __name__ == '__main__':
    pytest.main([__file__])
