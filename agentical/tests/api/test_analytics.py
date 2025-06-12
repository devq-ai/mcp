"""
Tests for Analytics & Monitoring API Endpoints

This module contains comprehensive tests for the analytics and monitoring endpoints,
including system metrics, workflow analytics, agent performance, and Logfire integration.

Features:
- System performance metrics tests
- Workflow execution analytics tests
- Agent performance monitoring tests
- Logfire observability metrics tests
- Custom analytics query tests
- Health check and monitoring tests
- WebSocket real-time metrics tests
- Data export functionality tests
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
from agentical.db.models.workflow import ExecutionStatus
from agentical.api.v1.endpoints.analytics import metrics_ws_manager


class TestSystemMetrics:
    """Test system performance metrics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.analytics.psutil')
    def test_get_system_metrics_success(self, mock_psutil, client):
        """Test successful system metrics retrieval."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.2
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=67.8,
            used=8589934592,  # 8GB
            total=12884901888  # 12GB
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            used=107374182400,  # 100GB
            total=322122547200,  # 300GB
            percent=33.3
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            _asdict=lambda: {"bytes_sent": 1024000, "bytes_recv": 2048000}
        )
        mock_psutil.net_connections.return_value = [MagicMock() for _ in range(150)]
        mock_psutil.boot_time.return_value = (datetime.now() - timedelta(days=1)).timestamp()
        mock_psutil.getloadavg.return_value = (0.5, 0.7, 0.8)

        response = client.get("/api/v1/analytics/system/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "disk_usage_percent" in data
        assert "network_io" in data
        assert "active_connections" in data
        assert "uptime_seconds" in data
        assert "load_average" in data
        assert data["cpu_usage_percent"] == 45.2
        assert data["memory_usage_percent"] == 67.8

    @patch('agentical.api.v1.endpoints.analytics.psutil')
    def test_get_system_metrics_windows_fallback(self, mock_psutil, client):
        """Test system metrics on Windows (no getloadavg)."""
        mock_psutil.cpu_percent.return_value = 35.5
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=55.0,
            used=4294967296,
            total=8589934592
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            used=53687091200,
            total=161061273600,
            percent=33.3
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            _asdict=lambda: {"bytes_sent": 512000, "bytes_recv": 1024000}
        )
        mock_psutil.net_connections.side_effect = OSError("Access denied")
        mock_psutil.boot_time.return_value = (datetime.now() - timedelta(hours=12)).timestamp()
        mock_psutil.getloadavg.side_effect = AttributeError("Windows doesn't have getloadavg")

        response = client.get("/api/v1/analytics/system/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["load_average"] == [0.0, 0.0, 0.0]  # Windows fallback
        assert data["active_connections"] == 0  # Access denied fallback

    @patch('agentical.api.v1.endpoints.analytics.psutil')
    def test_get_system_metrics_error(self, mock_psutil, client):
        """Test system metrics error handling."""
        mock_psutil.cpu_percent.side_effect = Exception("System error")

        response = client.get("/api/v1/analytics/system/metrics")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestWorkflowMetrics:
    """Test workflow analytics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        repo.get_workflow_metrics.return_value = {
            "total_executions": 150,
            "successful_executions": 120,
            "failed_executions": 30,
            "average_duration_seconds": 300.5,
            "success_rate_percent": 80.0,
            "executions_per_hour": [
                {"hour": "2024-01-01T00:00:00", "count": 10},
                {"hour": "2024-01-01T01:00:00", "count": 15}
            ],
            "most_used_workflows": [
                {"workflow_id": "wf1", "workflow_name": "Data Processing", "execution_count": 50},
                {"workflow_id": "wf2", "workflow_name": "Report Generation", "execution_count": 30}
            ],
            "error_distribution": {
                "ValidationError": 15,
                "TimeoutError": 10,
                "ConnectionError": 5
            },
            "performance_trends": [
                {"date": "2024-01-01", "avg_duration_seconds": 295.2, "execution_count": 25, "success_rate": 85.0}
            ]
        }
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_workflow_metrics_success(self, mock_get_repo, client, mock_analytics_repo):
        """Test successful workflow metrics retrieval."""
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/workflows/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_executions"] == 150
        assert data["successful_executions"] == 120
        assert data["failed_executions"] == 30
        assert data["success_rate_percent"] == 80.0
        assert len(data["executions_per_hour"]) == 2
        assert len(data["most_used_workflows"]) == 2
        assert "ValidationError" in data["error_distribution"]

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_workflow_metrics_with_filters(self, mock_get_repo, client, mock_analytics_repo):
        """Test workflow metrics with date and workflow filters."""
        mock_get_repo.return_value = mock_analytics_repo

        from_date = "2024-01-01T00:00:00Z"
        to_date = "2024-01-02T00:00:00Z"
        workflow_ids = "wf1,wf2"

        response = client.get(
            f"/api/v1/analytics/workflows/metrics"
            f"?from_date={from_date}&to_date={to_date}&workflow_ids={workflow_ids}"
        )

        assert response.status_code == status.HTTP_200_OK
        mock_analytics_repo.get_workflow_metrics.assert_called_once()
        call_args = mock_analytics_repo.get_workflow_metrics.call_args
        assert call_args[1]["workflow_ids"] == ["wf1", "wf2"]

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_workflow_metrics_error(self, mock_get_repo, client):
        """Test workflow metrics error handling."""
        mock_repo = AsyncMock()
        mock_repo.get_workflow_metrics.side_effect = Exception("Database error")
        mock_get_repo.return_value = mock_repo

        response = client.get("/api/v1/analytics/workflows/metrics")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestAgentMetrics:
    """Test agent performance metrics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        repo.get_agent_metrics.return_value = {
            "total_executions": 500,
            "successful_executions": 450,
            "failed_executions": 50,
            "average_response_time_ms": 1250.5,
            "success_rate_percent": 90.0,
            "most_active_agents": [
                {"agent_id": "agent1", "agent_name": "Code Agent", "agent_type": "code_agent", "execution_count": 200},
                {"agent_id": "agent2", "agent_name": "Data Agent", "agent_type": "data_science_agent", "execution_count": 150}
            ],
            "error_types": {
                "TimeoutError": 25,
                "ValidationError": 15,
                "ProcessingError": 10
            },
            "performance_by_agent_type": [
                {"agent_type": "code_agent", "total_executions": 200, "successful_executions": 190, "success_rate": 95.0, "avg_response_time_ms": 1100.0},
                {"agent_type": "data_science_agent", "total_executions": 150, "successful_executions": 135, "success_rate": 90.0, "avg_response_time_ms": 1800.0}
            ]
        }
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_agent_metrics_success(self, mock_get_repo, client, mock_analytics_repo):
        """Test successful agent metrics retrieval."""
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/agents/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_executions"] == 500
        assert data["successful_executions"] == 450
        assert data["average_response_time_ms"] == 1250.5
        assert data["success_rate_percent"] == 90.0
        assert len(data["most_active_agents"]) == 2
        assert len(data["performance_by_agent_type"]) == 2
        assert "TimeoutError" in data["error_types"]

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_agent_metrics_with_filters(self, mock_get_repo, client, mock_analytics_repo):
        """Test agent metrics with date and type filters."""
        mock_get_repo.return_value = mock_analytics_repo

        agent_types = "code_agent,data_science_agent"
        response = client.get(f"/api/v1/analytics/agents/metrics?agent_types={agent_types}")

        assert response.status_code == status.HTTP_200_OK
        mock_analytics_repo.get_agent_metrics.assert_called_once()
        call_args = mock_analytics_repo.get_agent_metrics.call_args
        assert call_args[1]["agent_types"] == ["code_agent", "data_science_agent"]


class TestLogfireMetrics:
    """Test Logfire observability metrics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        repo.get_logfire_metrics.return_value = {
            "total_spans": 15420,
            "error_spans": 234,
            "average_span_duration_ms": 125.5,
            "spans_by_service": {
                "agentical-api": 8500,
                "workflow-engine": 4200,
                "agent-executor": 2720
            },
            "error_rate_percent": 1.52,
            "top_errors": [
                {"error_type": "ValidationError", "count": 89, "percentage": 38.0},
                {"error_type": "TimeoutError", "count": 67, "percentage": 28.6}
            ],
            "performance_insights": [
                {"insight": "Workflow execution time increased 15%", "severity": "medium", "recommendation": "Review optimization"}
            ],
            "trace_analysis": {
                "average_trace_duration_ms": 892.3,
                "slowest_operations": [
                    {"operation": "database_query", "avg_duration_ms": 245.6},
                    {"operation": "llm_inference", "avg_duration_ms": 1850.2}
                ]
            }
        }
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_logfire_metrics_success(self, mock_get_repo, client, mock_analytics_repo):
        """Test successful Logfire metrics retrieval."""
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/logfire/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_spans"] == 15420
        assert data["error_spans"] == 234
        assert data["error_rate_percent"] == 1.52
        assert "agentical-api" in data["spans_by_service"]
        assert len(data["top_errors"]) == 2
        assert len(data["performance_insights"]) == 1
        assert "trace_analysis" in data

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_get_logfire_metrics_with_services(self, mock_get_repo, client, mock_analytics_repo):
        """Test Logfire metrics with service name filters."""
        mock_get_repo.return_value = mock_analytics_repo

        service_names = "agentical-api,workflow-engine"
        response = client.get(f"/api/v1/analytics/logfire/metrics?service_names={service_names}")

        assert response.status_code == status.HTTP_200_OK
        mock_analytics_repo.get_logfire_metrics.assert_called_once()
        call_args = mock_analytics_repo.get_logfire_metrics.call_args
        assert call_args[1]["service_names"] == ["agentical-api", "workflow-engine"]


class TestCustomMetricsQuery:
    """Test custom analytics query endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        repo.get_available_metrics.return_value = [
            "workflow_executions",
            "workflow_duration",
            "agent_executions",
            "system_cpu_usage"
        ]
        repo.query_metrics.return_value = [
            {"metric": "workflow_executions", "timestamp": "2024-01-01T00:00:00Z", "value": 10.0, "aggregation": "count"},
            {"metric": "workflow_executions", "timestamp": "2024-01-01T01:00:00Z", "value": 15.0, "aggregation": "count"}
        ]
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_query_custom_metrics_success(self, mock_get_repo, client, mock_analytics_repo):
        """Test successful custom metrics query."""
        mock_get_repo.return_value = mock_analytics_repo

        query_data = {
            "metrics": ["workflow_executions", "agent_executions"],
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z"
            },
            "granularity": "1h",
            "aggregation": "count"
        }

        response = client.post("/api/v1/analytics/query", json=query_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "result_count" in data
        assert len(data["results"]) == 2

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_query_custom_metrics_invalid_metric(self, mock_get_repo, client, mock_analytics_repo):
        """Test custom metrics query with invalid metric."""
        mock_get_repo.return_value = mock_analytics_repo

        query_data = {
            "metrics": ["invalid_metric"],
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z"
            }
        }

        response = client.post("/api/v1/analytics/query", json=query_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_query_custom_metrics_validation_error(self, client):
        """Test custom metrics query with validation errors."""
        invalid_query = {
            "metrics": [],  # Empty metrics list
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2023-12-31T23:59:59Z"  # End before start
            }
        }

        response = client.post("/api/v1/analytics/query", json=invalid_query)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestHealthCheck:
    """Test comprehensive health check endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        repo.health_check.return_value = True
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    @patch('agentical.api.v1.endpoints.analytics.psutil')
    def test_comprehensive_health_check_healthy(self, mock_psutil, mock_get_repo, client, mock_analytics_repo):
        """Test comprehensive health check with all systems healthy."""
        mock_get_repo.return_value = mock_analytics_repo
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=70.0)
        mock_psutil.boot_time.return_value = (datetime.now() - timedelta(days=1)).timestamp()

        response = client.get("/api/v1/analytics/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["overall_health_score"] >= 90.0
        assert "database" in data["checks"]
        assert "system_resources" in data["checks"]
        assert "logfire" in data["checks"]
        assert data["checks"]["database"]["status"] == "healthy"

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    @patch('agentical.api.v1.endpoints.analytics.psutil')
    def test_comprehensive_health_check_degraded(self, mock_psutil, mock_get_repo, client):
        """Test comprehensive health check with degraded performance."""
        mock_repo = AsyncMock()
        mock_repo.health_check.return_value = True
        mock_get_repo.return_value = mock_repo

        # High resource usage
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=95.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=90.0)
        mock_psutil.boot_time.return_value = (datetime.now() - timedelta(days=1)).timestamp()

        response = client.get("/api/v1/analytics/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] in ["degraded", "unhealthy"]
        assert data["overall_health_score"] < 90.0

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_comprehensive_health_check_database_error(self, mock_get_repo, client):
        """Test comprehensive health check with database error."""
        mock_repo = AsyncMock()
        mock_repo.health_check.side_effect = Exception("Database connection failed")
        mock_get_repo.return_value = mock_repo

        with patch('agentical.api.v1.endpoints.analytics.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.virtual_memory.return_value = MagicMock(percent=40.0)
            mock_psutil.disk_usage.return_value = MagicMock(percent=50.0)
            mock_psutil.boot_time.return_value = (datetime.now() - timedelta(days=1)).timestamp()

            response = client.get("/api/v1/analytics/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["checks"]["database"]["status"] == "unhealthy"
            assert "error" in data["checks"]["database"]


class TestMetricsWebSocket:
    """Test real-time metrics streaming via WebSocket."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_websocket_manager_connect(self):
        """Test WebSocket manager connection handling."""
        mock_websocket = MagicMock(spec=WebSocket)
        client_id = "client_123"
        config = {"interval_seconds": 30, "metrics": ["system", "workflows"]}

        # Test that connection tracking works
        assert client_id not in metrics_ws_manager.active_connections

        # In real test, would need to mock websocket.accept()
        # metrics_ws_manager.connect(mock_websocket, client_id, config)

        # Test disconnect
        metrics_ws_manager.disconnect(mock_websocket, client_id)
        assert client_id not in metrics_ws_manager.active_connections

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Test WebSocket broadcast functionality."""
        test_metrics = {
            "system": {"cpu_usage": 45.0, "memory_usage": 60.0},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Test broadcast to empty connections (should not raise error)
        await metrics_ws_manager.broadcast_metrics(test_metrics)

    def test_websocket_endpoint_invalid_params(self, client):
        """Test WebSocket endpoint with invalid parameters."""
        # Test invalid interval
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/analytics/metrics/stream?interval_seconds=1"):
                pass

    @patch('agentical.api.v1.endpoints.analytics.get_system_metrics')
    def test_websocket_metrics_streaming(self, mock_get_system_metrics, client):
        """Test WebSocket metrics streaming functionality."""
        mock_get_system_metrics.return_value = MagicMock(
            dict=lambda: {"cpu_usage_percent": 45.0, "memory_usage_percent": 60.0}
        )

        # This would be a more complex test in practice
        # testing the actual WebSocket communication
        pass


class TestMetricsExport:
    """Test metrics data export functionality."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @pytest.fixture
    def mock_analytics_repo(self):
        """Mock analytics repository."""
        repo = AsyncMock()
        return repo

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_export_metrics_json(self, mock_get_repo, client, mock_analytics_repo):
        """Test metrics export in JSON format."""
        mock_analytics_repo.export_metrics.return_value = '{"test": "data"}'
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/export/metrics?format=json")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers["content-disposition"]

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_export_metrics_csv(self, mock_get_repo, client, mock_analytics_repo):
        """Test metrics export in CSV format."""
        mock_analytics_repo.export_metrics.return_value = "metric,value\ntest,123"
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/export/metrics?format=csv")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/csv"

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_export_metrics_prometheus(self, mock_get_repo, client, mock_analytics_repo):
        """Test metrics export in Prometheus format."""
        mock_analytics_repo.export_metrics.return_value = "# HELP test_metric\ntest_metric 123"
        mock_get_repo.return_value = mock_analytics_repo

        response = client.get("/api/v1/analytics/export/metrics?format=prometheus")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/plain"

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_export_metrics_with_filters(self, mock_get_repo, client, mock_analytics_repo):
        """Test metrics export with date and type filters."""
        mock_analytics_repo.export_metrics.return_value = '{"filtered": "data"}'
        mock_get_repo.return_value = mock_analytics_repo

        from_date = "2024-01-01T00:00:00Z"
        to_date = "2024-01-02T00:00:00Z"
        metric_types = "workflows,agents"

        response = client.get(
            f"/api/v1/analytics/export/metrics"
            f"?format=json&from_date={from_date}&to_date={to_date}&metric_types={metric_types}"
        )

        assert response.status_code == status.HTTP_200_OK
        mock_analytics_repo.export_metrics.assert_called_once()
        call_args = mock_analytics_repo.export_metrics.call_args
        assert call_args[1]["metric_types"] == ["workflows", "agents"]

    def test_export_metrics_invalid_format(self, client):
        """Test metrics export with invalid format."""
        response = client.get("/api/v1/analytics/export/metrics?format=invalid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAnalyticsErrorHandling:
    """Test error handling in analytics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_database_error_handling(self, mock_get_repo, client):
        """Test handling of database errors."""
        mock_repo = AsyncMock()
        mock_repo.get_workflow_metrics.side_effect = Exception("Database connection failed")
        mock_get_repo.return_value = mock_repo

        response = client.get("/api/v1/analytics/workflows/metrics")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @patch('agentical.api.v1.endpoints.analytics.get_analytics_repository')
    def test_analytics_repository_error(self, mock_get_repo, client):
        """Test handling of analytics repository errors."""
        mock_repo = AsyncMock()
        mock_repo.query_metrics.side_effect = Exception("Query execution failed")
        mock_repo.get_available_metrics.return_value = ["test_metric"]
        mock_get_repo.return_value = mock_repo

        query_data = {
            "metrics": ["test_metric"],
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z"
            }
        }

        response = client.post("/api/v1/analytics/query", json=query_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestAnalyticsValidation:
    """Test validation in analytics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_metrics_query_time_range_validation(self, client):
        """Test metrics query time range validation."""
        invalid_query = {
            "metrics": ["test_metric"],
            "time_range": {
                "start_time": "2024-01-02T00:00:00Z",
                "end_time": "2024-01-01T00:00:00Z"  # End before start
            }
        }

        response = client.post("/api/v1/analytics/query", json=invalid_query)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_metrics_query_empty_metrics(self, client):
        """Test metrics query with empty metrics list."""
        invalid_query = {
            "metrics": [],  # Empty list
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z"
            }
        }

        response = client.post("/api/v1/analytics/query", json=invalid_query)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_granularity(self, client):
        """Test metrics query with invalid granularity."""
        invalid_query = {
            "metrics": ["test_metric"],
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z"
            },
            "granularity": "invalid_granularity"
        }

        response = client.post("/api/v1/analytics/query", json=invalid_query)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_aggregation(self, client):
        """Test metrics query with invalid aggregation."""
        invalid_query = {
            "metrics": ["test_metric"],
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z"
            },
            "aggregation": "invalid_aggregation"
        }

        response = client.post("/api/v1/analytics/query", json=invalid_query)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Integration tests that would run with actual database
@pytest.mark.integration
class TestAnalyticsIntegration:
    """Integration tests for analytics endpoints with real database."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_full_analytics_pipeline(self, client):
        """Test complete analytics pipeline from data collection to export."""
        # This would be an integration test that:
        # 1. Creates test data (workflows, executions, agents)
        # 2. Queries various analytics endpoints
        # 3. Verifies data consistency across endpoints
        # 4. Tests export functionality
        # 5. Validates real-time metrics streaming

        # For now, this is a placeholder
        pass

    def test_analytics_performance_under_load(self, client):
        """Test analytics system performance under load."""
        # This would test the system's performance
        # with large datasets and concurrent queries
        pass

    def test_real_logfire_integration(self, client):
        """Test real Logfire integration if available."""
        # This would test actual Logfire API integration
        # if credentials and connection are available
        pass


# Performance tests
@pytest.mark.performance
class TestAnalyticsPerformance:
    """Performance tests for analytics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_metrics_query_performance(self, client):
        """Test performance of complex metrics queries."""
        # This would test query performance with:
        # - Large time ranges
        # - Multiple metrics
        # - Complex aggregations
        # - Various granularities
        pass

    def test_export_performance(self, client):
        """Test performance of data export with large datasets."""
        # This would test export performance for:
        # - Large date ranges
        # - Multiple data types
        # - Different export formats
        pass

    def test_websocket_performance(self, client):
        """Test WebSocket streaming performance."""
        # This would test:
        # - Multiple concurrent connections
        # - High-frequency updates
        # - Large payload streaming
        pass


# Security tests
@pytest.mark.security
class TestAnalyticsSecurity:
    """Security tests for analytics endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection in analytics queries."""
        # This would test that malicious input in filters
        # and parameters doesn't lead to SQL injection
        pass

    def test_data_access_controls(self, client):
        """Test data access controls in analytics endpoints."""
        # This would test that users can only access
        # analytics data they're authorized to see
        pass

    def test_rate_limiting(self, client):
        """Test rate limiting on analytics endpoints."""
        # This would test that analytics endpoints
        # have appropriate rate limiting to prevent abuse
        pass


if __name__ == "__main__":
    pytest.main([__file__])
