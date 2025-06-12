"""
Test suite for Playbook Analytics API Endpoints

This module contains comprehensive tests for the new analytical endpoints
added in Task 9.2, including playbook analysis, expansion, detailed metrics,
and comprehensive reporting functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from uuid import uuid4

from agentical.main import app
from agentical.db.repositories.playbook import AsyncPlaybookRepository
from agentical.db.models.playbook import (
    Playbook, PlaybookStatus, PlaybookCategory, PlaybookStepType, PlaybookExecutionStatus
)


# Test client
client = TestClient(app)


class TestPlaybookAnalysisEndpoint:
    """Test cases for POST /playbooks/{id}/analyze endpoint."""

    @pytest.fixture
    def mock_playbook(self):
        """Create a mock playbook for testing."""
        playbook = Mock()
        playbook.id = uuid4()
        playbook.name = "Test Playbook"
        playbook.description = "Test description"
        playbook.status = PlaybookStatus.ACTIVE
        playbook.category = PlaybookCategory.AUTOMATION

        # Mock steps
        steps = []
        for i in range(5):
            step = Mock()
            step.id = uuid4()
            step.name = f"Step {i+1}"
            step.description = f"Description for step {i+1}"
            step.step_type = PlaybookStepType.ACTION if i < 3 else PlaybookStepType.CONDITIONAL
            steps.append(step)
        playbook.steps = steps

        # Mock variables
        playbook.variables = [Mock(), Mock()]

        return playbook

    @pytest.fixture
    def mock_repo(self, mock_playbook):
        """Create a mock repository for testing."""
        repo = AsyncMock(spec=AsyncPlaybookRepository)
        repo.get_with_executions.return_value = mock_playbook
        repo.calculate_dependency_depth.return_value = 2
        repo.get_recent_executions.return_value = [
            Mock(duration_seconds=120, status=PlaybookExecutionStatus.COMPLETED),
            Mock(duration_seconds=150, status=PlaybookExecutionStatus.COMPLETED),
            Mock(duration_seconds=180, status=PlaybookExecutionStatus.FAILED),
        ]
        repo.analyze_naming_consistency.return_value = 0.8
        repo.analyze_step_modularity.return_value = 0.7
        repo.get_resource_usage_stats.return_value = {
            "avg_cpu_usage": 45.2,
            "avg_memory_mb": 512.8
        }
        repo.identify_parallelization_opportunities.return_value = [
            {"type": "parallel_execution", "steps": ["step1", "step2"]}
        ]
        repo.get_step_execution_stats.return_value = {
            "avg_duration": 30.0,
            "failure_rate": 0.1,
            "complexity_score": 3.0
        }
        repo.get_category_statistics.return_value = {
            "avg_complexity": 5.5,
            "avg_performance": 7.0,
            "percentile_rank": 60.0
        }
        return repo

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_analyze_playbook_comprehensive(self, mock_get_repo, mock_repo):
        """Test comprehensive playbook analysis."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {
            "analysis_type": "comprehensive",
            "include_suggestions": True,
            "compare_with_category": True
        }

        response = client.post(f"/v1/playbooks/{playbook_id}/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "playbook_id" in data
        assert "analysis_timestamp" in data
        assert "complexity_score" in data
        assert "performance_score" in data
        assert "maintainability_score" in data
        assert "execution_efficiency" in data
        assert "step_analysis" in data
        assert "bottlenecks" in data
        assert "suggestions" in data
        assert "category_comparison" in data

        # Verify data types and ranges
        assert isinstance(data["complexity_score"], (int, float))
        assert 0 <= data["complexity_score"] <= 10
        assert isinstance(data["performance_score"], (int, float))
        assert 0 <= data["performance_score"] <= 10
        assert isinstance(data["suggestions"], list)

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_analyze_playbook_not_found(self, mock_get_repo):
        """Test analysis with non-existent playbook."""
        mock_repo = AsyncMock()
        mock_repo.get_with_executions.return_value = None
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {"analysis_type": "performance"}

        response = client.post(f"/v1/playbooks/{playbook_id}/analyze", json=request_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_analyze_playbook_without_category_comparison(self, mock_get_repo, mock_repo):
        """Test analysis without category comparison."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {
            "analysis_type": "performance",
            "compare_with_category": False
        }

        response = client.post(f"/v1/playbooks/{playbook_id}/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["category_comparison"] is None


class TestPlaybookExpansionEndpoint:
    """Test cases for POST /playbooks/{id}/expand endpoint."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository for expansion testing."""
        repo = AsyncMock(spec=AsyncPlaybookRepository)
        mock_playbook = Mock()
        mock_playbook.id = uuid4()
        mock_playbook.name = "Original Playbook"
        mock_playbook.description = "Original description"
        mock_playbook.steps = [Mock(), Mock(), Mock()]
        mock_playbook.variables = [Mock()]

        repo.get_with_executions.return_value = mock_playbook
        repo.calculate_complexity_score.return_value = 6.5
        repo.generate_structural_variation.return_value = {
            "name": "Optimized Variation",
            "description": "Performance optimized version",
            "rationale": "Added parallel execution"
        }
        repo.generate_optimized_version.return_value = {
            "name": "Parallel Optimization",
            "description": "Optimized for parallel execution",
            "estimated_improvement": "35% faster execution"
        }
        repo.generate_alternative_approaches.return_value = [
            {
                "approach_name": "Event-Driven",
                "approach_description": "Event-driven architecture",
                "benefits": ["Better scalability"]
            }
        ]
        return repo

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_expand_playbook_variations(self, mock_get_repo, mock_repo):
        """Test playbook expansion with variations."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {
            "expansion_type": "variations",
            "target_improvements": ["performance", "reliability"],
            "max_variations": 2
        }

        response = client.post(f"/v1/playbooks/{playbook_id}/expand", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "playbook_id" in data
        assert "expansion_timestamp" in data
        assert "expansion_type" in data
        assert "original_playbook" in data
        assert "generated_variations" in data
        assert "improvement_rationale" in data
        assert "estimated_benefits" in data

        # Verify content
        assert data["expansion_type"] == "variations"
        assert isinstance(data["generated_variations"], list)
        assert isinstance(data["improvement_rationale"], list)
        assert isinstance(data["estimated_benefits"], dict)

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_expand_playbook_optimizations(self, mock_get_repo, mock_repo):
        """Test playbook expansion with optimizations."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {
            "expansion_type": "optimizations",
            "max_variations": 3
        }

        response = client.post(f"/v1/playbooks/{playbook_id}/expand", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["expansion_type"] == "optimizations"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_expand_playbook_alternatives(self, mock_get_repo, mock_repo):
        """Test playbook expansion with alternatives."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())
        request_data = {
            "expansion_type": "alternatives",
            "max_variations": 2
        }

        response = client.post(f"/v1/playbooks/{playbook_id}/expand", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["expansion_type"] == "alternatives"


class TestPlaybookDetailedMetricsEndpoint:
    """Test cases for GET /playbooks/{id}/metrics endpoint."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository for metrics testing."""
        repo = AsyncMock(spec=AsyncPlaybookRepository)
        mock_playbook = Mock()
        mock_playbook.id = uuid4()
        mock_playbook.steps = [
            Mock(id=uuid4(), name="Step 1", step_type=PlaybookStepType.ACTION),
            Mock(id=uuid4(), name="Step 2", step_type=PlaybookStepType.CONDITIONAL)
        ]

        repo.get_by_id.return_value = mock_playbook
        repo.get_execution_metrics.return_value = {
            "total_executions": 50,
            "success_rate": 0.88,
            "avg_execution_time": 145.6
        }
        repo.get_resource_utilization.return_value = {
            "cpu_usage": {"avg": 42.5, "max": 85.2},
            "memory_usage": {"avg_mb": 456.8, "max_mb": 1024.0}
        }
        repo.get_step_metrics.return_value = {
            "avg_duration": 25.4,
            "success_rate": 0.92,
            "error_rate": 0.08
        }
        repo.analyze_error_patterns.return_value = [
            {
                "error_type": "TimeoutError",
                "frequency": 12,
                "percentage": 45.5
            }
        ]
        repo.analyze_success_patterns.return_value = [
            {
                "pattern_type": "Optimal Timing",
                "success_rate": 0.95
            }
        ]
        repo.get_trend_analysis.return_value = {
            "performance_trend": {"direction": "improving", "change_percentage": 12.5}
        }
        repo.get_comparative_metrics.return_value = {
            "performance_percentile": 75,
            "success_rate_percentile": 82
        }
        return repo

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_detailed_metrics(self, mock_get_repo, mock_repo):
        """Test getting detailed playbook metrics."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())

        response = client.get(f"/v1/playbooks/{playbook_id}/metrics?days=30&include_trends=true")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "playbook_id" in data
        assert "metrics_timestamp" in data
        assert "execution_metrics" in data
        assert "resource_utilization" in data
        assert "step_performance" in data
        assert "error_patterns" in data
        assert "success_patterns" in data
        assert "trend_analysis" in data
        assert "comparative_metrics" in data

        # Verify data content
        assert isinstance(data["step_performance"], list)
        assert isinstance(data["error_patterns"], list)
        assert isinstance(data["success_patterns"], list)

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_detailed_metrics_without_trends(self, mock_get_repo, mock_repo):
        """Test getting metrics without trend analysis."""
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())

        response = client.get(f"/v1/playbooks/{playbook_id}/metrics?include_trends=false")

        assert response.status_code == 200
        data = response.json()
        assert "trend_analysis" in data  # Should be empty dict when trends disabled

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_detailed_metrics_not_found(self, mock_get_repo):
        """Test getting metrics for non-existent playbook."""
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        mock_get_repo.return_value = mock_repo

        playbook_id = str(uuid4())

        response = client.get(f"/v1/playbooks/{playbook_id}/metrics")

        assert response.status_code == 404


class TestPlaybookReportsEndpoint:
    """Test cases for GET /playbooks/reports endpoint."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository for reports testing."""
        repo = AsyncMock(spec=AsyncPlaybookRepository)
        repo.get_summary_statistics.return_value = {
            "total_playbooks": 156,
            "active_playbooks": 124,
            "overall_success_rate": 0.863
        }
        repo.get_category_breakdown.return_value = [
            {
                "category": "automation",
                "playbook_count": 45,
                "success_rate": 0.89
            }
        ]
        repo.get_performance_rankings.return_value = [
            {
                "rank": 1,
                "playbook_name": "Fast Processor",
                "performance_score": 9.8
            }
        ]
        repo.analyze_usage_patterns.return_value = {
            "peak_usage_hours": [9, 10, 14, 15],
            "avg_daily_executions": 94.8
        }
        repo.get_system_trend_analysis.return_value = {
            "overall_performance": {"trend": "improving", "change_percentage": 8.5}
        }
        repo.generate_system_recommendations.return_value = [
            {
                "category": "performance",
                "priority": "high",
                "recommendation": "Implement caching"
            }
        ]
        repo.calculate_health_indicators.return_value = {
            "overall_health_score": 8.7,
            "health_indicators": {
                "system_availability": {"score": 9.8, "status": "excellent"}
            }
        }
        return repo

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_comprehensive_reports(self, mock_get_repo, mock_repo):
        """Test getting comprehensive playbook reports."""
        mock_get_repo.return_value = mock_repo

        response = client.get("/v1/playbooks/reports?report_type=comprehensive&days=30")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "report_timestamp" in data
        assert "report_type" in data
        assert "summary_statistics" in data
        assert "category_breakdown" in data
        assert "performance_rankings" in data
        assert "usage_patterns" in data
        assert "trend_analysis" in data
        assert "recommendations" in data
        assert "health_indicators" in data

        # Verify data content
        assert data["report_type"] == "comprehensive"
        assert isinstance(data["category_breakdown"], list)
        assert isinstance(data["performance_rankings"], list)
        assert isinstance(data["recommendations"], list)

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_performance_reports(self, mock_get_repo, mock_repo):
        """Test getting performance-focused reports."""
        mock_get_repo.return_value = mock_repo

        response = client.get("/v1/playbooks/reports?report_type=performance")

        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "performance"

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_reports_without_trends(self, mock_get_repo, mock_repo):
        """Test getting reports without trend analysis."""
        mock_get_repo.return_value = mock_repo

        response = client.get("/v1/playbooks/reports?include_trends=false")

        assert response.status_code == 200
        data = response.json()
        assert "trend_analysis" in data  # Should be empty dict when trends disabled

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_get_reports_custom_timeframe(self, mock_get_repo, mock_repo):
        """Test getting reports with custom timeframe."""
        mock_get_repo.return_value = mock_repo

        response = client.get("/v1/playbooks/reports?days=90")

        assert response.status_code == 200
        data = response.json()

        # Verify repository was called with correct parameters
        mock_repo.get_summary_statistics.assert_called_with(90)
        mock_repo.get_category_breakdown.assert_called_with(90)


class TestAnalyticalEndpointsIntegration:
    """Integration tests for all analytical endpoints."""

    @patch('agentical.api.v1.endpoints.playbooks.get_playbook_repository')
    def test_analytical_endpoints_workflow(self, mock_get_repo):
        """Test complete analytical workflow."""
        # Setup comprehensive mock repository
        repo = AsyncMock(spec=AsyncPlaybookRepository)

        # Mock playbook data
        mock_playbook = Mock()
        mock_playbook.id = uuid4()
        mock_playbook.name = "Test Workflow Playbook"
        mock_playbook.steps = [Mock() for _ in range(3)]
        mock_playbook.variables = [Mock()]

        # Setup all repository methods
        repo.get_with_executions.return_value = mock_playbook
        repo.get_by_id.return_value = mock_playbook
        repo.calculate_dependency_depth.return_value = 1
        repo.get_recent_executions.return_value = []
        repo.analyze_naming_consistency.return_value = 0.8
        repo.analyze_step_modularity.return_value = 0.7
        repo.get_resource_usage_stats.return_value = {}
        repo.identify_parallelization_opportunities.return_value = []
        repo.get_step_execution_stats.return_value = {}
        repo.get_category_statistics.return_value = {}
        repo.calculate_complexity_score.return_value = 5.0
        repo.generate_structural_variation.return_value = {}
        repo.get_execution_metrics.return_value = {}
        repo.get_resource_utilization.return_value = {}
        repo.get_step_metrics.return_value = {}
        repo.analyze_error_patterns.return_value = []
        repo.analyze_success_patterns.return_value = []
        repo.get_trend_analysis.return_value = {}
        repo.get_comparative_metrics.return_value = {}
        repo.get_summary_statistics.return_value = {}
        repo.get_category_breakdown.return_value = []
        repo.get_performance_rankings.return_value = []
        repo.analyze_usage_patterns.return_value = {}
        repo.get_system_trend_analysis.return_value = {}
        repo.generate_system_recommendations.return_value = []
        repo.calculate_health_indicators.return_value = {}

        mock_get_repo.return_value = repo

        playbook_id = str(uuid4())

        # 1. Analyze the playbook
        analyze_response = client.post(
            f"/v1/playbooks/{playbook_id}/analyze",
            json={"analysis_type": "comprehensive"}
        )
        assert analyze_response.status_code == 200

        # 2. Expand the playbook
        expand_response = client.post(
            f"/v1/playbooks/{playbook_id}/expand",
            json={"expansion_type": "optimizations"}
        )
        assert expand_response.status_code == 200

        # 3. Get detailed metrics
        metrics_response = client.get(f"/v1/playbooks/{playbook_id}/metrics")
        assert metrics_response.status_code == 200

        # 4. Generate reports
        reports_response = client.get("/v1/playbooks/reports")
        assert reports_response.status_code == 200

        # Verify all endpoints returned valid data
        analyze_data = analyze_response.json()
        expand_data = expand_response.json()
        metrics_data = metrics_response.json()
        reports_data = reports_response.json()

        assert analyze_data["playbook_id"] == playbook_id
        assert expand_data["playbook_id"] == playbook_id
        assert metrics_data["playbook_id"] == playbook_id
        assert "report_timestamp" in reports_data

    def test_endpoint_parameter_validation(self):
        """Test parameter validation for analytical endpoints."""
        playbook_id = str(uuid4())

        # Test invalid days parameter
        response = client.get(f"/v1/playbooks/{playbook_id}/metrics?days=0")
        assert response.status_code == 422  # Validation error

        response = client.get(f"/v1/playbooks/{playbook_id}/metrics?days=1000")
        assert response.status_code == 422  # Validation error

        # Test invalid max_variations
        response = client.post(
            f"/v1/playbooks/{playbook_id}/expand",
            json={"max_variations": 0}
        )
        assert response.status_code == 422  # Validation error

        response = client.post(
            f"/v1/playbooks/{playbook_id}/expand",
            json={"max_variations": 15}
        )
        assert response.status_code == 422  # Validation error


# Test fixtures and utilities
@pytest.fixture
def sample_playbook_data():
    """Sample playbook data for testing."""
    return {
        "name": "Test Analytics Playbook",
        "description": "Playbook for testing analytical endpoints",
        "category": "automation",
        "steps": [
            {
                "name": "Initialize",
                "description": "Initialize the process",
                "step_type": "action"
            },
            {
                "name": "Process Data",
                "description": "Process the input data",
                "step_type": "action"
            },
            {
                "name": "Validate Results",
                "description": "Validate processing results",
                "step_type": "conditional"
            }
        ],
        "variables": [
            {
                "name": "input_data",
                "type": "string",
                "required": True
            }
        ]
    }


@pytest.fixture
def mock_execution_history():
    """Mock execution history for testing."""
    return [
        {
            "execution_id": str(uuid4()),
            "status": "completed",
            "duration_seconds": 120,
            "started_at": datetime.utcnow() - timedelta(days=1),
            "completed_at": datetime.utcnow() - timedelta(days=1, hours=-2)
        },
        {
            "execution_id": str(uuid4()),
            "status": "completed",
            "duration_seconds": 150,
            "started_at": datetime.utcnow() - timedelta(days=2),
            "completed_at": datetime.utcnow() - timedelta(days=2, hours=-2.5)
        },
        {
            "execution_id": str(uuid4()),
            "status": "failed",
            "duration_seconds": 90,
            "started_at": datetime.utcnow() - timedelta(days=3),
            "completed_at": datetime.utcnow() - timedelta(days=3, hours=-1.5)
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
