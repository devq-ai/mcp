"""
Test Suite for Task 1.4: Health Check & Monitoring Endpoints

This module contains comprehensive tests for the health monitoring system
implementation, verifying all health endpoints, dependency checking,
metrics collection, and Kubernetes probe compatibility.

Test Coverage:
- Basic health check endpoints
- Kubernetes health probes (liveness, readiness, startup)
- Detailed health checks with dependency status
- Application metrics collection
- Performance monitoring
- Error handling in health checks
- Security considerations
- Integration with existing middleware
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
from typing import Dict, Any

# Mock external dependencies
class MockLogfire:
    @staticmethod
    def configure(**kwargs): pass
    @staticmethod
    def instrument_fastapi(*args, **kwargs): pass
    @staticmethod
    def error(message, **kwargs): pass
    @staticmethod
    def info(message, **kwargs): pass
    @staticmethod
    def span(name, **kwargs):
        class MockSpan:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return MockSpan()

class MockPsutil:
    @staticmethod
    def cpu_percent(interval=1):
        return 45.2
    
    @staticmethod
    def virtual_memory():
        class Memory:
            percent = 67.5
            available = 8 * 1024**3  # 8GB
        return Memory()
    
    @staticmethod
    def disk_usage(path):
        class Disk:
            percent = 55.0
            free = 100 * 1024**3  # 100GB
        return Disk()

import sys
sys.modules['logfire'] = MockLogfire()
sys.modules['psutil'] = MockPsutil()

# Now import the modules under test
from agentical.api.health import (
    router as health_router,
    HealthResponse,
    DetailedHealthResponse,
    ReadinessResponse,
    MetricsResponse,
    DependencyCheck,
    MetricsStore,
    metrics,
    dependency_checker
)


class TestHealthModels:
    """Test health response models"""
    
    def test_health_response_model(self):
        """Test HealthResponse model validation"""
        response = HealthResponse(
            status="healthy",
            uptime=3600.5
        )
        
        assert response.status == "healthy"
        assert response.uptime == 3600.5
        assert response.version == "1.0.0"
        assert isinstance(response.timestamp, datetime)
        
    def test_detailed_health_response_model(self):
        """Test DetailedHealthResponse model validation"""
        checks = {
            "database": {"status": "healthy", "url": "test"},
            "services": {"status": "degraded", "count": 5}
        }
        summary = {"healthy": 1, "degraded": 1}
        
        response = DetailedHealthResponse(
            status="degraded",
            uptime=1800.0,
            checks=checks,
            summary=summary
        )
        
        assert response.status == "degraded"
        assert response.uptime == 1800.0
        assert response.checks == checks
        assert response.summary == summary
        
    def test_readiness_response_model(self):
        """Test ReadinessResponse model validation"""
        checks = {"database": "healthy", "mcp_servers": "configured"}
        
        response = ReadinessResponse(
            ready=True,
            checks=checks,
            message="Application ready for traffic"
        )
        
        assert response.ready is True
        assert response.checks == checks
        assert "ready" in response.message
        
    def test_metrics_response_model(self):
        """Test MetricsResponse model validation"""
        response = MetricsResponse(
            uptime=7200.0,
            requests={"GET /api": 100, "POST /api": 50},
            errors={"GET /api": 5, "POST /api": 2},
            response_times={"GET /api": 150.5, "POST /api": 200.2},
            system={"cpu_percent": 45.0, "memory_percent": 60.0},
            health_history=[{"timestamp": datetime.utcnow(), "status": "healthy"}]
        )
        
        assert response.uptime == 7200.0
        assert response.requests["GET /api"] == 100
        assert response.system["cpu_percent"] == 45.0
        assert len(response.health_history) == 1


class TestMetricsStore:
    """Test metrics storage functionality"""
    
    def test_metrics_store_initialization(self):
        """Test MetricsStore initialization"""
        store = MetricsStore()
        
        assert len(store.requests_total) == 0
        assert len(store.errors_total) == 0
        assert len(store.response_times) == 0
        assert len(store.health_check_results) == 0
        assert isinstance(store.start_time, datetime)
        
    def test_record_request(self):
        """Test request recording"""
        store = MetricsStore()
        
        store.record_request("/api/test", 200, 150.5)
        store.record_request("/api/test", 404, 50.2)
        store.record_request("/api/other", 500, 300.0)
        
        assert store.requests_total["/api/test"] == 2
        assert store.requests_total["/api/other"] == 1
        assert store.errors_total["/api/test"] == 1  # 404 is an error
        assert store.errors_total["/api/other"] == 1  # 500 is an error
        
        test_times = list(store.response_times["/api/test"])
        assert 150.5 in test_times
        assert 50.2 in test_times
        
    def test_record_health_check(self):
        """Test health check recording"""
        store = MetricsStore()
        
        checks = {"database": "healthy", "services": "degraded"}
        store.record_health_check("degraded", checks)
        
        assert len(store.health_check_results) == 1
        result = store.health_check_results[0]
        assert result["status"] == "degraded"
        assert result["checks"] == checks
        assert isinstance(result["timestamp"], datetime)
        
    def test_get_uptime(self):
        """Test uptime calculation"""
        store = MetricsStore()
        # Small delay to ensure uptime > 0
        time.sleep(0.01)
        uptime = store.get_uptime()
        
        assert uptime > 0
        assert uptime < 1  # Should be very small for test
        
    def test_get_avg_response_time(self):
        """Test average response time calculation"""
        store = MetricsStore()
        
        # Add some response times
        store.record_request("/api/test", 200, 100.0)
        store.record_request("/api/test", 200, 200.0)
        store.record_request("/api/test", 200, 300.0)
        
        avg_time = store.get_avg_response_time("/api/test")
        assert avg_time == 200.0  # (100 + 200 + 300) / 3
        
        # Test overall average
        store.record_request("/api/other", 200, 150.0)
        overall_avg = store.get_avg_response_time()
        assert overall_avg == 187.5  # (100 + 200 + 300 + 150) / 4


class TestDependencyCheck:
    """Test dependency checking functionality"""
    
    def test_dependency_check_initialization(self):
        """Test DependencyCheck initialization"""
        checker = DependencyCheck()
        
        assert checker.timeout == 5.0
        assert checker.cache_ttl == 30
        assert len(checker.cache) == 0
        
    @pytest.mark.asyncio
    async def test_check_with_cache(self):
        """Test caching functionality"""
        checker = DependencyCheck()
        
        # Mock check function
        async def mock_check():
            return {"status": "healthy", "test": True}
        
        # First call should execute the function
        result1 = await checker.check_with_cache("test_service", mock_check)
        assert result1["status"] == "healthy"
        assert "test_service" in checker.cache
        
        # Second call should use cache
        async def mock_check_different():
            return {"status": "different"}
        
        result2 = await checker.check_with_cache("test_service", mock_check_different)
        assert result2["status"] == "healthy"  # Should be cached result
        
    @pytest.mark.asyncio
    async def test_check_database(self):
        """Test database connectivity check"""
        checker = DependencyCheck()
        
        with patch.dict('os.environ', {'SURREALDB_URL': 'ws://localhost:8000/rpc'}):
            result = await checker.check_database()
            
            assert result["status"] == "configured"
            assert result["url"] == "ws://localhost:8000/rpc"
            assert result["type"] == "surrealdb"
            
    @pytest.mark.asyncio
    async def test_check_database_no_config(self):
        """Test database check without configuration"""
        checker = DependencyCheck()
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('os.getenv', return_value=None):
                result = await checker.check_database()
                
                assert result["status"] == "error"
                assert "No database URL configured" in result["error"]
                
    @pytest.mark.asyncio
    async def test_check_mcp_servers(self):
        """Test MCP server configuration check"""
        checker = DependencyCheck()
        
        # Mock MCP configuration
        mock_config = {
            "mcp_servers": {
                "ptolemies-mcp": {"command": "test"},
                "filesystem": {"command": "test"},
                "git": {"command": "test"},
                "memory": {"command": "test"}
            }
        }
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open_config(mock_config)):
                result = await checker.check_mcp_servers()
                
                assert result["status"] == "healthy"
                assert result["total_servers"] == 4
                assert result["key_servers_available"] >= 3
                
    @pytest.mark.asyncio
    async def test_check_mcp_servers_no_config(self):
        """Test MCP servers check without configuration"""
        checker = DependencyCheck()
        
        with patch('os.path.exists', return_value=False):
            result = await checker.check_mcp_servers()
            
            assert result["status"] == "error"
            assert "MCP configuration not found" in result["error"]
            
    @pytest.mark.asyncio
    async def test_check_ptolemies_kb(self):
        """Test Ptolemies knowledge base check"""
        checker = DependencyCheck()
        
        with patch.dict('os.environ', {'PTOLEMIES_URL': 'http://localhost:8001'}):
            # Test API connection success
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                result = await checker.check_ptolemies_kb()
                
                assert result["status"] == "healthy"
                assert result["connection"] == "api"
                assert result["documents"] == 597
                
    @pytest.mark.asyncio
    async def test_check_external_services(self):
        """Test external services check"""
        checker = DependencyCheck()
        
        with patch.dict('os.environ', {
            'LOGFIRE_TOKEN': 'test_token',
            'ANTHROPIC_API_KEY': 'test_key'
        }):
            result = await checker.check_external_services()
            
            assert result["status"] == "configured"
            assert "logfire" in result["services"]
            assert "anthropic" in result["services"]
            assert result["count"] == 2
            
    @pytest.mark.asyncio
    async def test_check_external_services_minimal(self):
        """Test external services check with no services configured"""
        checker = DependencyCheck()
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('os.getenv', return_value=None):
                result = await checker.check_external_services()
                
                assert result["status"] == "minimal"
                assert result["services"] == []
                assert result["count"] == 0
                
    @pytest.mark.asyncio
    async def test_check_system_resources(self):
        """Test system resource monitoring"""
        checker = DependencyCheck()
        
        result = await checker.check_system_resources()
        
        assert result["status"] in ["healthy", "warning", "critical"]
        assert "cpu_percent" in result
        assert "memory_percent" in result
        assert "disk_percent" in result
        assert "memory_available_gb" in result
        assert "disk_free_gb" in result
        
        # Check status logic
        if max(result["cpu_percent"], result["memory_percent"], result["disk_percent"]) > 90:
            assert result["status"] == "critical"
        elif max(result["cpu_percent"], result["memory_percent"], result["disk_percent"]) > 80:
            assert result["status"] == "warning"
        else:
            assert result["status"] == "healthy"


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def setup_method(self):
        """Set up test client"""
        from fastapi import FastAPI
        self.app = FastAPI()
        self.app.include_router(health_router)
        self.client = TestClient(self.app)
        
    def test_basic_health_endpoint(self):
        """Test basic health check endpoint"""
        response = self.client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
        assert data["version"] == "1.0.0"
        
    def test_liveness_probe_endpoint(self):
        """Test Kubernetes liveness probe"""
        response = self.client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert data["version"] == "1.0.0"
        
    def test_startup_probe_endpoint(self):
        """Test Kubernetes startup probe"""
        response = self.client.get("/health/startup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "startup_complete" in data
        assert "uptime" in data
        assert "status" in data
        
    def test_readiness_probe_endpoint(self):
        """Test Kubernetes readiness probe"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock successful dependency checks
            mock_check.side_effect = [
                {"status": "healthy"},  # database
                {"status": "healthy"}   # mcp_servers
            ]
            
            response = self.client.get("/health/ready")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "ready" in data
            assert "checks" in data
            assert "message" in data
            
    def test_readiness_probe_not_ready(self):
        """Test readiness probe when dependencies are not ready"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock failed dependency checks
            mock_check.side_effect = [
                {"status": "error", "error": "Database connection failed"},  # database
                {"status": "degraded"}   # mcp_servers
            ]
            
            response = self.client.get("/health/ready")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["ready"] is False
            assert "not ready" in data["message"].lower()
            
    def test_detailed_health_endpoint(self):
        """Test detailed health check endpoint"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock all dependency checks
            mock_checks = [
                {"status": "healthy"},      # database
                {"status": "configured"},   # mcp_servers
                {"status": "accessible"},   # ptolemies
                {"status": "configured"},   # external_services
                {"status": "healthy"}       # system
            ]
            mock_check.side_effect = mock_checks
            
            response = self.client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "checks" in data
            assert "summary" in data
            assert "uptime" in data
            
            # Verify all expected checks are present
            expected_checks = ["database", "mcp_servers", "ptolemies_kb", 
                             "external_services", "system_resources", "application"]
            for check in expected_checks:
                assert check in data["checks"]
                
    def test_detailed_health_with_errors(self):
        """Test detailed health check with some services failing"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock mixed dependency check results
            mock_checks = [
                {"status": "error", "error": "Connection failed"},  # database
                {"status": "healthy"},    # mcp_servers
                {"status": "degraded"},   # ptolemies
                {"status": "configured"}, # external_services
                {"status": "warning"}     # system
            ]
            mock_check.side_effect = mock_checks
            
            response = self.client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"  # Should be unhealthy due to error
            assert data["summary"]["error"] >= 1
            
    def test_metrics_endpoint(self):
        """Test application metrics endpoint"""
        # Pre-populate some metrics
        metrics.record_request("/api/test", 200, 150.0)
        metrics.record_request("/api/test", 404, 50.0)
        metrics.record_health_check("healthy", {"database": "healthy"})
        
        response = self.client.get("/health/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "uptime" in data
        assert "requests" in data
        assert "errors" in data
        assert "response_times" in data
        assert "system" in data
        assert "health_history" in data
        
        # Verify metrics data
        assert data["requests"]["/api/test"] == 2
        assert data["errors"]["/api/test"] == 1
        
    def test_health_status_summary(self):
        """Test health status summary endpoint"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock successful checks
            mock_check.side_effect = [
                {"status": "healthy"},     # database
                {"status": "configured"}   # mcp_servers
            ]
            
            response = self.client.get("/health/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "UP"
            assert "timestamp" in data
            assert "uptime" in data
            assert data["version"] == "1.0.0"
            
    def test_health_status_summary_down(self):
        """Test health status summary when services are down"""
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock failed checks
            mock_check.side_effect = [
                {"status": "error", "error": "Database down"},  # database
                {"status": "error", "error": "MCP servers down"}  # mcp_servers
            ]
            
            response = self.client.get("/health/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "DOWN"
            
    def test_record_custom_metric(self):
        """Test custom metric recording endpoint"""
        response = self.client.post("/health/metrics/record", params={
            "endpoint": "/api/custom",
            "status_code": 201,
            "response_time": 89.5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"


class TestHealthEndpointIntegration:
    """Integration tests for health endpoints"""
    
    def test_health_endpoint_performance(self):
        """Test that health endpoints respond quickly"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(health_router)
        client = TestClient(app)
        
        # Test basic health endpoint performance
        start_time = time.time()
        response = client.get("/health/")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond in under 100ms
        
        # Test liveness probe performance
        start_time = time.time()
        response = client.get("/health/live")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond in under 100ms
        
    def test_health_endpoints_concurrent_access(self):
        """Test health endpoints under concurrent access"""
        from fastapi import FastAPI
        import threading
        
        app = FastAPI()
        app.include_router(health_router)
        client = TestClient(app)
        
        results = []
        
        def make_request():
            response = client.get("/health/")
            results.append(response.status_code)
        
        # Start multiple concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)
        
    def test_health_endpoint_error_handling(self):
        """Test health endpoint error handling"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(health_router)
        client = TestClient(app)
        
        # Test with invalid paths
        response = client.get("/health/nonexistent")
        assert response.status_code == 404
        
        # Test metrics endpoint with mocked system error
        with patch('agentical.api.health.dependency_checker.check_system_resources') as mock_check:
            mock_check.side_effect = Exception("System check failed")
            
            response = client.get("/health/metrics")
            assert response.status_code == 500


class TestHealthEndpointSecurity:
    """Test security aspects of health endpoints"""
    
    def test_health_endpoints_no_sensitive_data(self):
        """Test that health endpoints don't expose sensitive data"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(health_router)
        client = TestClient(app)
        
        # Test detailed health endpoint
        with patch.object(dependency_checker, 'check_with_cache') as mock_check:
            # Mock check that might contain sensitive data
            mock_checks = [
                {"status": "error", "error": "Connection failed to secret-host:5432"},
                {"status": "healthy"},
                {"status": "configured"},
                {"status": "configured"},
                {"status": "healthy"}
            ]
            mock_check.side_effect = mock_checks
            
            response = client.get("/health/detailed")
            assert response.status_code == 200
            
            # Check that response doesn't contain obvious sensitive patterns
            response_text = response.text.lower()
            sensitive_patterns = ["password", "secret", "key", "token"]
            
            # Allow the word "secret-host" as it's part of our test, but not actual secrets
            for pattern in sensitive_patterns:
                if pattern == "secret" and "secret-host" in response_text:
                    continue  # This is our test hostname
                assert pattern not in response_text, f"Sensitive pattern '{pattern}' found in response"
                
    def test_metrics_endpoint_safe_data(self):
        """Test that metrics endpoint only exposes safe operational data"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(health_router)
        client = TestClient(app)
        
        response = client.get("/health/metrics")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that only expected metric types are present
        expected_keys = ["timestamp", "uptime", "requests", "errors", 
                        "response_times", "system", "health_history"]
        for key in data.keys():
            assert key in expected_keys, f"Unexpected metric key: {key}"


# Helper functions
def mock_open_config(config_data):
    """Helper to mock file opening with JSON config"""
    import json
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(config_data))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])