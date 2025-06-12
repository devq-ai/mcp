"""
Performance Monitoring Tests

Comprehensive test suite for the performance monitoring system,
including metrics collection, alerting, resource monitoring, and health checks.
"""

import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import psutil

from src.monitoring.performance import (
    PerformanceMetrics,
    PerformanceAlertManager,
    ResourceMonitor,
    PerformanceMonitor,
    performance_monitor
)
from src.monitoring.health import health_router


class TestPerformanceMetrics:
    """Test PerformanceMetrics class functionality."""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert len(metrics.request_times) == 0
        assert len(metrics.error_counts) == 0
        assert len(metrics.endpoint_metrics) == 0
        assert len(metrics.agent_metrics) == 0
        assert len(metrics.tool_metrics) == 0
        assert len(metrics.resource_history) == 0
    
    def test_add_request_metric(self):
        """Test adding HTTP request metrics."""
        metrics = PerformanceMetrics()
        
        # Add successful request
        metrics.add_request_metric(
            endpoint="/api/test",
            method="GET",
            duration=0.150,
            status_code=200,
            request_size=100,
            response_size=500
        )
        
        assert len(metrics.request_times) == 1
        assert "GET /api/test" in metrics.endpoint_metrics
        
        endpoint_data = metrics.endpoint_metrics["GET /api/test"]
        assert endpoint_data["count"] == 1
        assert endpoint_data["total_time"] == 0.150
        assert endpoint_data["errors"] == 0
        assert endpoint_data["min_time"] == 0.150
        assert endpoint_data["max_time"] == 0.150
    
    def test_add_request_metric_with_error(self):
        """Test adding HTTP request metrics with error."""
        metrics = PerformanceMetrics()
        
        # Add error request
        metrics.add_request_metric(
            endpoint="/api/test",
            method="POST",
            duration=0.300,
            status_code=500
        )
        
        assert len(metrics.request_times) == 1
        assert metrics.error_counts[500] == 1
        
        endpoint_data = metrics.endpoint_metrics["POST /api/test"]
        assert endpoint_data["errors"] == 1
    
    def test_add_agent_metric(self):
        """Test adding agent execution metrics."""
        metrics = PerformanceMetrics()
        
        # Add successful agent execution
        metrics.add_agent_metric(
            agent_type="research_agent",
            duration=2.5,
            success=True,
            tokens_used=1500,
            tools_called=["web_search", "document_analysis"]
        )
        
        assert "research_agent" in metrics.agent_metrics
        agent_data = metrics.agent_metrics["research_agent"]
        
        assert agent_data["executions"] == 1
        assert agent_data["total_time"] == 2.5
        assert agent_data["success_count"] == 1
        assert agent_data["error_count"] == 0
        assert agent_data["avg_tokens"] == 1500
        assert agent_data["tool_usage"]["web_search"] == 1
        assert agent_data["tool_usage"]["document_analysis"] == 1
    
    def test_add_tool_metric(self):
        """Test adding tool usage metrics."""
        metrics = PerformanceMetrics()
        
        # Add successful tool usage
        metrics.add_tool_metric(
            tool_name="web_search",
            duration=1.2,
            success=True
        )
        
        assert "web_search" in metrics.tool_metrics
        tool_data = metrics.tool_metrics["web_search"]
        
        assert tool_data["calls"] == 1
        assert tool_data["total_time"] == 1.2
        assert tool_data["success_count"] == 1
        assert tool_data["error_count"] == 0
    
    def test_add_resource_snapshot(self):
        """Test adding resource usage snapshots."""
        metrics = PerformanceMetrics()
        
        metrics.add_resource_snapshot(
            cpu_percent=45.2,
            memory_percent=68.4,
            memory_mb=1024.5,
            disk_percent=75.0
        )
        
        assert len(metrics.resource_history) == 1
        snapshot = metrics.resource_history[0]
        
        assert snapshot["cpu_percent"] == 45.2
        assert snapshot["memory_percent"] == 68.4
        assert snapshot["memory_mb"] == 1024.5
        assert snapshot["disk_percent"] == 75.0
        assert "timestamp" in snapshot
    
    def test_get_response_time_percentiles(self):
        """Test response time percentile calculations."""
        metrics = PerformanceMetrics()
        
        # Add multiple requests with varying response times
        for duration in [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]:
            metrics.add_request_metric("/test", "GET", duration, 200)
        
        percentiles = metrics.get_response_time_percentiles()
        
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        
        # Values should be in milliseconds
        assert percentiles["p50"] > 0
        assert percentiles["p95"] >= percentiles["p50"]
        assert percentiles["p99"] >= percentiles["p95"]
    
    def test_get_current_metrics_summary(self):
        """Test comprehensive metrics summary."""
        metrics = PerformanceMetrics()
        
        # Add some test data
        metrics.add_request_metric("/api/test", "GET", 0.2, 200)
        metrics.add_agent_metric("test_agent", 1.5, True, tokens_used=1000)
        metrics.add_tool_metric("test_tool", 0.8, True)
        metrics.add_resource_snapshot(50.0, 70.0, 2048.0)
        
        summary = metrics.get_current_metrics_summary()
        
        assert "response_times" in summary
        assert "error_rate" in summary
        assert "total_requests" in summary
        assert "resource_usage" in summary
        assert "top_endpoints" in summary
        assert "agent_performance" in summary
        assert "tool_performance" in summary


class TestPerformanceAlertManager:
    """Test PerformanceAlertManager functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        alert_manager = PerformanceAlertManager()
        
        assert "response_time_p95_ms" in alert_manager.alert_thresholds
        assert "error_rate_5min" in alert_manager.alert_thresholds
        assert "memory_usage_percent" in alert_manager.alert_thresholds
        assert len(alert_manager.alert_history) == 0
        assert len(alert_manager.alert_cooldowns) == 0
    
    @pytest.mark.asyncio
    async def test_response_time_alerts(self):
        """Test response time threshold alerts."""
        alert_manager = PerformanceAlertManager()
        metrics = PerformanceMetrics()
        
        # Add requests that exceed P95 threshold
        for _ in range(10):
            metrics.add_request_metric("/slow", "GET", 1.5, 200)  # 1500ms > 1000ms threshold
        
        alerts = await alert_manager.check_performance_thresholds(metrics)
        
        # Should generate P95 response time alert
        response_time_alerts = [a for a in alerts if a.get("subtype") == "response_time"]
        assert len(response_time_alerts) > 0
        
        alert = response_time_alerts[0]
        assert alert["type"] == "performance"
        assert alert["severity"] in ["warning", "critical"]
        assert alert["value"] > alert["threshold"]
    
    @pytest.mark.asyncio
    async def test_error_rate_alerts(self):
        """Test error rate threshold alerts."""
        alert_manager = PerformanceAlertManager()
        metrics = PerformanceMetrics()
        
        # Add requests with high error rate (>5%)
        for _ in range(8):
            metrics.add_request_metric("/test", "GET", 0.2, 200)  # Successful
        for _ in range(2):
            metrics.add_request_metric("/test", "GET", 0.3, 500)  # Errors (20% error rate)
        
        alerts = await alert_manager.check_performance_thresholds(metrics)
        
        # Should generate error rate alert
        error_alerts = [a for a in alerts if a.get("subtype") == "error_rate"]
        assert len(error_alerts) > 0
        
        alert = error_alerts[0]
        assert alert["type"] == "reliability"
        assert alert["value"] > alert["threshold"]
    
    @pytest.mark.asyncio
    async def test_agent_performance_alerts(self):
        """Test agent performance alerts."""
        alert_manager = PerformanceAlertManager()
        metrics = PerformanceMetrics()
        
        # Add agent with high error rate
        for _ in range(7):
            metrics.add_agent_metric("slow_agent", 2.0, True)
        for _ in range(3):
            metrics.add_agent_metric("slow_agent", 2.0, False)  # 30% error rate
        
        alerts = await alert_manager.check_performance_thresholds(metrics)
        
        # Should generate agent error rate alert
        agent_alerts = [a for a in alerts if a.get("type") == "agent"]
        assert len(agent_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        alert_manager = PerformanceAlertManager()
        alert_manager.cooldown_duration = 1  # 1 second for testing
        current_time = time.time()
        
        # Set cooldown
        alert_manager._set_cooldown("test_alert", current_time)
        
        # Should be in cooldown immediately
        assert alert_manager._is_in_cooldown("test_alert", current_time)
        
        # Should not be in cooldown after duration
        assert not alert_manager._is_in_cooldown("test_alert", current_time + 2)


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        metrics = PerformanceMetrics()
        monitor = ResourceMonitor(metrics, collection_interval=1.0)
        
        assert monitor.metrics == metrics
        assert monitor.collection_interval == 1.0
        assert not monitor.running
        assert monitor.task is None
    
    @pytest.mark.asyncio
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.Process')
    async def test_collect_metrics(self, mock_process, mock_cpu, mock_memory):
        """Test resource metrics collection."""
        # Mock system resource data
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        mock_memory.return_value.used = 8 * 1024**3   # 8GB
        mock_memory.return_value.available = 8 * 1024**3  # 8GB
        mock_memory.return_value.percent = 50.0
        
        mock_cpu.return_value = 45.5
        
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 512 * 1024**2  # 512MB
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process_instance.num_threads.return_value = 8
        mock_process.return_value = mock_process_instance
        
        metrics = PerformanceMetrics()
        monitor = ResourceMonitor(metrics, collection_interval=1.0)
        
        await monitor.collect_metrics()
        
        # Verify metrics were collected
        assert len(metrics.resource_history) == 1
        snapshot = metrics.resource_history[0]
        
        assert snapshot["cpu_percent"] == 45.5
        assert snapshot["memory_percent"] == 50.0
        assert snapshot["memory_mb"] == 512.0


class TestPerformanceMonitor:
    """Test main PerformanceMonitor class."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.metrics is not None
        assert monitor.alert_manager is not None
        assert monitor.resource_monitor is not None
        assert not monitor.monitoring_active
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_create_performance_middleware(self):
        """Test performance middleware creation."""
        monitor = PerformanceMonitor()
        middleware = await monitor.create_performance_middleware()
        
        assert callable(middleware)
    
    def test_monitor_agent_performance_decorator(self):
        """Test agent performance monitoring decorator."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_agent_performance("test_agent")
        async def test_agent_function():
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        # Verify decorator was applied
        assert hasattr(test_agent_function, '__wrapped__')
    
    def test_monitor_tool_usage_decorator(self):
        """Test tool usage monitoring decorator."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_tool_usage("test_tool")
        async def test_tool_function():
            await asyncio.sleep(0.05)
            return {"data": "processed"}
        
        # Verify decorator was applied
        assert hasattr(test_tool_function, '__wrapped__')


class TestPerformanceIntegration:
    """Integration tests for performance monitoring system."""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application with performance monitoring."""
        app = FastAPI()
        
        # Add performance monitoring middleware
        @app.middleware("http")
        async def perf_middleware(request: Request, call_next):
            return await performance_monitor.create_performance_middleware()(request, call_next)
        
        # Add test endpoint
        @app.get("/test")
        async def test_endpoint():
            await asyncio.sleep(0.1)
            return {"message": "test"}
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(1.0)
            return {"message": "slow"}
        
        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test error")
        
        # Include health monitoring endpoints
        app.include_router(health_router)
        
        return app
    
    def test_performance_middleware_integration(self, test_app):
        """Test performance middleware integration with FastAPI."""
        client = TestClient(test_app)
        
        # Make request to test endpoint
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        
        # Verify metrics were collected
        assert len(performance_monitor.metrics.request_times) > 0
        assert "GET /test" in performance_monitor.metrics.endpoint_metrics
    
    def test_health_endpoints_integration(self, test_app):
        """Test health monitoring endpoints integration."""
        client = TestClient(test_app)
        
        # Test basic health check
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
    
    def test_performance_health_check_integration(self, test_app):
        """Test performance health check integration."""
        client = TestClient(test_app)
        
        # Make some requests to generate metrics
        client.get("/test")
        client.get("/test")
        
        # Test performance health check
        response = client.get("/health/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "performance_metrics" in data
        assert "health_indicators" in data
        assert "recommendations" in data
    
    def test_error_handling_in_middleware(self, test_app):
        """Test error handling in performance middleware."""
        client = TestClient(test_app)
        
        # Make request that will cause an error
        response = client.get("/error")
        assert response.status_code == 500
        
        # Verify error was recorded in metrics
        error_requests = [
            req for req in performance_monitor.metrics.request_times
            if req.get("status_code", 200) >= 400
        ]
        assert len(error_requests) > 0
    
    @pytest.mark.asyncio
    async def test_agent_monitoring_integration(self):
        """Test agent performance monitoring integration."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_agent_performance("integration_test_agent")
        async def test_agent():
            await asyncio.sleep(0.1)
            # Simulate agent result with metrics
            result = Mock()
            result.token_usage = 1500
            result.tools_called = ["test_tool_1", "test_tool_2"]
            return result
        
        # Execute monitored agent
        result = await test_agent()
        
        # Verify metrics were collected
        assert "integration_test_agent" in monitor.metrics.agent_metrics
        agent_data = monitor.metrics.agent_metrics["integration_test_agent"]
        assert agent_data["executions"] == 1
        assert agent_data["success_count"] == 1
    
    @pytest.mark.asyncio
    async def test_tool_monitoring_integration(self):
        """Test tool performance monitoring integration."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_tool_usage("integration_test_tool")
        async def test_tool():
            await asyncio.sleep(0.05)
            return {"status": "success"}
        
        # Execute monitored tool
        result = await test_tool()
        
        # Verify metrics were collected
        assert "integration_test_tool" in monitor.metrics.tool_metrics
        tool_data = monitor.metrics.tool_metrics["integration_test_tool"]
        assert tool_data["calls"] == 1
        assert tool_data["success_count"] == 1


class TestPerformanceOptimization:
    """Test performance monitoring system performance."""
    
    @pytest.mark.asyncio
    async def test_monitoring_overhead(self):
        """Test that monitoring overhead is minimal."""
        monitor = PerformanceMonitor()
        
        # Time without monitoring
        start_time = time.time()
        for _ in range(100):
            await asyncio.sleep(0.001)
        baseline_time = time.time() - start_time
        
        # Time with monitoring
        @monitor.monitor_agent_performance("overhead_test")
        async def monitored_operation():
            await asyncio.sleep(0.001)
            return {"result": "test"}
        
        start_time = time.time()
        for _ in range(100):
            await monitored_operation()
        monitored_time = time.time() - start_time
        
        # Overhead should be less than 20%
        overhead_ratio = (monitored_time - baseline_time) / baseline_time
        assert overhead_ratio < 0.20, f"Monitoring overhead too high: {overhead_ratio:.2%}"
    
    def test_metrics_memory_usage(self):
        """Test that metrics collection doesn't cause memory leaks."""
        metrics = PerformanceMetrics()
        
        # Add many metrics to test memory management
        for i in range(2000):  # More than maxlen
            metrics.add_request_metric(f"/test_{i % 10}", "GET", 0.1, 200)
            metrics.add_resource_snapshot(50.0, 60.0, 1024.0)
        
        # Verify collections are bounded
        assert len(metrics.request_times) <= 1000
        assert len(metrics.resource_history) <= 100
    
    @pytest.mark.asyncio
    async def test_concurrent_metrics_collection(self):
        """Test thread safety of metrics collection."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_agent_performance("concurrent_test")
        async def concurrent_agent():
            await asyncio.sleep(0.01)
            return {"result": "success"}
        
        # Run multiple concurrent operations
        tasks = [concurrent_agent() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 50
        
        # Verify metrics were collected correctly
        agent_data = monitor.metrics.agent_metrics["concurrent_test"]
        assert agent_data["executions"] == 50
        assert agent_data["success_count"] == 50


@pytest.mark.asyncio
async def test_global_performance_monitor():
    """Test global performance monitor instance."""
    from src.monitoring.performance import performance_monitor
    
    # Verify global instance exists
    assert performance_monitor is not None
    assert isinstance(performance_monitor, PerformanceMonitor)
    
    # Test global instance functionality
    summary = performance_monitor.get_performance_summary()
    assert isinstance(summary, dict)
    
    alerts = await performance_monitor.check_alerts()
    assert isinstance(alerts, list)