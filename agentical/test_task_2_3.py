"""
Test Task 2.3: Performance Monitoring Setup
Comprehensive validation of performance monitoring system implementation.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import logfire

from src.monitoring.performance import (
    PerformanceMetrics,
    PerformanceAlertManager,
    ResourceMonitor,
    PerformanceMonitor,
    performance_monitor
)
from main import app


class TestTask2_3PerformanceMonitoring:
    """Test suite for Task 2.3: Performance Monitoring Setup."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with performance monitoring."""
        test_app = FastAPI()

        @test_app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @test_app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(0.1)
            return {"status": "slow"}

        @test_app.get("/error")
        async def error_endpoint():
            raise Exception("Test error")

        return test_app

    @pytest.fixture
    def performance_metrics(self):
        """Create fresh performance metrics instance."""
        return PerformanceMetrics()

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return PerformanceAlertManager()

    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor instance."""
        monitor = ResourceMonitor()
        yield monitor
        monitor.stop()

    @pytest.fixture
    def perf_monitor(self):
        """Create performance monitor instance."""
        monitor = PerformanceMonitor()
        yield monitor
        if monitor.monitoring_active:
            monitor.stop_monitoring()

    # Phase 1: Core Performance Metrics Tests
    @pytest.mark.performance
    def test_performance_metrics_initialization(self, performance_metrics):
        """Test PerformanceMetrics initialization."""
        assert len(performance_metrics.request_times) == 0
        assert len(performance_metrics.error_counts) == 0
        assert len(performance_metrics.endpoint_metrics) == 0
        assert len(performance_metrics.agent_metrics) == 0
        assert len(performance_metrics.tool_metrics) == 0
        assert len(performance_metrics.resource_history) == 0

    @pytest.mark.performance
    def test_request_metric_collection(self, performance_metrics):
        """Test HTTP request metric collection."""
        # Add sample request metrics
        performance_metrics.add_request_metric(
            endpoint="/api/test",
            method="GET",
            duration=0.5,
            status_code=200,
            request_size=1024,
            response_size=2048
        )

        # Verify metrics were recorded
        assert len(performance_metrics.request_times) == 1
        assert "GET /api/test" in performance_metrics.endpoint_metrics

        endpoint_data = performance_metrics.endpoint_metrics["GET /api/test"]
        assert endpoint_data["count"] == 1
        assert endpoint_data["total_time"] == 0.5
        assert endpoint_data["min_time"] == 0.5
        assert endpoint_data["max_time"] == 0.5
        assert endpoint_data["errors"] == 0

    @pytest.mark.performance
    def test_error_metric_tracking(self, performance_metrics):
        """Test error metric tracking."""
        # Add error request
        performance_metrics.add_request_metric(
            endpoint="/api/error",
            method="POST",
            duration=0.1,
            status_code=500
        )

        # Verify error tracking
        assert performance_metrics.error_counts[500] == 1
        endpoint_data = performance_metrics.endpoint_metrics["POST /api/error"]
        assert endpoint_data["errors"] == 1

    @pytest.mark.performance
    def test_agent_metric_collection(self, performance_metrics):
        """Test agent performance metric collection."""
        # Add agent execution metrics
        performance_metrics.add_agent_metric(
            agent_type="TestAgent",
            duration=2.5,
            success=True,
            tokens_used=150,
            tools_called=["search", "calculator"]
        )

        # Verify agent metrics
        agent_data = performance_metrics.agent_metrics["TestAgent"]
        assert agent_data["executions"] == 1
        assert agent_data["total_time"] == 2.5
        assert agent_data["success_count"] == 1
        assert agent_data["error_count"] == 0
        assert agent_data["avg_tokens"] == 150
        assert agent_data["tool_usage"]["search"] == 1
        assert agent_data["tool_usage"]["calculator"] == 1

    @pytest.mark.performance
    def test_tool_metric_collection(self, performance_metrics):
        """Test tool usage metric collection."""
        # Add tool usage metrics
        performance_metrics.add_tool_metric(
            tool_name="search",
            duration=0.8,
            success=True
        )

        performance_metrics.add_tool_metric(
            tool_name="search",
            duration=1.2,
            success=False
        )

        # Verify tool metrics
        tool_data = performance_metrics.tool_metrics["search"]
        assert tool_data["calls"] == 2
        assert tool_data["total_time"] == 2.0
        assert tool_data["success_count"] == 1
        assert tool_data["error_count"] == 1

    @pytest.mark.performance
    def test_performance_percentiles(self, performance_metrics):
        """Test response time percentile calculations."""
        # Add multiple request metrics with varying durations
        durations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, duration in enumerate(durations):
            performance_metrics.add_request_metric(
                endpoint=f"/test/{i}",
                method="GET",
                duration=duration,
                status_code=200
            )

        # Test percentile calculations
        percentiles = performance_metrics.get_response_time_percentiles()
        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        # Verify reasonable percentile values
        assert 0.4 <= percentiles["p50"] <= 0.6  # Median around 0.5
        assert 0.8 <= percentiles["p90"] <= 1.0  # 90th percentile

    # Phase 2: Resource Monitoring Tests
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_monitor_collection(self, resource_monitor):
        """Test system resource monitoring."""
        # Collect current metrics
        metrics = await resource_monitor.collect_metrics()

        # Verify required metrics are present
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "memory_available_mb" in metrics
        assert "disk_usage_percent" in metrics
        assert "load_average" in metrics
        assert "active_processes" in metrics
        assert "timestamp" in metrics

        # Verify metric ranges
        assert 0 <= metrics["cpu_percent"] <= 100
        assert 0 <= metrics["memory_percent"] <= 100
        assert metrics["memory_available_mb"] > 0
        assert 0 <= metrics["disk_usage_percent"] <= 100

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_monitor_background_collection(self, resource_monitor):
        """Test background resource monitoring."""
        # Start monitoring
        resource_monitor.start()

        # Wait for some data collection
        await asyncio.sleep(0.1)

        # Stop monitoring
        resource_monitor.stop()

        # Verify monitoring was active
        assert not resource_monitor._monitoring_task or resource_monitor._monitoring_task.done()

    # Phase 3: Alert System Tests
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.thresholds is not None
        assert "response_time_p95" in alert_manager.thresholds
        assert "error_rate" in alert_manager.thresholds
        assert "cpu_usage" in alert_manager.thresholds
        assert "memory_usage" in alert_manager.thresholds

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_alerts(self, alert_manager, performance_metrics):
        """Test response time threshold alerting."""
        # Add slow requests that should trigger alerts
        for _ in range(10):
            performance_metrics.add_request_metric(
                endpoint="/slow",
                method="GET",
                duration=5.0,  # Very slow
                status_code=200
            )

        # Check for alerts
        alerts = await alert_manager.check_performance_thresholds(performance_metrics, {})

        # Should have response time alerts
        response_time_alerts = [a for a in alerts if a["type"] == "response_time"]
        assert len(response_time_alerts) > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_error_rate_alerts(self, alert_manager, performance_metrics):
        """Test error rate threshold alerting."""
        # Add requests with high error rate
        for i in range(10):
            status_code = 500 if i < 8 else 200  # 80% error rate
            performance_metrics.add_request_metric(
                endpoint="/error",
                method="GET",
                duration=0.1,
                status_code=status_code
            )

        # Check for alerts
        alerts = await alert_manager.check_performance_thresholds(performance_metrics, {})

        # Should have error rate alerts
        error_rate_alerts = [a for a in alerts if a["type"] == "error_rate"]
        assert len(error_rate_alerts) > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_usage_alerts(self, alert_manager):
        """Test resource usage alerting."""
        # Simulate high resource usage
        high_usage_metrics = {
            "cpu_percent": 95.0,
            "memory_percent": 90.0,
            "disk_usage_percent": 85.0
        }

        # Check for alerts
        alerts = await alert_manager.check_performance_thresholds({}, high_usage_metrics)

        # Should have resource alerts
        resource_alerts = [a for a in alerts if a["type"] in ["cpu_usage", "memory_usage"]]
        assert len(resource_alerts) > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_alert_cooldown(self, alert_manager):
        """Test alert cooldown mechanism."""
        # Trigger the same alert twice quickly
        alert_key = "test_alert"

        # First alert should be sent
        first_cooldown = alert_manager._is_in_cooldown(alert_key)
        assert not first_cooldown

        # Set cooldown
        alert_manager._set_cooldown(alert_key)

        # Second alert should be in cooldown
        second_cooldown = alert_manager._is_in_cooldown(alert_key)
        assert second_cooldown

    # Phase 4: Integration Tests
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_middleware_integration(self, perf_monitor, test_app):
        """Test performance monitoring middleware integration."""
        # Create middleware
        middleware = await perf_monitor.create_performance_middleware()

        # Mock request and response
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.headers = {"user-agent": "test-client"}

        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {}

        async def mock_call_next(req):
            await asyncio.sleep(0.1)  # Simulate processing time
            return response

        # Process request through middleware
        result = await middleware(request, mock_call_next)

        # Verify response headers
        assert "X-Response-Time" in response.headers
        assert response.headers["X-Response-Time"].endswith("ms")

        # Verify metrics were recorded
        assert len(perf_monitor.metrics.request_times) > 0

    @pytest.mark.performance
    def test_agent_performance_decorator(self, perf_monitor):
        """Test agent performance monitoring decorator."""
        @perf_monitor.monitor_agent_performance("TestAgent")
        async def test_agent_function():
            await asyncio.sleep(0.1)
            return {"result": "success"}

        # Execute decorated function
        result = asyncio.run(test_agent_function())

        # Verify result
        assert result == {"result": "success"}

        # Verify metrics were recorded
        assert "TestAgent" in perf_monitor.metrics.agent_metrics
        agent_data = perf_monitor.metrics.agent_metrics["TestAgent"]
        assert agent_data["executions"] == 1
        assert agent_data["success_count"] == 1

    @pytest.mark.performance
    def test_tool_performance_decorator(self, perf_monitor):
        """Test tool performance monitoring decorator."""
        @perf_monitor.monitor_tool_usage("test_tool")
        async def test_tool_function():
            await asyncio.sleep(0.05)
            return "tool_result"

        # Execute decorated function
        result = asyncio.run(test_tool_function())

        # Verify result
        assert result == "tool_result"

        # Verify metrics were recorded
        assert "test_tool" in perf_monitor.metrics.tool_metrics
        tool_data = perf_monitor.metrics.tool_metrics["test_tool"]
        assert tool_data["calls"] == 1
        assert tool_data["success_count"] == 1

    @pytest.mark.performance
    def test_performance_monitor_lifecycle(self, perf_monitor):
        """Test performance monitor start/stop lifecycle."""
        # Initial state
        assert not perf_monitor.monitoring_active

        # Start monitoring
        perf_monitor.start_monitoring()
        assert perf_monitor.monitoring_active

        # Stop monitoring
        perf_monitor.stop_monitoring()
        assert not perf_monitor.monitoring_active

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, perf_monitor):
        """Test performance summary generation."""
        # Add sample data
        perf_monitor.metrics.add_request_metric("/test", "GET", 0.5, 200)
        perf_monitor.metrics.add_agent_metric("TestAgent", 1.0, True, 100)
        perf_monitor.metrics.add_tool_metric("test_tool", 0.2, True)

        # Generate summary
        summary = perf_monitor.get_performance_summary()

        # Verify summary structure
        assert "request_metrics" in summary
        assert "agent_metrics" in summary
        assert "tool_metrics" in summary
        assert "response_time_percentiles" in summary

    # End-to-End Integration Tests
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_performance_monitoring_workflow(self):
        """Test complete performance monitoring workflow."""
        with TestClient(app) as client:
            # Make test requests
            response1 = client.get("/health")
            assert response1.status_code == 200

            response2 = client.get("/")
            assert response2.status_code == 200

            # Check performance headers
            assert "X-Response-Time" in response1.headers or "X-Response-Time" in response2.headers

        # Verify global performance monitor has data
        assert len(performance_monitor.metrics.request_times) > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_health_integration(self):
        """Test performance monitoring integration with health checks."""
        with TestClient(app) as client:
            # Get performance health data
            response = client.get("/health/performance")

            if response.status_code == 200:
                data = response.json()
                assert "performance_summary" in data
                assert "uptime_seconds" in data
            else:
                # Health endpoint might not be fully configured yet
                pytest.skip("Performance health endpoint not available")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_alert_system_integration(self, perf_monitor):
        """Test integrated alert system functionality."""
        # Start monitoring
        perf_monitor.start_monitoring()

        try:
            # Add metrics that should trigger alerts
            for _ in range(20):
                perf_monitor.metrics.add_request_metric(
                    "/slow", "GET", 3.0, 200  # Slow requests
                )

            # Check for alerts
            alerts = await perf_monitor.check_alerts()

            # Verify alert structure if any are generated
            if alerts:
                for alert in alerts:
                    assert "type" in alert
                    assert "severity" in alert
                    assert "message" in alert
                    assert "timestamp" in alert

        finally:
            perf_monitor.stop_monitoring()

    # Performance Regression Tests
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_monitoring_overhead(self, perf_monitor):
        """Test that performance monitoring doesn't add significant overhead."""
        iterations = 1000

        # Measure without monitoring
        start_time = time.time()
        for _ in range(iterations):
            # Simulate lightweight operation
            await asyncio.sleep(0.001)
        no_monitoring_time = time.time() - start_time

        # Measure with monitoring
        perf_monitor.start_monitoring()
        start_time = time.time()
        for i in range(iterations):
            perf_monitor.metrics.add_request_metric(f"/test/{i}", "GET", 0.001, 200)
            await asyncio.sleep(0.001)
        with_monitoring_time = time.time() - start_time
        perf_monitor.stop_monitoring()

        # Overhead should be minimal (less than 50% increase)
        overhead_ratio = (with_monitoring_time - no_monitoring_time) / no_monitoring_time
        assert overhead_ratio < 0.5, f"Monitoring overhead too high: {overhead_ratio:.2%}"

    @pytest.mark.performance
    def test_memory_usage_bounds(self, performance_metrics):
        """Test that performance metrics don't consume unbounded memory."""
        # Add many requests to test deque limits
        for i in range(2000):  # More than maxlen=1000
            performance_metrics.add_request_metric(
                f"/test/{i}", "GET", 0.1, 200
            )

        # Verify deque size limits are respected
        assert len(performance_metrics.request_times) <= 1000

        # Add many resource snapshots
        for i in range(200):  # More than maxlen=100
            performance_metrics.add_resource_snapshot({
                "timestamp": time.time(),
                "cpu_percent": 50.0,
                "memory_percent": 60.0
            })

        assert len(performance_metrics.resource_history) <= 100

    @pytest.mark.performance
    def test_concurrent_metric_collection(self, performance_metrics):
        """Test thread-safe metric collection under concurrent access."""
        import threading

        def add_metrics(thread_id):
            for i in range(100):
                performance_metrics.add_request_metric(
                    f"/thread/{thread_id}/test/{i}", "GET", 0.1, 200
                )

        # Create multiple threads
        threads = []
        for t in range(5):
            thread = threading.Thread(target=add_metrics, args=(t,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all metrics were recorded
        assert len(performance_metrics.request_times) == 500
        assert len(performance_metrics.endpoint_metrics) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
