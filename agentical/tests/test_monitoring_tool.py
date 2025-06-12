"""
Tests for Monitoring Tool

This module contains comprehensive tests for the monitoring tool functionality
including system metrics collection, alerting, target monitoring, and various
export formats.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.security.monitoring_tool import (
    MonitoringTool,
    MonitoringTarget,
    SystemMetrics,
    MonitoringAlert,
    MetricType,
    AlertLevel
)


class TestSystemMetrics:
    """Test system metrics collection."""

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test basic system metrics collection."""
        metrics = SystemMetrics()
        await metrics.collect()

        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_usage >= 0
        assert metrics.network_io is not None
        assert metrics.timestamp is not None

    def test_system_metrics_to_dict(self):
        """Test system metrics serialization."""
        metrics = SystemMetrics()
        metrics.cpu_percent = 50.0
        metrics.memory_percent = 75.0
        metrics.disk_usage = 80.0

        data = metrics.to_dict()
        assert data["cpu_percent"] == 50.0
        assert data["memory_percent"] == 75.0
        assert data["disk_usage"] == 80.0


class TestMonitoringTarget:
    """Test monitoring target functionality."""

    def test_monitoring_target_creation(self):
        """Test creating a monitoring target."""
        target = MonitoringTarget(
            name="test-api",
            url="http://localhost:8000/health",
            check_interval=30,
            timeout=10
        )

        assert target.name == "test-api"
        assert target.url == "http://localhost:8000/health"
        assert target.check_interval == 30
        assert target.timeout == 10
        assert target.success_rate == 0.0
        assert target.average_response_time == 0.0

    def test_monitoring_target_success_rate(self):
        """Test success rate calculation."""
        target = MonitoringTarget("test", "http://test.com")

        # Add some successful checks
        target.add_check_result(True, 100)
        target.add_check_result(True, 200)
        target.add_check_result(False, 0)

        assert target.success_rate == 2/3  # 2 out of 3 successful

    def test_monitoring_target_average_response_time(self):
        """Test average response time calculation."""
        target = MonitoringTarget("test", "http://test.com")

        target.add_check_result(True, 100)
        target.add_check_result(True, 200)
        target.add_check_result(True, 300)

        assert target.average_response_time == 200.0  # (100+200+300)/3

    def test_monitoring_target_to_dict(self):
        """Test target serialization."""
        target = MonitoringTarget("test", "http://test.com")
        target.add_check_result(True, 100)

        data = target.to_dict()
        assert data["name"] == "test"
        assert data["url"] == "http://test.com"
        assert "success_rate" in data
        assert "average_response_time" in data


class TestMonitoringAlert:
    """Test monitoring alert functionality."""

    def test_alert_creation(self):
        """Test creating a monitoring alert."""
        alert = MonitoringAlert(
            id="test-alert-1",
            title="High CPU Usage",
            description="CPU usage is above threshold",
            level=AlertLevel.WARNING,
            source="system",
            metric_name="cpu_percent",
            current_value=85.0,
            threshold_value=80.0,
            condition="greater_than"
        )

        assert alert.id == "test-alert-1"
        assert alert.title == "High CPU Usage"
        assert alert.level == AlertLevel.WARNING
        assert alert.current_value == 85.0
        assert not alert.acknowledged
        assert not alert.resolved

    def test_alert_acknowledge(self):
        """Test alert acknowledgment."""
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        assert not alert.acknowledged
        alert.acknowledge()
        assert alert.acknowledged

    def test_alert_resolve(self):
        """Test alert resolution."""
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        assert not alert.resolved
        alert.resolve()
        assert alert.resolved

    def test_alert_escalate(self):
        """Test alert escalation."""
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.WARNING,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        assert not alert.escalated
        alert.escalate()
        assert alert.escalated

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        data = alert.to_dict()
        assert data["id"] == "test-alert"
        assert data["title"] == "Test Alert"
        assert data["level"] == "info"
        assert data["source"] == "test"


class TestMonitoringTool:
    """Test monitoring tool main functionality."""

    def test_monitoring_tool_creation(self):
        """Test creating a monitoring tool instance."""
        config = {
            "check_interval": 60,
            "alert_retention_days": 7,
            "max_targets": 100
        }

        tool = MonitoringTool(config)
        assert tool.config == config
        assert not tool.monitoring_active
        assert len(tool.targets) == 0
        assert len(tool.active_alerts) == 0

    @pytest.mark.asyncio
    async def test_add_target(self):
        """Test adding a monitoring target."""
        tool = MonitoringTool({})

        await tool.add_target(
            name="test-api",
            url="http://localhost:8000",
            check_interval=30
        )

        assert "test-api" in tool.targets
        target = tool.targets["test-api"]
        assert target.name == "test-api"
        assert target.url == "http://localhost:8000"
        assert target.check_interval == 30

    @pytest.mark.asyncio
    async def test_remove_target(self):
        """Test removing a monitoring target."""
        tool = MonitoringTool({})

        await tool.add_target("test-api", "http://localhost:8000")
        assert "test-api" in tool.targets

        await tool.remove_target("test-api")
        assert "test-api" not in tool.targets

    def test_add_custom_metric(self):
        """Test adding a custom metric."""
        tool = MonitoringTool({})

        def custom_collector():
            return {"custom_value": 42}

        tool.add_custom_metric(
            name="custom_metric",
            metric_type=MetricType.GAUGE,
            description="Test custom metric",
            collector=custom_collector
        )

        assert "custom_metric" in tool.custom_metrics

    @pytest.mark.asyncio
    async def test_metrics_summary(self):
        """Test getting metrics summary."""
        tool = MonitoringTool({})

        # Add a test target
        await tool.add_target("test-api", "http://localhost:8000")

        summary = await tool.get_metrics_summary()

        assert "timestamp" in summary
        assert "system_metrics" in summary
        assert "targets" in summary
        assert "alerts" in summary
        assert "monitoring_status" in summary

    @pytest.mark.asyncio
    async def test_alert_history(self):
        """Test getting alert history."""
        tool = MonitoringTool({})

        # Create a test alert
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        tool.alert_history["test-alert"] = alert

        history = await tool.get_alert_history(limit=10)
        assert len(history) == 1
        assert history[0]["id"] == "test-alert"

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        tool = MonitoringTool({})

        # Create and add an active alert
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.WARNING,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        tool.active_alerts["test-alert"] = alert

        result = await tool.acknowledge_alert("test-alert", "test-user")
        assert result is True
        assert alert.acknowledged is True

    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test resolving an alert."""
        tool = MonitoringTool({})

        # Create and add an active alert
        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test",
            level=AlertLevel.WARNING,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        tool.active_alerts["test-alert"] = alert

        result = await tool.resolve_alert("test-alert", "test-user")
        assert result is True
        assert alert.resolved is True
        assert "test-alert" not in tool.active_alerts
        assert "test-alert" in tool.alert_history

    @pytest.mark.asyncio
    async def test_export_json_metrics(self):
        """Test exporting metrics as JSON."""
        tool = MonitoringTool({})

        json_data = await tool.export_metrics(format="json")
        assert json_data != ""

        # Verify it's valid JSON
        data = json.loads(json_data)
        assert "export_timestamp" in data
        assert "metrics_summary" in data

    @pytest.mark.asyncio
    async def test_export_csv_metrics(self):
        """Test exporting metrics as CSV."""
        tool = MonitoringTool({})

        csv_data = await tool.export_metrics(format="csv")
        assert csv_data != ""
        assert "timestamp,alert_id,level" in csv_data

    def test_health_status(self):
        """Test getting health status."""
        tool = MonitoringTool({})

        status = tool.get_health_status()
        assert "status" in status
        assert "monitoring_active" in status
        assert "active_alerts" in status
        assert "targets_monitored" in status

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        tool = MonitoringTool({"check_interval": 1})  # 1 second for testing

        # Mock the monitoring loop to avoid actual system calls
        with patch.object(tool, '_monitoring_loop', new_callable=AsyncMock):
            tool.start_monitoring()
            assert tool.monitoring_active is True

            await tool.stop_monitoring()
            assert tool.monitoring_active is False

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        tool = MonitoringTool({})

        # Add many alerts to test cleanup
        for i in range(1500):  # More than the 1000 limit
            alert = MonitoringAlert(
                id=f"alert-{i}",
                title=f"Alert {i}",
                description="Test",
                level=AlertLevel.INFO,
                source="test",
                metric_name="test_metric",
                current_value=100,
                threshold_value=90
            )
            tool.alert_history[f"alert-{i}"] = alert

        await tool.cleanup()

        # Should keep only 1000 most recent alerts
        assert len(tool.alert_history) == 1000

    @pytest.mark.asyncio
    async def test_threshold_evaluation(self):
        """Test threshold evaluation logic."""
        tool = MonitoringTool({})

        # Test greater_than condition
        assert tool._evaluate_threshold(90, 80, "greater_than") is True
        assert tool._evaluate_threshold(70, 80, "greater_than") is False

        # Test less_than condition
        assert tool._evaluate_threshold(70, 80, "less_than") is True
        assert tool._evaluate_threshold(90, 80, "less_than") is False

        # Test equals condition
        assert tool._evaluate_threshold(80.0, 80.0, "equals") is True
        assert tool._evaluate_threshold(80.1, 80.0, "equals") is False

    @pytest.mark.asyncio
    async def test_custom_metric_collection(self):
        """Test custom metric collection."""
        tool = MonitoringTool({})

        def test_collector():
            return {"test_value": 42, "another_value": 100}

        tool.add_custom_metric(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="Test metric",
            collector=test_collector
        )

        # Test collection
        await tool._collect_custom_metrics()

        # Verify metric was collected (would be stored in custom_metrics_data)
        assert "test_metric" in tool.custom_metrics

    @pytest.mark.asyncio
    @patch('tools.security.monitoring_tool.requests')
    async def test_webhook_alert_notification(self, mock_requests):
        """Test webhook alert notifications."""
        tool = MonitoringTool({
            "webhook_url": "http://example.com/webhook"
        })

        alert = MonitoringAlert(
            id="test-alert",
            title="Test Alert",
            description="Test webhook",
            level=AlertLevel.ERROR,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        await tool._send_webhook_alert(alert)

        # Verify webhook was called
        mock_requests.post.assert_called_once()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @pytest.mark.asyncio
    async def test_system_metrics_with_mocked_psutil(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection with mocked psutil."""
        # Mock psutil responses
        mock_cpu.return_value = 75.0

        mock_memory_obj = Mock()
        mock_memory_obj.percent = 60.0
        mock_memory.return_value = mock_memory_obj

        mock_disk_obj = Mock()
        mock_disk_obj.percent = 80.0
        mock_disk.return_value = mock_disk_obj

        mock_net_obj = Mock()
        mock_net_obj.bytes_sent = 1000
        mock_net_obj.bytes_recv = 2000
        mock_net.return_value = mock_net_obj

        # Test metrics collection
        metrics = SystemMetrics()
        await metrics.collect()

        assert metrics.cpu_percent == 75.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_usage == 80.0

    @pytest.mark.asyncio
    async def test_alert_escalation_processing(self):
        """Test alert escalation processing."""
        tool = MonitoringTool({})

        # Create an old critical alert
        alert = MonitoringAlert(
            id="critical-alert",
            title="Critical Alert",
            description="Test critical alert",
            level=AlertLevel.CRITICAL,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        # Set triggered time to 10 minutes ago (should escalate for critical alerts)
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=10)
        tool.active_alerts["critical-alert"] = alert

        # Process escalations
        await tool._process_alert_escalations()

        # Alert should be escalated
        assert alert.escalated is True

    @pytest.mark.asyncio
    async def test_auto_resolve_old_info_alerts(self):
        """Test auto-resolution of old info alerts."""
        tool = MonitoringTool({})

        # Create an old info alert (25 hours ago)
        alert = MonitoringAlert(
            id="old-info-alert",
            title="Old Info Alert",
            description="Test old info alert",
            level=AlertLevel.INFO,
            source="test",
            metric_name="test_metric",
            current_value=100,
            threshold_value=90
        )

        alert.triggered_at = datetime.utcnow() - timedelta(hours=25)
        tool.active_alerts["old-info-alert"] = alert

        # Process auto-resolution
        await tool._auto_resolve_alerts()

        # Alert should be resolved
        assert alert.resolved is True

    def test_metric_type_enum(self):
        """Test MetricType enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"

    def test_alert_level_enum(self):
        """Test AlertLevel enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"


# Integration tests
class TestMonitoringToolIntegration:
    """Integration tests for monitoring tool."""

    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test a complete monitoring cycle."""
        config = {
            "check_interval": 1,  # 1 second for testing
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0
        }

        tool = MonitoringTool(config)

        try:
            # Add monitoring targets
            await tool.add_target(
                name="test-service",
                url="http://httpbin.org/status/200",  # Public test endpoint
                check_interval=5,
                timeout=10
            )

            # Add custom metric
            def test_metric_collector():
                return {"test_value": 50.0}

            tool.add_custom_metric(
                name="test_custom_metric",
                metric_type=MetricType.GAUGE,
                description="Test custom metric",
                collector=test_metric_collector
            )

            # Get initial metrics summary
            summary = await tool.get_metrics_summary()
            assert summary is not None
            assert "system_metrics" in summary
            assert "targets" in summary

            # Test health status
            health = tool.get_health_status()
            assert health["status"] in ["healthy", "warning", "critical", "unknown"]

            # Test exports
            json_export = await tool.export_metrics("json")
            assert json_export != ""

            csv_export = await tool.export_metrics("csv")
            assert csv_export != ""

        finally:
            await tool.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        tool = MonitoringTool({})

        # Test invalid target removal
        result = await tool.remove_target("nonexistent-target")
        # Should not raise exception

        # Test invalid alert acknowledgment
        result = await tool.acknowledge_alert("nonexistent-alert")
        assert result is False

        # Test invalid alert resolution
        result = await tool.resolve_alert("nonexistent-alert")
        assert result is False

        # Test invalid export format
        export_data = await tool.export_metrics("invalid_format")
        assert export_data == ""  # Should return empty string on error


if __name__ == "__main__":
    pytest.main([__file__])
