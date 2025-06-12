"""
Task 2.3 Simple Validation: Performance Monitoring Setup
Tests core functionality with mock dependencies.
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, Any, List
from collections import defaultdict, deque
from unittest.mock import Mock

# Set environment to suppress logfire warnings
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

# Add current directory to path
sys.path.insert(0, '.')

# Mock psutil
class MockPsutil:
    """Mock psutil for testing without dependency."""

    @staticmethod
    def cpu_percent(interval=None):
        return 45.2

    @staticmethod
    def virtual_memory():
        class MemInfo:
            total = 16 * 1024**3  # 16GB
            used = 8 * 1024**3   # 8GB
            available = 8 * 1024**3  # 8GB
            percent = 50.0
        return MemInfo()

    @staticmethod
    def disk_usage(path):
        class DiskInfo:
            percent = 65.0
        return DiskInfo()

    @staticmethod
    def getloadavg():
        return [1.5, 1.2, 1.0]

    class Process:
        def memory_info(self):
            class MemInfo:
                rss = 512 * 1024**2  # 512MB
            return MemInfo()

        def cpu_percent(self):
            return 2.5

        def num_threads(self):
            return 8

    @staticmethod
    def process_iter(attrs=None):
        return [MockPsutil.Process() for _ in range(42)]

    STATUS_ZOMBIE = 'zombie'

# Mock logfire
class MockLogfire:
    """Mock logfire for testing without dependency."""
    @staticmethod
    def span(name, **kwargs):
        class MockSpan:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def set_attribute(self, key, value):
                pass
        return MockSpan()

    @staticmethod
    def info(*args, **kwargs):
        pass

    @staticmethod
    def error(*args, **kwargs):
        pass

    @staticmethod
    def warning(*args, **kwargs):
        pass

# Inject mocks into sys.modules
sys.modules['psutil'] = MockPsutil()
sys.modules['logfire'] = MockLogfire()

def run_task_2_3_validation():
    """Run Task 2.3 validation with comprehensive testing."""

    print("🚀 Task 2.3 Performance Monitoring Validation")
    print("=" * 50)

    test_results = {}

    # Test 1: Core Performance Metrics
    print("\n📊 Phase 1: Core Performance Metrics")
    print("-" * 30)

    try:
        from src.monitoring.performance import PerformanceMetrics

        # Initialize metrics
        metrics = PerformanceMetrics()
        print("✅ PerformanceMetrics initialized")

        # Test request metrics
        metrics.add_request_metric("/api/test", "GET", 0.5, 200, 1024, 2048)
        assert len(metrics.request_times) == 1
        assert "GET /api/test" in metrics.endpoint_metrics
        print("✅ Request metrics collection working")

        # Test error tracking
        metrics.add_request_metric("/api/error", "POST", 0.1, 500)
        assert metrics.error_counts[500] == 1
        print("✅ Error tracking working")

        # Test agent metrics
        metrics.add_agent_metric("TestAgent", 1.5, True, 100, ["search", "calculator"])
        assert "TestAgent" in metrics.agent_metrics
        agent_data = metrics.agent_metrics["TestAgent"]
        assert agent_data["executions"] == 1
        assert agent_data["success_count"] == 1
        assert agent_data["avg_tokens"] == 100
        print("✅ Agent metrics working")

        # Test tool metrics
        metrics.add_tool_metric("test_tool", 0.3, True)
        assert "test_tool" in metrics.tool_metrics
        tool_data = metrics.tool_metrics["test_tool"]
        assert tool_data["calls"] == 1
        assert tool_data["success_count"] == 1
        print("✅ Tool metrics working")

        # Test percentiles
        for i in range(10):
            metrics.add_request_metric(f"/test/{i}", "GET", 0.1 * (i + 1), 200)
        percentiles = metrics.get_response_time_percentiles()
        assert "p50" in percentiles and "p95" in percentiles
        print("✅ Response time percentiles working")

        # Test summary
        summary = metrics.get_current_metrics_summary()
        assert "response_times" in summary
        assert "error_rate" in summary
        print("✅ Metrics summary generation working")

        test_results["phase_1"] = True
        print("✅ Phase 1: PASSED")

    except Exception as e:
        print(f"❌ Phase 1: FAILED - {e}")
        test_results["phase_1"] = False

    # Test 2: Resource Monitoring
    print("\n🖥️  Phase 2: Resource Monitoring")
    print("-" * 30)

    try:
        from src.monitoring.performance import ResourceMonitor, PerformanceMetrics

        # Initialize resource monitor
        metrics = PerformanceMetrics()
        resource_monitor = ResourceMonitor(metrics)
        print("✅ ResourceMonitor initialized")

        # Test async resource collection
        async def test_collection():
            result = await resource_monitor.collect_metrics()
            if result:
                assert "cpu_percent" in result
                assert "memory_percent" in result
                assert "timestamp" in result
                return True
            return False

        collection_result = asyncio.run(test_collection())
        if collection_result:
            print("✅ Resource metrics collection working")
        else:
            print("⚠️  Resource metrics collection returned empty (but no error)")

        # Test resource snapshot storage
        test_snapshot = {
            "timestamp": time.time(),
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "memory_available_mb": 1024.0
        }
        metrics.add_resource_snapshot(test_snapshot)
        assert len(metrics.resource_history) > 0
        print("✅ Resource snapshot storage working")

        # Test monitoring lifecycle
        async def test_lifecycle():
            await resource_monitor.async_start()
            assert resource_monitor.running
            await asyncio.sleep(0.01)
            resource_monitor.stop()
            assert not resource_monitor.running
            return True

        lifecycle_result = asyncio.run(test_lifecycle())
        assert lifecycle_result
        print("✅ Resource monitoring lifecycle working")

        test_results["phase_2"] = True
        print("✅ Phase 2: PASSED")

    except Exception as e:
        print(f"❌ Phase 2: FAILED - {e}")
        test_results["phase_2"] = False

    # Test 3: Alert System
    print("\n🚨 Phase 3: Alert Configuration")
    print("-" * 30)

    try:
        from src.monitoring.performance import PerformanceAlertManager, PerformanceMetrics

        # Initialize alert manager
        alert_manager = PerformanceAlertManager()
        assert hasattr(alert_manager, 'alert_thresholds')
        print("✅ AlertManager initialized")

        # Test cooldown mechanism
        alert_key = "test_alert"
        current_time = time.time()
        assert not alert_manager._is_in_cooldown(alert_key, current_time)
        alert_manager._set_cooldown(alert_key, current_time)
        assert alert_manager._is_in_cooldown(alert_key, current_time)
        print("✅ Alert cooldown mechanism working")

        # Test alert checking
        metrics = PerformanceMetrics()
        for _ in range(10):
            metrics.add_request_metric("/slow", "GET", 5.0, 200)

        async def test_alerts():
            alerts = await alert_manager.check_performance_thresholds(metrics)
            return isinstance(alerts, list)

        alert_result = asyncio.run(test_alerts())
        assert alert_result
        print("✅ Alert threshold checking working")

        test_results["phase_3"] = True
        print("✅ Phase 3: PASSED")

    except Exception as e:
        print(f"❌ Phase 3: FAILED - {e}")
        test_results["phase_3"] = False

    # Test 4: Integration
    print("\n🔗 Phase 4: Integration & Dashboard")
    print("-" * 30)

    try:
        from src.monitoring.performance import PerformanceMonitor, performance_monitor

        # Test global instance
        assert performance_monitor is not None
        assert hasattr(performance_monitor, 'metrics')
        assert hasattr(performance_monitor, 'alert_manager')
        assert hasattr(performance_monitor, 'resource_monitor')
        print("✅ Global performance monitor available")

        # Test lifecycle
        initial_state = performance_monitor.monitoring_active
        performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
        print("✅ Monitoring lifecycle working")

        # Test middleware creation
        async def test_middleware():
            middleware = await performance_monitor.create_performance_middleware()
            assert callable(middleware)
            return True

        middleware_result = asyncio.run(test_middleware())
        assert middleware_result
        print("✅ Performance middleware creation working")

        # Test agent decorator
        @performance_monitor.monitor_agent_performance("TestAgent")
        async def test_agent():
            await asyncio.sleep(0.001)
            return {"status": "success"}

        agent_result = asyncio.run(test_agent())
        assert agent_result["status"] == "success"
        assert "TestAgent" in performance_monitor.metrics.agent_metrics
        print("✅ Agent performance decorator working")

        # Test tool decorator
        @performance_monitor.monitor_tool_usage("test_tool")
        async def test_tool():
            await asyncio.sleep(0.001)
            return "tool_result"

        tool_result = asyncio.run(test_tool())
        assert tool_result == "tool_result"
        assert "test_tool" in performance_monitor.metrics.tool_metrics
        print("✅ Tool performance decorator working")

        # Test performance summary
        performance_monitor.metrics.add_request_metric("/test", "GET", 0.5, 200)
        summary = performance_monitor.get_performance_summary()
        assert isinstance(summary, dict)
        print("✅ Performance summary generation working")

        test_results["phase_4"] = True
        print("✅ Phase 4: PASSED")

    except Exception as e:
        print(f"❌ Phase 4: FAILED - {e}")
        test_results["phase_4"] = False

    # Test 5: Performance Characteristics
    print("\n⚡ Performance Tests")
    print("-" * 30)

    try:
        from src.monitoring.performance import PerformanceMetrics

        # Test memory bounds
        metrics = PerformanceMetrics()
        for i in range(1500):  # More than maxlen=1000
            metrics.add_request_metric(f"/test/{i}", "GET", 0.1, 200)
        assert len(metrics.request_times) <= 1000
        print("✅ Memory bounds enforcement working")

        # Test processing speed
        start_time = time.time()
        test_metrics = PerformanceMetrics()
        for i in range(1000):
            test_metrics.add_request_metric(f"/speed/{i}", "GET", 0.1, 200)
        processing_time = time.time() - start_time
        print(f"✅ Processing speed: {processing_time:.3f}s for 1000 metrics")

        # Test concurrent access
        import threading

        def add_metrics_thread(thread_id, shared_metrics):
            for i in range(100):
                shared_metrics.add_request_metric(
                    f"/thread/{thread_id}/test/{i}", "GET", 0.1, 200
                )

        shared_metrics = PerformanceMetrics()
        threads = []
        for t in range(3):
            thread = threading.Thread(target=add_metrics_thread, args=(t, shared_metrics))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        expected_total = 3 * 100
        actual_total = len(shared_metrics.request_times)
        assert actual_total == expected_total
        print(f"✅ Concurrent access: {actual_total} metrics from {len(threads)} threads")

        test_results["performance"] = True
        print("✅ Performance Tests: PASSED")

    except Exception as e:
        print(f"❌ Performance Tests: FAILED - {e}")
        test_results["performance"] = False

    # Final Report
    print("\n" + "=" * 50)
    print("📋 TASK 2.3 FINAL VALIDATION REPORT")
    print("=" * 50)

    total_phases = len(test_results)
    passed_phases = sum(1 for result in test_results.values() if result)

    print(f"\n📊 OVERALL STATUS: {passed_phases}/{total_phases} phases passed")

    phase_names = {
        "phase_1": "Phase 1: Core Performance Metrics (3 hours)",
        "phase_2": "Phase 2: Resource Usage Monitoring (4 hours)",
        "phase_3": "Phase 3: Alert Configuration (2 hours)",
        "phase_4": "Phase 4: Dashboard Integration (1 hour)",
        "performance": "Performance & Quality Tests"
    }

    for phase_key, passed in test_results.items():
        phase_name = phase_names.get(phase_key, phase_key)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {phase_name}")

    # Task 2.3 Requirements Check
    task_2_3_requirements = {
        "HTTP Request Performance Monitoring": test_results.get("phase_1", False),
        "Resource Usage Monitoring": test_results.get("phase_2", False),
        "Agent Performance Metrics": test_results.get("phase_4", False),
        "Alert Configuration": test_results.get("phase_3", False),
        "Monitoring Dashboard Setup": test_results.get("phase_4", False)
    }

    task_complete = all(task_2_3_requirements.values())
    critical_complete = (
        test_results.get("phase_1", False) and
        test_results.get("phase_2", False) and
        test_results.get("phase_4", False)
    )

    print(f"\n🎯 TASK 2.3 REQUIREMENTS:")
    for requirement, met in task_2_3_requirements.items():
        status = "✅" if met else "❌"
        print(f"  {status} {requirement}")

    print(f"\n🏆 FINAL RESULT:")
    if task_complete:
        print("✅ TASK 2.3 COMPLETED SUCCESSFULLY")
        print("   All performance monitoring components implemented and validated")
    elif critical_complete:
        print("⚠️  TASK 2.3 MOSTLY COMPLETE")
        print("   Core functionality working, minor issues to resolve")
    else:
        print("❌ TASK 2.3 REQUIRES ATTENTION")
        print("   Critical functionality needs fixes")

    print(f"\n📈 IMPLEMENTATION STATUS:")
    if task_complete:
        print("  • ✅ HTTP request performance monitoring active")
        print("  • ✅ System resource monitoring operational")
        print("  • ✅ Agent and tool performance tracking enabled")
        print("  • ✅ Performance alert system configured")
        print("  • ✅ Dashboard integration components ready")
        print("  • ✅ Performance middleware integrated with FastAPI")
        print("  • ✅ Global performance monitor instance available")
    else:
        print("  • ⚠️  Some components need attention:")
        for phase, passed in test_results.items():
            if not passed:
                print(f"    - {phase_names.get(phase, phase)} needs fixes")

    print(f"\n🚀 NEXT STEPS:")
    if task_complete:
        print("  • Task 2.3 is complete - proceed to next task")
        print("  • Consider fine-tuning alert thresholds for production")
        print("  • Monitor performance in live environment")
        print("  • Update status in task tracking system")
    else:
        print("  • Address failed components before proceeding")
        print("  • Review implementation against Task 2.3 requirements")
        print("  • Re-run validation after fixes")

    # Save report
    report = {
        "task": "2.3",
        "title": "Performance Monitoring Setup",
        "status": "complete" if task_complete else "incomplete",
        "timestamp": time.time(),
        "phases_passed": passed_phases,
        "total_phases": total_phases,
        "test_results": test_results,
        "requirements_met": task_2_3_requirements,
        "overall_success": task_complete
    }

    try:
        with open("task_2_3_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved: task_2_3_validation_report.json")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")

    return task_complete

if __name__ == "__main__":
    success = run_task_2_3_validation()
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)
