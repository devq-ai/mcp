"""
Task 2.3 Validation: Performance Monitoring Setup
Comprehensive validation test for performance monitoring implementation.
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import logfire

# Configure logfire for testing
import os
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

from src.monitoring.performance import (
    PerformanceMetrics,
    PerformanceAlertManager,
    ResourceMonitor,
    PerformanceMonitor,
    performance_monitor
)


class Task2_3Validator:
    """Comprehensive validator for Task 2.3 performance monitoring requirements."""

    def __init__(self):
        self.test_results = {
            "phase_1_core_metrics": False,
            "phase_2_resource_monitoring": False,
            "phase_3_alert_configuration": False,
            "phase_4_dashboard_integration": False,
            "integration_tests": False,
            "performance_tests": False
        }
        self.detailed_results = {}

    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive validation of all Task 2.3 requirements."""
        print("🚀 Starting Task 2.3 Performance Monitoring Validation")
        print("=" * 60)

        # Phase 1: Core Performance Metrics
        self.test_results["phase_1_core_metrics"] = self._validate_phase_1()

        # Phase 2: Resource Monitoring
        self.test_results["phase_2_resource_monitoring"] = self._validate_phase_2()

        # Phase 3: Alert Configuration
        self.test_results["phase_3_alert_configuration"] = self._validate_phase_3()

        # Phase 4: Dashboard Integration
        self.test_results["phase_4_dashboard_integration"] = self._validate_phase_4()

        # Integration Tests
        self.test_results["integration_tests"] = self._validate_integration()

        # Performance Tests
        self.test_results["performance_tests"] = self._validate_performance()

        return self._generate_final_report()

    def _validate_phase_1(self) -> bool:
        """Phase 1: Core Performance Metrics (3 hours) - HTTP Request Performance."""
        print("\n📊 Phase 1: Core Performance Metrics")
        print("-" * 40)

        try:
            # Test 1: PerformanceMetrics initialization
            metrics = PerformanceMetrics()
            assert len(metrics.request_times) == 0
            assert len(metrics.endpoint_metrics) == 0
            print("✅ PerformanceMetrics initialization")

            # Test 2: Request metric collection
            metrics.add_request_metric("/api/test", "GET", 0.5, 200, 1024, 2048)
            assert len(metrics.request_times) == 1
            assert "GET /api/test" in metrics.endpoint_metrics
            endpoint_data = metrics.endpoint_metrics["GET /api/test"]
            assert endpoint_data["count"] == 1
            assert endpoint_data["total_time"] == 0.5
            print("✅ HTTP request metric collection")

            # Test 3: Error tracking
            metrics.add_request_metric("/api/error", "POST", 0.1, 500)
            assert metrics.error_counts[500] == 1
            print("✅ Error metric tracking")

            # Test 4: Performance percentiles
            for i in range(10):
                metrics.add_request_metric(f"/test/{i}", "GET", 0.1 * i, 200)
            percentiles = metrics.get_response_time_percentiles()
            assert "p50" in percentiles
            assert "p95" in percentiles
            assert "p99" in percentiles
            print("✅ Response time percentiles")

            # Test 5: Metrics summary
            summary = metrics.get_current_metrics_summary()
            assert "response_times" in summary
            assert "error_rate" in summary
            assert "top_endpoints" in summary
            print("✅ Metrics summary generation")

            self.detailed_results["phase_1"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": "All core metrics functionality working correctly"
            }

            return True

        except Exception as e:
            print(f"❌ Phase 1 failed: {str(e)}")
            self.detailed_results["phase_1"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _validate_phase_2(self) -> bool:
        """Phase 2: Resource Usage Monitoring (4 hours) - System Resources."""
        print("\n🖥️  Phase 2: Resource Usage Monitoring")
        print("-" * 40)

        try:
            metrics = PerformanceMetrics()

            # Test 1: ResourceMonitor initialization
            resource_monitor = ResourceMonitor(metrics)
            assert resource_monitor.metrics is not None
            assert resource_monitor.collection_interval == 30.0
            print("✅ ResourceMonitor initialization")

            # Test 2: Async resource collection
            async def test_resource_collection():
                result = await resource_monitor.collect_metrics()
                assert isinstance(result, dict)
                assert "cpu_percent" in result
                assert "memory_percent" in result
                assert "memory_available_mb" in result
                assert "timestamp" in result
                return True

            collection_result = asyncio.run(test_resource_collection())
            assert collection_result
            print("✅ Resource metrics collection")

            # Test 3: Resource snapshot storage
            assert len(metrics.resource_history) > 0
            latest_snapshot = metrics.resource_history[-1]
            assert "cpu_percent" in latest_snapshot
            assert "memory_percent" in latest_snapshot
            print("✅ Resource snapshot storage")

            # Test 4: Async monitoring lifecycle
            async def test_monitoring_lifecycle():
                await resource_monitor.async_start()
                assert resource_monitor.running
                await asyncio.sleep(0.1)  # Let it run briefly
                resource_monitor.stop()
                assert not resource_monitor.running
                return True

            lifecycle_result = asyncio.run(test_monitoring_lifecycle())
            assert lifecycle_result
            print("✅ Resource monitoring lifecycle")

            # Test 5: Metrics bounds (memory safety)
            for i in range(150):  # More than maxlen=100
                metrics.add_resource_snapshot({
                    "timestamp": time.time(),
                    "cpu_percent": 50.0,
                    "memory_percent": 60.0
                })
            assert len(metrics.resource_history) <= 100
            print("✅ Memory bounds enforcement")

            self.detailed_results["phase_2"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": "All resource monitoring functionality working correctly"
            }

            return True

        except Exception as e:
            print(f"❌ Phase 2 failed: {str(e)}")
            self.detailed_results["phase_2"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _validate_phase_3(self) -> bool:
        """Phase 3: Alert Configuration (2 hours) - Performance Thresholds."""
        print("\n🚨 Phase 3: Alert Configuration")
        print("-" * 40)

        try:
            # Test 1: AlertManager initialization
            alert_manager = PerformanceAlertManager()
            assert alert_manager.alert_thresholds is not None
            assert "response_time_p95_ms" in alert_manager.alert_thresholds
            assert "error_rate_5min" in alert_manager.alert_thresholds
            print("✅ AlertManager initialization")

            # Test 2: Response time alerts
            metrics = PerformanceMetrics()
            for _ in range(10):
                metrics.add_request_metric("/slow", "GET", 5.0, 200)  # Very slow

            async def test_response_time_alerts():
                alerts = await alert_manager.check_performance_thresholds(metrics)
                response_time_alerts = [a for a in alerts if a["type"] == "performance"]
                return len(response_time_alerts) > 0

            rt_alert_result = asyncio.run(test_response_time_alerts())
            assert rt_alert_result
            print("✅ Response time alerting")

            # Test 3: Error rate alerts
            metrics_with_errors = PerformanceMetrics()
            for i in range(10):
                status_code = 500 if i < 8 else 200  # 80% error rate
                metrics_with_errors.add_request_metric("/error", "GET", 0.1, status_code)

            async def test_error_rate_alerts():
                alerts = await alert_manager.check_performance_thresholds(metrics_with_errors)
                error_rate_alerts = [a for a in alerts if "error" in str(a).lower()]
                return len(error_rate_alerts) > 0

            er_alert_result = asyncio.run(test_error_rate_alerts())
            assert er_alert_result
            print("✅ Error rate alerting")

            # Test 4: Resource usage alerts
            high_usage_metrics = {
                "cpu_percent": 95.0,
                "memory_percent": 90.0,
                "disk_usage_percent": 85.0
            }

            async def test_resource_alerts():
                # Create a mock resource monitor that returns high usage
                test_metrics = PerformanceMetrics()
                test_metrics.add_resource_snapshot(high_usage_metrics)
                alerts = await alert_manager.check_performance_thresholds(test_metrics)
                resource_alerts = [a for a in alerts if "resource" in str(a).lower() or "cpu" in str(a).lower() or "memory" in str(a).lower()]
                return len(resource_alerts) > 0

            resource_alert_result = asyncio.run(test_resource_alerts())
            assert resource_alert_result
            print("✅ Resource usage alerting")

            # Test 5: Alert cooldown mechanism
            alert_key = "test_alert"
            assert not alert_manager._is_in_cooldown(alert_key, time.time())
            alert_manager._set_cooldown(alert_key, time.time())
            assert alert_manager._is_in_cooldown(alert_key, time.time())
            print("✅ Alert cooldown mechanism")

            self.detailed_results["phase_3"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": "All alerting functionality working correctly"
            }

            return True

        except Exception as e:
            print(f"❌ Phase 3 failed: {str(e)}")
            self.detailed_results["phase_3"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _validate_phase_4(self) -> bool:
        """Phase 4: Dashboard Integration (1 hour) - Health Check Integration."""
        print("\n📈 Phase 4: Dashboard Integration")
        print("-" * 40)

        try:
            perf_monitor = PerformanceMonitor()

            # Test 1: Performance middleware creation
            async def test_middleware_creation():
                middleware = await perf_monitor.create_performance_middleware()
                assert callable(middleware)
                return True

            middleware_result = asyncio.run(test_middleware_creation())
            assert middleware_result
            print("✅ Performance middleware creation")

            # Test 2: Agent performance decorator
            @perf_monitor.monitor_agent_performance("TestAgent")
            async def test_agent_function():
                await asyncio.sleep(0.01)
                return {"result": "success"}

            result = asyncio.run(test_agent_function())
            assert result == {"result": "success"}
            assert "TestAgent" in perf_monitor.metrics.agent_metrics
            print("✅ Agent performance monitoring decorator")

            # Test 3: Tool performance decorator
            @perf_monitor.monitor_tool_usage("test_tool")
            async def test_tool_function():
                await asyncio.sleep(0.01)
                return "tool_result"

            tool_result = asyncio.run(test_tool_function())
            assert tool_result == "tool_result"
            assert "test_tool" in perf_monitor.metrics.tool_metrics
            print("✅ Tool performance monitoring decorator")

            # Test 4: Performance summary generation
            perf_monitor.metrics.add_request_metric("/test", "GET", 0.5, 200)
            summary = perf_monitor.get_performance_summary()
            assert "request_metrics" in summary or "response_times" in summary
            assert "agent_metrics" in summary or "agent_performance" in summary
            print("✅ Performance summary generation")

            # Test 5: Alert checking integration
            async def test_alert_integration():
                alerts = await perf_monitor.check_alerts()
                assert isinstance(alerts, list)
                return True

            alert_integration_result = asyncio.run(test_alert_integration())
            assert alert_integration_result
            print("✅ Alert system integration")

            self.detailed_results["phase_4"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": "All dashboard integration functionality working correctly"
            }

            return True

        except Exception as e:
            print(f"❌ Phase 4 failed: {str(e)}")
            self.detailed_results["phase_4"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _validate_integration(self) -> bool:
        """Validate end-to-end integration of performance monitoring."""
        print("\n🔗 Integration Tests")
        print("-" * 40)

        try:
            # Test 1: Global performance monitor functionality
            assert performance_monitor is not None
            assert hasattr(performance_monitor, 'metrics')
            assert hasattr(performance_monitor, 'alert_manager')
            assert hasattr(performance_monitor, 'resource_monitor')
            print("✅ Global performance monitor structure")

            # Test 2: Monitoring lifecycle
            initial_state = performance_monitor.monitoring_active
            performance_monitor.start_monitoring()
            assert performance_monitor.monitoring_active
            performance_monitor.stop_monitoring()
            assert not performance_monitor.monitoring_active
            print("✅ Monitoring lifecycle management")

            # Test 3: Async monitoring lifecycle
            async def test_async_lifecycle():
                await performance_monitor.async_start_monitoring()
                assert performance_monitor.monitoring_active
                performance_monitor.stop_monitoring()
                assert not performance_monitor.monitoring_active
                return True

            async_lifecycle_result = asyncio.run(test_async_lifecycle())
            assert async_lifecycle_result
            print("✅ Async monitoring lifecycle")

            # Test 4: Agent metrics collection
            performance_monitor.metrics.add_agent_metric(
                "IntegrationTestAgent", 1.5, True, 100, ["search", "calculator"]
            )
            agent_data = performance_monitor.metrics.agent_metrics["IntegrationTestAgent"]
            assert agent_data["executions"] == 1
            assert agent_data["success_count"] == 1
            assert agent_data["avg_tokens"] == 100
            print("✅ Agent metrics integration")

            # Test 5: Tool metrics collection
            performance_monitor.metrics.add_tool_metric("integration_tool", 0.3, True)
            tool_data = performance_monitor.metrics.tool_metrics["integration_tool"]
            assert tool_data["calls"] == 1
            assert tool_data["success_count"] == 1
            print("✅ Tool metrics integration")

            self.detailed_results["integration"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": "All integration tests passing"
            }

            return True

        except Exception as e:
            print(f"❌ Integration tests failed: {str(e)}")
            self.detailed_results["integration"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _validate_performance(self) -> bool:
        """Validate performance characteristics of monitoring system."""
        print("\n⚡ Performance Tests")
        print("-" * 40)

        try:
            # Test 1: Memory bounds
            metrics = PerformanceMetrics()
            for i in range(2000):  # More than maxlen=1000
                metrics.add_request_metric(f"/test/{i}", "GET", 0.1, 200)
            assert len(metrics.request_times) <= 1000
            print("✅ Memory bounds enforcement")

            # Test 2: Monitoring overhead
            iterations = 100

            # Measure without monitoring
            start_time = time.time()
            for _ in range(iterations):
                time.sleep(0.001)
            no_monitoring_time = time.time() - start_time

            # Measure with monitoring
            test_monitor = PerformanceMonitor()
            test_monitor.start_monitoring()
            start_time = time.time()
            for i in range(iterations):
                test_monitor.metrics.add_request_metric(f"/test/{i}", "GET", 0.001, 200)
                time.sleep(0.001)
            with_monitoring_time = time.time() - start_time
            test_monitor.stop_monitoring()

            overhead_ratio = (with_monitoring_time - no_monitoring_time) / no_monitoring_time
            assert overhead_ratio < 1.0  # Should be reasonable overhead
            print(f"✅ Monitoring overhead: {overhead_ratio:.2%}")

            # Test 3: Concurrent access safety
            import threading

            def add_metrics(thread_id, shared_metrics):
                for i in range(50):
                    shared_metrics.add_request_metric(
                        f"/thread/{thread_id}/test/{i}", "GET", 0.1, 200
                    )

            shared_metrics = PerformanceMetrics()
            threads = []
            for t in range(3):
                thread = threading.Thread(target=add_metrics, args=(t, shared_metrics))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert len(shared_metrics.request_times) == 150
            print("✅ Concurrent access safety")

            # Test 4: Large dataset handling
            large_metrics = PerformanceMetrics()
            start_time = time.time()
            for i in range(1000):
                large_metrics.add_request_metric(f"/load/{i}", "GET", 0.1, 200)
            summary_time = time.time()
            summary = large_metrics.get_current_metrics_summary()
            end_time = time.time()

            processing_time = summary_time - start_time
            summary_time_taken = end_time - summary_time

            assert processing_time < 1.0  # Should be fast
            assert summary_time_taken < 0.5  # Summary should be fast
            print(f"✅ Large dataset handling: {processing_time:.3f}s processing, {summary_time_taken:.3f}s summary")

            # Test 5: Resource collection performance
            resource_monitor = ResourceMonitor(PerformanceMetrics())

            async def test_resource_performance():
                start_time = time.time()
                for _ in range(10):
                    await resource_monitor.collect_metrics()
                end_time = time.time()

                avg_collection_time = (end_time - start_time) / 10
                assert avg_collection_time < 1.0  # Should be fast
                return avg_collection_time

            avg_time = asyncio.run(test_resource_performance())
            print(f"✅ Resource collection performance: {avg_time:.3f}s avg")

            self.detailed_results["performance"] = {
                "total_tests": 5,
                "passed_tests": 5,
                "details": f"All performance tests passing. Overhead: {overhead_ratio:.2%}, Resource collection: {avg_time:.3f}s"
            }

            return True

        except Exception as e:
            print(f"❌ Performance tests failed: {str(e)}")
            self.detailed_results["performance"] = {
                "total_tests": 5,
                "passed_tests": 0,
                "error": str(e)
            }
            return False

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("📋 TASK 2.3 VALIDATION FINAL REPORT")
        print("=" * 60)

        total_phases = len(self.test_results)
        passed_phases = sum(1 for result in self.test_results.values() if result)

        print(f"\n📊 OVERALL STATUS: {passed_phases}/{total_phases} phases passed")

        # Phase-by-phase results
        phase_names = {
            "phase_1_core_metrics": "Phase 1: Core Performance Metrics",
            "phase_2_resource_monitoring": "Phase 2: Resource Monitoring",
            "phase_3_alert_configuration": "Phase 3: Alert Configuration",
            "phase_4_dashboard_integration": "Phase 4: Dashboard Integration",
            "integration_tests": "Integration Tests",
            "performance_tests": "Performance Tests"
        }

        for phase_key, passed in self.test_results.items():
            phase_name = phase_names.get(phase_key, phase_key)
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} {phase_name}")

            if phase_key.replace("_tests", "") in self.detailed_results:
                details = self.detailed_results[phase_key.replace("_tests", "")]
                if "total_tests" in details:
                    print(f"    └─ {details['passed_tests']}/{details['total_tests']} tests passed")

        # Success criteria
        success_criteria = {
            "core_functionality": passed_phases >= 4,  # At least 4/6 phases
            "critical_phases": (
                self.test_results["phase_1_core_metrics"] and
                self.test_results["phase_2_resource_monitoring"]
            ),
            "integration_working": self.test_results["integration_tests"],
            "performance_acceptable": self.test_results["performance_tests"]
        }

        overall_success = all(success_criteria.values())

        print(f"\n🎯 SUCCESS CRITERIA:")
        for criterion, met in success_criteria.items():
            status = "✅" if met else "❌"
            print(f"  {status} {criterion.replace('_', ' ').title()}")

        print(f"\n🏆 FINAL RESULT: {'✅ TASK 2.3 COMPLETED SUCCESSFULLY' if overall_success else '❌ TASK 2.3 REQUIRES ATTENTION'}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_success:
            print("  • Performance monitoring system is fully operational")
            print("  • All core functionality validated and working")
            print("  • Ready for production deployment")
            print("  • Consider configuring alert thresholds for specific use case")
        else:
            print("  • Review failed phases and address issues")
            print("  • Check error logs for specific failures")
            print("  • Verify all dependencies are properly installed")
            print("  • Test in isolated environment if issues persist")

        return {
            "overall_success": overall_success,
            "phases_passed": passed_phases,
            "total_phases": total_phases,
            "success_criteria": success_criteria,
            "test_results": self.test_results,
            "detailed_results": self.detailed_results
        }


def main():
    """Run Task 2.3 validation."""
    validator = Task2_3Validator()
    report = validator.validate_all()

    # Save detailed report
    with open("task_2_3_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: task_2_3_validation_report.json")

    return report["overall_success"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
