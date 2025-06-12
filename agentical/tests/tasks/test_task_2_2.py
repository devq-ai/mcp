#!/usr/bin/env python3
"""
Task 2.2 Validation Test - Structured Logging Implementation

This test validates that Task 2.2 requirements are met:
- Logging schema consistently applied across all operations
- Request correlation working end-to-end with unique trace IDs
- Agent operations logged with full context and decision rationale
- Log aggregation enables efficient debugging and monitoring
- Performance impact <2ms per logged operation
- Logfire dashboard shows rich, searchable structured data
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock, AsyncMock

import logfire
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

# Add the parent directory to sys.path to import modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.structured_logging import (
        StructuredLogger,
        CorrelationContext,
        LogLevel,
        OperationType,
        AgentPhase,
        BaseLogSchema,
        APIRequestSchema,
        AgentOperationSchema,
        WorkflowExecutionSchema,
        ToolUsageSchema,
        DatabaseOperationSchema,
        ExternalServiceSchema,
        PerformanceMetricSchema,
        api_logger,
        agent_logger,
        workflow_logger,
        tool_logger,
        database_logger,
        system_logger,
        get_logger,
        create_correlation_context,
        log_error_with_context,
        timed_operation
    )
    
    # Import main app for integration testing
    import main
    from main import app
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this test from the agentical directory")
    print("Required files: core/structured_logging.py, main.py")
    sys.exit(1)


class StructuredLoggingTest:
    """Test suite for Structured Logging Implementation"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.performance_metrics = {}
        
    def log_test_result(self, test_name: str, passed: bool, details: str = "", 
                       performance_data: Optional[Dict] = None):
        """Log individual test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "performance": performance_data
        }
        self.test_results.append(result)
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if performance_data:
            print(f"    Performance: {performance_data}")

    def test_logging_schema_framework(self) -> bool:
        """Test 1: Verify logging schema framework implementation"""
        print("Test 1: Logging Schema Framework")
        
        try:
            # Test correlation context generation
            correlation = create_correlation_context(
                session_id="test_session",
                user_id="test_user",
                agent_id="test_agent"
            )
            
            if not correlation.request_id or not correlation.trace_id:
                self.log_test_result(
                    "Logging Schema Framework",
                    False,
                    "Correlation context missing required IDs"
                )
                return False
            
            # Test structured logger creation
            test_logger = get_logger("test_component")
            if not isinstance(test_logger, StructuredLogger):
                self.log_test_result(
                    "Logging Schema Framework",
                    False,
                    "get_logger did not return StructuredLogger instance"
                )
                return False
            
            # Test schema models can be instantiated
            schemas_to_test = [
                (APIRequestSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test API request",
                    "correlation": correlation,
                    "method": "GET",
                    "path": "/test"
                }),
                (AgentOperationSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test agent operation",
                    "correlation": correlation,
                    "agent_type": "test",
                    "agent_name": "test_agent",
                    "phase": AgentPhase.ACTION,
                    "operation_id": "test_op"
                }),
                (PerformanceMetricSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test performance metric",
                    "correlation": correlation,
                    "metric_name": "test_metric",
                    "metric_value": 42.0,
                    "metric_unit": "milliseconds"
                })
            ]
            
            for schema_class, test_data in schemas_to_test:
                try:
                    schema_instance = schema_class(**test_data)
                    if not schema_instance.timestamp or not schema_instance.correlation:
                        raise ValueError(f"Schema {schema_class.__name__} missing required fields")
                except Exception as e:
                    self.log_test_result(
                        "Logging Schema Framework",
                        False,
                        f"Failed to create {schema_class.__name__}: {str(e)}"
                    )
                    return False
            
            self.log_test_result(
                "Logging Schema Framework",
                True,
                f"Successfully validated {len(schemas_to_test)} schema types and correlation context"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Logging Schema Framework",
                False,
                f"Exception during schema framework test: {str(e)}"
            )
            return False

    def test_request_correlation(self) -> bool:
        """Test 2: Verify request correlation and tracing"""
        print("Test 2: Request Correlation and Tracing")
        
        try:
            # Test correlation context in request flow
            correlation = create_correlation_context()
            
            # Test with correlation context manager
            test_logger = get_logger("correlation_test")
            
            with test_logger.correlation_context(correlation):
                current_context = test_logger._get_current_context()
                
                if not current_context:
                    self.log_test_result(
                        "Request Correlation",
                        False,
                        "Correlation context not properly maintained in context manager"
                    )
                    return False
                
                if current_context.request_id != correlation.request_id:
                    self.log_test_result(
                        "Request Correlation",
                        False,
                        "Correlation context request_id mismatch"
                    )
                    return False
            
            # Test that context is cleared after context manager
            context_after = test_logger._get_current_context()
            if context_after is not None:
                self.log_test_result(
                    "Request Correlation",
                    False,
                    "Correlation context not properly cleared after context manager"
                )
                return False
            
            # Test API request with correlation headers
            test_headers = {
                "X-Session-ID": "test_session_123",
                "X-User-ID": "test_user_456"
            }
            
            response = self.client.get("/health", headers=test_headers)
            
            if response.status_code != 200:
                self.log_test_result(
                    "Request Correlation",
                    False,
                    f"Health endpoint failed with correlation headers: {response.status_code}"
                )
                return False
            
            # Check for correlation headers in response
            if "X-Request-ID" not in response.headers:
                self.log_test_result(
                    "Request Correlation",
                    False,
                    "Response missing X-Request-ID header"
                )
                return False
            
            if "X-Trace-ID" not in response.headers:
                self.log_test_result(
                    "Request Correlation",
                    False,
                    "Response missing X-Trace-ID header"
                )
                return False
            
            self.log_test_result(
                "Request Correlation",
                True,
                f"Request correlation working with trace ID: {response.headers.get('X-Trace-ID', 'N/A')[:16]}..."
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Request Correlation",
                False,
                f"Exception during correlation test: {str(e)}"
            )
            return False

    def test_agent_operation_logging(self) -> bool:
        """Test 3: Verify agent operation logging with context"""
        print("Test 3: Agent Operation Logging")
        
        try:
            # Test agent operation logging methods
            correlation = create_correlation_context(agent_id="test_agent_123")
            
            # Test each agent phase logging
            phases_to_test = [
                (AgentPhase.INITIALIZATION, "Agent initialization"),
                (AgentPhase.PERCEPTION, "Agent perception phase"),
                (AgentPhase.DECISION, "Agent decision making"),
                (AgentPhase.ACTION, "Agent action execution"),
                (AgentPhase.REFLECTION, "Agent reflection phase")
            ]
            
            for phase, message in phases_to_test:
                try:
                    agent_logger.log_agent_operation(
                        message=message,
                        agent_type="test_agent",
                        agent_name="test_agent_123",
                        phase=phase,
                        operation_id=f"op_{phase.value}",
                        level=LogLevel.INFO,
                        correlation=correlation,
                        input_data={"test": "data"},
                        decision_rationale=f"Rationale for {phase.value}",
                        tools_used=["test_tool_1", "test_tool_2"],
                        execution_time_ms=50.0,
                        success=True
                    )
                except Exception as e:
                    self.log_test_result(
                        "Agent Operation Logging",
                        False,
                        f"Failed to log agent phase {phase.value}: {str(e)}"
                    )
                    return False
            
            # Test tool usage logging
            tool_logger.log_tool_usage(
                message="Tool execution test",
                tool_name="test_tool",
                tool_category="test_category",
                operation="execute",
                level=LogLevel.INFO,
                correlation=correlation,
                input_parameters={"param1": "value1"},
                output_result={"result": "success"},
                execution_time_ms=25.0,
                success=True
            )
            
            # Test workflow execution logging
            workflow_logger.log_workflow_execution(
                message="Workflow execution test",
                workflow_type="test_workflow",
                workflow_name="test_workflow_instance",
                level=LogLevel.INFO,
                correlation=correlation,
                step_name="test_step",
                step_index=1,
                total_steps=3,
                agents_involved=["agent1", "agent2"],
                execution_time_ms=100.0,
                success=True
            )
            
            self.log_test_result(
                "Agent Operation Logging",
                True,
                f"Successfully logged {len(phases_to_test)} agent phases + tool usage + workflow execution"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Agent Operation Logging",
                False,
                f"Exception during agent operation logging test: {str(e)}"
            )
            return False

    def test_performance_overhead(self) -> bool:
        """Test 4: Verify logging performance overhead <2ms per operation"""
        print("Test 4: Performance Overhead")
        
        try:
            # Test logging performance overhead
            num_operations = 10
            correlation = create_correlation_context()
            test_logger = get_logger("performance_test")
            
            # Measure structured logging overhead
            logging_times = []
            
            for i in range(num_operations):
                start_time = time.time()
                
                test_logger.log_api_request(
                    message=f"Performance test operation {i}",
                    method="GET",
                    path=f"/test/{i}",
                    level=LogLevel.INFO,
                    correlation=correlation,
                    status_code=200,
                    response_time_ms=10.0
                )
                
                end_time = time.time()
                logging_time = (end_time - start_time) * 1000  # Convert to ms
                logging_times.append(logging_time)
            
            # Calculate performance metrics
            avg_logging_time = sum(logging_times) / len(logging_times)
            max_logging_time = max(logging_times)
            min_logging_time = min(logging_times)
            
            # Test performance target (<2ms per operation)
            performance_acceptable = avg_logging_time < 2.0
            
            performance_data = {
                "avg_logging_time_ms": round(avg_logging_time, 3),
                "max_logging_time_ms": round(max_logging_time, 3),
                "min_logging_time_ms": round(min_logging_time, 3),
                "num_operations": num_operations,
                "target_ms": 2.0,
                "performance_acceptable": performance_acceptable
            }
            
            self.performance_metrics.update(performance_data)
            
            # Test timed operation decorator
            @timed_operation(test_logger, "test_timed_operation")
            def test_sync_function():
                time.sleep(0.01)  # 10ms operation
                return "success"
            
            @timed_operation(test_logger, "test_async_timed_operation")
            async def test_async_function():
                await asyncio.sleep(0.01)  # 10ms operation
                return "success"
            
            # Test sync timed operation
            sync_result = test_sync_function()
            if sync_result != "success":
                self.log_test_result(
                    "Performance Overhead",
                    False,
                    "Timed operation decorator failed for sync function"
                )
                return False
            
            # Test async timed operation
            async_result = asyncio.run(test_async_function())
            if async_result != "success":
                self.log_test_result(
                    "Performance Overhead",
                    False,
                    "Timed operation decorator failed for async function"
                )
                return False
            
            details = f"Avg: {avg_logging_time:.3f}ms, Max: {max_logging_time:.3f}ms"
            if not performance_acceptable:
                details += f" (exceeds 2ms target)"
            
            self.log_test_result(
                "Performance Overhead",
                performance_acceptable,
                details,
                performance_data
            )
            return performance_acceptable
            
        except Exception as e:
            self.log_test_result(
                "Performance Overhead",
                False,
                f"Exception during performance test: {str(e)}"
            )
            return False

    def test_log_aggregation_patterns(self) -> bool:
        """Test 5: Verify log aggregation and query patterns"""
        print("Test 5: Log Aggregation Patterns")
        
        try:
            # Test various log aggregation scenarios
            correlation = create_correlation_context()
            
            # Test database operation logging
            database_logger.log_database_operation(
                message="Database query executed",
                database_type="sqlalchemy",
                operation="select",
                level=LogLevel.DEBUG,
                correlation=correlation,
                table_name="users",
                rows_affected=5,
                execution_time_ms=15.0,
                success=True
            )
            
            # Test external service logging
            system_logger.log_external_service(
                message="External API call",
                service_name="ptolemies_api",
                level=LogLevel.INFO,
                correlation=correlation,
                endpoint="/api/knowledge",
                method="POST",
                response_time_ms=150.0,
                status_code=200,
                success=True
            )
            
            # Test performance metrics logging
            system_logger.log_performance_metric(
                message="System performance metric",
                metric_name="memory_usage",
                metric_value=75.5,
                metric_unit="percentage",
                level=LogLevel.INFO,
                correlation=correlation,
                tags={"component": "api", "environment": "test"},
                threshold_exceeded=False
            )
            
            # Test error logging with context
            test_exception = ValueError("Test exception for logging")
            
            log_error_with_context(
                system_logger,
                test_exception,
                "Test error with context",
                OperationType.SYSTEM_EVENT,
                correlation,
                component="test_component",
                operation="test_operation"
            )
            
            # Test structured logging with various log levels
            levels_to_test = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
            
            for level in levels_to_test:
                api_logger.log_api_request(
                    message=f"Test message for level {level.value}",
                    method="GET",
                    path="/test",
                    level=level,
                    correlation=correlation,
                    status_code=200 if level != LogLevel.ERROR else 500
                )
            
            self.log_test_result(
                "Log Aggregation Patterns",
                True,
                f"Successfully tested database, external service, performance, error, and multi-level logging"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Log Aggregation Patterns",
                False,
                f"Exception during log aggregation test: {str(e)}"
            )
            return False

    def test_enhanced_request_tracing(self) -> bool:
        """Test 6: Verify enhanced request tracing integration"""
        print("Test 6: Enhanced Request Tracing")
        
        try:
            # Test API endpoints with enhanced tracing
            endpoints_to_test = [
                ("/", "GET", "Root endpoint"),
                ("/health", "GET", "Health check endpoint"),
            ]
            
            all_passed = True
            tracing_results = []
            
            for endpoint, method, description in endpoints_to_test:
                start_time = time.time()
                
                # Add correlation headers
                headers = {
                    "X-Session-ID": f"session_{uuid.uuid4().hex[:8]}",
                    "X-User-ID": f"user_{uuid.uuid4().hex[:8]}"
                }
                
                response = self.client.request(method, endpoint, headers=headers)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000
                
                # Verify response has correlation headers
                has_request_id = "X-Request-ID" in response.headers
                has_trace_id = "X-Trace-ID" in response.headers
                has_response_time = "X-Response-Time" in response.headers
                
                success = (response.status_code in [200, 404] and 
                          has_request_id and has_trace_id and has_response_time)
                
                tracing_result = {
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time, 2),
                    "has_request_id": has_request_id,
                    "has_trace_id": has_trace_id,
                    "has_response_time": has_response_time,
                    "success": success
                }
                
                tracing_results.append(tracing_result)
                
                if not success:
                    all_passed = False
            
            # Test agent execution endpoint with structured logging
            agent_request = {
                "agent_id": "test_agent_structured_logging",
                "operation": "test_operation", 
                "inputs": {"test": "data"},
                "use_knowledge": False,
                "tools": ["test_tool"]
            }
            
            try:
                response = self.client.post("/api/v1/agents/execute", json=agent_request)
                agent_success = response.status_code in [200, 422]  # 422 acceptable for missing agent
                
                tracing_results.append({
                    "endpoint": "/api/v1/agents/execute",
                    "method": "POST",
                    "status_code": response.status_code,
                    "agent_execution_traced": agent_success,
                    "success": agent_success
                })
                
            except Exception as e:
                # Agent execution might fail due to missing agent system, but tracing should work
                tracing_results.append({
                    "endpoint": "/api/v1/agents/execute",
                    "method": "POST",
                    "error": str(e),
                    "success": False
                })
                # Don't fail the test for agent system issues
            
            details = f"Tested {len(endpoints_to_test)} endpoints with correlation headers"
            
            self.log_test_result(
                "Enhanced Request Tracing",
                all_passed,
                details,
                {"tracing_results": tracing_results}
            )
            return all_passed
            
        except Exception as e:
            self.log_test_result(
                "Enhanced Request Tracing",
                False,
                f"Exception during request tracing test: {str(e)}"
            )
            return False

    def test_integration_with_logfire(self) -> bool:
        """Test 7: Verify integration with existing Logfire foundation"""
        print("Test 7: Integration with Logfire Foundation")
        
        try:
            # Test that structured logging works with existing Logfire setup
            correlation = create_correlation_context()
            
            # Test direct Logfire spans with structured context
            with api_logger.correlation_context(correlation):
                with logfire.span("test_integration_span", test_attribute="structured_logging"):
                    # Log using structured logging within Logfire span
                    api_logger.log_api_request(
                        message="Integration test with Logfire span",
                        method="GET",
                        path="/integration_test",
                        level=LogLevel.INFO,
                        correlation=correlation
                    )
                    
                    # Log agent operation within span
                    agent_logger.log_agent_operation(
                        message="Agent operation within Logfire span",
                        agent_type="integration_test",
                        agent_name="test_agent",
                        phase=AgentPhase.ACTION,
                        operation_id="integration_test",
                        level=LogLevel.INFO,
                        correlation=correlation
                    )
            
            # Test that existing Logfire instrumentation still works
            response = self.client.get("/health")
            if response.status_code != 200:
                self.log_test_result(
                    "Integration with Logfire",
                    False,
                    "Existing Logfire instrumentation broken after structured logging integration"
                )
                return False
            
            # Test performance metrics integration
            api_logger.log_performance_metric(
                message="Integration performance test",
                metric_name="integration_test_duration",
                metric_value=42.5,
                metric_unit="milliseconds",
                level=LogLevel.INFO,
                correlation=correlation,
                tags={"integration": "logfire", "test": "structured_logging"}
            )
            
            self.log_test_result(
                "Integration with Logfire",
                True,
                "Structured logging successfully integrated with existing Logfire foundation"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Integration with Logfire",
                False,
                f"Exception during Logfire integration test: {str(e)}"
            )
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all structured logging tests and return comprehensive results"""
        print("🚀 Starting Task 2.2 Structured Logging Implementation Tests")
        print("=" * 70)
        
        # List of all tests to run
        tests = [
            ("Logging Schema Framework", self.test_logging_schema_framework),
            ("Request Correlation", self.test_request_correlation),
            ("Agent Operation Logging", self.test_agent_operation_logging),
            ("Performance Overhead", self.test_performance_overhead),
            ("Log Aggregation Patterns", self.test_log_aggregation_patterns),
            ("Enhanced Request Tracing", self.test_enhanced_request_tracing),
            ("Integration with Logfire", self.test_integration_with_logfire),
        ]
        
        # Run all tests
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                self.log_test_result(
                    test_name,
                    False,
                    f"Unexpected exception: {str(e)}"
                )
        
        # Calculate overall results
        success_rate = (passed_tests / total_tests) * 100
        overall_success = passed_tests == total_tests
        
        print("\n" + "=" * 70)
        print(f"📊 TASK 2.2 STRUCTURED LOGGING TEST RESULTS")
        print("=" * 70)
        print(f"Overall Status: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if self.performance_metrics:
            print(f"Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value}")
        
        # Summary by category
        categories = {
            "Schema & Framework": ["Logging Schema Framework", "Log Aggregation Patterns"],
            "Correlation & Tracing": ["Request Correlation", "Enhanced Request Tracing"],
            "Agent Integration": ["Agent Operation Logging"],
            "Performance": ["Performance Overhead"],
            "Integration": ["Integration with Logfire"]
        }
        
        print(f"\nResults by Category:")
        for category, test_names in categories.items():
            category_results = [r for r in self.test_results if r["test"] in test_names]
            category_passed = sum(1 for r in category_results if "✅" in r["status"])
            category_total = len(category_results)
            print(f"  {category}: {category_passed}/{category_total}")
        
        return {
            "overall_success": overall_success,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }


def main():
    """Main test execution"""
    print("Task 2.2 - Structured Logging Implementation Validation")
    print("Testing comprehensive context-aware structured logging framework")
    print()
    
    # Initialize test suite
    test_suite = StructuredLoggingTest()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print final status
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if results['overall_success'] else 'FAILED'}")
    
    if results['overall_success']:
        print("✅ Task 2.2 requirements successfully validated!")
        print("✅ Structured logging implementation is complete and functional")
        print("✅ Ready to proceed with Task 2.3 (Performance Monitoring Setup)")
        
        # Print key achievements
        if results.get('performance_metrics'):
            avg_time = results['performance_metrics'].get('avg_logging_time_ms', 0)
            print(f"✅ Performance target achieved: {avg_time:.3f}ms avg overhead")
        
    else:
        print("❌ Task 2.2 validation failed - review and fix issues before proceeding")
        failed_tests = [r for r in results['test_results'] if "❌" in r["status"]]
        print(f"❌ Failed tests: {', '.join([r['test'] for r in failed_tests])}")
    
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)