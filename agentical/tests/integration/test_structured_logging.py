#!/usr/bin/env python3
"""
Simplified Structured Logging Test for Task 2.2

This test validates the structured logging framework without full app dependencies:
- Logging schema framework functionality
- Correlation context management
- Performance overhead measurement
- Integration with Logfire SDK
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

import logfire
from pydantic import BaseModel

# Configure Logfire for testing
logfire.configure(
    token="test_token_for_validation",
    project_name="test_project",
    service_name="test_service",
    environment="test"
)

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
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this test from the agentical directory")
    print("Required files: core/structured_logging.py")
    sys.exit(1)


class StructuredLoggingTest:
    """Test suite for Structured Logging Framework"""
    
    def __init__(self):
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

    def test_correlation_context(self) -> bool:
        """Test 1: Verify correlation context functionality"""
        print("Test 1: Correlation Context Management")
        
        try:
            # Test correlation context generation
            correlation = create_correlation_context(
                session_id="test_session",
                user_id="test_user",
                agent_id="test_agent"
            )
            
            # Verify required fields
            if not correlation.request_id or not correlation.trace_id:
                self.log_test_result(
                    "Correlation Context",
                    False,
                    "Correlation context missing required IDs"
                )
                return False
            
            # Test context manager functionality
            test_logger = get_logger("test_component")
            
            with test_logger.correlation_context(correlation):
                current_context = test_logger._get_current_context()
                
                if not current_context or current_context.request_id != correlation.request_id:
                    self.log_test_result(
                        "Correlation Context",
                        False,
                        "Correlation context not properly maintained in context manager"
                    )
                    return False
            
            # Test that context is cleared after context manager
            context_after = test_logger._get_current_context()
            if context_after is not None:
                self.log_test_result(
                    "Correlation Context",
                    False,
                    "Correlation context not properly cleared after context manager"
                )
                return False
            
            self.log_test_result(
                "Correlation Context",
                True,
                f"Successfully created and managed correlation context: {correlation.request_id[:16]}..."
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Correlation Context",
                False,
                f"Exception during correlation context test: {str(e)}"
            )
            return False

    def test_schema_validation(self) -> bool:
        """Test 2: Verify schema framework and validation"""
        print("Test 2: Schema Framework Validation")
        
        try:
            correlation = create_correlation_context()
            
            # Test all schema types
            schemas_to_test = [
                (APIRequestSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test API request",
                    "correlation": correlation,
                    "component": "test_component",
                    "method": "GET",
                    "path": "/test",
                    "status_code": 200,
                    "response_time_ms": 150.0
                }),
                (AgentOperationSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test agent operation",
                    "correlation": correlation,
                    "component": "test_component",
                    "agent_type": "test_agent",
                    "agent_name": "test_agent_123",
                    "phase": AgentPhase.ACTION,
                    "operation_id": "test_operation",
                    "execution_time_ms": 250.0,
                    "success": True
                }),
                (WorkflowExecutionSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test workflow execution",
                    "correlation": correlation,
                    "component": "test_component",
                    "workflow_type": "test_workflow",
                    "workflow_name": "test_instance",
                    "step_name": "test_step",
                    "execution_time_ms": 500.0,
                    "success": True
                }),
                (ToolUsageSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test tool usage",
                    "correlation": correlation,
                    "component": "test_component",
                    "tool_name": "test_tool",
                    "tool_category": "testing",
                    "operation": "execute",
                    "execution_time_ms": 100.0,
                    "success": True
                }),
                (DatabaseOperationSchema, {
                    "level": LogLevel.DEBUG,
                    "message": "Test database operation",
                    "correlation": correlation,
                    "component": "test_component",
                    "database_type": "postgresql",
                    "operation": "select",
                    "table_name": "test_table",
                    "execution_time_ms": 50.0,
                    "success": True
                }),
                (ExternalServiceSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test external service call",
                    "correlation": correlation,
                    "component": "test_component",
                    "service_name": "test_api",
                    "endpoint": "/api/test",
                    "method": "POST",
                    "response_time_ms": 300.0,
                    "status_code": 200,
                    "success": True
                }),
                (PerformanceMetricSchema, {
                    "level": LogLevel.INFO,
                    "message": "Test performance metric",
                    "correlation": correlation,
                    "component": "test_component",
                    "metric_name": "test_metric",
                    "metric_value": 75.5,
                    "metric_unit": "percentage",
                    "tags": {"component": "test", "environment": "test"}
                })
            ]
            
            successful_schemas = 0
            
            for schema_class, test_data in schemas_to_test:
                try:
                    schema_instance = schema_class(**test_data)
                    
                    # Verify required fields
                    if not schema_instance.timestamp or not schema_instance.correlation:
                        raise ValueError(f"Schema {schema_class.__name__} missing required fields")
                    
                    # Verify schema can be serialized
                    schema_dict = schema_instance.dict()
                    if not schema_dict:
                        raise ValueError(f"Schema {schema_class.__name__} failed to serialize")
                    
                    successful_schemas += 1
                    
                except Exception as e:
                    self.log_test_result(
                        "Schema Validation",
                        False,
                        f"Failed to validate {schema_class.__name__}: {str(e)}"
                    )
                    return False
            
            self.log_test_result(
                "Schema Validation",
                True,
                f"Successfully validated {successful_schemas}/{len(schemas_to_test)} schema types"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Schema Validation",
                False,
                f"Exception during schema validation test: {str(e)}"
            )
            return False

    def test_structured_logging_operations(self) -> bool:
        """Test 3: Verify structured logging operations"""
        print("Test 3: Structured Logging Operations")
        
        try:
            correlation = create_correlation_context()
            
            # Test API request logging
            api_logger.log_api_request(
                message="Test API request logging",
                method="GET",
                path="/test/api",
                level=LogLevel.INFO,
                correlation=correlation,
                status_code=200,
                response_time_ms=120.0
            )
            
            # Test agent operation logging with all phases
            agent_phases = [
                (AgentPhase.INITIALIZATION, "Agent initialization"),
                (AgentPhase.PERCEPTION, "Agent perception phase"),
                (AgentPhase.DECISION, "Agent decision making"),
                (AgentPhase.ACTION, "Agent action execution"),
                (AgentPhase.REFLECTION, "Agent reflection phase")
            ]
            
            for phase, message in agent_phases:
                agent_logger.log_agent_operation(
                    message=message,
                    agent_type="test_agent",
                    agent_name="test_agent_123",
                    phase=phase,
                    operation_id=f"op_{phase.value}",
                    level=LogLevel.INFO,
                    correlation=correlation,
                    execution_time_ms=50.0,
                    success=True
                )
            
            # Test workflow execution logging
            workflow_logger.log_workflow_execution(
                message="Test workflow execution",
                workflow_type="test_workflow",
                workflow_name="test_instance",
                level=LogLevel.INFO,
                correlation=correlation,
                step_name="test_step",
                step_index=1,
                total_steps=3,
                execution_time_ms=200.0,
                success=True
            )
            
            # Test tool usage logging
            tool_logger.log_tool_usage(
                message="Test tool usage",
                tool_name="test_tool",
                tool_category="testing",
                operation="execute",
                level=LogLevel.INFO,
                correlation=correlation,
                input_parameters={"param1": "value1"},
                output_result={"result": "success"},
                execution_time_ms=75.0,
                success=True
            )
            
            # Test database operation logging
            database_logger.log_database_operation(
                message="Test database query",
                database_type="postgresql",
                operation="select",
                level=LogLevel.DEBUG,
                correlation=correlation,
                table_name="test_table",
                rows_affected=5,
                execution_time_ms=25.0,
                success=True
            )
            
            # Test external service logging
            system_logger.log_external_service(
                message="Test external API call",
                service_name="test_api",
                level=LogLevel.INFO,
                correlation=correlation,
                endpoint="/api/test",
                method="POST",
                response_time_ms=180.0,
                status_code=200,
                success=True
            )
            
            # Test performance metrics logging
            system_logger.log_performance_metric(
                message="Test performance metric",
                metric_name="response_time",
                metric_value=125.0,
                metric_unit="milliseconds",
                level=LogLevel.INFO,
                correlation=correlation,
                tags={"endpoint": "/test", "method": "GET"}
            )
            
            self.log_test_result(
                "Structured Logging Operations",
                True,
                f"Successfully tested {len(agent_phases) + 6} different logging operations"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Structured Logging Operations",
                False,
                f"Exception during structured logging operations test: {str(e)}"
            )
            return False

    def test_performance_overhead(self) -> bool:
        """Test 4: Verify logging performance overhead <2ms per operation"""
        print("Test 4: Performance Overhead")
        
        try:
            num_operations = 20
            correlation = create_correlation_context()
            test_logger = get_logger("performance_test")
            
            # Measure structured logging performance
            logging_times = []
            
            for i in range(num_operations):
                start_time = time.time()
                
                # Log a structured API request
                test_logger.log_api_request(
                    message=f"Performance test operation {i}",
                    method="GET",
                    path=f"/test/{i}",
                    level=LogLevel.INFO,
                    correlation=correlation,
                    status_code=200,
                    response_time_ms=10.0 + i
                )
                
                end_time = time.time()
                logging_time = (end_time - start_time) * 1000  # Convert to ms
                logging_times.append(logging_time)
            
            # Calculate performance metrics
            avg_time = sum(logging_times) / len(logging_times)
            max_time = max(logging_times)
            min_time = min(logging_times)
            
            # Test performance target (<2ms per operation)
            performance_acceptable = avg_time < 2.0
            
            performance_data = {
                "avg_logging_time_ms": round(avg_time, 3),
                "max_logging_time_ms": round(max_time, 3),
                "min_logging_time_ms": round(min_time, 3),
                "num_operations": num_operations,
                "target_ms": 2.0,
                "performance_acceptable": performance_acceptable
            }
            
            self.performance_metrics.update(performance_data)
            
            details = f"Avg: {avg_time:.3f}ms, Max: {max_time:.3f}ms, Min: {min_time:.3f}ms"
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

    def test_error_handling(self) -> bool:
        """Test 5: Verify error handling and logging"""
        print("Test 5: Error Handling and Logging")
        
        try:
            correlation = create_correlation_context()
            test_logger = get_logger("error_test")
            
            # Test error logging with context
            test_exception = ValueError("Test exception for structured logging")
            
            log_error_with_context(
                test_logger,
                test_exception,
                "Test error with full context",
                OperationType.SYSTEM_EVENT,
                correlation,
                component="test_component",
                operation="test_operation",
                additional_context="error_test_data"
            )
            
            # Test error logging in agent operations
            agent_logger.log_agent_operation(
                message="Test agent operation error",
                agent_type="test_agent",
                agent_name="error_test_agent",
                phase=AgentPhase.ACTION,
                operation_id="error_test_op",
                level=LogLevel.ERROR,
                correlation=correlation,
                success=False,
                error_details={
                    "exception_type": "ValueError",
                    "exception_message": "Test error message",
                    "operation": "error_test"
                }
            )
            
            # Test different log levels
            log_levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
            
            for level in log_levels:
                test_logger.log_api_request(
                    message=f"Test log level: {level.value}",
                    method="GET",
                    path="/test/levels",
                    level=level,
                    correlation=correlation,
                    status_code=200 if level not in [LogLevel.ERROR, LogLevel.CRITICAL] else 500
                )
            
            self.log_test_result(
                "Error Handling",
                True,
                f"Successfully tested error logging and {len(log_levels)} log levels"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Error Handling",
                False,
                f"Exception during error handling test: {str(e)}"
            )
            return False

    def test_timed_operation_decorator(self) -> bool:
        """Test 6: Verify timed operation decorator functionality"""
        print("Test 6: Timed Operation Decorator")
        
        try:
            test_logger = get_logger("timer_test")
            
            # Test sync timed operation
            @timed_operation(test_logger, "sync_test_operation")
            def sync_test_function(duration: float = 0.01):
                time.sleep(duration)
                return "sync_success"
            
            # Test async timed operation
            @timed_operation(test_logger, "async_test_operation")
            async def async_test_function(duration: float = 0.01):
                await asyncio.sleep(duration)
                return "async_success"
            
            # Execute sync function
            sync_result = sync_test_function(0.02)  # 20ms operation
            if sync_result != "sync_success":
                self.log_test_result(
                    "Timed Operation Decorator",
                    False,
                    "Sync timed operation failed to return expected result"
                )
                return False
            
            # Execute async function
            async_result = asyncio.run(async_test_function(0.015))  # 15ms operation
            if async_result != "async_success":
                self.log_test_result(
                    "Timed Operation Decorator",
                    False,
                    "Async timed operation failed to return expected result"
                )
                return False
            
            # Test error handling in timed operation
            @timed_operation(test_logger, "error_test_operation")
            def error_test_function():
                raise ValueError("Test error in timed operation")
            
            try:
                error_test_function()
                self.log_test_result(
                    "Timed Operation Decorator",
                    False,
                    "Timed operation should have raised an exception"
                )
                return False
            except ValueError:
                # Expected behavior - exception should be re-raised
                pass
            
            self.log_test_result(
                "Timed Operation Decorator",
                True,
                "Successfully tested sync, async, and error scenarios for timed operations"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Timed Operation Decorator",
                False,
                f"Exception during timed operation test: {str(e)}"
            )
            return False

    def test_logfire_integration(self) -> bool:
        """Test 7: Verify integration with Logfire SDK"""
        print("Test 7: Logfire SDK Integration")
        
        try:
            correlation = create_correlation_context()
            test_logger = get_logger("integration_test")
            
            # Test structured logging within Logfire span
            with logfire.span("test_structured_logging_span", test_attribute="integration"):
                with test_logger.correlation_context(correlation):
                    # Log various operation types within the span
                    test_logger.log_api_request(
                        message="API request within Logfire span",
                        method="POST",
                        path="/integration/test",
                        level=LogLevel.INFO,
                        correlation=correlation,
                        status_code=201
                    )
                    
                    agent_logger.log_agent_operation(
                        message="Agent operation within Logfire span",
                        agent_type="integration_agent",
                        agent_name="span_test_agent",
                        phase=AgentPhase.ACTION,
                        operation_id="span_test_op",
                        level=LogLevel.INFO,
                        correlation=correlation,
                        success=True
                    )
                    
                    # Nested span test
                    with logfire.span("nested_operation", operation="validation"):
                        test_logger.log_performance_metric(
                            message="Performance metric in nested span",
                            metric_name="nested_operation_time",
                            metric_value=42.0,
                            metric_unit="milliseconds",
                            level=LogLevel.INFO,
                            correlation=correlation
                        )
            
            # Test manual logfire calls with structured context
            logfire.info("Direct Logfire logging test", 
                        structured_logging="enabled",
                        correlation_id=correlation.request_id)
            
            self.log_test_result(
                "Logfire Integration",
                True,
                "Successfully integrated structured logging with Logfire spans and direct logging"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Logfire Integration",
                False,
                f"Exception during Logfire integration test: {str(e)}"
            )
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all structured logging tests and return comprehensive results"""
        print("🚀 Starting Structured Logging Framework Tests for Task 2.2")
        print("=" * 70)
        
        # List of all tests to run
        tests = [
            ("Correlation Context", self.test_correlation_context),
            ("Schema Validation", self.test_schema_validation),
            ("Structured Logging Operations", self.test_structured_logging_operations),
            ("Performance Overhead", self.test_performance_overhead),
            ("Error Handling", self.test_error_handling),
            ("Timed Operation Decorator", self.test_timed_operation_decorator),
            ("Logfire Integration", self.test_logfire_integration),
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
        print(f"📊 STRUCTURED LOGGING FRAMEWORK TEST RESULTS")
        print("=" * 70)
        print(f"Overall Status: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if self.performance_metrics:
            print(f"Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value}")
        
        # Summary by category
        categories = {
            "Core Framework": ["Correlation Context", "Schema Validation"],
            "Logging Operations": ["Structured Logging Operations", "Error Handling"],
            "Performance": ["Performance Overhead", "Timed Operation Decorator"],
            "Integration": ["Logfire Integration"]
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
    print("Task 2.2 - Structured Logging Framework Validation")
    print("Testing comprehensive structured logging implementation")
    print()
    
    # Initialize test suite
    test_suite = StructuredLoggingTest()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print final status
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if results['overall_success'] else 'FAILED'}")
    
    if results['overall_success']:
        print("✅ Task 2.2 structured logging framework successfully validated!")
        print("✅ All logging schemas, correlation, and performance targets met")
        print("✅ Ready to proceed with Task 2.3 (Performance Monitoring Setup)")
        
        # Print key achievements
        if results.get('performance_metrics'):
            avg_time = results['performance_metrics'].get('avg_logging_time_ms', 0)
            print(f"✅ Performance target achieved: {avg_time:.3f}ms avg overhead (target: <2ms)")
        
    else:
        print("❌ Task 2.2 validation failed - review and fix issues before proceeding")
        failed_tests = [r for r in results['test_results'] if "❌" in r["status"]]
        print(f"❌ Failed tests: {', '.join([r['test'] for r in failed_tests])}")
    
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)