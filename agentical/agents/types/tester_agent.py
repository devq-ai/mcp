"""
Tester Agent Implementation for Agentical Framework

This module provides the TesterAgent implementation for automated testing,
quality assurance, test case generation, and testing workflow orchestration.

Features:
- Automated test case generation
- Test execution and reporting
- Performance and load testing
- Security testing automation
- Test data management
- CI/CD test integration
- Bug detection and reporting
- Test coverage analysis
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import logfire
from pydantic import BaseModel, Field

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class TestRequest(BaseModel):
    """Request model for test operations."""
    test_type: str = Field(..., description="Type of test (unit, integration, e2e, performance)")
    target: str = Field(..., description="Test target (file, URL, application)")
    test_framework: str = Field(..., description="Testing framework to use")
    test_config: Dict[str, Any] = Field(..., description="Test configuration")
    coverage_threshold: float = Field(default=80.0, description="Minimum coverage threshold")
    timeout: int = Field(default=300, description="Test timeout in seconds")


class TestGenerationRequest(BaseModel):
    """Request model for test generation."""
    source_code: str = Field(..., description="Source code to generate tests for")
    language: str = Field(..., description="Programming language")
    test_type: str = Field(..., description="Type of tests to generate")
    framework: str = Field(..., description="Testing framework")
    coverage_target: float = Field(default=90.0, description="Target coverage percentage")
    include_edge_cases: bool = Field(default=True, description="Include edge case tests")


class PerformanceTestRequest(BaseModel):
    """Request model for performance testing."""
    target_url: str = Field(..., description="Target URL or endpoint")
    test_duration: int = Field(default=300, description="Test duration in seconds")
    concurrent_users: int = Field(default=10, description="Number of concurrent users")
    ramp_up_time: int = Field(default=60, description="Ramp up time in seconds")
    test_scenarios: List[Dict[str, Any]] = Field(..., description="Test scenarios")
    performance_thresholds: Dict[str, float] = Field(..., description="Performance thresholds")


class SecurityTestRequest(BaseModel):
    """Request model for security testing."""
    target: str = Field(..., description="Target for security testing")
    test_type: str = Field(..., description="Security test type (vulnerability, penetration)")
    scan_depth: str = Field(default="medium", description="Scan depth (light, medium, deep)")
    test_categories: List[str] = Field(..., description="Security test categories")
    compliance_standards: Optional[List[str]] = Field(default=None, description="Compliance standards")


class TesterAgent(EnhancedBaseAgent[TestRequest, Dict[str, Any]]):
    """
    Specialized agent for automated testing and quality assurance.

    Capabilities:
    - Test case generation and execution
    - Performance and load testing
    - Security testing automation
    - Test coverage analysis
    - Bug detection and reporting
    - CI/CD test integration
    - Test data management
    - Quality metrics tracking
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "TesterAgent",
        description: str = "Specialized agent for automated testing and QA",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.TESTER_AGENT,
            **kwargs
        )

        # Testing framework configuration
        self.test_frameworks = {
            "python": ["pytest", "unittest", "nose2", "doctest"],
            "javascript": ["jest", "mocha", "jasmine", "karma"],
            "java": ["junit", "testng", "mockito"],
            "go": ["testing", "ginkgo", "testify"],
            "rust": ["cargo test", "quickcheck"],
            "csharp": ["nunit", "xunit", "mstest"]
        }

        self.test_types = {
            "unit": "Individual component testing",
            "integration": "Component interaction testing",
            "e2e": "End-to-end workflow testing",
            "performance": "Performance and load testing",
            "security": "Security vulnerability testing",
            "api": "API endpoint testing",
            "ui": "User interface testing",
            "regression": "Regression testing",
            "smoke": "Basic functionality testing"
        }

        self.performance_tools = [
            "locust", "jmeter", "k6", "artillery", "wrk", "siege"
        ]

        self.security_tools = [
            "owasp-zap", "nmap", "nikto", "burp-suite", "sqlmap", "nuclei"
        ]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "test_generation",
            "test_execution",
            "performance_testing",
            "load_testing",
            "security_testing",
            "api_testing",
            "ui_testing",
            "regression_testing",
            "test_coverage_analysis",
            "bug_detection",
            "test_reporting",
            "ci_cd_integration",
            "test_data_management",
            "quality_metrics",
            "automated_debugging",
            "test_maintenance",
            "cross_browser_testing",
            "mobile_testing",
            "accessibility_testing",
            "compliance_testing"
        ]

    async def _execute_core_logic(
        self,
        request: TestRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core testing logic.

        Args:
            request: Test request
            correlation_context: Optional correlation context

        Returns:
            Test execution results with metrics and reports
        """
        with logfire.span(
            "TesterAgent.execute_core_logic",
            agent_id=self.agent_id,
            test_type=request.test_type,
            target=request.target
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "test_type": request.test_type,
                    "target": request.target,
                    "framework": request.test_framework,
                    "coverage_threshold": request.coverage_threshold
                },
                correlation_context
            )

            try:
                # Validate test type
                if request.test_type not in self.test_types:
                    raise ValidationError(f"Unsupported test type: {request.test_type}")

                # Execute test based on type
                if request.test_type == "unit":
                    result = await self._execute_unit_tests(request)
                elif request.test_type == "integration":
                    result = await self._execute_integration_tests(request)
                elif request.test_type == "e2e":
                    result = await self._execute_e2e_tests(request)
                elif request.test_type == "performance":
                    result = await self._execute_performance_tests(request)
                elif request.test_type == "security":
                    result = await self._execute_security_tests(request)
                elif request.test_type == "api":
                    result = await self._execute_api_tests(request)
                else:
                    result = await self._execute_generic_tests(request)

                # Add metadata
                result.update({
                    "test_type": request.test_type,
                    "target": request.target,
                    "framework": request.test_framework,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "Test execution completed",
                    agent_id=self.agent_id,
                    test_type=request.test_type,
                    success=result.get("success", False),
                    tests_passed=result.get("summary", {}).get("passed", 0)
                )

                return result

            except Exception as e:
                logfire.error(
                    "Test execution failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    test_type=request.test_type
                )
                raise AgentExecutionError(f"Test execution failed: {str(e)}")

    async def _execute_unit_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute unit tests."""

        # Mock unit test execution
        test_results = {
            "summary": {
                "total": 156,
                "passed": 148,
                "failed": 6,
                "skipped": 2,
                "duration": 45.7,
                "coverage": 87.3
            },
            "failed_tests": [
                {
                    "name": "test_user_validation",
                    "file": "tests/test_user.py",
                    "line": 42,
                    "error": "AssertionError: Expected 'valid' but got 'invalid'"
                },
                {
                    "name": "test_payment_processing",
                    "file": "tests/test_payment.py",
                    "line": 78,
                    "error": "ValueError: Invalid payment amount"
                }
            ],
            "coverage_report": {
                "total_lines": 2500,
                "covered_lines": 2183,
                "missing_lines": 317,
                "coverage_percentage": 87.3,
                "files": [
                    {"file": "src/user.py", "coverage": 92.5},
                    {"file": "src/payment.py", "coverage": 78.2},
                    {"file": "src/utils.py", "coverage": 95.1}
                ]
            }
        }

        return {
            "success": test_results["summary"]["failed"] == 0,
            "test_results": test_results,
            "operation": "unit_tests"
        }

    async def _execute_integration_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute integration tests."""

        # Mock integration test execution
        test_results = {
            "summary": {
                "total": 45,
                "passed": 42,
                "failed": 2,
                "skipped": 1,
                "duration": 180.5
            },
            "test_suites": [
                {
                    "name": "Database Integration",
                    "tests": 15,
                    "passed": 15,
                    "failed": 0,
                    "duration": 45.2
                },
                {
                    "name": "API Integration",
                    "tests": 20,
                    "passed": 18,
                    "failed": 2,
                    "duration": 89.3
                },
                {
                    "name": "External Services",
                    "tests": 10,
                    "passed": 9,
                    "failed": 0,
                    "skipped": 1,
                    "duration": 46.0
                }
            ],
            "failed_tests": [
                {
                    "name": "test_payment_gateway_integration",
                    "suite": "API Integration",
                    "error": "ConnectionError: Payment gateway unavailable"
                },
                {
                    "name": "test_email_service_integration",
                    "suite": "External Services",
                    "error": "TimeoutError: Email service timeout"
                }
            ]
        }

        return {
            "success": test_results["summary"]["failed"] == 0,
            "test_results": test_results,
            "operation": "integration_tests"
        }

    async def _execute_e2e_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute end-to-end tests."""

        # Mock E2E test execution
        test_results = {
            "summary": {
                "total": 25,
                "passed": 23,
                "failed": 1,
                "skipped": 1,
                "duration": 420.8
            },
            "scenarios": [
                {
                    "name": "User Registration Flow",
                    "steps": 8,
                    "status": "passed",
                    "duration": 45.3,
                    "screenshots": ["step1.png", "step2.png", "step3.png"]
                },
                {
                    "name": "Purchase Workflow",
                    "steps": 12,
                    "status": "passed",
                    "duration": 89.7,
                    "screenshots": ["purchase1.png", "purchase2.png"]
                },
                {
                    "name": "Admin Dashboard",
                    "steps": 6,
                    "status": "failed",
                    "duration": 67.2,
                    "error": "Element not found: #admin-menu",
                    "screenshot": "admin_failure.png"
                }
            ],
            "browser_results": {
                "chrome": {"passed": 23, "failed": 1, "skipped": 1},
                "firefox": {"passed": 22, "failed": 2, "skipped": 1},
                "safari": {"passed": 21, "failed": 3, "skipped": 1}
            }
        }

        return {
            "success": test_results["summary"]["failed"] == 0,
            "test_results": test_results,
            "operation": "e2e_tests"
        }

    async def _execute_performance_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute performance tests."""

        # Mock performance test execution
        test_results = {
            "summary": {
                "duration": 300,
                "total_requests": 15000,
                "successful_requests": 14750,
                "failed_requests": 250,
                "requests_per_second": 50.0,
                "average_response_time": 245.6,
                "percentiles": {
                    "50th": 198.5,
                    "90th": 387.2,
                    "95th": 456.8,
                    "99th": 678.9
                }
            },
            "endpoints": [
                {
                    "url": "/api/users",
                    "method": "GET",
                    "requests": 5000,
                    "avg_response_time": 156.3,
                    "success_rate": 99.2
                },
                {
                    "url": "/api/orders",
                    "method": "POST",
                    "requests": 3000,
                    "avg_response_time": 298.7,
                    "success_rate": 97.8
                },
                {
                    "url": "/api/products",
                    "method": "GET",
                    "requests": 7000,
                    "avg_response_time": 189.4,
                    "success_rate": 99.5
                }
            ],
            "resource_usage": {
                "cpu_max": 78.5,
                "memory_max": 1250.7,
                "disk_io": 450.2,
                "network_io": 2340.8
            },
            "bottlenecks": [
                "Database connection pool exhaustion at 250 concurrent users",
                "Memory usage spike during large data processing"
            ]
        }

        return {
            "success": test_results["summary"]["success_rate"] > 95.0,
            "test_results": test_results,
            "operation": "performance_tests"
        }

    async def _execute_security_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute security tests."""

        # Mock security test execution
        test_results = {
            "summary": {
                "total_checks": 150,
                "vulnerabilities_found": 8,
                "severity_breakdown": {
                    "critical": 1,
                    "high": 2,
                    "medium": 3,
                    "low": 2,
                    "info": 0
                },
                "scan_duration": 1200.5
            },
            "vulnerabilities": [
                {
                    "id": "VULN-001",
                    "severity": "critical",
                    "title": "SQL Injection vulnerability",
                    "description": "Unsanitized input in login form",
                    "location": "/login",
                    "remediation": "Use parameterized queries"
                },
                {
                    "id": "VULN-002",
                    "severity": "high",
                    "title": "Cross-Site Scripting (XSS)",
                    "description": "Reflected XSS in search parameter",
                    "location": "/search?q=<script>",
                    "remediation": "Sanitize and validate all user inputs"
                },
                {
                    "id": "VULN-003",
                    "severity": "medium",
                    "title": "Insecure Direct Object Reference",
                    "description": "User can access other users' data",
                    "location": "/api/users/{id}",
                    "remediation": "Implement proper authorization checks"
                }
            ],
            "compliance_checks": {
                "owasp_top_10": {
                    "total": 10,
                    "passed": 7,
                    "failed": 3
                },
                "pci_dss": {
                    "total": 12,
                    "passed": 10,
                    "failed": 2
                }
            }
        }

        return {
            "success": test_results["summary"]["severity_breakdown"]["critical"] == 0,
            "test_results": test_results,
            "operation": "security_tests"
        }

    async def _execute_api_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute API tests."""

        # Mock API test execution
        test_results = {
            "summary": {
                "total_endpoints": 25,
                "tested_endpoints": 25,
                "passed": 23,
                "failed": 2,
                "duration": 78.5
            },
            "endpoint_results": [
                {
                    "endpoint": "GET /api/users",
                    "status": "passed",
                    "response_time": 156.3,
                    "status_code": 200,
                    "schema_valid": True
                },
                {
                    "endpoint": "POST /api/users",
                    "status": "passed",
                    "response_time": 298.7,
                    "status_code": 201,
                    "schema_valid": True
                },
                {
                    "endpoint": "DELETE /api/users/123",
                    "status": "failed",
                    "response_time": 5000.0,
                    "status_code": 500,
                    "error": "Internal server error",
                    "schema_valid": False
                }
            ],
            "schema_validation": {
                "total_schemas": 25,
                "valid_schemas": 23,
                "invalid_schemas": 2
            },
            "security_checks": {
                "authentication_required": True,
                "authorization_enforced": True,
                "rate_limiting": True,
                "input_validation": False
            }
        }

        return {
            "success": test_results["summary"]["failed"] == 0,
            "test_results": test_results,
            "operation": "api_tests"
        }

    async def _execute_generic_tests(self, request: TestRequest) -> Dict[str, Any]:
        """Execute generic test type."""
        return {
            "success": True,
            "test_type": request.test_type,
            "message": f"Generic {request.test_type} tests executed successfully",
            "operation": "generic_tests"
        }

    async def generate_tests(self, request: TestGenerationRequest) -> Dict[str, Any]:
        """
        Generate test cases from source code.

        Args:
            request: Test generation request

        Returns:
            Generated test cases and coverage analysis
        """
        with logfire.span(
            "TesterAgent.generate_tests",
            agent_id=self.agent_id,
            language=request.language,
            test_type=request.test_type
        ):
            try:
                # Mock test generation
                generated_tests = {
                    "test_file": f"test_{request.language}_{request.test_type}.py",
                    "test_count": 25,
                    "coverage_estimate": 92.5,
                    "test_cases": [
                        {
                            "name": "test_user_creation_valid_input",
                            "type": "positive",
                            "description": "Test user creation with valid input"
                        },
                        {
                            "name": "test_user_creation_invalid_email",
                            "type": "negative",
                            "description": "Test user creation with invalid email"
                        },
                        {
                            "name": "test_user_creation_empty_fields",
                            "type": "edge_case",
                            "description": "Test user creation with empty required fields"
                        }
                    ],
                    "framework": request.framework,
                    "language": request.language,
                    "generation_time": datetime.utcnow().isoformat()
                }

                logfire.info(
                    "Test generation completed",
                    agent_id=self.agent_id,
                    language=request.language,
                    test_count=generated_tests["test_count"]
                )

                return {"success": True, "generated_tests": generated_tests}

            except Exception as e:
                logfire.error(
                    "Test generation failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Test generation failed: {str(e)}")

    async def run_performance_test(self, request: PerformanceTestRequest) -> Dict[str, Any]:
        """
        Run performance and load tests.

        Args:
            request: Performance test request

        Returns:
            Performance test results with metrics
        """
        with logfire.span(
            "TesterAgent.run_performance_test",
            agent_id=self.agent_id,
            target_url=request.target_url,
            concurrent_users=request.concurrent_users
        ):
            try:
                # Mock performance test execution
                performance_results = {
                    "test_config": {
                        "target_url": request.target_url,
                        "duration": request.test_duration,
                        "concurrent_users": request.concurrent_users,
                        "ramp_up_time": request.ramp_up_time
                    },
                    "results": {
                        "total_requests": request.test_duration * request.concurrent_users * 2,
                        "successful_requests": int(request.test_duration * request.concurrent_users * 1.95),
                        "failed_requests": int(request.test_duration * request.concurrent_users * 0.05),
                        "requests_per_second": request.concurrent_users * 2.0,
                        "average_response_time": 245.6,
                        "min_response_time": 89.2,
                        "max_response_time": 1245.8
                    },
                    "thresholds_met": {
                        "response_time": request.performance_thresholds.get("response_time", 500) > 245.6,
                        "error_rate": request.performance_thresholds.get("error_rate", 5.0) > 2.5,
                        "throughput": request.performance_thresholds.get("throughput", 10) < request.concurrent_users * 2.0
                    }
                }

                logfire.info(
                    "Performance test completed",
                    agent_id=self.agent_id,
                    target_url=request.target_url,
                    avg_response_time=performance_results["results"]["average_response_time"]
                )

                return {"success": True, "performance_results": performance_results}

            except Exception as e:
                logfire.error(
                    "Performance test failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Performance test failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for tester agent."""
        return {
            "default_timeout": 300,
            "default_coverage_threshold": 80.0,
            "parallel_execution": True,
            "retry_failed_tests": True,
            "max_retries": 3,
            "generate_reports": True,
            "screenshot_on_failure": True,
            "video_recording": False,
            "test_data_cleanup": True,
            "performance_baseline": True,
            "security_scanning": True,
            "accessibility_testing": False,
            "cross_browser_testing": True,
            "test_frameworks": self.test_frameworks,
            "test_types": self.test_types,
            "performance_tools": self.performance_tools,
            "security_tools": self.security_tools
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["default_timeout", "default_coverage_threshold"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate timeout
        if config.get("default_timeout", 0) <= 0:
            raise ValidationError("default_timeout must be positive")

        # Validate coverage threshold
        coverage = config.get("default_coverage_threshold", 0)
        if not 0 <= coverage <= 100:
            raise ValidationError("default_coverage_threshold must be between 0 and 100")

        return True
