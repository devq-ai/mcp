"""
UAT Agent Implementation for Agentical Framework

This module provides the UatAgent implementation for user acceptance testing
coordination, test planning, execution monitoring, and stakeholder management.

Features:
- UAT test planning and scenario design
- Test execution coordination and monitoring
- User feedback collection and analysis
- Defect tracking and resolution
- Stakeholder communication and reporting
- Test environment management
- Acceptance criteria validation
- Sign-off process automation
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime
import asyncio
import json
from pathlib import Path

import logfire
from pydantic import BaseModel, Field

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class UatTestRequest(BaseModel):
    """Request model for UAT operations."""
    application: str = Field(..., description="Application under test")
    test_scenarios: List[str] = Field(..., description="Test scenarios to execute")
    user_personas: List[str] = Field(..., description="User personas for testing")
    acceptance_criteria: List[str] = Field(..., description="Acceptance criteria")
    test_duration: int = Field(default=5, description="Test duration in days")
    environment: str = Field(default="uat", description="Test environment")


class FeedbackCollectionRequest(BaseModel):
    """Request model for feedback collection operations."""
    test_session_id: str = Field(..., description="Test session identifier")
    feedback_methods: List[str] = Field(..., description="Methods for collecting feedback")
    participants: List[str] = Field(..., description="Test participants")
    survey_questions: Optional[List[str]] = Field(default=None, description="Survey questions")


class DefectTrackingRequest(BaseModel):
    """Request model for defect tracking operations."""
    defect_id: Optional[str] = Field(default=None, description="Defect identifier")
    defect_data: Dict[str, Any] = Field(..., description="Defect information")
    severity: str = Field(..., description="Defect severity")
    priority: str = Field(..., description="Defect priority")
    assigned_to: Optional[str] = Field(default=None, description="Assignee")


class UatAgent(EnhancedBaseAgent[UatTestRequest, Dict[str, Any]]):
    """
    Specialized agent for user acceptance testing coordination and management.

    Capabilities:
    - UAT test planning and design
    - Test execution coordination
    - User feedback collection and analysis
    - Defect tracking and management
    - Stakeholder communication
    - Test environment coordination
    - Acceptance validation
    - Sign-off process management
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "UatAgent",
        description: str = "Specialized agent for user acceptance testing coordination",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.UAT_AGENT,
            **kwargs
        )

        # UAT-specific configuration
        self.test_types = {
            "functional": "Functional requirement testing",
            "usability": "User experience and usability testing",
            "workflow": "Business workflow validation",
            "integration": "End-to-end integration testing",
            "regression": "Regression testing of existing features",
            "performance": "User-perceived performance testing"
        }

        self.user_roles = {
            "end_user": "Primary application users",
            "admin": "System administrators",
            "manager": "Business managers and supervisors",
            "guest": "Guest or anonymous users",
            "power_user": "Advanced users with special permissions",
            "stakeholder": "Business stakeholders and decision makers"
        }

        self.feedback_methods = [
            "surveys", "interviews", "focus_groups", "usability_sessions",
            "bug_reports", "feature_requests", "satisfaction_ratings"
        ]

        self.severity_levels = ["critical", "high", "medium", "low", "cosmetic"]
        self.priority_levels = ["urgent", "high", "medium", "low", "deferred"]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "test_planning",
            "scenario_design",
            "user_coordination",
            "feedback_collection",
            "defect_tracking",
            "acceptance_validation",
            "test_execution_monitoring",
            "stakeholder_reporting",
            "stakeholder_communication",
            "test_data_management",
            "environment_coordination",
            "sign_off_management",
            "requirement_validation",
            "business_process_testing",
            "user_training_coordination",
            "documentation_validation",
            "compliance_testing",
            "accessibility_validation",
            "cross_browser_testing",
            "mobile_uat_testing"
        ]

    async def _execute_core_logic(
        self,
        request: UatTestRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core UAT logic.

        Args:
            request: UAT test request
            correlation_context: Optional correlation context

        Returns:
            UAT execution results with test outcomes and recommendations
        """
        with logfire.span(
            "UatAgent.execute_core_logic",
            agent_id=self.agent_id,
            application=request.application,
            test_scenarios_count=len(request.test_scenarios)
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "application": request.application,
                    "test_scenarios": len(request.test_scenarios),
                    "user_personas": len(request.user_personas),
                    "acceptance_criteria": len(request.acceptance_criteria),
                    "test_duration": request.test_duration
                },
                correlation_context
            )

            try:
                # Execute UAT coordination
                result = await self._coordinate_uat_execution(request)

                # Add metadata
                result.update({
                    "application": request.application,
                    "test_duration": request.test_duration,
                    "environment": request.environment,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "UAT execution completed",
                    agent_id=self.agent_id,
                    application=request.application,
                    scenarios_passed=result.get("test_summary", {}).get("scenarios_passed", 0),
                    scenarios_failed=result.get("test_summary", {}).get("scenarios_failed", 0)
                )

                return result

            except Exception as e:
                logfire.error(
                    "UAT execution failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    application=request.application
                )
                raise AgentExecutionError(f"UAT execution failed: {str(e)}")

    async def _coordinate_uat_execution(self, request: UatTestRequest) -> Dict[str, Any]:
        """Coordinate UAT test execution across scenarios and personas."""

        # Mock UAT execution results
        scenarios_passed = max(0, len(request.test_scenarios) - 2)
        scenarios_failed = min(2, len(request.test_scenarios))

        uat_results = {
            "test_summary": {
                "total_scenarios": len(request.test_scenarios),
                "scenarios_passed": scenarios_passed,
                "scenarios_failed": scenarios_failed,
                "user_personas_tested": len(request.user_personas),
                "completion_rate": (scenarios_passed / len(request.test_scenarios)) * 100 if request.test_scenarios else 0,
                "test_duration_actual": request.test_duration,
                "environment": request.environment
            },
            "scenario_results": [],
            "defects_found": [],
            "user_feedback": {
                "overall_satisfaction": 7.8,
                "ease_of_use": 8.2,
                "functionality_rating": 7.5,
                "performance_rating": 6.9,
                "suggestions_count": 12
            },
            "acceptance_status": "conditional" if scenarios_failed > 0 else "accepted",
            "recommendations": []
        }

        # Generate scenario results
        for i, scenario in enumerate(request.test_scenarios):
            persona = request.user_personas[i % len(request.user_personas)]
            status = "failed" if i < scenarios_failed else "passed"

            scenario_result = {
                "scenario": scenario,
                "status": status,
                "user_persona": persona,
                "execution_time": f"{20 + i * 5} minutes",
                "feedback": self._generate_scenario_feedback(scenario, status),
                "issues": self._generate_scenario_issues(scenario, status)
            }
            uat_results["scenario_results"].append(scenario_result)

        # Generate defects for failed scenarios
        for i in range(scenarios_failed):
            defect = {
                "id": f"UAT-{i+1:03d}",
                "severity": "medium" if i % 2 == 0 else "low",
                "priority": "high" if i == 0 else "medium",
                "description": f"Issue found in scenario: {request.test_scenarios[i]}",
                "scenario": request.test_scenarios[i],
                "user_persona": request.user_personas[i % len(request.user_personas)],
                "steps_to_reproduce": [
                    "Navigate to the application",
                    "Follow the test scenario steps",
                    "Observe the issue"
                ],
                "expected_result": "Scenario should complete successfully",
                "actual_result": "Error or unexpected behavior encountered"
            }
            uat_results["defects_found"].append(defect)

        # Generate recommendations
        if scenarios_failed > 0:
            uat_results["recommendations"].extend([
                "Address critical and high-priority defects before release",
                "Re-test failed scenarios after defect resolution",
                "Consider additional user training for complex workflows"
            ])

        if uat_results["user_feedback"]["performance_rating"] < 7.0:
            uat_results["recommendations"].append("Investigate and improve application performance")

        uat_results["recommendations"].extend([
            "Document user feedback for future enhancements",
            "Schedule follow-up UAT session if needed",
            "Prepare user documentation and training materials"
        ])

        return {
            "success": True,
            "uat_results": uat_results,
            "operation": "uat_execution"
        }

    def _generate_scenario_feedback(self, scenario: str, status: str) -> str:
        """Generate realistic feedback for test scenarios."""
        if status == "passed":
            feedbacks = [
                "Scenario completed smoothly as expected",
                "User interface is intuitive for this workflow",
                "Good performance and responsiveness",
                "All functionality works as described"
            ]
        else:
            feedbacks = [
                "Encountered unexpected error during execution",
                "Some functionality doesn't work as expected",
                "Performance issues observed",
                "User interface could be improved"
            ]

        import random
        return random.choice(feedbacks)

    def _generate_scenario_issues(self, scenario: str, status: str) -> List[str]:
        """Generate issues for test scenarios."""
        if status == "passed":
            return []

        issues = [
            "Performance slower than expected",
            "Error message not user-friendly",
            "Navigation could be clearer",
            "Missing validation feedback"
        ]

        import random
        return random.sample(issues, min(2, len(issues)))

    async def collect_feedback(self, request: FeedbackCollectionRequest) -> Dict[str, Any]:
        """
        Collect and analyze user feedback from UAT sessions.

        Args:
            request: Feedback collection request

        Returns:
            Feedback collection and analysis results
        """
        with logfire.span(
            "UatAgent.collect_feedback",
            agent_id=self.agent_id,
            test_session_id=request.test_session_id
        ):
            try:
                # Mock feedback collection results
                feedback_results = {
                    "test_session_id": request.test_session_id,
                    "participants_count": len(request.participants),
                    "feedback_methods_used": request.feedback_methods,
                    "response_rate": 85.7,  # percentage
                    "feedback_summary": {
                        "total_responses": 24,
                        "positive_feedback": 18,
                        "negative_feedback": 4,
                        "neutral_feedback": 2,
                        "suggestions": 12
                    },
                    "sentiment_analysis": {
                        "overall_sentiment": "positive",
                        "satisfaction_score": 7.8,
                        "likelihood_to_recommend": 8.2,
                        "ease_of_use_rating": 7.5
                    },
                    "common_themes": [
                        {
                            "theme": "User Interface",
                            "mentions": 15,
                            "sentiment": "positive",
                            "sample_comment": "The interface is clean and intuitive"
                        },
                        {
                            "theme": "Performance",
                            "mentions": 8,
                            "sentiment": "negative",
                            "sample_comment": "Some operations take longer than expected"
                        },
                        {
                            "theme": "Functionality",
                            "mentions": 12,
                            "sentiment": "positive",
                            "sample_comment": "All features work as expected"
                        }
                    ],
                    "improvement_suggestions": [
                        "Improve loading times for data-heavy operations",
                        "Add more keyboard shortcuts for power users",
                        "Enhance mobile responsiveness",
                        "Provide better error messages"
                    ]
                }

                logfire.info(
                    "Feedback collection completed",
                    agent_id=self.agent_id,
                    test_session_id=request.test_session_id,
                    total_responses=feedback_results["feedback_summary"]["total_responses"]
                )

                return {"success": True, "feedback_results": feedback_results}

            except Exception as e:
                logfire.error(
                    "Feedback collection failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Feedback collection failed: {str(e)}")

    async def track_defects(self, request: DefectTrackingRequest) -> Dict[str, Any]:
        """
        Track and manage defects found during UAT.

        Args:
            request: Defect tracking request

        Returns:
            Defect tracking and management results
        """
        with logfire.span(
            "UatAgent.track_defects",
            agent_id=self.agent_id,
            defect_id=request.defect_id,
            severity=request.severity
        ):
            try:
                # Mock defect tracking results
                if request.defect_id:
                    # Update existing defect
                    defect_results = {
                        "action": "update",
                        "defect_id": request.defect_id,
                        "status": "updated",
                        "updated_fields": list(request.defect_data.keys()),
                        "assigned_to": request.assigned_to,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                else:
                    # Create new defect
                    new_defect_id = f"UAT-{datetime.utcnow().strftime('%Y%m%d')}-{hash(str(request.defect_data)) % 1000:03d}"
                    defect_results = {
                        "action": "create",
                        "defect_id": new_defect_id,
                        "status": "created",
                        "severity": request.severity,
                        "priority": request.priority,
                        "assigned_to": request.assigned_to,
                        "created_date": datetime.utcnow().isoformat(),
                        "estimated_resolution": "3-5 business days"
                    }

                defect_results.update({
                    "defect_data": request.defect_data,
                    "workflow_stage": "triage" if not request.defect_id else "in_progress",
                    "stakeholders_notified": True
                })

                logfire.info(
                    "Defect tracking completed",
                    agent_id=self.agent_id,
                    action=defect_results["action"],
                    defect_id=defect_results["defect_id"]
                )

                return {"success": True, "defect_results": defect_results}

            except Exception as e:
                logfire.error(
                    "Defect tracking failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Defect tracking failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for UAT agent."""
        return {
            "default_test_duration": 5,  # days
            "parallel_testing": True,
            "feedback_collection_method": "digital",
            "defect_tracking_integration": True,
            "automated_reporting": True,
            "stakeholder_notifications": True,
            "test_environment_management": True,
            "sign_off_workflow": True,
            "user_training_coordination": True,
            "documentation_validation": True,
            "accessibility_testing": False,
            "mobile_testing": True,
            "cross_browser_testing": True,
            "test_types": self.test_types,
            "user_roles": self.user_roles,
            "feedback_methods": self.feedback_methods,
            "severity_levels": self.severity_levels,
            "priority_levels": self.priority_levels
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["default_test_duration", "feedback_collection_method"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate test duration
        if config.get("default_test_duration", 0) <= 0:
            raise ValidationError("default_test_duration must be positive")

        # Validate feedback collection method
        valid_methods = ["digital", "manual", "hybrid"]
        if config.get("feedback_collection_method") not in valid_methods:
            raise ValidationError(f"Invalid feedback collection method. Must be one of: {valid_methods}")

        return True
