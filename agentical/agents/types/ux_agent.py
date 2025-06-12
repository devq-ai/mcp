"""
UX Agent Implementation for Agentical Framework

This module provides the UxAgent implementation for user experience design,
analysis, usability testing, and interface optimization operations.

Features:
- User experience analysis and evaluation
- Usability testing coordination and analysis
- Interface design review and optimization
- User journey mapping and analysis
- Accessibility auditing and compliance
- Information architecture evaluation
- Visual design analysis and feedback
- Conversion optimization recommendations
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


class UxAnalysisRequest(BaseModel):
    """Request model for UX analysis operations."""
    interface_type: str = Field(..., description="Type of interface (web, mobile, desktop, voice)")
    analysis_focus: str = Field(..., description="Focus area for analysis")
    user_goals: List[str] = Field(..., description="User goals to evaluate")
    design_artifacts: Optional[List[str]] = Field(default=None, description="Design artifacts to analyze")
    target_audience: Optional[str] = Field(default=None, description="Target audience demographics")
    accessibility_level: str = Field(default="AA", description="WCAG accessibility level to evaluate")


class UsabilityTestRequest(BaseModel):
    """Request model for usability testing operations."""
    test_type: str = Field(..., description="Type of usability test")
    participants_count: int = Field(..., description="Number of test participants")
    test_scenarios: List[str] = Field(..., description="Test scenarios to execute")
    success_metrics: List[str] = Field(..., description="Success metrics to measure")
    test_duration: int = Field(default=60, description="Test duration in minutes")


class DesignReviewRequest(BaseModel):
    """Request model for design review operations."""
    design_files: List[str] = Field(..., description="Design files to review")
    review_criteria: List[str] = Field(..., description="Review criteria and guidelines")
    brand_guidelines: Optional[Dict[str, Any]] = Field(default=None, description="Brand guidelines")
    design_system: Optional[str] = Field(default=None, description="Design system reference")
    target_platform: str = Field(..., description="Target platform for design")


class UxAgent(EnhancedBaseAgent[UxAnalysisRequest, Dict[str, Any]]):
    """
    Specialized agent for user experience design and analysis.

    Capabilities:
    - User experience analysis and evaluation
    - Usability testing coordination
    - Interface design review and optimization
    - User journey mapping and analysis
    - Accessibility auditing and compliance
    - Information architecture evaluation
    - Visual design analysis
    - Conversion optimization
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "UxAgent",
        description: str = "Specialized agent for UX design and analysis",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.UX_AGENT,
            **kwargs
        )

        # UX-specific configuration
        self.interface_types = {
            "web": "Web applications and websites",
            "mobile": "Mobile applications (iOS/Android)",
            "desktop": "Desktop applications",
            "voice": "Voice user interfaces",
            "ar_vr": "Augmented and virtual reality interfaces",
            "iot": "Internet of Things interfaces",
            "kiosk": "Self-service kiosk interfaces"
        }

        self.analysis_focuses = {
            "usability": "Overall usability and ease of use",
            "accessibility": "Accessibility compliance and inclusive design",
            "information_architecture": "Content organization and navigation",
            "visual_design": "Visual hierarchy, typography, and aesthetics",
            "interaction_design": "User interactions and micro-interactions",
            "user_flow": "User journey and task flow optimization",
            "conversion": "Conversion rate optimization",
            "performance": "Perceived performance and loading experience"
        }

        self.usability_metrics = [
            "task_completion_rate", "time_on_task", "error_rate",
            "satisfaction_score", "first_click_accuracy", "navigation_efficiency",
            "learnability", "memorability", "error_recovery"
        ]

        self.accessibility_standards = {
            "WCAG_2.1_A": "Web Content Accessibility Guidelines 2.1 Level A",
            "WCAG_2.1_AA": "Web Content Accessibility Guidelines 2.1 Level AA",
            "WCAG_2.1_AAA": "Web Content Accessibility Guidelines 2.1 Level AAA",
            "Section_508": "Section 508 Rehabilitation Act",
            "ADA": "Americans with Disabilities Act compliance"
        }

        self.design_principles = [
            "clarity", "consistency", "feedback", "efficiency",
            "forgiveness", "accessibility", "simplicity", "discoverability"
        ]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "usability_analysis",
            "accessibility_audit",
            "user_journey_mapping",
            "information_architecture_review",
            "visual_design_analysis",
            "interaction_design_review",
            "user_research_planning",
            "persona_development",
            "wireframe_analysis",
            "prototype_evaluation",
            "design_system_analysis",
            "conversion_optimization",
            "heuristic_evaluation",
            "cognitive_walkthrough",
            "task_analysis",
            "competitive_analysis",
            "a_b_testing_design",
            "user_testing_coordination",
            "design_critique",
            "ux_metrics_analysis"
        ]

    async def _execute_core_logic(
        self,
        request: UxAnalysisRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core UX analysis logic.

        Args:
            request: UX analysis request
            correlation_context: Optional correlation context

        Returns:
            UX analysis results with insights and recommendations
        """
        with logfire.span(
            "UxAgent.execute_core_logic",
            agent_id=self.agent_id,
            interface_type=request.interface_type,
            analysis_focus=request.analysis_focus
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "interface_type": request.interface_type,
                    "analysis_focus": request.analysis_focus,
                    "user_goals": len(request.user_goals),
                    "accessibility_level": request.accessibility_level
                },
                correlation_context
            )

            try:
                # Validate interface type and analysis focus
                if request.interface_type not in self.interface_types:
                    raise ValidationError(f"Unsupported interface type: {request.interface_type}")

                if request.analysis_focus not in self.analysis_focuses:
                    raise ValidationError(f"Unsupported analysis focus: {request.analysis_focus}")

                # Execute analysis based on focus area
                if request.analysis_focus == "usability":
                    result = await self._analyze_usability(request)
                elif request.analysis_focus == "accessibility":
                    result = await self._audit_accessibility(request)
                elif request.analysis_focus == "information_architecture":
                    result = await self._analyze_information_architecture(request)
                elif request.analysis_focus == "visual_design":
                    result = await self._analyze_visual_design(request)
                elif request.analysis_focus == "user_flow":
                    result = await self._analyze_user_flow(request)
                elif request.analysis_focus == "conversion":
                    result = await self._analyze_conversion_optimization(request)
                else:
                    result = await self._perform_comprehensive_analysis(request)

                # Add metadata
                result.update({
                    "interface_type": request.interface_type,
                    "analysis_focus": request.analysis_focus,
                    "user_goals": request.user_goals,
                    "accessibility_level": request.accessibility_level,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "UX analysis completed",
                    agent_id=self.agent_id,
                    interface_type=request.interface_type,
                    analysis_focus=request.analysis_focus,
                    overall_score=result.get("overall_score", 0)
                )

                return result

            except Exception as e:
                logfire.error(
                    "UX analysis failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    interface_type=request.interface_type
                )
                raise AgentExecutionError(f"UX analysis failed: {str(e)}")

    async def _analyze_usability(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Analyze interface usability and user experience."""

        # Mock usability analysis results
        usability_analysis = {
            "overall_score": 7.8,
            "usability_metrics": {
                "task_completion_rate": 87.5,
                "average_task_time": 125,  # seconds
                "error_rate": 6.2,
                "satisfaction_score": 8.1,
                "first_click_accuracy": 78.5,
                "navigation_efficiency": 82.3
            },
            "heuristic_evaluation": {
                "visibility_of_system_status": 8.5,
                "match_system_real_world": 7.8,
                "user_control_freedom": 7.2,
                "consistency_standards": 8.9,
                "error_prevention": 6.8,
                "recognition_vs_recall": 8.1,
                "flexibility_efficiency": 7.5,
                "aesthetic_minimalist_design": 8.7,
                "help_users_errors": 6.9,
                "help_documentation": 7.3
            },
            "user_pain_points": [
                {
                    "area": "Navigation",
                    "severity": "medium",
                    "description": "Users struggle to find secondary menu items",
                    "frequency": "35% of users",
                    "impact": "Increased task completion time"
                },
                {
                    "area": "Form Input",
                    "severity": "low",
                    "description": "Validation messages could be clearer",
                    "frequency": "12% of users",
                    "impact": "Minor confusion during input"
                }
            ],
            "strengths": [
                "Clear visual hierarchy guides user attention effectively",
                "Consistent interaction patterns throughout the interface",
                "Good use of progressive disclosure for complex features",
                "Responsive design works well across devices"
            ],
            "recommendations": [
                "Improve secondary navigation visibility and organization",
                "Enhance form validation feedback with clearer messaging",
                "Add loading states for better perceived performance",
                "Implement breadcrumb navigation for deeper page levels"
            ]
        }

        return {
            "success": True,
            "usability_analysis": usability_analysis,
            "operation": "usability_analysis"
        }

    async def _audit_accessibility(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Audit accessibility compliance and inclusive design."""

        # Mock accessibility audit results
        accessibility_audit = {
            "compliance_level": request.accessibility_level,
            "overall_score": 8.2,
            "wcag_compliance": {
                "level_a": {
                    "total_criteria": 25,
                    "passed": 23,
                    "failed": 2,
                    "not_applicable": 0,
                    "compliance_rate": 92.0
                },
                "level_aa": {
                    "total_criteria": 13,
                    "passed": 11,
                    "failed": 2,
                    "not_applicable": 0,
                    "compliance_rate": 84.6
                }
            },
            "accessibility_issues": [
                {
                    "criterion": "1.4.3 Contrast (Minimum)",
                    "level": "AA",
                    "severity": "medium",
                    "description": "Insufficient color contrast for some text elements",
                    "affected_elements": ["secondary buttons", "help text"],
                    "recommendation": "Increase contrast ratio to at least 4.5:1"
                },
                {
                    "criterion": "2.4.6 Headings and Labels",
                    "level": "AA",
                    "severity": "low",
                    "description": "Some form labels could be more descriptive",
                    "affected_elements": ["search input", "filter dropdown"],
                    "recommendation": "Provide clear, descriptive labels for all form controls"
                }
            ],
            "assistive_technology": {
                "screen_reader_compatibility": 8.5,
                "keyboard_navigation": 9.2,
                "voice_control": 7.8,
                "motor_accessibility": 8.7
            },
            "inclusive_design_score": 8.0,
            "recommendations": [
                "Improve color contrast for better visibility",
                "Add more descriptive labels and instructions",
                "Implement skip navigation links",
                "Ensure all interactive elements are keyboard accessible",
                "Add alternative text for all images and icons"
            ]
        }

        return {
            "success": True,
            "accessibility_audit": accessibility_audit,
            "operation": "accessibility_audit"
        }

    async def _analyze_information_architecture(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Analyze information architecture and content organization."""

        # Mock information architecture analysis
        ia_analysis = {
            "overall_score": 7.5,
            "navigation_structure": {
                "depth_analysis": {
                    "average_depth": 3.2,
                    "maximum_depth": 5,
                    "optimal_range": "2-4 levels",
                    "score": 8.0
                },
                "breadth_analysis": {
                    "average_items_per_level": 6.5,
                    "maximum_items": 9,
                    "optimal_range": "5-9 items",
                    "score": 8.5
                }
            },
            "content_organization": {
                "categorization_clarity": 7.8,
                "labeling_consistency": 8.2,
                "search_findability": 6.9,
                "content_hierarchy": 8.1
            },
            "user_mental_models": {
                "alignment_score": 7.3,
                "category_expectations": "well-aligned",
                "terminology_familiarity": 8.0,
                "conceptual_clarity": 7.6
            },
            "navigation_patterns": {
                "primary_navigation": "clear and consistent",
                "secondary_navigation": "needs improvement",
                "contextual_navigation": "good",
                "search_functionality": "adequate"
            },
            "issues_identified": [
                {
                    "area": "Secondary Navigation",
                    "issue": "Inconsistent placement and visibility",
                    "impact": "medium",
                    "recommendation": "Standardize secondary nav patterns"
                },
                {
                    "area": "Search Results",
                    "issue": "Limited filtering and sorting options",
                    "impact": "low",
                    "recommendation": "Add faceted search capabilities"
                }
            ],
            "recommendations": [
                "Standardize secondary navigation patterns across sections",
                "Improve search functionality with better filtering",
                "Add contextual help and guided discovery",
                "Implement breadcrumb navigation for complex hierarchies"
            ]
        }

        return {
            "success": True,
            "ia_analysis": ia_analysis,
            "operation": "information_architecture_analysis"
        }

    async def _analyze_visual_design(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Analyze visual design elements and aesthetics."""

        # Mock visual design analysis
        visual_analysis = {
            "overall_score": 8.1,
            "design_elements": {
                "color_scheme": {
                    "score": 8.5,
                    "harmony": "excellent",
                    "contrast": "good",
                    "accessibility": "needs improvement",
                    "brand_alignment": "strong"
                },
                "typography": {
                    "score": 8.7,
                    "hierarchy": "clear",
                    "readability": "excellent",
                    "consistency": "good",
                    "font_choices": "appropriate"
                },
                "layout": {
                    "score": 7.8,
                    "grid_system": "consistent",
                    "spacing": "good",
                    "alignment": "excellent",
                    "responsive_behavior": "good"
                },
                "imagery": {
                    "score": 7.5,
                    "quality": "high",
                    "relevance": "good",
                    "consistency": "fair",
                    "optimization": "needs improvement"
                }
            },
            "visual_hierarchy": {
                "primary_elements": "clearly emphasized",
                "secondary_elements": "appropriately de-emphasized",
                "content_scanning": "supports F-pattern reading",
                "call_to_action_prominence": "good"
            },
            "brand_consistency": {
                "score": 8.3,
                "logo_usage": "consistent",
                "color_application": "mostly consistent",
                "voice_and_tone": "aligned",
                "imagery_style": "needs standardization"
            },
            "aesthetic_evaluation": {
                "modern_appeal": 8.2,
                "professional_appearance": 8.7,
                "trustworthiness": 8.1,
                "emotional_connection": 7.4
            },
            "recommendations": [
                "Improve color contrast for accessibility compliance",
                "Standardize image styles and treatments",
                "Optimize images for better performance",
                "Enhance visual feedback for interactive elements",
                "Consider adding subtle animations for better engagement"
            ]
        }

        return {
            "success": True,
            "visual_analysis": visual_analysis,
            "operation": "visual_design_analysis"
        }

    async def _analyze_user_flow(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Analyze user flows and journey optimization."""

        # Mock user flow analysis
        flow_analysis = {
            "overall_score": 7.6,
            "user_journeys": [
                {
                    "journey_name": "New User Registration",
                    "steps": 5,
                    "completion_rate": 78.5,
                    "average_time": "4 minutes",
                    "friction_points": [
                        {
                            "step": "Email Verification",
                            "issue": "Users miss verification email",
                            "impact": "22% drop-off rate"
                        }
                    ],
                    "optimization_score": 7.2
                },
                {
                    "journey_name": "Product Purchase",
                    "steps": 8,
                    "completion_rate": 65.3,
                    "average_time": "7 minutes",
                    "friction_points": [
                        {
                            "step": "Checkout Form",
                            "issue": "Too many required fields",
                            "impact": "15% abandon at checkout"
                        },
                        {
                            "step": "Payment Options",
                            "issue": "Limited payment methods",
                            "impact": "8% prefer alternative payment"
                        }
                    ],
                    "optimization_score": 6.8
                }
            ],
            "flow_efficiency": {
                "average_steps_to_goal": 6.2,
                "optimal_steps": 4.5,
                "efficiency_score": 72.6,
                "redundant_steps": 2
            },
            "conversion_funnels": {
                "awareness_to_interest": 85.2,
                "interest_to_consideration": 68.7,
                "consideration_to_purchase": 45.3,
                "overall_conversion": 26.5
            },
            "user_behavior_patterns": {
                "common_entry_points": ["homepage", "product_pages", "search_results"],
                "exit_patterns": ["checkout_page", "pricing_page", "contact_form"],
                "backtracking_frequency": "moderate",
                "help_seeking_behavior": "low"
            },
            "recommendations": [
                "Simplify registration process with social login options",
                "Reduce checkout form fields to essential information only",
                "Add progress indicators for multi-step processes",
                "Implement exit-intent interventions",
                "Provide contextual help at friction points"
            ]
        }

        return {
            "success": True,
            "flow_analysis": flow_analysis,
            "operation": "user_flow_analysis"
        }

    async def _analyze_conversion_optimization(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Analyze conversion optimization opportunities."""

        # Mock conversion optimization analysis
        conversion_analysis = {
            "overall_score": 7.1,
            "conversion_metrics": {
                "primary_goal_conversion": 3.2,  # percentage
                "secondary_goal_conversion": 12.8,
                "micro_conversion_rate": 18.5,
                "goal_completion_time": "8.5 minutes"
            },
            "cta_analysis": {
                "visibility_score": 8.2,
                "clarity_score": 7.8,
                "urgency_score": 6.5,
                "trust_signals": 7.9,
                "positioning_effectiveness": 8.1
            },
            "form_optimization": {
                "completion_rate": 68.7,
                "abandonment_points": [
                    {"field": "phone_number", "abandonment_rate": 15.2},
                    {"field": "company_size", "abandonment_rate": 8.7}
                ],
                "average_completion_time": "3.2 minutes",
                "error_recovery_rate": 82.3
            },
            "trust_factors": {
                "security_indicators": 8.5,
                "social_proof": 6.8,
                "testimonials_impact": 7.2,
                "guarantee_prominence": 7.9,
                "contact_information_clarity": 8.7
            },
            "optimization_opportunities": [
                {
                    "area": "Call-to-Action Buttons",
                    "current_performance": "7.8/10",
                    "improvement_potential": "15-20% increase",
                    "recommended_changes": [
                        "Use more action-oriented copy",
                        "Increase button size and contrast",
                        "Add urgency indicators"
                    ]
                },
                {
                    "area": "Form Fields",
                    "current_performance": "6.9/10",
                    "improvement_potential": "25-30% increase",
                    "recommended_changes": [
                        "Reduce required fields",
                        "Implement progressive disclosure",
                        "Add field validation feedback"
                    ]
                }
            ],
            "a_b_test_suggestions": [
                "Test different CTA button colors and text",
                "Compare single-page vs multi-step forms",
                "Test social proof placement and format",
                "Experiment with pricing presentation"
            ]
        }

        return {
            "success": True,
            "conversion_analysis": conversion_analysis,
            "operation": "conversion_optimization_analysis"
        }

    async def _perform_comprehensive_analysis(self, request: UxAnalysisRequest) -> Dict[str, Any]:
        """Perform comprehensive UX analysis across all areas."""

        # Mock comprehensive analysis combining all focus areas
        comprehensive_analysis = {
            "overall_score": 7.7,
            "analysis_summary": {
                "usability_score": 7.8,
                "accessibility_score": 8.2,
                "visual_design_score": 8.1,
                "information_architecture_score": 7.5,
                "user_flow_score": 7.6,
                "conversion_score": 7.1
            },
            "priority_improvements": [
                {
                    "priority": 1,
                    "area": "Conversion Optimization",
                    "impact": "High",
                    "effort": "Medium",
                    "description": "Optimize checkout process and CTAs"
                },
                {
                    "priority": 2,
                    "area": "Information Architecture",
                    "impact": "Medium",
                    "effort": "High",
                    "description": "Restructure secondary navigation"
                },
                {
                    "priority": 3,
                    "area": "Accessibility",
                    "impact": "Medium",
                    "effort": "Low",
                    "description": "Improve color contrast and labels"
                }
            ],
            "user_satisfaction_prediction": {
                "current_estimated_satisfaction": 7.4,
                "potential_satisfaction": 8.6,
                "improvement_opportunity": 16.2  # percentage
            }
        }

        return {
            "success": True,
            "comprehensive_analysis": comprehensive_analysis,
            "operation": "comprehensive_ux_analysis"
        }

    async def conduct_usability_test(self, request: UsabilityTestRequest) -> Dict[str, Any]:
        """
        Conduct usability testing and analyze results.

        Args:
            request: Usability test request

        Returns:
            Usability test results and insights
        """
        with logfire.span(
            "UxAgent.conduct_usability_test",
            agent_id=self.agent_id,
            test_type=request.test_type,
            participants_count=request.participants_count
        ):
            try:
                # Mock usability test results
                test_results = {
                    "test_overview": {
                        "test_type": request.test_type,
                        "participants": request.participants_count,
                        "scenarios_tested": len(request.test_scenarios),
                        "success_metrics": request.success_metrics,
                        "test_duration": request.test_duration
                    },
                    "quantitative_results": {
                        "overall_success_rate": 82.5,
                        "average_task_completion_time": 145,  # seconds
                        "error_rate": 8.3,
                        "satisfaction_rating": 7.8,
                        "ease_of_use_rating": 8.1
                    },
                    "qualitative_insights": {
                        "positive_feedback": [
                            "Interface is clean and intuitive",
                            "Navigation makes sense",
                            "Good visual feedback for actions"
                        ],
                        "negative_feedback": [
                            "Some features are hard to discover",
                            "Error messages could be clearer",
                            "Mobile experience needs improvement"
                        ],
                        "suggestions": [
                            "Add onboarding tour for new users",
                            "Improve search functionality",
                            "Better mobile responsiveness"
                        ]
                    },
                    "behavior_observations": {
                        "common_user_paths": ["homepage -> search -> results -> detail"],
                        "frequent_errors": ["clicking wrong navigation item", "form validation issues"],
                        "hesitation_points": ["checkout process", "account creation"],
                        "abandonment_triggers": ["complex forms", "unclear pricing"]
                    },
                    "recommendations": [
                        "Implement progressive onboarding",
                        "Enhance mobile user experience",
                        "Simplify checkout and registration flows",
                        "Improve feature discoverability",
                        "Add contextual help and guidance"
                    ]
                }

                logfire.info(
                    "Usability test completed",
                    agent_id=self.agent_id,
                    test_type=request.test_type,
                    success_rate=test_results["quantitative_results"]["overall_success_rate"]
                )

                return {"success": True, "usability_test_results": test_results}

            except Exception as e:
                logfire.error(
                    "Usability test failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Usability test failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for UX agent."""
        return {
            "analysis_depth": "comprehensive",
            "include_accessibility_audit": True,
            "generate_wireframes": False,
            "user_testing_integration": True,
            "analytics_integration": True,
            "design_system_validation": True,
            "mobile_first_analysis": True,
            "conversion_optimization_focus": True,
            "a_b_testing_recommendations": True,
            "heuristic_evaluation_enabled": True,
            "cognitive_walkthrough_enabled": True,
            "persona_based_analysis": True,
            "competitive_benchmarking": False,
            "interface_types": self.interface_types,
            "analysis_focuses": self.analysis_focuses,
            "usability_metrics": self.usability_metrics,
            "accessibility_standards": self.accessibility_standards,
            "design_principles": self.design_principles
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["analysis_depth"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate analysis depth
        valid_depths = ["basic", "standard", "comprehensive"]
        if config.get("analysis_depth") not in valid_depths:
            raise ValidationError(f"Invalid analysis depth. Must be one of: {valid_depths}")

        return True
