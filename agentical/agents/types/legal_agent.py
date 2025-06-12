"""
Legal Agent Implementation for Agentical Framework

This module provides the LegalAgent implementation for legal document analysis,
compliance checking, contract review, and legal research operations.

Features:
- Legal document analysis and review
- Contract management and analysis
- Compliance monitoring and reporting
- Legal research and case law analysis
- Risk assessment and mitigation
- Regulatory compliance checking
- Legal workflow automation
- Due diligence support
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


class LegalDocumentRequest(BaseModel):
    """Request model for legal document operations."""
    document_type: str = Field(..., description="Type of legal document")
    document_content: str = Field(..., description="Document content or path")
    analysis_type: str = Field(..., description="Type of analysis (review, compliance, risk)")
    jurisdiction: str = Field(default="US", description="Legal jurisdiction")
    practice_area: Optional[str] = Field(default=None, description="Legal practice area")
    urgency_level: str = Field(default="normal", description="Urgency level (low, normal, high, critical)")


class ContractAnalysisRequest(BaseModel):
    """Request model for contract analysis."""
    contract_text: str = Field(..., description="Contract text to analyze")
    contract_type: str = Field(..., description="Type of contract")
    party_role: str = Field(..., description="Role of requesting party (buyer, seller, vendor)")
    key_terms: Optional[List[str]] = Field(default=None, description="Key terms to focus on")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance level")
    compliance_frameworks: Optional[List[str]] = Field(default=None, description="Compliance frameworks")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""
    entity_type: str = Field(..., description="Type of entity (corporation, partnership, etc.)")
    industry: str = Field(..., description="Industry sector")
    regulations: List[str] = Field(..., description="Applicable regulations")
    jurisdiction: str = Field(..., description="Legal jurisdiction")
    assessment_scope: str = Field(..., description="Scope of compliance assessment")
    current_policies: Optional[Dict[str, Any]] = Field(default=None, description="Current policies")


class LegalAgent(EnhancedBaseAgent[LegalDocumentRequest, Dict[str, Any]]):
    """
    Specialized agent for legal document analysis and compliance operations.

    Capabilities:
    - Legal document review and analysis
    - Contract management and risk assessment
    - Compliance monitoring and reporting
    - Legal research and case analysis
    - Regulatory compliance checking
    - Due diligence support
    - Legal workflow automation
    - Risk mitigation strategies
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "LegalAgent",
        description: str = "Specialized agent for legal operations and compliance",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.LEGAL_AGENT,
            **kwargs
        )

        # Legal-specific configuration
        self.document_types = {
            "contracts": ["employment", "service", "sales", "licensing", "nda"],
            "corporate": ["articles", "bylaws", "board_resolutions", "securities"],
            "litigation": ["complaints", "motions", "briefs", "depositions"],
            "regulatory": ["filings", "permits", "licenses", "compliance_reports"],
            "intellectual_property": ["patents", "trademarks", "copyrights", "trade_secrets"]
        }

        self.practice_areas = [
            "corporate_law", "contract_law", "employment_law", "intellectual_property",
            "litigation", "regulatory_compliance", "tax_law", "real_estate",
            "securities_law", "merger_acquisition", "privacy_law", "environmental_law"
        ]

        self.jurisdictions = [
            "US_federal", "california", "new_york", "delaware", "texas",
            "EU", "UK", "canada", "australia", "singapore"
        ]

        self.compliance_frameworks = [
            "gdpr", "ccpa", "sox", "pci_dss", "hipaa", "ferpa",
            "iso_27001", "nist", "coso", "basel_iii"
        ]

        self.risk_levels = ["low", "medium", "high", "critical"]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "document_review",
            "contract_analysis",
            "compliance_monitoring",
            "legal_research",
            "risk_assessment",
            "due_diligence",
            "regulatory_analysis",
            "policy_development",
            "legal_workflow_automation",
            "case_law_research",
            "document_drafting",
            "clause_analysis",
            "liability_assessment",
            "intellectual_property_analysis",
            "merger_due_diligence",
            "employment_law_compliance",
            "privacy_compliance",
            "litigation_support",
            "contract_negotiation_support",
            "regulatory_filing_assistance"
        ]

    async def _execute_core_logic(
        self,
        request: LegalDocumentRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core legal analysis logic.

        Args:
            request: Legal document request
            correlation_context: Optional correlation context

        Returns:
            Legal analysis results with findings and recommendations
        """
        with logfire.span(
            "LegalAgent.execute_core_logic",
            agent_id=self.agent_id,
            document_type=request.document_type,
            analysis_type=request.analysis_type
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "document_type": request.document_type,
                    "analysis_type": request.analysis_type,
                    "jurisdiction": request.jurisdiction,
                    "practice_area": request.practice_area
                },
                correlation_context
            )

            try:
                # Validate document type
                all_doc_types = [doc for docs in self.document_types.values() for doc in docs]
                if request.document_type not in all_doc_types:
                    # Allow for generic document types not in predefined list
                    pass

                # Execute analysis based on type
                if request.analysis_type == "review":
                    result = await self._review_document(request)
                elif request.analysis_type == "compliance":
                    result = await self._check_compliance(request)
                elif request.analysis_type == "risk":
                    result = await self._assess_risk(request)
                elif request.analysis_type == "contract_analysis":
                    result = await self._analyze_contract(request)
                else:
                    result = await self._perform_general_analysis(request)

                # Add metadata
                result.update({
                    "document_type": request.document_type,
                    "analysis_type": request.analysis_type,
                    "jurisdiction": request.jurisdiction,
                    "practice_area": request.practice_area,
                    "urgency_level": request.urgency_level,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "Legal analysis completed",
                    agent_id=self.agent_id,
                    document_type=request.document_type,
                    analysis_type=request.analysis_type,
                    risk_level=result.get("risk_assessment", {}).get("overall_risk", "unknown")
                )

                return result

            except Exception as e:
                logfire.error(
                    "Legal analysis failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    document_type=request.document_type
                )
                raise AgentExecutionError(f"Legal analysis failed: {str(e)}")

    async def _review_document(self, request: LegalDocumentRequest) -> Dict[str, Any]:
        """Review legal document for issues and recommendations."""

        # Mock document review results
        review_results = {
            "document_summary": {
                "word_count": 2450,
                "page_count": 12,
                "sections": 8,
                "complexity_score": 7.5,
                "readability_score": 6.2
            },
            "key_findings": [
                {
                    "section": "Termination Clause",
                    "issue": "Ambiguous termination conditions",
                    "severity": "medium",
                    "recommendation": "Clarify specific termination triggers"
                },
                {
                    "section": "Liability Limitation",
                    "issue": "Excessive liability exposure",
                    "severity": "high",
                    "recommendation": "Add mutual liability caps"
                }
            ],
            "compliance_issues": [
                {
                    "regulation": "State contract law",
                    "issue": "Missing required disclosures",
                    "severity": "medium",
                    "citation": "Section 123.45 of State Code"
                }
            ],
            "missing_clauses": [
                "Force majeure provision",
                "Dispute resolution mechanism",
                "Governing law clause"
            ],
            "risk_indicators": [
                "Unlimited liability exposure",
                "Vague performance standards",
                "Inadequate intellectual property protection"
            ]
        }

        return {
            "success": True,
            "review_results": review_results,
            "operation": "document_review"
        }

    async def _check_compliance(self, request: LegalDocumentRequest) -> Dict[str, Any]:
        """Check document compliance with applicable regulations."""

        # Mock compliance check results
        compliance_results = {
            "overall_compliance": "Partial",
            "compliance_score": 7.2,
            "frameworks_checked": ["GDPR", "CCPA", "SOX"],
            "compliant_areas": [
                {
                    "area": "Data protection disclosures",
                    "framework": "GDPR",
                    "status": "compliant",
                    "confidence": 0.95
                },
                {
                    "area": "Financial reporting requirements",
                    "framework": "SOX",
                    "status": "compliant",
                    "confidence": 0.88
                }
            ],
            "non_compliant_areas": [
                {
                    "area": "Consumer rights notifications",
                    "framework": "CCPA",
                    "status": "non_compliant",
                    "severity": "medium",
                    "remediation": "Add required privacy policy language"
                }
            ],
            "recommendations": [
                "Update privacy policy language per CCPA requirements",
                "Add data retention policy section",
                "Include consumer rights contact information"
            ]
        }

        return {
            "success": True,
            "compliance_results": compliance_results,
            "operation": "compliance_check"
        }

    async def _assess_risk(self, request: LegalDocumentRequest) -> Dict[str, Any]:
        """Assess legal and business risks in document."""

        # Mock risk assessment results
        risk_assessment = {
            "overall_risk": "Medium",
            "risk_score": 6.5,
            "risk_categories": {
                "financial_risk": {
                    "level": "high",
                    "score": 8.2,
                    "factors": ["Unlimited liability", "Payment terms unfavorable"]
                },
                "operational_risk": {
                    "level": "medium",
                    "score": 5.8,
                    "factors": ["Vague deliverables", "Tight timelines"]
                },
                "legal_risk": {
                    "level": "medium",
                    "score": 6.1,
                    "factors": ["Governing law unclear", "Dispute resolution undefined"]
                },
                "reputational_risk": {
                    "level": "low",
                    "score": 3.2,
                    "factors": ["Standard confidentiality provisions"]
                }
            },
            "critical_risks": [
                {
                    "risk": "Unlimited liability exposure",
                    "impact": "High financial loss potential",
                    "probability": "Medium",
                    "mitigation": "Negotiate liability caps"
                }
            ],
            "risk_mitigation_strategies": [
                "Add liability limitation clauses",
                "Include force majeure provisions",
                "Clarify performance standards",
                "Add termination rights"
            ]
        }

        return {
            "success": True,
            "risk_assessment": risk_assessment,
            "operation": "risk_assessment"
        }

    async def _analyze_contract(self, request: LegalDocumentRequest) -> Dict[str, Any]:
        """Perform comprehensive contract analysis."""

        # Mock contract analysis results
        contract_analysis = {
            "contract_summary": {
                "parties": ["Company A", "Company B"],
                "contract_value": "$500,000",
                "term": "24 months",
                "key_obligations": {
                    "party_a": ["Provide services", "Meet SLA requirements"],
                    "party_b": ["Make payments", "Provide access"]
                }
            },
            "favorable_terms": [
                {
                    "clause": "Payment terms",
                    "benefit": "Net 30 payment terms favor cash flow",
                    "significance": "medium"
                }
            ],
            "unfavorable_terms": [
                {
                    "clause": "Liability provisions",
                    "issue": "Disproportionate liability allocation",
                    "significance": "high",
                    "negotiation_priority": 1
                }
            ],
            "negotiation_points": [
                {
                    "clause": "Termination rights",
                    "current": "90-day notice required",
                    "recommended": "30-day notice with cause",
                    "rationale": "More flexible exit strategy"
                }
            ],
            "deal_breakers": [
                "Unlimited indemnification obligations"
            ],
            "approval_recommendation": "Conditional - subject to negotiation of liability terms"
        }

        return {
            "success": True,
            "contract_analysis": contract_analysis,
            "operation": "contract_analysis"
        }

    async def _perform_general_analysis(self, request: LegalDocumentRequest) -> Dict[str, Any]:
        """Perform general legal document analysis."""

        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "document_type": request.document_type,
            "general_findings": [
                f"Document appears to be a standard {request.document_type}",
                "Overall structure follows conventional format",
                "Recommend detailed review by subject matter expert"
            ],
            "operation": "general_analysis"
        }

    async def analyze_contract(self, request: ContractAnalysisRequest) -> Dict[str, Any]:
        """
        Perform detailed contract analysis with risk assessment.

        Args:
            request: Contract analysis request

        Returns:
            Comprehensive contract analysis results
        """
        with logfire.span(
            "LegalAgent.analyze_contract",
            agent_id=self.agent_id,
            contract_type=request.contract_type
        ):
            try:
                # Mock comprehensive contract analysis
                analysis_results = {
                    "executive_summary": {
                        "contract_type": request.contract_type,
                        "party_role": request.party_role,
                        "overall_assessment": "Favorable with modifications",
                        "key_concerns": 2,
                        "negotiation_items": 5
                    },
                    "clause_analysis": [
                        {
                            "clause_type": "Payment Terms",
                            "current_terms": "Net 45 days",
                            "assessment": "Unfavorable",
                            "recommendation": "Negotiate to Net 30",
                            "priority": "High"
                        },
                        {
                            "clause_type": "Intellectual Property",
                            "current_terms": "Work for hire",
                            "assessment": "Standard",
                            "recommendation": "Add IP warranty",
                            "priority": "Medium"
                        }
                    ],
                    "risk_analysis": {
                        "financial_exposure": "$2.5M maximum",
                        "performance_risk": "Medium",
                        "legal_risk": "Low",
                        "mitigation_required": True
                    }
                }

                logfire.info(
                    "Contract analysis completed",
                    agent_id=self.agent_id,
                    contract_type=request.contract_type,
                    overall_assessment=analysis_results["executive_summary"]["overall_assessment"]
                )

                return {"success": True, "contract_analysis": analysis_results}

            except Exception as e:
                logfire.error(
                    "Contract analysis failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Contract analysis failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for legal agent."""
        return {
            "default_jurisdiction": "US_federal",
            "risk_tolerance": "medium",
            "compliance_frameworks": self.compliance_frameworks,
            "review_depth": "comprehensive",
            "include_precedent_analysis": True,
            "generate_redlines": True,
            "provide_negotiation_guidance": True,
            "citation_format": "bluebook",
            "confidentiality_protection": "high",
            "document_retention_days": 2555,  # 7 years
            "auto_compliance_checking": True,
            "risk_scoring_enabled": True,
            "document_types": self.document_types,
            "practice_areas": self.practice_areas,
            "jurisdictions": self.jurisdictions,
            "risk_levels": self.risk_levels
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["default_jurisdiction", "risk_tolerance", "review_depth"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate jurisdiction
        if config.get("default_jurisdiction") not in self.jurisdictions:
            raise ValidationError(f"Unsupported jurisdiction: {config.get('default_jurisdiction')}")

        # Validate risk tolerance
        if config.get("risk_tolerance") not in self.risk_levels:
            raise ValidationError(f"Invalid risk tolerance: {config.get('risk_tolerance')}")

        # Validate retention period
        if config.get("document_retention_days", 0) <= 0:
            raise ValidationError("document_retention_days must be positive")

        return True
