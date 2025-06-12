"""
InfoSec Agent Implementation for Agentical Framework

This module provides the InfoSecAgent implementation for security analysis,
threat assessment, vulnerability scanning, and security compliance operations.

Features:
- Vulnerability scanning and assessment
- Penetration testing coordination
- Security compliance checking
- Threat analysis and monitoring
- Incident response support
- Security audit automation
- Risk assessment and mitigation
- Security policy validation
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


class SecurityScanRequest(BaseModel):
    """Request model for security scanning operations."""
    target: str = Field(..., description="Target for security scan (IP, domain, application)")
    scan_type: str = Field(..., description="Type of security scan")
    depth: str = Field(default="medium", description="Scan depth (light, medium, deep)")
    compliance_standards: Optional[List[str]] = Field(default=None, description="Compliance standards to check")
    exclude_checks: Optional[List[str]] = Field(default=None, description="Security checks to exclude")
    output_format: str = Field(default="json", description="Output format for results")


class ThreatAnalysisRequest(BaseModel):
    """Request model for threat analysis operations."""
    indicators: List[str] = Field(..., description="Indicators of compromise or threat indicators")
    analysis_type: str = Field(..., description="Type of threat analysis")
    threat_intelligence_sources: Optional[List[str]] = Field(default=None, description="TI sources to use")
    time_range: Optional[str] = Field(default="24h", description="Time range for analysis")


class ComplianceAuditRequest(BaseModel):
    """Request model for compliance audit operations."""
    framework: str = Field(..., description="Compliance framework to audit against")
    scope: str = Field(..., description="Audit scope (network, application, organization)")
    audit_type: str = Field(..., description="Type of audit (internal, external, certification)")
    requirements: Optional[List[str]] = Field(default=None, description="Specific requirements to focus on")


class InfoSecAgent(EnhancedBaseAgent[SecurityScanRequest, Dict[str, Any]]):
    """
    Specialized agent for information security analysis and threat assessment.

    Capabilities:
    - Vulnerability scanning and assessment
    - Penetration testing coordination
    - Security compliance auditing
    - Threat analysis and intelligence
    - Incident response support
    - Security monitoring and alerting
    - Risk assessment and management
    - Security policy enforcement
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "InfoSecAgent",
        description: str = "Specialized agent for information security and threat assessment",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.INFOSEC_AGENT,
            **kwargs
        )

        # InfoSec-specific configuration
        self.scan_types = {
            "vulnerability": "Vulnerability assessment and scanning",
            "penetration": "Penetration testing and exploitation",
            "compliance": "Compliance and regulatory auditing",
            "malware": "Malware detection and analysis",
            "network": "Network security assessment",
            "web_app": "Web application security testing",
            "social_engineering": "Social engineering assessment",
            "wireless": "Wireless network security testing"
        }

        self.security_tools = {
            "scanners": ["nmap", "nessus", "openvas", "qualys", "rapid7"],
            "penetration": ["metasploit", "burp_suite", "owasp_zap", "sqlmap", "nikto"],
            "compliance": ["lynis", "docker_bench", "kube_bench", "prowler"],
            "malware": ["clamav", "yara", "virustotal", "hybrid_analysis"],
            "network": ["wireshark", "tcpdump", "ntopng", "snort", "suricata"]
        }

        self.compliance_frameworks = {
            "iso_27001": "ISO 27001 Information Security Management",
            "nist_csf": "NIST Cybersecurity Framework",
            "pci_dss": "Payment Card Industry Data Security Standard",
            "sox": "Sarbanes-Oxley Act compliance",
            "gdpr": "General Data Protection Regulation",
            "hipaa": "Health Insurance Portability and Accountability Act",
            "cis_controls": "CIS Critical Security Controls",
            "nist_800_53": "NIST SP 800-53 Security Controls"
        }

        self.threat_categories = [
            "malware", "phishing", "insider_threat", "apt", "ddos",
            "data_breach", "ransomware", "supply_chain", "zero_day"
        ]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "vulnerability_scanning",
            "penetration_testing",
            "threat_assessment",
            "security_auditing",
            "compliance_checking",
            "malware_analysis",
            "network_security_analysis",
            "incident_response",
            "security_monitoring",
            "risk_assessment",
            "security_policy_review",
            "access_control_audit",
            "data_protection_audit",
            "security_training_assessment",
            "threat_intelligence_analysis",
            "forensic_analysis",
            "security_architecture_review",
            "encryption_assessment",
            "authentication_analysis",
            "authorization_review"
        ]

    async def _execute_core_logic(
        self,
        request: SecurityScanRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core InfoSec logic.

        Args:
            request: Security scan request
            correlation_context: Optional correlation context

        Returns:
            Security analysis results with findings and recommendations
        """
        with logfire.span(
            "InfoSecAgent.execute_core_logic",
            agent_id=self.agent_id,
            target=request.target,
            scan_type=request.scan_type
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "target": request.target,
                    "scan_type": request.scan_type,
                    "depth": request.depth,
                    "compliance_standards": request.compliance_standards
                },
                correlation_context
            )

            try:
                # Validate scan type
                if request.scan_type not in self.scan_types:
                    raise ValidationError(f"Unsupported scan type: {request.scan_type}")

                # Execute scan based on type
                if request.scan_type == "vulnerability":
                    result = await self._perform_vulnerability_scan(request)
                elif request.scan_type == "penetration":
                    result = await self._perform_penetration_test(request)
                elif request.scan_type == "compliance":
                    result = await self._perform_compliance_audit(request)
                elif request.scan_type == "malware":
                    result = await self._perform_malware_scan(request)
                elif request.scan_type == "network":
                    result = await self._perform_network_scan(request)
                else:
                    result = await self._perform_generic_security_scan(request)

                # Add metadata
                result.update({
                    "target": request.target,
                    "scan_type": request.scan_type,
                    "depth": request.depth,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "Security scan completed",
                    agent_id=self.agent_id,
                    target=request.target,
                    scan_type=request.scan_type,
                    vulnerabilities_found=result.get("summary", {}).get("total_vulnerabilities", 0)
                )

                return result

            except Exception as e:
                logfire.error(
                    "Security scan failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    target=request.target,
                    scan_type=request.scan_type
                )
                raise AgentExecutionError(f"Security scan failed: {str(e)}")

    async def _perform_vulnerability_scan(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform vulnerability scanning and assessment."""

        # Mock vulnerability scan results
        scan_results = {
            "summary": {
                "total_vulnerabilities": 12,
                "critical": 2,
                "high": 4,
                "medium": 5,
                "low": 1,
                "informational": 0,
                "scan_duration": 1800,
                "scan_coverage": "95%"
            },
            "vulnerabilities": [
                {
                    "id": "CVE-2024-0001",
                    "severity": "critical",
                    "title": "Remote Code Execution in Web Server",
                    "description": "Buffer overflow vulnerability allows remote code execution",
                    "cvss_score": 9.8,
                    "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    "affected_component": "Apache HTTP Server 2.4.41",
                    "remediation": "Upgrade to Apache HTTP Server 2.4.50 or later",
                    "references": ["https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-0001"],
                    "exploit_available": True
                },
                {
                    "id": "CVE-2023-9999",
                    "severity": "high",
                    "title": "SQL Injection in Login Form",
                    "description": "SQL injection vulnerability in user authentication",
                    "cvss_score": 8.2,
                    "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:L/A:N",
                    "affected_component": "Custom Web Application",
                    "remediation": "Implement parameterized queries and input validation",
                    "references": [],
                    "exploit_available": False
                }
            ],
            "compliance_status": {
                "iso_27001": "non_compliant",
                "pci_dss": "partially_compliant",
                "nist_csf": "partially_compliant",
                "overall_score": 6.5
            },
            "recommendations": [
                "Implement critical security patches immediately",
                "Establish regular vulnerability scanning schedule",
                "Enhance input validation and sanitization",
                "Implement web application firewall",
                "Conduct security awareness training"
            ],
            "remediation_priorities": [
                {
                    "priority": 1,
                    "vulnerabilities": ["CVE-2024-0001"],
                    "impact": "Complete system compromise possible",
                    "timeline": "Immediate (within 24 hours)"
                },
                {
                    "priority": 2,
                    "vulnerabilities": ["CVE-2023-9999"],
                    "impact": "Data breach and unauthorized access",
                    "timeline": "Within 1 week"
                }
            ]
        }

        return {
            "success": True,
            "scan_results": scan_results,
            "operation": "vulnerability_scan"
        }

    async def _perform_penetration_test(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform penetration testing assessment."""

        # Mock penetration test results
        pentest_results = {
            "summary": {
                "test_duration": 5,  # days
                "systems_tested": 15,
                "vulnerabilities_exploited": 3,
                "systems_compromised": 2,
                "data_accessed": True,
                "privilege_escalation": True
            },
            "attack_chain": [
                {
                    "step": 1,
                    "technique": "Reconnaissance",
                    "description": "Information gathering and target enumeration",
                    "tools_used": ["nmap", "dirb", "nikto"],
                    "findings": "Discovered multiple open services and web directories"
                },
                {
                    "step": 2,
                    "technique": "Initial Access",
                    "description": "Exploited SQL injection vulnerability",
                    "tools_used": ["sqlmap"],
                    "findings": "Gained database access and extracted user credentials"
                },
                {
                    "step": 3,
                    "technique": "Privilege Escalation",
                    "description": "Escalated privileges using local exploit",
                    "tools_used": ["metasploit"],
                    "findings": "Obtained administrative access to web server"
                }
            ],
            "compromised_systems": [
                {
                    "system": "web-server-01",
                    "ip": "192.168.1.100",
                    "access_level": "administrator",
                    "data_accessed": ["user_database", "configuration_files"],
                    "persistence_established": True
                }
            ],
            "business_impact": {
                "confidentiality": "high",
                "integrity": "medium",
                "availability": "low",
                "financial_impact": "Potential data breach costs: $500K - $2M",
                "regulatory_impact": "GDPR/CCPA violation potential"
            },
            "recommendations": [
                "Implement comprehensive input validation",
                "Deploy endpoint detection and response (EDR)",
                "Establish network segmentation",
                "Implement privileged access management",
                "Conduct regular security awareness training"
            ]
        }

        return {
            "success": True,
            "pentest_results": pentest_results,
            "operation": "penetration_test"
        }

    async def _perform_compliance_audit(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform compliance audit assessment."""

        frameworks = request.compliance_standards or ["iso_27001", "nist_csf"]

        # Mock compliance audit results
        audit_results = {
            "summary": {
                "frameworks_audited": frameworks,
                "total_controls": 150,
                "compliant_controls": 120,
                "non_compliant_controls": 20,
                "not_applicable": 10,
                "overall_compliance": 80.0
            },
            "framework_results": {},
            "critical_gaps": [
                {
                    "control": "Access Control Management",
                    "framework": "ISO 27001",
                    "gap": "Lack of regular access reviews",
                    "risk_level": "high",
                    "remediation": "Implement quarterly access reviews"
                },
                {
                    "control": "Incident Response",
                    "framework": "NIST CSF",
                    "gap": "No formal incident response plan",
                    "risk_level": "high",
                    "remediation": "Develop and test incident response procedures"
                }
            ],
            "recommendations": [
                "Develop formal security policies and procedures",
                "Implement security awareness training program",
                "Establish regular vulnerability assessments",
                "Create incident response and disaster recovery plans",
                "Implement continuous monitoring capabilities"
            ]
        }

        # Add framework-specific results
        for framework in frameworks:
            audit_results["framework_results"][framework] = {
                "compliance_percentage": 82.5,
                "critical_findings": 2,
                "high_findings": 5,
                "medium_findings": 8,
                "status": "partially_compliant"
            }

        return {
            "success": True,
            "audit_results": audit_results,
            "operation": "compliance_audit"
        }

    async def _perform_malware_scan(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform malware detection and analysis."""

        # Mock malware scan results
        malware_results = {
            "summary": {
                "files_scanned": 50000,
                "malware_detected": 3,
                "suspicious_files": 7,
                "clean_files": 49990,
                "scan_duration": 3600
            },
            "detections": [
                {
                    "file_path": "/tmp/suspicious_file.exe",
                    "malware_type": "trojan",
                    "threat_name": "Trojan.Generic.12345",
                    "severity": "high",
                    "action_taken": "quarantined",
                    "signatures_matched": ["YARA_Rule_Trojan", "Heuristic_Suspicious"]
                },
                {
                    "file_path": "/var/log/backdoor.sh",
                    "malware_type": "backdoor",
                    "threat_name": "Backdoor.Linux.Shell",
                    "severity": "critical",
                    "action_taken": "quarantined",
                    "signatures_matched": ["Custom_Backdoor_Rule"]
                }
            ],
            "suspicious_activity": [
                {
                    "activity": "Unusual network connections",
                    "description": "Multiple connections to suspicious IPs",
                    "risk_level": "medium",
                    "recommended_action": "Monitor and investigate"
                }
            ],
            "recommendations": [
                "Implement real-time malware protection",
                "Update antivirus signatures regularly",
                "Restrict executable file permissions",
                "Monitor network traffic for suspicious patterns",
                "Implement application whitelisting"
            ]
        }

        return {
            "success": True,
            "malware_results": malware_results,
            "operation": "malware_scan"
        }

    async def _perform_network_scan(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform network security assessment."""

        # Mock network scan results
        network_results = {
            "summary": {
                "hosts_discovered": 25,
                "services_identified": 78,
                "open_ports": 156,
                "vulnerable_services": 8,
                "security_issues": 12
            },
            "host_discovery": [
                {
                    "ip": "192.168.1.10",
                    "hostname": "web-server-01",
                    "os": "Linux Ubuntu 20.04",
                    "status": "up",
                    "open_ports": [22, 80, 443, 3306],
                    "services": ["ssh", "http", "https", "mysql"]
                },
                {
                    "ip": "192.168.1.20",
                    "hostname": "db-server-01",
                    "os": "Windows Server 2019",
                    "status": "up",
                    "open_ports": [135, 139, 445, 1433],
                    "services": ["rpc", "netbios", "smb", "mssql"]
                }
            ],
            "security_findings": [
                {
                    "type": "weak_ssl_configuration",
                    "host": "192.168.1.10",
                    "port": 443,
                    "description": "Weak SSL/TLS configuration detected",
                    "severity": "medium",
                    "recommendation": "Update SSL/TLS configuration"
                },
                {
                    "type": "default_credentials",
                    "host": "192.168.1.20",
                    "port": 1433,
                    "description": "Default database credentials detected",
                    "severity": "high",
                    "recommendation": "Change default passwords immediately"
                }
            ],
            "network_topology": {
                "subnets": ["192.168.1.0/24", "10.0.1.0/24"],
                "vlans": ["VLAN_100_SERVERS", "VLAN_200_WORKSTATIONS"],
                "firewalls": ["pfSense-01", "pfSense-02"],
                "segmentation_score": 7.5
            },
            "recommendations": [
                "Implement network segmentation",
                "Deploy intrusion detection system",
                "Configure proper firewall rules",
                "Disable unnecessary services",
                "Implement network access control"
            ]
        }

        return {
            "success": True,
            "network_results": network_results,
            "operation": "network_scan"
        }

    async def _perform_generic_security_scan(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """Perform generic security assessment."""

        return {
            "success": True,
            "scan_type": request.scan_type,
            "target": request.target,
            "message": f"Generic {request.scan_type} scan completed successfully",
            "recommendations": [
                "Review and implement security best practices",
                "Conduct regular security assessments",
                "Maintain up-to-date security documentation"
            ],
            "operation": "generic_security_scan"
        }

    async def analyze_threat_intelligence(self, request: ThreatAnalysisRequest) -> Dict[str, Any]:
        """
        Analyze threat intelligence and indicators.

        Args:
            request: Threat analysis request

        Returns:
            Threat intelligence analysis results
        """
        with logfire.span(
            "InfoSecAgent.analyze_threat_intelligence",
            agent_id=self.agent_id,
            analysis_type=request.analysis_type
        ):
            try:
                # Mock threat intelligence analysis
                threat_analysis = {
                    "indicators_analyzed": len(request.indicators),
                    "malicious_indicators": 3,
                    "suspicious_indicators": 2,
                    "clean_indicators": len(request.indicators) - 5,
                    "threat_attribution": {
                        "apt_group": "APT-29",
                        "confidence": "medium",
                        "techniques": ["T1566.001", "T1059.001", "T1083"],
                        "campaign": "Operation CloudHopper 2024"
                    },
                    "ioc_analysis": [
                        {
                            "indicator": "malicious-domain.com",
                            "type": "domain",
                            "verdict": "malicious",
                            "first_seen": "2024-01-15",
                            "last_seen": "2024-01-20",
                            "reputation_score": 1,
                            "associated_malware": ["TrickBot", "Emotet"]
                        }
                    ],
                    "recommendations": [
                        "Block identified malicious indicators",
                        "Implement enhanced monitoring for suspicious indicators",
                        "Update threat hunting rules based on TTPs",
                        "Share intelligence with threat intelligence platforms"
                    ]
                }

                logfire.info(
                    "Threat intelligence analysis completed",
                    agent_id=self.agent_id,
                    indicators_analyzed=threat_analysis["indicators_analyzed"],
                    malicious_found=threat_analysis["malicious_indicators"]
                )

                return {"success": True, "threat_analysis": threat_analysis}

            except Exception as e:
                logfire.error(
                    "Threat intelligence analysis failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Threat intelligence analysis failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for InfoSec agent."""
        return {
            "default_scan_depth": "medium",
            "timeout_seconds": 3600,
            "max_concurrent_scans": 5,
            "report_format": "json",
            "auto_remediation": False,
            "compliance_checking": True,
            "threat_intelligence": True,
            "vulnerability_database_update": True,
            "scan_scheduling": True,
            "notification_on_critical": True,
            "data_retention_days": 365,
            "scan_types": self.scan_types,
            "security_tools": self.security_tools,
            "compliance_frameworks": self.compliance_frameworks,
            "threat_categories": self.threat_categories
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["default_scan_depth", "timeout_seconds", "max_concurrent_scans"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate scan depth
        valid_depths = ["light", "medium", "deep"]
        if config.get("default_scan_depth") not in valid_depths:
            raise ValidationError(f"Invalid scan depth. Must be one of: {valid_depths}")

        # Validate timeout
        if config.get("timeout_seconds", 0) <= 0:
            raise ValidationError("timeout_seconds must be positive")

        # Validate concurrent scans
        if config.get("max_concurrent_scans", 0) <= 0:
            raise ValidationError("max_concurrent_scans must be positive")

        return True
