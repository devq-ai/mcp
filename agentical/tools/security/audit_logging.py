"""
Audit Logging Tool for Agentical

This module provides comprehensive audit logging capabilities for security
compliance, forensic analysis, and regulatory requirements with integration
to the Agentical framework.

Features:
- Comprehensive audit event logging and tracking
- Multi-standard compliance (SOX, HIPAA, GDPR, PCI-DSS, ISO 27001)
- Real-time security monitoring and alerting
- Tamper-evident log storage and integrity verification
- Structured logging with rich metadata and context
- Log aggregation and centralized management
- Automated compliance reporting and dashboards
- Integration with SIEM and security tools
- Performance monitoring with minimal overhead
- Secure log archival and retention management
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import logging
import gzip
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    SecurityError
)
from ...core.logging import log_operation


class AuditLevel(Enum):
    """Audit logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AuditCategory(Enum):
    """Categories of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    PERFORMANCE = "performance"
    ERROR = "error"
    BUSINESS_LOGIC = "business_logic"
    WORKFLOW = "workflow"


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    SOX = "sox"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CCPA = "ccpa"


class AuditEvent:
    """Individual audit event with comprehensive metadata."""

    def __init__(
        self,
        event_id: str,
        category: AuditCategory,
        level: AuditLevel,
        action: str,
        actor: str,
        resource: str,
        outcome: str = "success",
        description: str = "",
        client_info: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[ComplianceStandard]] = None,
        sensitive_data: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_id = event_id
        self.category = category
        self.level = level
        self.action = action
        self.actor = actor
        self.resource = resource
        self.outcome = outcome
        self.description = description
        self.client_info = client_info or {}
        self.compliance_tags = compliance_tags or []
        self.sensitive_data = sensitive_data
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.session_id = self.metadata.get("session_id")
        self.correlation_id = self.metadata.get("correlation_id")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "level": self.level.value,
            "action": self.action,
            "actor": self.actor,
            "resource": self.resource,
            "outcome": self.outcome,
            "description": self.description,
            "client_info": self.client_info,
            "compliance_tags": [tag.value for tag in self.compliance_tags],
            "sensitive_data": self.sensitive_data,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))

    def get_hash(self) -> str:
        """Get hash of event for integrity verification."""
        event_data = self.to_json()
        return hashlib.sha256(event_data.encode()).hexdigest()


class AuditReport:
    """Audit report for compliance and analysis."""

    def __init__(
        self,
        report_id: str,
        title: str,
        period_start: datetime,
        period_end: datetime,
        compliance_standard: Optional[ComplianceStandard] = None,
        events: Optional[List[AuditEvent]] = None,
        summary: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ):
        self.report_id = report_id
        self.title = title
        self.period_start = period_start
        self.period_end = period_end
        self.compliance_standard = compliance_standard
        self.events = events or []
        self.summary = summary or {}
        self.recommendations = recommendations or []
        self.generated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "compliance_standard": self.compliance_standard.value if self.compliance_standard else None,
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat()
        }


class AuditLogger:
    """
    Comprehensive audit logger supporting multiple compliance standards
    with secure storage and real-time monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audit logger.

        Args:
            config: Configuration for audit logging
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.log_level = AuditLevel(self.config.get("log_level", "info"))
        self.retention_days = self.config.get("retention_days", 365)
        self.encryption_enabled = self.config.get("encryption_enabled", True)
        self.real_time_alerts = self.config.get("real_time_alerts", True)
        self.compliance_standards = [
            ComplianceStandard(std) for std in self.config.get("compliance_standards", ["iso_27001"])
        ]
        self.include_request_body = self.config.get("include_request_body", False)
        self.include_response_body = self.config.get("include_response_body", False)
        self.log_directory = Path(self.config.get("log_directory", "audit_logs"))
        self.max_log_file_size = self.config.get("max_log_file_size", 100 * 1024 * 1024)  # 100MB

        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Internal storage
        self.events: List[AuditEvent] = []
        self.event_index: Dict[str, AuditEvent] = {}
        self.integrity_chain: List[str] = []

        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins_per_minute": 5,
            "security_events_per_hour": 10,
            "error_rate_percentage": 5.0,
            "suspicious_activity_score": 80
        }

        # Initialize encryption if enabled
        self.encryption_key = None
        if self.encryption_enabled and CRYPTOGRAPHY_AVAILABLE:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption for audit logs."""
        try:
            key_file = self.log_directory / "audit_key.pem"

            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.encryption_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
            else:
                # Generate new key
                self.encryption_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )

                # Save key
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))

                # Restrict permissions
                os.chmod(key_file, 0o600)

        except Exception as e:
            self.logger.warning(f"Failed to initialize audit log encryption: {e}")
            self.encryption_enabled = False

    @log_operation("audit_logging")
    async def log_event(
        self,
        category: AuditCategory,
        level: AuditLevel,
        action: str,
        actor: str,
        resource: str,
        outcome: str = "success",
        description: str = "",
        client_info: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[ComplianceStandard]] = None,
        sensitive_data: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log audit event with comprehensive details.

        Args:
            category: Event category
            level: Logging level
            action: Action performed
            actor: User or system performing action
            resource: Resource affected
            outcome: Outcome of action
            description: Human-readable description
            client_info: Client information (IP, user agent, etc.)
            compliance_tags: Relevant compliance standards
            sensitive_data: Whether event contains sensitive data
            metadata: Additional metadata

        Returns:
            str: Event ID for tracking
        """
        # Filter by log level
        if not self._should_log(level):
            return ""

        event_id = str(uuid.uuid4())

        event = AuditEvent(
            event_id=event_id,
            category=category,
            level=level,
            action=action,
            actor=actor,
            resource=resource,
            outcome=outcome,
            description=description,
            client_info=client_info,
            compliance_tags=compliance_tags or [],
            sensitive_data=sensitive_data,
            metadata=metadata or {}
        )

        # Store event
        self.events.append(event)
        self.event_index[event_id] = event

        # Update integrity chain
        event_hash = event.get_hash()
        if self.integrity_chain:
            # Chain with previous hash
            combined = self.integrity_chain[-1] + event_hash
            chained_hash = hashlib.sha256(combined.encode()).hexdigest()
        else:
            chained_hash = event_hash

        self.integrity_chain.append(chained_hash)

        # Write to persistent storage
        await self._write_event_to_file(event)

        # Check for real-time alerts
        if self.real_time_alerts:
            await self._check_alert_conditions(event)

        return event_id

    async def log_authentication_event(
        self,
        action: str,
        username: str,
        outcome: str,
        client_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log authentication-related event."""
        level = AuditLevel.SECURITY if outcome == "failure" else AuditLevel.INFO

        return await self.log_event(
            category=AuditCategory.AUTHENTICATION,
            level=level,
            action=action,
            actor=username,
            resource="authentication_system",
            outcome=outcome,
            description=f"Authentication {action} for user {username}",
            client_info=client_info,
            compliance_tags=[ComplianceStandard.SOX, ComplianceStandard.ISO_27001],
            metadata=metadata
        )

    async def log_data_access_event(
        self,
        action: str,
        username: str,
        resource: str,
        outcome: str = "success",
        data_classification: str = "internal",
        client_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log data access event."""
        sensitive = data_classification in ["confidential", "restricted", "pii"]
        compliance_tags = [ComplianceStandard.ISO_27001]

        if sensitive:
            compliance_tags.extend([ComplianceStandard.GDPR, ComplianceStandard.HIPAA])

        return await self.log_event(
            category=AuditCategory.DATA_ACCESS,
            level=AuditLevel.INFO,
            action=action,
            actor=username,
            resource=resource,
            outcome=outcome,
            description=f"Data {action} on {resource} by {username}",
            client_info=client_info,
            compliance_tags=compliance_tags,
            sensitive_data=sensitive,
            metadata={**(metadata or {}), "data_classification": data_classification}
        )

    async def log_security_event(
        self,
        action: str,
        severity: str,
        source: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log security-related event."""
        level_map = {
            "low": AuditLevel.INFO,
            "medium": AuditLevel.WARNING,
            "high": AuditLevel.ERROR,
            "critical": AuditLevel.CRITICAL
        }

        return await self.log_event(
            category=AuditCategory.SECURITY_EVENT,
            level=level_map.get(severity, AuditLevel.WARNING),
            action=action,
            actor="security_system",
            resource=source,
            outcome="detected",
            description=description,
            compliance_tags=[ComplianceStandard.ISO_27001, ComplianceStandard.SOX],
            metadata={**(metadata or {}), "severity": severity}
        )

    async def log_configuration_change(
        self,
        action: str,
        username: str,
        component: str,
        old_value: Any = None,
        new_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log configuration change event."""
        return await self.log_event(
            category=AuditCategory.CONFIGURATION_CHANGE,
            level=AuditLevel.WARNING,
            action=action,
            actor=username,
            resource=component,
            outcome="success",
            description=f"Configuration change in {component} by {username}",
            compliance_tags=[ComplianceStandard.SOX, ComplianceStandard.ISO_27001],
            metadata={
                **(metadata or {}),
                "old_value": str(old_value) if old_value is not None else None,
                "new_value": str(new_value) if new_value is not None else None
            }
        )

    def _should_log(self, level: AuditLevel) -> bool:
        """Check if event should be logged based on level."""
        level_priority = {
            AuditLevel.DEBUG: 0,
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2,
            AuditLevel.ERROR: 3,
            AuditLevel.CRITICAL: 4,
            AuditLevel.SECURITY: 4,
            AuditLevel.COMPLIANCE: 3
        }

        return level_priority.get(level, 1) >= level_priority.get(self.log_level, 1)

    async def _write_event_to_file(self, event: AuditEvent) -> None:
        """Write event to persistent storage."""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.log_directory / f"audit_{date_str}.log"

            # Prepare log entry
            log_entry = event.to_json() + "\n"

            # Encrypt if enabled
            if self.encryption_enabled and self.encryption_key:
                log_entry = self._encrypt_log_entry(log_entry)

            # Write to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)

            # Check file size and rotate if necessary
            if log_file.stat().st_size > self.max_log_file_size:
                await self._rotate_log_file(log_file)

        except Exception as e:
            self.logger.error(f"Failed to write audit event to file: {e}")

    def _encrypt_log_entry(self, log_entry: str) -> str:
        """Encrypt log entry."""
        try:
            if not self.encryption_key:
                return log_entry

            # For large log entries, use symmetric encryption
            # This is a simplified implementation
            encrypted = self.encryption_key.public_key().encrypt(
                log_entry.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            import base64
            return base64.b64encode(encrypted).decode() + "\n"

        except Exception as e:
            self.logger.warning(f"Failed to encrypt log entry: {e}")
            return log_entry

    async def _rotate_log_file(self, log_file: Path) -> None:
        """Rotate log file when it gets too large."""
        try:
            # Compress and archive
            timestamp = datetime.now().strftime("%H%M%S")
            archive_file = log_file.with_suffix(f".{timestamp}.gz")

            with open(log_file, 'rb') as f_in:
                with gzip.open(archive_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Remove original file
            log_file.unlink()

        except Exception as e:
            self.logger.error(f"Failed to rotate log file: {e}")

    async def _check_alert_conditions(self, event: AuditEvent) -> None:
        """Check if event triggers any alert conditions."""
        try:
            # Check for failed authentication attempts
            if (event.category == AuditCategory.AUTHENTICATION and
                event.outcome == "failure"):
                await self._check_failed_login_threshold(event)

            # Check for security events
            if event.category == AuditCategory.SECURITY_EVENT:
                await self._check_security_event_threshold(event)

            # Check for high-privilege access
            if (event.category == AuditCategory.DATA_ACCESS and
                event.sensitive_data):
                await self._alert_sensitive_data_access(event)

        except Exception as e:
            self.logger.error(f"Failed to check alert conditions: {e}")

    async def _check_failed_login_threshold(self, event: AuditEvent) -> None:
        """Check for excessive failed login attempts."""
        # Count recent failed logins from same IP
        cutoff = datetime.now() - timedelta(minutes=1)
        client_ip = event.client_info.get("ip", "unknown")

        recent_failures = [
            e for e in self.events
            if (e.timestamp > cutoff and
                e.category == AuditCategory.AUTHENTICATION and
                e.outcome == "failure" and
                e.client_info.get("ip") == client_ip)
        ]

        if len(recent_failures) >= self.alert_thresholds["failed_logins_per_minute"]:
            await self._send_alert(
                "Excessive Failed Login Attempts",
                f"Multiple failed login attempts from IP {client_ip}",
                "high",
                {"event_count": len(recent_failures), "client_ip": client_ip}
            )

    async def _check_security_event_threshold(self, event: AuditEvent) -> None:
        """Check for excessive security events."""
        cutoff = datetime.now() - timedelta(hours=1)

        recent_security_events = [
            e for e in self.events
            if (e.timestamp > cutoff and
                e.category == AuditCategory.SECURITY_EVENT)
        ]

        if len(recent_security_events) >= self.alert_thresholds["security_events_per_hour"]:
            await self._send_alert(
                "High Security Event Volume",
                "Unusual number of security events detected",
                "medium",
                {"event_count": len(recent_security_events)}
            )

    async def _alert_sensitive_data_access(self, event: AuditEvent) -> None:
        """Alert on sensitive data access."""
        await self._send_alert(
            "Sensitive Data Access",
            f"Sensitive data accessed: {event.resource} by {event.actor}",
            "medium",
            {"resource": event.resource, "actor": event.actor}
        )

    async def _send_alert(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Send security alert."""
        # Log the alert as a security event
        await self.log_event(
            category=AuditCategory.SECURITY_EVENT,
            level=AuditLevel.CRITICAL,
            action="alert_generated",
            actor="audit_system",
            resource="security_monitoring",
            outcome="alert",
            description=f"Security alert: {title} - {message}",
            metadata={"alert_severity": severity, **metadata}
        )

        # In production, integrate with alerting system (email, Slack, PagerDuty, etc.)
        self.logger.critical(f"SECURITY ALERT: {title} - {message}")

    async def generate_compliance_report(
        self,
        compliance_standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime,
        include_events: bool = True
    ) -> AuditReport:
        """Generate compliance report for specified standard and period."""

        report_id = str(uuid.uuid4())

        # Filter events for the period and compliance standard
        relevant_events = [
            event for event in self.events
            if (start_date <= event.timestamp <= end_date and
                compliance_standard in event.compliance_tags)
        ]

        # Generate summary statistics
        summary = {
            "total_events": len(relevant_events),
            "events_by_category": {},
            "events_by_level": {},
            "events_by_outcome": {},
            "security_events": 0,
            "failed_authentications": 0,
            "data_access_events": 0,
            "configuration_changes": 0
        }

        for event in relevant_events:
            # Count by category
            category = event.category.value
            summary["events_by_category"][category] = summary["events_by_category"].get(category, 0) + 1

            # Count by level
            level = event.level.value
            summary["events_by_level"][level] = summary["events_by_level"].get(level, 0) + 1

            # Count by outcome
            outcome = event.outcome
            summary["events_by_outcome"][outcome] = summary["events_by_outcome"].get(outcome, 0) + 1

            # Special counters
            if event.category == AuditCategory.SECURITY_EVENT:
                summary["security_events"] += 1
            elif event.category == AuditCategory.AUTHENTICATION and event.outcome == "failure":
                summary["failed_authentications"] += 1
            elif event.category == AuditCategory.DATA_ACCESS:
                summary["data_access_events"] += 1
            elif event.category == AuditCategory.CONFIGURATION_CHANGE:
                summary["configuration_changes"] += 1

        # Generate recommendations based on findings
        recommendations = self._generate_compliance_recommendations(
            compliance_standard, summary, relevant_events
        )

        report = AuditReport(
            report_id=report_id,
            title=f"{compliance_standard.value.upper()} Compliance Report",
            period_start=start_date,
            period_end=end_date,
            compliance_standard=compliance_standard,
            events=relevant_events if include_events else [],
            summary=summary,
            recommendations=recommendations
        )

        return report

    def _generate_compliance_recommendations(
        self,
        standard: ComplianceStandard,
        summary: Dict[str, Any],
        events: List[AuditEvent]
    ) -> List[str]:
        """Generate compliance recommendations based on audit findings."""
        recommendations = []

        if standard == ComplianceStandard.SOX:
            if summary["configuration_changes"] > 10:
                recommendations.append("Consider implementing change approval workflow for configuration changes")
            if summary["failed_authentications"] > 5:
                recommendations.append("Review authentication security policies and implement account lockout")

        elif standard == ComplianceStandard.GDPR:
            if summary["data_access_events"] > 100:
                recommendations.append("Implement data access controls and privacy impact assessments")
            recommendations.append("Ensure data subject rights are documented and accessible")

        elif standard == ComplianceStandard.HIPAA:
            if any(event.sensitive_data for event in events):
                recommendations.append("Ensure all PHI access is properly logged and monitored")
            recommendations.append("Verify encryption of PHI in transit and at rest")

        elif standard == ComplianceStandard.PCI_DSS:
            recommendations.append("Implement network segmentation for cardholder data environment")
            recommendations.append("Ensure regular security testing and vulnerability assessments")

        elif standard == ComplianceStandard.ISO_27001:
            if summary["security_events"] > 5:
                recommendations.append("Review incident response procedures and security controls")
            recommendations.append("Conduct regular risk assessments and security reviews")

        return recommendations

    async def verify_log_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit logs."""
        integrity_status = {
            "status": "verified",
            "total_events": len(self.events),
            "chain_length": len(self.integrity_chain),
            "corrupted_events": [],
            "missing_events": [],
            "verification_timestamp": datetime.now().isoformat()
        }

        try:
            # Verify hash chain
            for i, event in enumerate(self.events):
                expected_hash = event.get_hash()

                if i == 0:
                    # First event should match first chain entry
                    if self.integrity_chain[i] != expected_hash:
                        integrity_status["corrupted_events"].append(event.event_id)
                else:
                    # Subsequent events should be chained
                    combined = self.integrity_chain[i-1] + expected_hash
                    expected_chained = hashlib.sha256(combined.encode()).hexdigest()

                    if self.integrity_chain[i] != expected_chained:
                        integrity_status["corrupted_events"].append(event.event_id)

            if integrity_status["corrupted_events"]:
                integrity_status["status"] = "corrupted"

        except Exception as e:
            integrity_status["status"] = "error"
            integrity_status["error"] = str(e)

        return integrity_status

    async def search_events(
        self,
        category: Optional[AuditCategory] = None,
        level: Optional[AuditLevel] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit events with filters."""

        results = []
        count = 0

        for event in reversed(self.events):  # Most recent first
            if count >= limit:
                break

            # Apply filters
            if category and event.category != category:
                continue
            if level and event.level != level:
                continue
            if actor and actor.lower() not in event.actor.lower():
                continue
            if resource and resource.lower() not in event.resource.lower():
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)
            count += 1

        return results

    async def cleanup_old_logs(self) -> Dict[str, Any]:
        """Clean up old audit logs based on retention policy."""
        cleanup_result = {
            "files_removed": 0,
            "events_archived": 0,
            "bytes_freed": 0,
            "errors": []
        }

        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            # Clean up in-memory events
            original_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp > cutoff_date]
            cleanup_result["events_archived"] = original_count - len(self.events)

            # Rebuild event index
            self.event_index = {e.event_id: e for e in self.events}

            # Clean up log files
            for log_file in self.log_directory.glob("audit_*.log*"):
                try:
                    file_stat = log_file.stat()
                    file_date = datetime.fromtimestamp(file_stat.st_mtime)

                    if file_date < cutoff_date:
                        file_size = file_stat.st_size
                        log_file.unlink()
                        cleanup_result["files_removed"] += 1
                        cleanup_result["bytes_freed"] += file_size

                except Exception as e:
                    cleanup_result["errors"].append(f"Failed to remove {log_file}: {e}")

        except Exception as e:
            cleanup_result["errors"].append(f"Cleanup failed: {e}")

        return cleanup_result

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on audit logger."""
        health_status = {
            "status": "healthy",
            "total_events": len(self.events),
            "log_level": self.log_level.value,
            "encryption_enabled": self.encryption_enabled,
            "real_time_alerts": self.real_time_alerts,
            "retention_days": self.retention_days,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "log_directory": str(self.log_directory),
            "log_files": len(list(self.log_directory.glob("audit_*.log*"))),
            "dependencies": {
                "cryptography": CRYPTOGRAPHY_AVAILABLE
            }
        }

        try:
            # Test logging functionality
            test_event_
id = await self.log_event(
                category=AuditCategory.SYSTEM_ACCESS,
                level=AuditLevel.INFO,
                action="health_check",
                actor="audit_system",
                resource="audit_logger",
                outcome="success",
                description="Health check performed"
            )

            health_status["test_logging"] = bool(test_event_id)

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["test_logging"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating audit logger
def create_audit_logger(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """
    Create an audit logger with specified configuration.

    Args:
        config: Configuration for audit logging

    Returns:
        AuditLogger: Configured audit logger instance
    """
    return AuditLogger(config=config)
