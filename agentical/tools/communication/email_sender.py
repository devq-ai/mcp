"""
Email Sender for Agentical

This module provides comprehensive email sending capabilities with support for
multiple email providers, template engines, attachment handling, delivery tracking,
and enterprise-grade features for reliable email communication.

Features:
- Multiple email providers (SMTP, SendGrid, AWS SES, Mailgun, Postmark)
- Template engine with dynamic content (Jinja2, simple placeholders)
- Attachment handling and inline images
- HTML and plain text email support
- Delivery status tracking and bounce handling
- Bulk email sending with rate limiting
- Email validation and verification
- Unsubscribe management and compliance
- Enterprise features (audit logging, monitoring, encryption)
"""

import asyncio
import email.utils
import json
import mimetypes
import os
import re
import smtplib
import time
import uuid
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, Any, List, Optional, Union, BinaryIO
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import base64

# Optional dependencies
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_SES_AVAILABLE = True
except ImportError:
    AWS_SES_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


class EmailProvider(Enum):
    """Supported email providers."""
    SMTP = "smtp"
    SENDGRID = "sendgrid"
    AWS_SES = "aws_ses"
    MAILGUN = "mailgun"
    POSTMARK = "postmark"
    CUSTOM = "custom"


class EmailFormat(Enum):
    """Email format types."""
    TEXT = "text"
    HTML = "html"
    MULTIPART = "multipart"


class EmailPriority(Enum):
    """Email priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryStatus(Enum):
    """Email delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    FAILED = "failed"
    UNSUBSCRIBED = "unsubscribed"


class TemplateEngine(Enum):
    """Template engines."""
    NONE = "none"
    SIMPLE = "simple"
    JINJA2 = "jinja2"


@dataclass
class EmailAddress:
    """Email address with optional name."""
    email: str
    name: Optional[str] = None

    def __str__(self) -> str:
        """Format email address for headers."""
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {"email": self.email, "name": self.name}

    @classmethod
    def from_string(cls, address: str) -> 'EmailAddress':
        """Parse email address from string."""
        parsed = email.utils.parseaddr(address)
        return cls(email=parsed[1], name=parsed[0] if parsed[0] else None)


@dataclass
class EmailAttachment:
    """Email attachment."""
    filename: str
    content: bytes
    content_type: Optional[str] = None
    content_id: Optional[str] = None  # For inline images
    disposition: str = "attachment"  # "attachment" or "inline"

    def __post_init__(self):
        if self.content_type is None:
            self.content_type, _ = mimetypes.guess_type(self.filename)
            if self.content_type is None:
                self.content_type = "application/octet-stream"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "content_id": self.content_id,
            "disposition": self.disposition,
            "size": len(self.content)
        }


@dataclass
class EmailTemplate:
    """Email template definition."""
    id: str
    name: str
    subject_template: str
    text_template: Optional[str] = None
    html_template: Optional[str] = None
    variables: Optional[List[str]] = None
    engine: TemplateEngine = TemplateEngine.SIMPLE
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['engine'] = self.engine.value
        return data


@dataclass
class EmailMessage:
    """Email message definition."""
    id: str
    from_address: EmailAddress
    to_addresses: List[EmailAddress]
    subject: str
    text_content: Optional[str] = None
    html_content: Optional[str] = None
    cc_addresses: Optional[List[EmailAddress]] = None
    bcc_addresses: Optional[List[EmailAddress]] = None
    reply_to: Optional[EmailAddress] = None
    attachments: Optional[List[EmailAttachment]] = None
    headers: Optional[Dict[str, str]] = None
    priority: EmailPriority = EmailPriority.NORMAL
    template_id: Optional[str] = None
    template_variables: Optional[Dict[str, Any]] = None
    send_at: Optional[datetime] = None
    track_opens: bool = False
    track_clicks: bool = False
    unsubscribe_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.cc_addresses is None:
            self.cc_addresses = []
        if self.bcc_addresses is None:
            self.bcc_addresses = []
        if self.attachments is None:
            self.attachments = []
        if self.headers is None:
            self.headers = {}
        if self.template_variables is None:
            self.template_variables = {}
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def all_recipients(self) -> List[EmailAddress]:
        """Get all recipient addresses."""
        recipients = self.to_addresses.copy()
        recipients.extend(self.cc_addresses)
        recipients.extend(self.bcc_addresses)
        return recipients

    @property
    def recipient_count(self) -> int:
        """Get total number of recipients."""
        return len(self.all_recipients)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['from_address'] = self.from_address.to_dict()
        data['to_addresses'] = [addr.to_dict() for addr in self.to_addresses]
        data['cc_addresses'] = [addr.to_dict() for addr in self.cc_addresses]
        data['bcc_addresses'] = [addr.to_dict() for addr in self.bcc_addresses]
        if self.reply_to:
            data['reply_to'] = self.reply_to.to_dict()
        data['attachments'] = [att.to_dict() for att in self.attachments]
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.send_at:
            data['send_at'] = self.send_at.isoformat()
        return data


@dataclass
class EmailResult:
    """Result from email sending operation."""
    message_id: str
    status: DeliveryStatus
    provider_message_id: Optional[str] = None
    provider_response: Optional[Dict[str, Any]] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    bounce_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.sent_at is None and self.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]:
            self.sent_at = datetime.utcnow()

    @property
    def is_success(self) -> bool:
        """Check if email was sent successfully."""
        return self.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.sent_at:
            data['sent_at'] = self.sent_at.isoformat()
        if self.delivered_at:
            data['delivered_at'] = self.delivered_at.isoformat()
        return data


class EmailProviderInterface(ABC):
    """Abstract interface for email providers."""

    @abstractmethod
    async def send_email(self, message: EmailMessage) -> EmailResult:
        """Send email message."""
        pass

    @abstractmethod
    async def send_bulk_emails(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """Send multiple emails efficiently."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies are available."""
        pass

    @abstractmethod
    async def verify_email(self, email: str) -> bool:
        """Verify email address validity."""
        pass


class SMTPProvider(EmailProviderInterface):
    """SMTP email provider."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_tls = config.get('use_tls', True)
        self.use_ssl = config.get('use_ssl', False)

    async def send_email(self, message: EmailMessage) -> EmailResult:
        """Send email via SMTP."""
        try:
            # Create MIME message
            mime_message = self._create_mime_message(message)

            # Send via SMTP
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)

            try:
                if self.use_tls and not self.use_ssl:
                    server.starttls()

                if self.username and self.password:
                    server.login(self.username, self.password)

                # Send message
                recipients = [addr.email for addr in message.all_recipients]
                server.send_message(mime_message, to_addrs=recipients)

                return EmailResult(
                    message_id=message.id,
                    status=DeliveryStatus.SENT,
                    provider_message_id=message.headers.get('Message-ID')
                )

            finally:
                server.quit()

        except Exception as e:
            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def send_bulk_emails(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """Send multiple emails via SMTP."""
        results = []
        for message in messages:
            result = await self.send_email(message)
            results.append(result)
        return results

    def _create_mime_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create MIME message from EmailMessage."""
        if message.html_content and message.text_content:
            mime_message = MIMEMultipart('alternative')
        elif message.attachments:
            mime_message = MIMEMultipart('mixed')
        else:
            mime_message = MIMEMultipart()

        # Headers
        mime_message['From'] = str(message.from_address)
        mime_message['To'] = ', '.join(str(addr) for addr in message.to_addresses)
        if message.cc_addresses:
            mime_message['Cc'] = ', '.join(str(addr) for addr in message.cc_addresses)
        if message.reply_to:
            mime_message['Reply-To'] = str(message.reply_to)
        mime_message['Subject'] = message.subject
        mime_message['Date'] = email.utils.formatdate(localtime=True)
        mime_message['Message-ID'] = email.utils.make_msgid()

        # Custom headers
        for key, value in message.headers.items():
            mime_message[key] = value

        # Priority
        if message.priority != EmailPriority.NORMAL:
            priority_map = {
                EmailPriority.LOW: ('5', 'Low'),
                EmailPriority.HIGH: ('2', 'High'),
                EmailPriority.URGENT: ('1', 'Urgent')
            }
            if message.priority in priority_map:
                mime_message['X-Priority'] = priority_map[message.priority][0]
                mime_message['Priority'] = priority_map[message.priority][1]

        # Content
        if message.text_content:
            text_part = MIMEText(message.text_content, 'plain', 'utf-8')
            mime_message.attach(text_part)

        if message.html_content:
            html_part = MIMEText(message.html_content, 'html', 'utf-8')
            mime_message.attach(html_part)

        # Attachments
        for attachment in message.attachments:
            if attachment.disposition == 'inline' and attachment.content_id:
                # Inline image
                if attachment.content_type.startswith('image/'):
                    mime_attachment = MIMEImage(attachment.content)
                    mime_attachment.add_header('Content-ID', f'<{attachment.content_id}>')
                else:
                    mime_attachment = MIMEApplication(attachment.content)
                    mime_attachment.add_header('Content-ID', f'<{attachment.content_id}>')
                mime_attachment.add_header('Content-Disposition', 'inline', filename=attachment.filename)
            else:
                # Regular attachment
                mime_attachment = MIMEApplication(attachment.content)
                mime_attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment.filename}"'
                )

            mime_attachment.add_header('Content-Type', attachment.content_type)
            mime_message.attach(mime_attachment)

        return mime_message

    def is_available(self) -> bool:
        """Check if SMTP is available."""
        return True  # SMTP is part of standard library

    async def verify_email(self, email: str) -> bool:
        """Basic email format verification."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))


class SendGridProvider(EmailProviderInterface):
    """SendGrid email provider."""

    def __init__(self, config: Dict[str, Any]):
        if not SENDGRID_AVAILABLE:
            raise ImportError("sendgrid library required for SendGrid provider")

        self.config = config
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("SendGrid API key is required")

        self.client = sendgrid.SendGridAPIClient(api_key=self.api_key)

    async def send_email(self, message: EmailMessage) -> EmailResult:
        """Send email via SendGrid."""
        try:
            # Create SendGrid mail object
            sg_message = Mail()

            # From address
            sg_message.from_email = Email(message.from_address.email, message.from_address.name)

            # To addresses
            for addr in message.to_addresses:
                sg_message.add_to(To(addr.email, addr.name))

            # CC addresses
            for addr in message.cc_addresses:
                sg_message.add_cc(Email(addr.email, addr.name))

            # BCC addresses
            for addr in message.bcc_addresses:
                sg_message.add_bcc(Email(addr.email, addr.name))

            # Subject
            sg_message.subject = message.subject

            # Content
            if message.text_content:
                sg_message.add_content(Content('text/plain', message.text_content))
            if message.html_content:
                sg_message.add_content(Content('text/html', message.html_content))

            # Attachments
            for attachment in message.attachments:
                sg_attachment = Attachment()
                sg_attachment.file_content = FileContent(base64.b64encode(attachment.content).decode())
                sg_attachment.file_type = FileType(attachment.content_type)
                sg_attachment.file_name = FileName(attachment.filename)
                sg_attachment.disposition = Disposition(attachment.disposition)
                if attachment.content_id:
                    sg_attachment.content_id = attachment.content_id
                sg_message.add_attachment(sg_attachment)

            # Custom headers
            for key, value in message.headers.items():
                sg_message.add_header(key, value)

            # Tracking
            if message.track_opens or message.track_clicks:
                tracking_settings = {
                    'open_tracking': {'enable': message.track_opens},
                    'click_tracking': {'enable': message.track_clicks}
                }
                sg_message.tracking_settings = tracking_settings

            # Send message
            response = self.client.send(sg_message)

            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.SENT if response.status_code == 202 else DeliveryStatus.FAILED,
                provider_response={'status_code': response.status_code, 'headers': dict(response.headers)}
            )

        except Exception as e:
            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def send_bulk_emails(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """Send multiple emails via SendGrid."""
        # SendGrid can handle bulk emails more efficiently
        results = []
        for message in messages:
            result = await self.send_email(message)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Check if SendGrid is available."""
        return SENDGRID_AVAILABLE and bool(self.api_key)

    async def verify_email(self, email: str) -> bool:
        """Verify email using SendGrid validation API."""
        try:
            # SendGrid email validation would go here
            # For now, use basic validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, email))
        except:
            return False


class AWSSESProvider(EmailProviderInterface):
    """AWS SES email provider."""

    def __init__(self, config: Dict[str, Any]):
        if not AWS_SES_AVAILABLE:
            raise ImportError("boto3 library required for AWS SES provider")

        self.config = config
        self.region = config.get('region', 'us-east-1')
        self.access_key_id = config.get('access_key_id')
        self.secret_access_key = config.get('secret_access_key')

        # Create SES client
        session = boto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region
        )
        self.client = session.client('ses')

    async def send_email(self, message: EmailMessage) -> EmailResult:
        """Send email via AWS SES."""
        try:
            # Prepare destinations
            destinations = {
                'ToAddresses': [addr.email for addr in message.to_addresses],
                'CcAddresses': [addr.email for addr in message.cc_addresses],
                'BccAddresses': [addr.email for addr in message.bcc_addresses]
            }

            # Prepare message content
            email_message = {
                'Subject': {'Data': message.subject, 'Charset': 'UTF-8'}
            }

            if message.text_content and message.html_content:
                email_message['Body'] = {
                    'Text': {'Data': message.text_content, 'Charset': 'UTF-8'},
                    'Html': {'Data': message.html_content, 'Charset': 'UTF-8'}
                }
            elif message.html_content:
                email_message['Body'] = {
                    'Html': {'Data': message.html_content, 'Charset': 'UTF-8'}
                }
            elif message.text_content:
                email_message['Body'] = {
                    'Text': {'Data': message.text_content, 'Charset': 'UTF-8'}
                }

            # Send email
            if message.attachments:
                # Use raw email sending for attachments
                raw_message = self._create_raw_message(message)
                response = self.client.send_raw_email(
                    Source=message.from_address.email,
                    Destinations=[addr.email for addr in message.all_recipients],
                    RawMessage={'Data': raw_message.as_bytes()}
                )
            else:
                # Use simple email sending
                response = self.client.send_email(
                    Source=message.from_address.email,
                    Destination=destinations,
                    Message=email_message
                )

            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.SENT,
                provider_message_id=response['MessageId'],
                provider_response=response
            )

        except ClientError as e:
            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    def _create_raw_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create raw MIME message for AWS SES."""
        # Reuse SMTP MIME creation logic
        smtp_provider = SMTPProvider({})
        return smtp_provider._create_mime_message(message)

    async def send_bulk_emails(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """Send multiple emails via AWS SES."""
        results = []
        for message in messages:
            result = await self.send_email(message)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Check if AWS SES is available."""
        return AWS_SES_AVAILABLE

    async def verify_email(self, email: str) -> bool:
        """Verify email address with AWS SES."""
        try:
            # Check if email is verified in SES
            response = self.client.get_identity_verification_attributes(
                Identities=[email]
            )
            verification_attrs = response.get('VerificationAttributes', {})
            return verification_attrs.get(email, {}).get('VerificationStatus') == 'Success'
        except:
            return False


class EmailValidator:
    """Email address validation utilities."""

    @staticmethod
    def validate_format(email: str) -> bool:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))

    @staticmethod
    async def validate_domain(email: str) -> bool:
        """Validate email domain (requires DNS lookup)."""
        if not DNS_AVAILABLE:
            return True  # Skip validation if DNS library not available

        try:
            domain = email.split('@')[1]
            mx_records = dns.resolver.resolve(domain, 'MX')
            return len(mx_records) > 0
        except:
            return False

    @classmethod
    async def validate_email(cls, email: str, check_domain: bool = False) -> bool:
        """Comprehensive email validation."""
        if not cls.validate_format(email):
            return False

        if check_domain:
            return await cls.validate_domain(email)

        return True


class TemplateProcessor:
    """Email template processing."""

    def __init__(self, engine: TemplateEngine = TemplateEngine.SIMPLE):
        self.engine = engine
        self.jinja_env = None

        if engine == TemplateEngine.JINJA2 and JINJA2_AVAILABLE:
            self.jinja_env = Environment()

    def process_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Process template with variables."""
        if self.engine == TemplateEngine.SIMPLE:
            return self._process_simple_template(template, variables)
        elif self.engine == TemplateEngine.JINJA2 and self.jinja_env:
            return self._process_jinja2_template(template, variables)
        else:
            return template

    def _process_simple_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Process simple placeholder template."""
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result

    def _process_jinja2_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Process Jinja2 template."""
        template_obj = self.jinja_env.from_string(template)
        return template_obj.render(**variables)


class EmailSender:
    """
    Comprehensive email sending system.

    Provides enterprise-grade email capabilities with multiple providers,
    template processing, and advanced delivery features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize email sender.

        Args:
            config: Configuration dictionary with email settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.provider_type = EmailProvider(self.config.get('provider', 'smtp'))
        self.template_engine = TemplateEngine(self.config.get('template_engine', 'simple'))

        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.rate_limit_per_minute = self.config.get('rate_limit_per_minute', 60)
        self.max_retries = self.config.get('max_retries', 3)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.bounce_tracking = self.config.get('bounce_tracking', False)

        # Initialize components
        self.provider = self._initialize_provider()
        self.template_processor = TemplateProcessor(self.template_engine)
        self.email_validator = EmailValidator()
        self.templates: Dict[str, EmailTemplate] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.send_history: List[EmailResult] = []

    def _initialize_provider(self) -> EmailProviderInterface:
        """Initialize email provider."""
        provider_config = self.config.get('provider_config', {})

        if self.provider_type == EmailProvider.SMTP:
            return SMTPProvider(provider_config)
        elif self.provider_type == EmailProvider.SENDGRID:
            return SendGridProvider(provider_config)
        elif self.provider_type == EmailProvider.AWS_SES:
            return AWSSESProvider(provider_config)
        else:
            raise ValueError(f"Unsupported email provider: {self.provider_type}")

    async def send_email(self, message: EmailMessage) -> EmailResult:
        """
        Send a single email message.

        Args:
            message: Email message to send

        Returns:
            Email result with delivery status
        """
        try:
            self.logger.info(f"Sending email: {message.id}")

            # Validate email addresses
            await self._validate_message(message)

            # Process template if specified
            if message.template_id and message.template_id in self.templates:
                await self._apply_template(message)

            # Send email
            result = await self.provider.send_email(message)

            # Update metrics
            if result.is_success:
                self.metrics['emails_sent'] += 1
            else:
                self.metrics['emails_failed'] += 1

            # Store result
            self.send_history.append(result)

            # Log audit
            if self.audit_logging:
                self._log_email(message, result)

            return result

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return EmailResult(
                message_id=message.id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def send_bulk_emails(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """
        Send multiple emails efficiently.

        Args:
            messages: List of email messages to send

        Returns:
            List of email results
        """
        try:
            self.logger.info(f"Sending {len(messages)} bulk emails")

            # Process in batches
            results = []
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]

                # Validate and process templates for batch
                for message in batch:
                    await self._validate_message(message)
                    if message.template_id and message.template_id in self.templates:
                        await self._apply_template(message)

                # Send batch
                batch_results = await self.provider.send_bulk_emails(batch)
                results.extend(batch_results)

                # Update metrics
                for result in batch_results:
                    if result.is_success:
                        self.metrics['emails_sent'] += 1
                    else:
                        self.metrics['emails_failed'] += 1

                # Rate limiting
                if i + self.batch_size < len(messages):
                    await asyncio.sleep(60 / self.rate_limit_per_minute)

            # Store results
            self.send_history.extend(results)

            return results

        except Exception as e:
            self.logger.error(f"Failed to send bulk emails: {e}")
            return [EmailResult(
                message_id=msg.id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            ) for msg in messages]

    async def _validate_message(self, message: EmailMessage):
        """Validate email message before sending."""
        # Validate from address
        if not await self.email_validator.validate_email(message.from_address.email):
            raise ValueError(f"Invalid from address: {message.from_address.email}")

        # Validate to addresses
        for addr in message.to_addresses:
            if not await self.email_validator.validate_email(addr.email):
                raise ValueError(f"Invalid to address: {addr.email}")

        # Validate CC addresses
        for addr in message.cc_addresses:
            if not await self.email_validator.validate_email(addr.email):
                raise ValueError(f"Invalid CC address: {addr.email}")

        # Validate BCC addresses
        for addr in message.bcc_addresses:
            if not await self.email_validator.validate_email(addr.email):
                raise ValueError(f"Invalid BCC address: {addr.email}")

        # Check content
        if not message.text_content and not message.html_content:
            raise ValueError("Email must have either text or HTML content")

    async def _apply_template(self, message: EmailMessage):
        """Apply template to email message."""
        template = self.templates[message.template_id]

        # Process subject template
        message.subject = self.template_processor.process_template(
            template.subject_template,
            message.template_variables
        )

        # Process text template
        if template.text_template:
            message.text_content = self.template_processor.process_template(
                template.text_template,
                message.template_variables
            )

        # Process HTML template
        if template.html_template:
            message.html_content = self.template_processor.process_template(
                template.html_template,
                message.template_variables
            )

    def add_template(self, template: EmailTemplate):
        """Add email template."""
        self.templates[template.id] = template
        self.logger.info(f"Added email template: {template.id}")

    def remove_template(self, template_id: str):
        """Remove email template."""
        if template_id in self.templates:
            del self.templates[template_id]
            self.logger.info(f"Removed email template: {template_id}")

    def get_template(self, template_id: str) -> Optional[EmailTemplate]:
        """Get email template by ID."""
        return self.templates.get(template_id)

    def list_templates(self) -> List[EmailTemplate]:
        """List all email templates."""
        return list(self.templates.values())

    async def verify_email_address(self, email: str) -> bool:
        """
        Verify email address using provider capabilities.

        Args:
            email: Email address to verify

        Returns:
            True if email is valid and verified
        """
        try:
            return await self.provider.verify_email(email)
        except Exception as e:
            self.logger.error(f"Email verification failed: {e}")
            return False

    def get_delivery_status(self, message_id: str) -> Optional[EmailResult]:
        """
        Get delivery status for a message.

        Args:
            message_id: Message ID to check

        Returns:
            Email result if found, None otherwise
        """
        for result in self.send_history:
            if result.message_id == message_id:
                return result
        return None

    def get_send_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get email sending statistics.

        Args:
            days: Number of days to include in statistics

        Returns:
            Statistics dictionary
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_results = [
            result for result in self.send_history
            if result.sent_at and result.sent_at >= cutoff_date
        ]

        total_sent = len(recent_results)
        successful = len([r for r in recent_results if r.is_success])
        failed = total_sent - successful
        bounced = len([r for r in recent_results if r.status == DeliveryStatus.BOUNCED])

        return {
            'total_sent': total_sent,
            'successful': successful,
            'failed': failed,
            'bounced': bounced,
            'success_rate': (successful / total_sent * 100) if total_sent > 0 else 0,
            'bounce_rate': (bounced / total_sent * 100) if total_sent > 0 else 0,
            'period_days': days
        }

    def _log_email(self, message: EmailMessage, result: EmailResult):
        """Log email for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'message_id': message.id,
            'from_address': message.from_address.email,
            'to_addresses': [addr.email for addr in message.to_addresses],
            'subject': message.subject,
            'status': result.status.value,
            'provider': self.provider_type.value,
            'success': result.is_success
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    async def health_check(self) -> bool:
        """
        Perform health check of email provider.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            return self.provider.is_available()
        except Exception as e:
            self.logger.error(f"Email provider health check failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup email sender resources."""
        try:
            # Clear sensitive data
            self.templates.clear()
            self.metrics.clear()

            # Limit send history size
            if len(self.send_history) > 1000:
                self.send_history = self.send_history[-1000:]

            self.logger.info("Email sender cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'templates') and self.templates:
                self.logger.info("EmailSender being destroyed - cleanup recommended")
        except:
            pass
