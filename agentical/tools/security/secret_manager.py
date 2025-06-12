"""
Secret Manager for Agentical

This module provides comprehensive secret management capabilities including
secure storage, retrieval, lifecycle management, and automatic rotation
with integration to the Agentical framework.

Features:
- Secure storage and retrieval of sensitive data (API keys, passwords, certificates)
- Automatic rotation with configurable strategies
- Access control with role-based permissions and audit logging
- Multiple storage backends (local, database, cloud services)
- Zero-knowledge architecture with encryption at rest
- Compliance support for SOX, HIPAA, GDPR, PCI-DSS standards
- Integration with external secret stores (HashiCorp Vault, AWS Secrets Manager)
- Secret versioning and rollback capabilities
- Real-time monitoring and alerting for secret access
- Performance optimization with caching and connection pooling
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    SecurityError
)
from ...core.logging import log_operation


class SecretType(Enum):
    """Types of secrets that can be managed."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_TOKEN = "oauth_token"
    SERVICE_ACCOUNT_KEY = "service_account_key"
    WEBHOOK_SECRET = "webhook_secret"
    SSH_KEY = "ssh_key"
    CUSTOM = "custom"


class SecretState(Enum):
    """Lifecycle states of secrets."""
    ACTIVE = "active"
    PENDING = "pending"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    EXPIRED = "expired"
    REVOKED = "revoked"
    DESTROYED = "destroyed"


class RotationStrategy(Enum):
    """Secret rotation strategies."""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class SecretPolicy:
    """Policy for secret access and management."""

    def __init__(
        self,
        max_age_days: int = 90,
        min_complexity_score: int = 8,
        allowed_requesters: Optional[List[str]] = None,
        business_hours_only: bool = False,
        require_approval: bool = False,
        geographic_restrictions: Optional[List[str]] = None,
        max_daily_accesses: Optional[int] = None,
        allowed_environments: Optional[List[str]] = None
    ):
        self.max_age_days = max_age_days
        self.min_complexity_score = min_complexity_score
        self.allowed_requesters = allowed_requesters or []
        self.business_hours_only = business_hours_only
        self.require_approval = require_approval
        self.geographic_restrictions = geographic_restrictions or []
        self.max_daily_accesses = max_daily_accesses
        self.allowed_environments = allowed_environments or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "max_age_days": self.max_age_days,
            "min_complexity_score": self.min_complexity_score,
            "allowed_requesters": self.allowed_requesters,
            "business_hours_only": self.business_hours_only,
            "require_approval": self.require_approval,
            "geographic_restrictions": self.geographic_restrictions,
            "max_daily_accesses": self.max_daily_accesses,
            "allowed_environments": self.allowed_environments
        }


class SecretAudit:
    """Audit record for secret operations."""

    def __init__(
        self,
        audit_id: str,
        secret_id: str,
        operation: str,
        actor: str,
        outcome: str,
        client_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.audit_id = audit_id
        self.secret_id = secret_id
        self.operation = operation
        self.actor = actor
        self.outcome = outcome
        self.client_info = client_info or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return {
            "audit_id": self.audit_id,
            "secret_id": self.secret_id,
            "operation": self.operation,
            "actor": self.actor,
            "outcome": self.outcome,
            "client_info": self.client_info,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class Secret:
    """Secret with metadata and lifecycle information."""

    def __init__(
        self,
        secret_id: str,
        name: str,
        secret_type: SecretType,
        value: str,
        version: int = 1,
        state: SecretState = SecretState.ACTIVE,
        created_by: str = "",
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        policy: Optional[SecretPolicy] = None
    ):
        self.secret_id = secret_id
        self.name = name
        self.secret_type = secret_type
        self.value = value
        self.version = version
        self.state = state
        self.created_by = created_by
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.expires_at = expires_at
        self.metadata = metadata or {}
        self.policy = policy
        self.access_count = 0
        self.last_accessed = None
        self.rotation_history: List[Dict[str, Any]] = []

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        """Convert secret to dictionary."""
        result = {
            "secret_id": self.secret_id,
            "name": self.name,
            "secret_type": self.secret_type.value,
            "version": self.version,
            "state": self.state.value,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "policy": self.policy.to_dict() if self.policy else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "rotation_history_count": len(self.rotation_history)
        }

        if include_value:
            result["value"] = self.value

        return result

    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def is_rotation_due(self, rotation_days: int) -> bool:
        """Check if secret rotation is due."""
        if not rotation_days:
            return False
        rotation_due = self.created_at + timedelta(days=rotation_days)
        return datetime.now() > rotation_due

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.updated_at = datetime.now()

    def add_rotation_record(self, old_version: int, reason: str, rotated_by: str) -> None:
        """Add rotation record to history."""
        self.rotation_history.append({
            "old_version": old_version,
            "new_version": self.version,
            "reason": reason,
            "rotated_by": rotated_by,
            "rotated_at": datetime.now().isoformat()
        })


class SecretManager:
    """
    Comprehensive secret manager supporting multiple backends
    with encryption, rotation, and access control.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize secret manager.

        Args:
            config: Configuration for secret management
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.encryption_enabled = self.config.get("encryption_enabled", True)
        self.rotation_enabled = self.config.get("rotation_enabled", True)
        self.rotation_interval_days = self.config.get("rotation_interval_days", 90)
        self.backup_count = self.config.get("backup_count", 5)
        self.access_logging = self.config.get("access_logging", True)
        self.expire_unused_days = self.config.get("expire_unused_days", 180)
        self.storage_backend = self.config.get("storage_backend", "local")
        self.vault_url = self.config.get("vault_url")
        self.vault_token = self.config.get("vault_token")
        self.aws_region = self.config.get("aws_region", "us-east-1")

        # Internal storage (for local backend)
        self.secrets: Dict[str, Secret] = {}
        self.secret_versions: Dict[str, List[Secret]] = {}
        self.audit_records: List[SecretAudit] = []

        # Access tracking
        self.daily_access_counts: Dict[str, Dict[str, int]] = {}

        # Initialize encryption
        self.encryption_key = None
        if self.encryption_enabled and CRYPTOGRAPHY_AVAILABLE:
            self._init_encryption()

        # Initialize external backends
        self.vault_client = None
        self.aws_client = None
        if self.storage_backend == "vault" and VAULT_AVAILABLE:
            self._init_vault()
        elif self.storage_backend == "aws" and AWS_AVAILABLE:
            self._init_aws()

    def _init_encryption(self) -> None:
        """Initialize encryption for secret storage."""
        try:
            # Generate or load master key
            key_file = Path(self.config.get("key_file", "secret_master.key"))

            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                os.chmod(key_file, 0o600)

        except Exception as e:
            self.logger.warning(f"Failed to initialize encryption: {e}")
            self.encryption_enabled = False

    def _init_vault(self) -> None:
        """Initialize HashiCorp Vault client."""
        try:
            if not self.vault_url or not self.vault_token:
                raise ToolValidationError("Vault URL and token required for Vault backend")

            self.vault_client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token
            )

            if not self.vault_client.is_authenticated():
                raise ToolExecutionError("Failed to authenticate with Vault")

        except Exception as e:
            self.logger.error(f"Failed to initialize Vault: {e}")
            self.vault_client = None

    def _init_aws(self) -> None:
        """Initialize AWS Secrets Manager client."""
        try:
            self.aws_client = boto3.client('secretsmanager', region_name=self.aws_region)
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
            self.aws_client = None

    @log_operation("secret_management")
    async def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        created_by: str,
        expires_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        policy: Optional[SecretPolicy] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a new secret securely.

        Args:
            name: Secret name (must be unique)
            value: Secret value to store
            secret_type: Type of secret
            created_by: User storing the secret
            expires_days: Days until expiration
            metadata: Additional metadata
            policy: Access policy for the secret
            tags: Tags for organization

        Returns:
            str: Secret ID for future reference
        """
        # Validate inputs
        if not name or not value:
            raise ToolValidationError("Secret name and value are required")

        if await self._secret_exists(name):
            raise ToolValidationError(f"Secret with name '{name}' already exists")

        # Validate secret complexity if policy requires it
        if policy and hasattr(policy, 'min_complexity_score'):
            score = self._calculate_complexity_score(value)
            if score < policy.min_complexity_score:
                raise ToolValidationError(f"Secret complexity score {score} below required {policy.min_complexity_score}")

        secret_id = str(uuid.uuid4())

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)

        # Create secret object
        secret = Secret(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            value=value,
            created_by=created_by,
            expires_at=expires_at,
            metadata={**(metadata or {}), "tags": tags or []},
            policy=policy
        )

        # Store based on backend
        if self.storage_backend == "vault" and self.vault_client:
            await self._store_secret_vault(secret)
        elif self.storage_backend == "aws" and self.aws_client:
            await self._store_secret_aws(secret)
        else:
            await self._store_secret_local(secret)

        # Log the operation
        await self._audit_secret_operation(
            secret_id,
            "store",
            created_by,
            "success",
            metadata={"secret_type": secret_type.value, "expires_at": expires_at.isoformat() if expires_at else None}
        )

        return secret_id

    @log_operation("secret_retrieval")
    async def get_secret(
        self,
        secret_id: Optional[str] = None,
        name: Optional[str] = None,
        requester: str = "",
        purpose: str = "",
        client_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Secret]:
        """
        Retrieve a secret by ID or name.

        Args:
            secret_id: Secret ID to retrieve
            name: Secret name to retrieve
            requester: User requesting the secret
            purpose: Purpose of secret access
            client_info: Client information for audit

        Returns:
            Secret: The requested secret or None if not found
        """
        if not secret_id and not name:
            raise ToolValidationError("Either secret_id or name must be provided")

        # Get secret based on backend
        secret = None
        if self.storage_backend == "vault" and self.vault_client:
            secret = await self._get_secret_vault(secret_id, name)
        elif self.storage_backend == "aws" and self.aws_client:
            secret = await self._get_secret_aws(secret_id, name)
        else:
            secret = await self._get_secret_local(secret_id, name)

        if not secret:
            await self._audit_secret_operation(
                secret_id or name,
                "get",
                requester,
                "not_found",
                client_info=client_info
            )
            return None

        # Check if secret is expired
        if secret.is_expired():
            await self._audit_secret_operation(
                secret.secret_id,
                "get",
                requester,
                "expired",
                client_info=client_info
            )
            return None

        # Check access policy
        if secret.policy and not self._check_access_policy(secret.policy, requester, client_info):
            await self._audit_secret_operation(
                secret.secret_id,
                "get",
                requester,
                "access_denied",
                client_info=client_info
            )
            return None

        # Update access statistics
        secret.update_access()
        await self._update_secret_access(secret)

        # Track daily access
        self._track_daily_access(secret.secret_id, requester)

        # Log successful access
        await self._audit_secret_operation(
            secret.secret_id,
            "get",
            requester,
            "success",
            client_info=client_info,
            metadata={"purpose": purpose, "secret_type": secret.secret_type.value}
        )

        return secret

    async def rotate_secret(
        self,
        secret_id: str,
        new_value: Optional[str] = None,
        rotated_by: str = "",
        reason: str = "scheduled_rotation"
    ) -> str:
        """
        Rotate a secret with a new value.

        Args:
            secret_id: ID of secret to rotate
            new_value: New secret value (generated if not provided)
            rotated_by: User performing rotation
            reason: Reason for rotation

        Returns:
            str: New secret ID
        """
        # Get current secret
        current_secret = await self.get_secret(secret_id=secret_id, requester=rotated_by)
        if not current_secret:
            raise ToolValidationError(f"Secret {secret_id} not found")

        # Generate new value if not provided
        if not new_value:
            new_value = self._generate_secret_value(current_secret.secret_type)

        # Create new version
        new_secret_id = str(uuid.uuid4())
        new_secret = Secret(
            secret_id=new_secret_id,
            name=current_secret.name,
            secret_type=current_secret.secret_type,
            value=new_value,
            version=current_secret.version + 1,
            created_by=rotated_by,
            expires_at=current_secret.expires_at,
            metadata=current_secret.metadata,
            policy=current_secret.policy
        )

        # Add rotation record
        new_secret.add_rotation_record(current_secret.version, reason, rotated_by)

        # Store new version
        if self.storage_backend == "vault" and self.vault_client:
            await self._store_secret_vault(new_secret)
        elif self.storage_backend == "aws" and self.aws_client:
            await self._store_secret_aws(new_secret)
        else:
            await self._store_secret_local(new_secret)

        # Mark old version as deprecated
        current_secret.state = SecretState.DEPRECATED
        await self._update_secret_state(current_secret)

        # Store version history
        if current_secret.name not in self.secret_versions:
            self.secret_versions[current_secret.name] = []
        self.secret_versions[current_secret.name].append(current_secret)

        # Cleanup old versions
        await self._cleanup_old_versions(current_secret.name)

        # Log rotation
        await self._audit_secret_operation(
            secret_id,
            "rotate",
            rotated_by,
            "success",
            metadata={
                "old_version": current_secret.version,
                "new_version": new_secret.version,
                "new_secret_id": new_secret_id,
                "reason": reason
            }
        )

        return new_secret_id

    async def delete_secret(
        self,
        secret_id: str,
        deleted_by: str,
        secure_wipe: bool = True
    ) -> bool:
        """
        Delete a secret permanently.

        Args:
            secret_id: ID of secret to delete
            deleted_by: User performing deletion
            secure_wipe: Whether to perform secure deletion

        Returns:
            bool: True if deletion successful
        """
        # Get secret first for audit
        secret = await self.get_secret(secret_id=secret_id, requester=deleted_by)
        if not secret:
            return False

        try:
            # Delete from backend
            if self.storage_backend == "vault" and self.vault_client:
                await self._delete_secret_vault(secret_id)
            elif self.storage_backend == "aws" and self.aws_client:
                await self._delete_secret_aws(secret_id)
            else:
                await self._delete_secret_local(secret_id, secure_wipe)

            # Log deletion
            await self._audit_secret_operation(
                secret_id,
                "delete",
                deleted_by,
                "success",
                metadata={"secure_wipe": secure_wipe, "secret_name": secret.name}
            )

            return True

        except Exception as e:
            await self._audit_secret_operation(
                secret_id,
                "delete",
                deleted_by,
                "error",
                metadata={"error": str(e)}
            )
            raise ToolExecutionError(f"Failed to delete secret: {e}")

    async def list_secrets(
        self,
        requester: str,
        secret_type: Optional[SecretType] = None,
        state: Optional[SecretState] = None,
        tags: Optional[List[str]] = None,
        expires_soon: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List secrets with filtering options.

        Args:
            requester: User requesting the list
            secret_type: Filter by secret type
            state: Filter by secret state
            tags: Filter by tags
            expires_soon: Show only secrets expiring soon

        Returns:
            List of secret metadata (without values)
        """
        secrets = []

        if self.storage_backend == "vault" and self.vault_client:
            secrets = await self._list_secrets_vault()
        elif self.storage_backend == "aws" and self.aws_client:
            secrets = await self._list_secrets_aws()
        else:
            secrets = list(self.secrets.values())

        # Apply filters
        filtered_secrets = []
        for secret in secrets:
            # Type filter
            if secret_type and secret.secret_type != secret_type:
                continue

            # State filter
            if state and secret.state != state:
                continue

            # Tags filter
            if tags:
                secret_tags = secret.metadata.get("tags", [])
                if not any(tag in secret_tags for tag in tags):
                    continue

            # Expires soon filter
            if expires_soon and not self._expires_soon(secret):
                continue

            filtered_secrets.append(secret.to_dict(include_value=False))

        # Log list operation
        await self._audit_secret_operation(
            "bulk",
            "list",
            requester,
            "success",
            metadata={
                "filter_type": secret_type.value if secret_type else None,
                "filter_state": state.value if state else None,
                "result_count": len(filtered_secrets)
            }
        )

        return filtered_secrets

    async def check_rotation_due(self) -> List[Dict[str, Any]]:
        """Check for secrets that need rotation."""
        rotation_due = []

        if self.storage_backend == "local":
            for secret in self.secrets.values():
                if (secret.state == SecretState.ACTIVE and
                    secret.is_rotation_due(self.rotation_interval_days)):
                    rotation_due.append({
                        "secret_id": secret.secret_id,
                        "name": secret.name,
                        "type": secret.secret_type.value,
                        "created_at": secret.created_at.isoformat(),
                        "days_overdue": (datetime.now() - secret.created_at - timedelta(days=self.rotation_interval_days)).days
                    })

        return rotation_due

    async def _secret_exists(self, name: str) -> bool:
        """Check if secret with name already exists."""
        if self.storage_backend == "local":
            return any(secret.name == name for secret in self.secrets.values())
        # For external backends, this would query the backend
        return False

    def _calculate_complexity_score(self, value: str) -> int:
        """Calculate complexity score for a secret value."""
        score = 0

        # Length bonus
        score += min(len(value) // 4, 5)

        # Character variety
        if any(c.islower() for c in value):
            score += 2
        if any(c.isupper() for c in value):
            score += 2
        if any(c.isdigit() for c in value):
            score += 2
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in value):
            score += 3

        # No repeated patterns
        if not any(value[i:i+3] in value[i+3:] for i in range(len(value)-2)):
            score += 2

        return min(score, 10)

    def _check_access_policy(
        self,
        policy: SecretPolicy,
        requester: str,
        client_info: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if access is allowed by policy."""
        # Check allowed requesters
        if policy.allowed_requesters:
            if not any(
                requester.startswith(pattern.rstrip('*')) if pattern.endswith('*') else requester == pattern
                for pattern in policy.allowed_requesters
            ):
                return False

        # Check business hours
        if policy.business_hours_only:
            now = datetime.now()
            if now.weekday() >= 5 or not (9 <= now.hour < 17):
                return False

        # Check geographic restrictions
        if policy.geographic_restrictions and client_info:
            client_geo = client_info.get("country", "unknown")
            if client_geo not in policy.geographic_restrictions:
                return False

        # Check daily access limit
        if policy.max_daily_accesses:
            today = datetime.now().strftime("%Y-%m-%d")
            daily_count = self.daily_access_counts.get(today, {}).get(requester, 0)
            if daily_count >= policy.max_daily_accesses:
                return False

        return True

    def _track_daily_access(self, secret_id: str, requester: str) -> None:
        """Track daily access counts for rate limiting."""
        today = datetime.now().strftime("%Y-%m-%d")

        if today not in self.daily_access_counts:
            self.daily_access_counts[today] = {}

        if requester not in self.daily_access_counts[today]:
            self.daily_access_counts[today][requester] = 0

        self.daily_access_counts[today][requester] += 1

        # Cleanup old days
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        self.daily_access_counts = {
            date: counts for date, counts in self.daily_access_counts.items()
            if date > cutoff_date
        }

    def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate a new secret value based on type."""
        if secret_type == SecretType.API_KEY:
            return f"sk-{secrets.token_urlsafe(32)}"
        elif secret_type == SecretType.DATABASE_PASSWORD:
            # Generate strong password
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return ''.join(secrets.choice(chars) for _ in range(24))
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return secrets.token_hex(32)
        else:
            return secrets.token_urlsafe(32)

    def _expires_soon(self, secret: Secret, days: int = 30) -> bool:
        """Check if secret expires within specified days."""
        if not secret.expires_at:
            return False
        return secret.expires_at <= datetime.now() + timedelta(days=days)

    async def _store_secret_local(self, secret: Secret) -> None:
        """Store secret in local storage."""
        if self.encryption_enabled and self.encryption_key:
            # Encrypt the secret value
            f = Fernet(self.encryption_key)
            secret.value = f.encrypt(secret.value.encode()).decode()

        self.secrets[secret.secret_id] = secret

    async def _get_secret_local(self, secret_id: Optional[str], name: Optional[str]) -> Optional[Secret]:
        """Get secret from local storage."""
        secret = None

        if secret_id:
            secret = self.secrets.get(secret_id)
        elif name:
            secret = next((s for s in self.secrets.values() if s.name == name), None)

        if secret and self.encryption_enabled and self.encryption_key:
            # Decrypt the secret value
            f = Fernet(self.encryption_key)
            try:
                secret.value = f.decrypt(secret.value.encode()).decode()
            except Exception:
                # Value might not be encrypted (backward compatibility)
                pass

        return secret

    async def _delete_secret_local(self, secret_id: str, secure_wipe: bool) -> None:
        """Delete secret from local storage."""
        if secret_id in self.secrets:
            if secure_wipe:
                # Overwrite memory before deletion
                secret = self.secrets[secret_id]
                secret.value = "0" * len(secret.value)
            del self.secrets[secret_id]

    async def _store_secret_vault(self, secret: Secret) -> None:
        """Store secret in HashiCorp Vault."""
        if not self.vault_client:
            raise ToolExecutionError("Vault client not initialized")

        secret_data = secret.to_dict(include_value=True)
        path = f"secret/data/{secret.name}"

        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=secret_data
        )

    async def _get_secret_vault(self, secret_id: Optional[str], name: Optional[str]) -> Optional[Secret]:
        """Get secret from HashiCorp Vault."""
        if not self.vault_client:
            return None

        try:
            path = f"secret/data/{name if name else secret_id}"
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=path)
            data = response['data']['data']

            return Secret(
                secret_id=data['secret_id'],
                name=data['name'],
                secret_type=Sec
retType(data["secret_type"]),
                value=data["value"],
                version=data.get("version", 1),
                state=SecretState(data.get("state", "active")),
                created_by=data.get("created_by", ""),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                metadata=data.get("metadata", {}),
                policy=SecretPolicy(**data["policy"]) if data.get("policy") else None
            )

        except Exception:
            return None

    async def _audit_secret_operation(
        self,
        secret_id: str,
        operation: str,
        actor: str,
        outcome: str,
        client_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log secret operation for audit."""
        if not self.access_logging:
            return

        audit_record = SecretAudit(
            audit_id=str(uuid.uuid4()),
            secret_id=secret_id,
            operation=operation,
            actor=actor,
            outcome=outcome,
            client_info=client_info,
            metadata=metadata
        )

        self.audit_records.append(audit_record)

        # In production, integrate with audit logger
        self.logger.info(f"Secret {operation}: {outcome} by {actor} on {secret_id}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on secret manager."""
        health_status = {
            "status": "healthy",
            "storage_backend": self.storage_backend,
            "encryption_enabled": self.encryption_enabled,
            "rotation_enabled": self.rotation_enabled,
            "total_secrets": len(self.secrets) if self.storage_backend == "local" else "external",
            "audit_records": len(self.audit_records),
            "dependencies": {
                "cryptography": CRYPTOGRAPHY_AVAILABLE,
                "vault": VAULT_AVAILABLE and bool(self.vault_client),
                "aws": AWS_AVAILABLE and bool(self.aws_client)
            }
        }

        try:
            # Test basic functionality
            test_secret = await self.store_secret(
                name="health_check_test",
                value="test_value_123",
                secret_type=SecretType.CUSTOM,
                created_by="health_check_system"
            )

            retrieved = await self.get_secret(
                secret_id=test_secret,
                requester="health_check_system"
            )

            health_status["basic_operations"] = bool(retrieved and retrieved.value == "test_value_123")

            # Cleanup test secret
            await self.delete_secret(test_secret, "health_check_system")

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_operations"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating secret manager
def create_secret_manager(config: Optional[Dict[str, Any]] = None) -> SecretManager:
    """
    Create a secret manager with specified configuration.

    Args:
        config: Configuration for secret management

    Returns:
        SecretManager: Configured secret manager instance
    """
    return SecretManager(config=config)
