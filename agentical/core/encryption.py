"""
Data Protection and Encryption System for Agentical

This module provides comprehensive data protection and encryption capabilities
including field-level encryption, secure data handling, key management,
and compliance with data protection standards.

Features:
- Field-level encryption for sensitive data
- Secure key derivation and management
- Multiple encryption algorithms support
- Data masking and anonymization
- Secure configuration handling
- PII (Personally Identifiable Information) protection
- Integration with database models and repositories
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

import logfire

from agentical.core.exceptions import SecurityError, ValidationError
from agentical.core.structured_logging import StructuredLogger

# Initialize logger
logger = StructuredLogger("encryption")

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    FERNET = "fernet"
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"

class DataClassification(Enum):
    """Data classification levels for encryption requirements."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"

# Classification-Algorithm mapping
CLASSIFICATION_ALGORITHMS = {
    DataClassification.PUBLIC: None,  # No encryption required
    DataClassification.INTERNAL: EncryptionAlgorithm.FERNET,
    DataClassification.CONFIDENTIAL: EncryptionAlgorithm.AES_256_GCM,
    DataClassification.RESTRICTED: EncryptionAlgorithm.AES_256_GCM,
    DataClassification.TOP_SECRET: EncryptionAlgorithm.CHACHA20_POLY1305,
}

class EncryptionKey:
    """Encryption key management and operations."""

    def __init__(self, key_data: bytes, algorithm: EncryptionAlgorithm, key_id: str = None):
        self.key_data = key_data
        self.algorithm = algorithm
        self.key_id = key_id or secrets.token_hex(16)
        self.created_at = datetime.utcnow()

    @classmethod
    def generate(cls, algorithm: EncryptionAlgorithm, key_id: str = None) -> 'EncryptionKey':
        """Generate a new encryption key for the specified algorithm."""

        if algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
        else:
            raise ValidationError(f"Unsupported algorithm: {algorithm}")

        return cls(key_data, algorithm, key_id)

    @classmethod
    def derive_from_password(
        cls,
        password: str,
        salt: bytes,
        algorithm: EncryptionAlgorithm,
        iterations: int = 100000
    ) -> 'EncryptionKey':
        """Derive encryption key from password using PBKDF2."""

        if algorithm == EncryptionAlgorithm.FERNET:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            key_data = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            key_data = kdf.derive(password.encode())

        key_id = hashlib.sha256(salt + password.encode()).hexdigest()[:16]
        return cls(key_data, algorithm, key_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert key to dictionary representation (without key data)."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "created_at": self.created_at.isoformat(),
        }

class EncryptionManager:
    """Main encryption management class for data protection operations."""

    def __init__(self, master_key: Optional[str] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise SecurityError("Cryptography library not available")

        self.master_key = master_key or os.getenv("AGENTICAL_MASTER_KEY")
        if not self.master_key:
            logger.warning("No master key provided - generating temporary key")
            self.master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

        self.keys: Dict[str, EncryptionKey] = {}
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM
        self._initialize_default_keys()

    def _initialize_default_keys(self):
        """Initialize default encryption keys."""

        with logfire.span("Initialize encryption keys"):
            try:
                # Generate salt from master key
                salt = hashlib.sha256(self.master_key.encode()).digest()

                # Create keys for each algorithm
                for algorithm in EncryptionAlgorithm:
                    key = EncryptionKey.derive_from_password(
                        password=self.master_key,
                        salt=salt + algorithm.value.encode(),
                        algorithm=algorithm,
                        iterations=100000
                    )
                    self.keys[algorithm.value] = key

                logger.info("Encryption keys initialized", key_count=len(self.keys))

            except Exception as e:
                logger.error("Failed to initialize encryption keys", error=str(e))
                raise SecurityError(f"Key initialization failed: {str(e)}")

    def get_key(self, algorithm: EncryptionAlgorithm) -> EncryptionKey:
        """Get encryption key for algorithm."""
        key = self.keys.get(algorithm.value)
        if not key:
            raise SecurityError(f"No key available for algorithm: {algorithm}")
        return key

    def encrypt_data(
        self,
        data: Union[str, bytes],
        algorithm: Optional[EncryptionAlgorithm] = None,
        classification: Optional[DataClassification] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Encrypt data using specified algorithm or classification level."""

        if classification:
            algorithm = CLASSIFICATION_ALGORITHMS.get(classification)
            if not algorithm:
                # No encryption required for this classification
                if isinstance(data, str):
                    data = data.encode()
                return data, {"encrypted": False, "classification": classification.value}

        algorithm = algorithm or self.default_algorithm

        with logfire.span("Encrypt data", algorithm=algorithm.value):
            try:
                if isinstance(data, str):
                    data = data.encode('utf-8')

                key = self.get_key(algorithm)

                if algorithm == EncryptionAlgorithm.FERNET:
                    fernet = Fernet(key.key_data)
                    encrypted_data = fernet.encrypt(data)

                elif algorithm == EncryptionAlgorithm.AES_256_GCM:
                    iv = secrets.token_bytes(12)  # 96-bit IV for GCM
                    cipher = Cipher(
                        algorithms.AES(key.key_data),
                        modes.GCM(iv),
                        backend=default_backend()
                    )
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(data) + encryptor.finalize()
                    encrypted_data = iv + encryptor.tag + ciphertext

                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    iv = secrets.token_bytes(16)  # 128-bit IV for CBC
                    # Pad data to block size
                    padding_length = 16 - (len(data) % 16)
                    padded_data = data + bytes([padding_length] * padding_length)

                    cipher = Cipher(
                        algorithms.AES(key.key_data),
                        modes.CBC(iv),
                        backend=default_backend()
                    )
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                    encrypted_data = iv + ciphertext

                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    nonce = secrets.token_bytes(12)  # 96-bit nonce
                    cipher = Cipher(
                        algorithms.ChaCha20(key.key_data, nonce),
                        mode=None,
                        backend=default_backend()
                    )
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(data) + encryptor.finalize()
                    encrypted_data = nonce + ciphertext

                else:
                    raise SecurityError(f"Unsupported encryption algorithm: {algorithm}")

                metadata = {
                    "encrypted": True,
                    "algorithm": algorithm.value,
                    "key_id": key.key_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if classification:
                    metadata["classification"] = classification.value

                return encrypted_data, metadata

            except Exception as e:
                logger.error("Data encryption failed", error=str(e), algorithm=algorithm.value)
                raise SecurityError(f"Encryption failed: {str(e)}")

    def decrypt_data(
        self,
        encrypted_data: bytes,
        metadata: Dict[str, Any]
    ) -> bytes:
        """Decrypt data using metadata information."""

        if not metadata.get("encrypted", False):
            return encrypted_data

        algorithm_str = metadata.get("algorithm")
        if not algorithm_str:
            raise SecurityError("Missing algorithm in metadata")

        try:
            algorithm = EncryptionAlgorithm(algorithm_str)
        except ValueError:
            raise SecurityError(f"Unknown algorithm: {algorithm_str}")

        with logfire.span("Decrypt data", algorithm=algorithm.value):
            try:
                key = self.get_key(algorithm)

                if algorithm == EncryptionAlgorithm.FERNET:
                    fernet = Fernet(key.key_data)
                    decrypted_data = fernet.decrypt(encrypted_data)

                elif algorithm == EncryptionAlgorithm.AES_256_GCM:
                    iv = encrypted_data[:12]
                    tag = encrypted_data[12:28]
                    ciphertext = encrypted_data[28:]

                    cipher = Cipher(
                        algorithms.AES(key.key_data),
                        modes.GCM(iv, tag),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    iv = encrypted_data[:16]
                    ciphertext = encrypted_data[16:]

                    cipher = Cipher(
                        algorithms.AES(key.key_data),
                        modes.CBC(iv),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

                    # Remove padding
                    padding_length = padded_data[-1]
                    decrypted_data = padded_data[:-padding_length]

                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    nonce = encrypted_data[:12]
                    ciphertext = encrypted_data[12:]

                    cipher = Cipher(
                        algorithms.ChaCha20(key.key_data, nonce),
                        mode=None,
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

                else:
                    raise SecurityError(f"Unsupported decryption algorithm: {algorithm}")

                return decrypted_data

            except Exception as e:
                logger.error("Data decryption failed", error=str(e), algorithm=algorithm.value)
                raise SecurityError(f"Decryption failed: {str(e)}")

    def encrypt_field(
        self,
        value: Any,
        field_classification: DataClassification = DataClassification.CONFIDENTIAL
    ) -> Dict[str, Any]:
        """Encrypt a field value and return encrypted package."""

        if value is None:
            return {"value": None, "encrypted": False}

        # Convert value to JSON string
        json_value = json.dumps(value, default=str)

        encrypted_data, metadata = self.encrypt_data(
            json_value,
            classification=field_classification
        )

        return {
            "value": base64.b64encode(encrypted_data).decode('utf-8'),
            "metadata": metadata
        }

    def decrypt_field(self, encrypted_package: Dict[str, Any]) -> Any:
        """Decrypt a field value from encrypted package."""

        if not encrypted_package.get("metadata", {}).get("encrypted", False):
            return encrypted_package.get("value")

        encrypted_data = base64.b64decode(encrypted_package["value"])
        metadata = encrypted_package["metadata"]

        decrypted_bytes = self.decrypt_data(encrypted_data, metadata)
        json_value = decrypted_bytes.decode('utf-8')

        return json.loads(json_value)

class DataMasker:
    """Data masking and anonymization utilities."""

    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address."""
        if not email or '@' not in email:
            return email

        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number."""
        if not phone:
            return phone

        # Remove non-digits
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) < 4:
            return '*' * len(phone)

        # Keep last 4 digits
        masked = '*' * (len(digits) - 4) + digits[-4:]

        # Restore original format
        result = phone
        digit_index = 0
        for i, char in enumerate(phone):
            if char.isdigit():
                if digit_index < len(masked):
                    result = result[:i] + masked[digit_index] + result[i+1:]
                    digit_index += 1

        return result

    @staticmethod
    def mask_credit_card(card_number: str) -> str:
        """Mask credit card number."""
        if not card_number:
            return card_number

        digits = ''.join(c for c in card_number if c.isdigit())
        if len(digits) < 4:
            return '*' * len(card_number)

        # Keep last 4 digits
        masked_digits = '*' * (len(digits) - 4) + digits[-4:]

        # Restore original format
        result = card_number
        digit_index = 0
        for i, char in enumerate(card_number):
            if char.isdigit():
                if digit_index < len(masked_digits):
                    result = result[:i] + masked_digits[digit_index] + result[i+1:]
                    digit_index += 1

        return result

    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """Mask Social Security Number."""
        if not ssn:
            return ssn

        digits = ''.join(c for c in ssn if c.isdigit())
        if len(digits) < 4:
            return '*' * len(ssn)

        # Keep last 4 digits
        masked_digits = '*' * (len(digits) - 4) + digits[-4:]

        # Restore original format
        result = ssn
        digit_index = 0
        for i, char in enumerate(ssn):
            if char.isdigit():
                if digit_index < len(masked_digits):
                    result = result[:i] + masked_digits[digit_index] + result[i+1:]
                    digit_index += 1

        return result

    @classmethod
    def mask_pii(cls, value: str, pii_type: PIIType) -> str:
        """Mask PII based on type."""

        if pii_type == PIIType.EMAIL:
            return cls.mask_email(value)
        elif pii_type == PIIType.PHONE:
            return cls.mask_phone(value)
        elif pii_type == PIIType.CREDIT_CARD:
            return cls.mask_credit_card(value)
        elif pii_type == PIIType.SSN:
            return cls.mask_ssn(value)
        elif pii_type in [PIIType.NAME, PIIType.ADDRESS]:
            # Simple masking for names and addresses
            if len(value) <= 2:
                return '*' * len(value)
            return value[0] + '*' * (len(value) - 2) + value[-1]
        else:
            # Default masking
            return '*' * min(len(value), 8)

class SecureConfigManager:
    """Secure configuration management with encryption."""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.encrypted_configs: Dict[str, Dict[str, Any]] = {}

    def set_config(
        self,
        key: str,
        value: Any,
        classification: DataClassification = DataClassification.CONFIDENTIAL
    ):
        """Set encrypted configuration value."""

        encrypted_package = self.encryption_manager.encrypt_field(value, classification)
        self.encrypted_configs[key] = encrypted_package

        logger.info("Configuration set", key=key, classification=classification.value)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get decrypted configuration value."""

        encrypted_package = self.encrypted_configs.get(key)
        if not encrypted_package:
            return default

        try:
            return self.encryption_manager.decrypt_field(encrypted_package)
        except Exception as e:
            logger.error("Failed to decrypt config", key=key, error=str(e))
            return default

    def delete_config(self, key: str) -> bool:
        """Delete configuration value."""

        if key in self.encrypted_configs:
            del self.encrypted_configs[key]
            logger.info("Configuration deleted", key=key)
            return True
        return False

    def list_config_keys(self) -> List[str]:
        """List available configuration keys."""
        return list(self.encrypted_configs.keys())

# Global instances
_encryption_manager: Optional[EncryptionManager] = None
_secure_config: Optional[SecureConfigManager] = None

def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager instance."""
    global _encryption_manager

    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()

    return _encryption_manager

def get_secure_config() -> SecureConfigManager:
    """Get global secure configuration manager instance."""
    global _secure_config

    if _secure_config is None:
        _secure_config = SecureConfigManager(get_encryption_manager())

    return _secure_config

# Utility functions
def encrypt_sensitive_data(
    data: Union[str, Dict[str, Any]],
    classification: DataClassification = DataClassification.CONFIDENTIAL
) -> Dict[str, Any]:
    """Encrypt sensitive data with specified classification."""
    manager = get_encryption_manager()
    return manager.encrypt_field(data, classification)

def decrypt_sensitive_data(encrypted_package: Dict[str, Any]) -> Any:
    """Decrypt sensitive data from encrypted package."""
    manager = get_encryption_manager()
    return manager.decrypt_field(encrypted_package)

def mask_sensitive_field(value: str, pii_type: PIIType) -> str:
    """Mask sensitive field based on PII type."""
    return DataMasker.mask_pii(value, pii_type)

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token."""
    return secrets.token_urlsafe(length)

def secure_hash(data: str, salt: Optional[str] = None) -> str:
    """Generate secure hash of data with optional salt."""
    if salt is None:
        salt = secrets.token_hex(16)

    combined = f"{salt}:{data}"
    hash_obj = hashlib.sha256(combined.encode())
    return f"{salt}:{hash_obj.hexdigest()}"

def verify_secure_hash(data: str, hashed: str) -> bool:
    """Verify data against secure hash."""
    try:
        salt, hash_value = hashed.split(':', 1)
        expected_hash = secure_hash(data, salt)
        return hmac.compare_digest(expected_hash, hashed)
    except ValueError:
        return False
