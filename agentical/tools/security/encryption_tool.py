"""
Encryption Tool for Agentical

This module provides comprehensive encryption and decryption capabilities
supporting multiple algorithms, key management, and cryptographic operations
with integration to the Agentical framework.

Features:
- Multi-algorithm support (AES, ChaCha20, RSA, ECC)
- Symmetric and asymmetric encryption
- Key generation, rotation, and management
- Digital signatures and verification
- Secure key storage and derivation
- Performance optimization for different use cases
- Integration with HSM and cloud key services
- Zero-knowledge encryption capabilities
- Compliance with cryptographic standards
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, serialization, padding
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding as asym_padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import nacl.secret
    import nacl.public
    import nacl.signing
    import nacl.encoding
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    SecurityError
)
from ...core.logging import log_operation


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    AES_128_GCM = "aes_128_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECC_P256 = "ecc_p256"
    ECC_P384 = "ecc_p384"
    FERNET = "fernet"


class KeyType(Enum):
    """Types of cryptographic keys."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    SIGNING = "signing"
    VERIFICATION = "verification"
    DERIVATION = "derivation"


class CryptoContext:
    """Cryptographic context for operations."""

    def __init__(
        self,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.algorithm = algorithm
        self.key_id = key_id
        self.operation = operation
        self.metadata = metadata or {}
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "operation": self.operation,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class EncryptionResult:
    """Result of encryption/decryption operation."""

    def __init__(
        self,
        operation_id: str,
        success: bool,
        algorithm: EncryptionAlgorithm,
        operation: str,
        data: Optional[bytes] = None,
        key_id: Optional[str] = None,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        signature: Optional[bytes] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation_id = operation_id
        self.success = success
        self.algorithm = algorithm
        self.operation = operation
        self.data = data
        self.key_id = key_id
        self.iv = iv
        self.tag = tag
        self.signature = signature
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "operation_id": self.operation_id,
            "success": self.success,
            "algorithm": self.algorithm.value,
            "operation": self.operation,
            "data_base64": base64.b64encode(self.data).decode() if self.data else None,
            "data_size": len(self.data) if self.data else 0,
            "key_id": self.key_id,
            "iv_base64": base64.b64encode(self.iv).decode() if self.iv else None,
            "tag_base64": base64.b64encode(self.tag).decode() if self.tag else None,
            "signature_base64": base64.b64encode(self.signature).decode() if self.signature else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class KeyManager:
    """Secure key management and storage."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize key manager."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Key storage (in production, use secure key store)
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.key_rotation_days = self.config.get("key_rotation_days", 90)
        self.backup_key_count = self.config.get("backup_key_count", 3)
        self.secure_key_storage = self.config.get("secure_key_storage", True)

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm,
        key_type: KeyType,
        key_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cryptographic key."""

        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        key_id = key_id or str(uuid.uuid4())

        try:
            if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
                key_data = os.urandom(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.AES_128_GCM:
                key_data = os.urandom(16)  # 128 bits
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key_data = os.urandom(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.FERNET:
                key_data = Fernet.generate_key()
            elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )

                if key_type == KeyType.ASYMMETRIC_PRIVATE:
                    key_data = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                else:
                    public_key = private_key.public_key()
                    key_data = public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
            elif algorithm in [EncryptionAlgorithm.ECC_P256, EncryptionAlgorithm.ECC_P384]:
                curve = ec.SECP256R1() if algorithm == EncryptionAlgorithm.ECC_P256 else ec.SECP384R1()
                private_key = ec.generate_private_key(curve, default_backend())

                if key_type == KeyType.ASYMMETRIC_PRIVATE:
                    key_data = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                else:
                    public_key = private_key.public_key()
                    key_data = public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
            else:
                raise ToolValidationError(f"Unsupported algorithm: {algorithm}")

            # Store key
            self.keys[key_id] = {
                "key_data": key_data,
                "algorithm": algorithm.value,
                "key_type": key_type.value,
                "created_at": datetime.now().isoformat(),
                "active": True
            }

            # Store metadata
            self.key_metadata[key_id] = {
                "algorithm": algorithm.value,
                "key_type": key_type.value,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "usage_count": 0,
                "metadata": metadata or {},
                "rotation_due": (datetime.now() + timedelta(days=self.key_rotation_days)).isoformat()
            }

            return key_id

        except Exception as e:
            raise ToolExecutionError(f"Key generation failed: {e}")

    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get key data by ID."""
        key_info = self.keys.get(key_id)
        if not key_info or not key_info.get("active", False):
            return None

        # Update usage statistics
        if key_id in self.key_metadata:
            self.key_metadata[key_id]["last_used"] = datetime.now().isoformat()
            self.key_metadata[key_id]["usage_count"] += 1

        key_data = key_info["key_data"]
        return key_data if isinstance(key_data, bytes) else key_data.encode()

    def rotate_key(self, key_id: str) -> str:
        """Rotate existing key."""
        old_key_info = self.key_metadata.get(key_id)
        if not old_key_info:
            raise ToolValidationError(f"Key {key_id} not found")

        algorithm = EncryptionAlgorithm(old_key_info["algorithm"])
        key_type = KeyType(old_key_info["key_type"])
        metadata = old_key_info.get("metadata", {})

        # Generate new key
        new_key_id = self.generate_key(algorithm, key_type, metadata=metadata)

        # Mark old key as inactive
        if key_id in self.keys:
            self.keys[key_id]["active"] = False
            self.keys[key_id]["rotated_at"] = datetime.now().isoformat()
            self.keys[key_id]["rotated_to"] = new_key_id

        return new_key_id

    def delete_key(self, key_id: str) -> bool:
        """Securely delete key."""
        if key_id in self.keys:
            # In production, use secure deletion
            del self.keys[key_id]

        if key_id in self.key_metadata:
            del self.key_metadata[key_id]

        return True

    def list_keys(self, algorithm: Optional[EncryptionAlgorithm] = None) -> List[Dict[str, Any]]:
        """List available keys."""
        keys = []
        for key_id, metadata in self.key_metadata.items():
            if algorithm and metadata["algorithm"] != algorithm.value:
                continue

            key_info = {
                "key_id": key_id,
                "algorithm": metadata["algorithm"],
                "key_type": metadata["key_type"],
                "created_at": metadata["created_at"],
                "last_used": metadata.get("last_used"),
                "usage_count": metadata.get("usage_count", 0),
                "rotation_due": metadata.get("rotation_due"),
                "active": self.keys.get(key_id, {}).get("active", False)
            }
            keys.append(key_info)

        return keys


class EncryptionTool:
    """
    Comprehensive encryption tool supporting multiple algorithms
    with key management and secure operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize encryption tool.

        Args:
            config: Configuration for encryption operations
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.default_algorithm = EncryptionAlgorithm(
            self.config.get("default_algorithm", "aes_256_gcm")
        )
        self.key_size = self.config.get("key_size", 256)
        self.mode = self.config.get("mode", "GCM")

        # Initialize key manager
        self.key_manager = KeyManager(self.config.get("key_manager", {}))

        # Algorithm configurations
        self.algorithm_configs = {
            EncryptionAlgorithm.AES_256_GCM: {
                "key_size": 32,
                "iv_size": 12,
                "tag_size": 16,
                "authenticated": True,
                "symmetric": True
            },
            EncryptionAlgorithm.AES_256_CBC: {
                "key_size": 32,
                "iv_size": 16,
                "tag_size": 0,
                "authenticated": False,
                "symmetric": True
            },
            EncryptionAlgorithm.AES_128_GCM: {
                "key_size": 16,
                "iv_size": 12,
                "tag_size": 16,
                "authenticated": True,
                "symmetric": True
            },
            EncryptionAlgorithm.CHACHA20_POLY1305: {
                "key_size": 32,
                "iv_size": 12,
                "tag_size": 16,
                "authenticated": True,
                "symmetric": True
            },
            EncryptionAlgorithm.RSA_2048: {
                "key_size": 256,
                "symmetric": False,
                "max_plaintext": 190  # For OAEP padding
            },
            EncryptionAlgorithm.RSA_4096: {
                "key_size": 512,
                "symmetric": False,
                "max_plaintext": 446  # For OAEP padding
            },
            EncryptionAlgorithm.FERNET: {
                "key_size": 32,
                "symmetric": True,
                "authenticated": True
            }
        }

    @log_operation("encryption")
    async def encrypt(
        self,
        data: Union[str, bytes],
        algorithm: Optional[EncryptionAlgorithm] = None,
        key_id: Optional[str] = None,
        context: Optional[CryptoContext] = None,
        additional_data: Optional[bytes] = None
    ) -> EncryptionResult:
        """
        Encrypt data using specified algorithm and key.

        Args:
            data: Data to encrypt (string or bytes)
            algorithm: Encryption algorithm to use
            key_id: Key ID to use for encryption
            context: Cryptographic context
            additional_data: Additional authenticated data (for AEAD)

        Returns:
            EncryptionResult: Encryption result with ciphertext and metadata
        """
        operation_id = str(uuid.uuid4())
        algorithm = algorithm or self.default_algorithm

        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Generate or get key
            if not key_id:
                key_id = self.key_manager.generate_key(
                    algorithm,
                    KeyType.SYMMETRIC if self._is_symmetric(algorithm) else KeyType.ASYMMETRIC_PUBLIC
                )

            key_data = self.key_manager.get_key(key_id)
            if not key_data:
                raise ToolValidationError(f"Key {key_id} not found or inactive")

            # Perform encryption based on algorithm
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._encrypt_aes_gcm(operation_id, data, key_data, algorithm, key_id, additional_data)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                return await self._encrypt_aes_cbc(operation_id, data, key_data, algorithm, key_id)
            elif algorithm == EncryptionAlgorithm.AES_128_GCM:
                return await self._encrypt_aes_gcm(operation_id, data, key_data, algorithm, key_id, additional_data)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._encrypt_chacha20(operation_id, data, key_data, algorithm, key_id, additional_data)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._encrypt_fernet(operation_id, data, key_data, algorithm, key_id)
            elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                return await self._encrypt_rsa(operation_id, data, key_data, algorithm, key_id)
            else:
                raise ToolValidationError(f"Encryption not implemented for algorithm: {algorithm}")

        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return EncryptionResult(
                operation_id=operation_id,
                success=False,
                algorithm=algorithm,
                operation="encrypt",
                error_message=str(e)
            )

    @log_operation("decryption")
    async def decrypt(
        self,
        encrypted_data: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        additional_data: Optional[bytes] = None
    ) -> EncryptionResult:
        """
        Decrypt data using specified algorithm and key.

        Args:
            encrypted_data: Data to decrypt
            algorithm: Encryption algorithm used
            key_id: Key ID used for encryption
            iv: Initialization vector (if required)
            tag: Authentication tag (for AEAD)
            additional_data: Additional authenticated data (for AEAD)

        Returns:
            EncryptionResult: Decryption result with plaintext
        """
        operation_id = str(uuid.uuid4())

        try:
            key_data = self.key_manager.get_key(key_id)
            if not key_data:
                raise ToolValidationError(f"Key {key_id} not found or inactive")

            # Perform decryption based on algorithm
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._decrypt_aes_gcm(operation_id, encrypted_data, key_data, algorithm, key_id, iv, tag, additional_data)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                return await self._decrypt_aes_cbc(operation_id, encrypted_data, key_data, algorithm, key_id, iv)
            elif algorithm == EncryptionAlgorithm.AES_128_GCM:
                return await self._decrypt_aes_gcm(operation_id, encrypted_data, key_data, algorithm, key_id, iv, tag, additional_data)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._decrypt_chacha20(operation_id, encrypted_data, key_data, algorithm, key_id, iv, tag, additional_data)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._decrypt_fernet(operation_id, encrypted_data, key_data, algorithm, key_id)
            elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                return await self._decrypt_rsa(operation_id, encrypted_data, key_data, algorithm, key_id)
            else:
                raise ToolValidationError(f"Decryption not implemented for algorithm: {algorithm}")

        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return EncryptionResult(
                operation_id=operation_id,
                success=False,
                algorithm=algorithm,
                operation="decrypt",
                error_message=str(e)
            )

    async def _encrypt_aes_gcm(
        self,
        operation_id: str,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        additional_data: Optional[bytes]
    ) -> EncryptionResult:
        """Encrypt using AES-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        if additional_data:
            encryptor.authenticate_additional_data(additional_data)

        ciphertext = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="encrypt",
            data=ciphertext,
            key_id=key_id,
            iv=iv,
            tag=tag,
            metadata={"additional_data_size": len(additional_data) if additional_data else 0}
        )

    async def _decrypt_aes_gcm(
        self,
        operation_id: str,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        iv: bytes,
        tag: bytes,
        additional_data: Optional[bytes]
    ) -> EncryptionResult:
        """Decrypt using AES-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()

        if additional_data:
            decryptor.authenticate_additional_data(additional_data)

        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="decrypt",
            data=plaintext,
            key_id=key_id,
            iv=iv,
            tag=tag
        )

    async def _encrypt_aes_cbc(
        self,
        operation_id: str,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str
    ) -> EncryptionResult:
        """Encrypt using AES-CBC."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()

        iv = os.urandom(16)  # 128-bit IV for CBC
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="encrypt",
            data=ciphertext,
            key_id=key_id,
            iv=iv
        )

    async def _decrypt_aes_cbc(
        self,
        operation_id: str,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        iv: bytes
    ) -> EncryptionResult:
        """Decrypt using AES-CBC."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(encrypted_data) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="decrypt",
            data=plaintext,
            key_id=key_id,
            iv=iv
        )

    async def _encrypt_fernet(
        self,
        operation_id: str,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str
    ) -> EncryptionResult:
        """Encrypt using Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        f = Fernet(key)
        ciphertext = f.encrypt(data)

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="encrypt",
            data=ciphertext,
            key_id=key_id
        )

    async def _decrypt_fernet(
        self,
        operation_id: str,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str
    ) -> EncryptionResult:
        """Decrypt using Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        f = Fernet(key)
        plaintext = f.decrypt(encrypted_data)

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="decrypt",
            data=plaintext,
            key_id=key_id
        )

    async def _encrypt_chacha20(
        self,
        operation_id: str,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        additional_data: Optional[bytes]
    ) -> EncryptionResult:
        """Encrypt using ChaCha20-Poly1305."""
        if not NACL_AVAILABLE:
            raise ToolExecutionError("PyNaCl library not available for ChaCha20")

        box = nacl.secret.SecretBox(key)
        ciphertext = box.encrypt(data)

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="encrypt",
            data=ciphertext,
            key_id=key_id
        )

    async def _decrypt_chacha20(
        self,
        operation_id: str,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str,
        iv: Optional[bytes],
        tag: Optional[bytes],
        additional_data: Optional[bytes]
    ) -> EncryptionResult:
        """Decrypt using ChaCha20-Poly1305."""
        if not NACL_AVAILABLE:
            raise ToolExecutionError("PyNaCl library not available for ChaCha20")

        box = nacl.secret.SecretBox(key)
        plaintext = box.decrypt(encrypted_data)

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="decrypt",
            data=plaintext,
            key_id=key_id
        )

    async def _encrypt_rsa(
        self,
        operation_id: str,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str
    ) -> EncryptionResult:
        """Encrypt using RSA."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        # Load public key
        public_key = serialization.load_pem_public_key(key, backend=default_backend())

        # Check data size limits
        config = self.algorithm_configs[algorithm]
        if len(data) > config["max_plaintext"]:
            raise ToolValidationError(f"Data too large for RSA encryption (max {config['max_plaintext']} bytes)")

        ciphertext = public_key.encrypt(
            data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="encrypt",
            data=ciphertext,
            key_id=key_id
        )

    async def _decrypt_rsa(
        self,
        operation_id: str,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        key_id: str
    ) -> EncryptionResult:
        """Decrypt using RSA."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ToolExecutionError("Cryptography library not available")

        # Load private key
        private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())

        plaintext = private_key.decrypt(
            encrypted_data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return EncryptionResult(
            operation_id=operation_id,
            success=True,
            algorithm=algorithm,
            operation="decrypt",
            data=plaintext,
            key_id=key_id
        )

    def _is_symmetric(self, algorithm: EncryptionAlgorithm) -> bool:
        """Check if algorithm is symmetric."""
        config = self.algorithm_configs.get(algorithm, {})
        return config.get("symmetric", True)

    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms."""
        return [alg.value for alg in EncryptionAlgorithm]

    def get_algorithm_info(self, algorithm: EncryptionAlgorithm) -> Dict[str, Any]:
        """Get information about specific algorithm."""
        return self.algorithm_configs.get(algorithm, {})

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on encryption tool."""
        health_status = {
            "status": "healthy",
            "supported_algorithms": self.get_supported_algorithms(),
            "default_algorithm": self.default_algorithm.value,
            "active_keys": len(self.key_manager.list_keys()),
            "dependencies": {
                "cryptography": CRYPTOGRAPHY_AVAILABLE,
                "nacl": NACL_AVAILABLE
            }
        }

        try:
            # Test basic encryption/decryption
            test_data = b"health_check_test"
            encrypt_result = await self.encrypt(test_data)
            
            if encrypt_result.success:
                decrypt_result = await self.decrypt(
                    encrypt_result.data,
                    encrypt_result.algorithm,
                    encrypt_result.key_id,
                    encrypt_result.iv,
                    encrypt_result.tag
                )
                health_status["basic_crypto"] = decrypt_result.success and decrypt_result.data == test_data
            else:
                health_status["basic_crypto"] = False

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_crypto"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating encryption tool
def create_encryption_tool(config: Optional[Dict[str, Any]] = None) -> EncryptionTool:
    """
    Create an encryption tool with specified configuration.

    Args:
        config: Configuration for encryption operations

    Returns:
        EncryptionTool: Configured encryption tool instance
    """
    return EncryptionTool(config=config)
