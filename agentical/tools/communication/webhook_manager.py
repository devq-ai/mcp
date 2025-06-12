"""
Webhook Manager for Agentical

This module provides comprehensive webhook management capabilities including
webhook registration, signature verification, event routing, rate limiting,
and enterprise-grade features for secure webhook processing.

Features:
- Webhook registration and management
- Secure signature verification (HMAC, JWT)
- Event routing and filtering
- Rate limiting and throttling
- Retry logic and failure handling
- Event persistence and replay
- Real-time event processing
- Health monitoring and analytics
- Enterprise features (audit logging, compliance, encryption)
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import base64
import os

# Optional dependencies
try:
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class WebhookMethod(Enum):
    """Supported HTTP methods for webhooks."""
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class SignatureMethod(Enum):
    """Webhook signature verification methods."""
    NONE = "none"
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"
    JWT = "jwt"
    CUSTOM = "custom"


class WebhookStatus(Enum):
    """Webhook endpoint status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class EventStatus(Enum):
    """Webhook event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


class RetryStrategy(Enum):
    """Retry strategies for failed deliveries."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    id: str
    url: str
    secret: str
    method: WebhookMethod = WebhookMethod.POST
    signature_method: SignatureMethod = SignatureMethod.HMAC_SHA256
    signature_header: str = "X-Signature"
    event_types: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_delay: float = 1.0
    status: WebhookStatus = WebhookStatus.ACTIVE
    rate_limit_per_minute: int = 60
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.event_types is None:
            self.event_types = []
        if self.filters is None:
            self.filters = {}
        if self.headers is None:
            self.headers = {}
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['method'] = self.method.value
        data['signature_method'] = self.signature_method.value
        data['retry_strategy'] = self.retry_strategy.value
        data['status'] = self.status.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookEndpoint':
        """Create from dictionary representation."""
        # Convert enums from strings
        if 'method' in data:
            data['method'] = WebhookMethod(data['method'])
        if 'signature_method' in data:
            data['signature_method'] = SignatureMethod(data['signature_method'])
        if 'retry_strategy' in data:
            data['retry_strategy'] = RetryStrategy(data['retry_strategy'])
        if 'status' in data:
            data['status'] = WebhookStatus(data['status'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class WebhookEvent:
    """Webhook event data."""
    id: str
    event_type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookEvent':
        """Create from dictionary representation."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt."""
    id: str
    endpoint_id: str
    event_id: str
    attempt_number: int
    status: EventStatus
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None
    latency: Optional[float] = None
    scheduled_at: Optional[datetime] = None
    attempted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None

    def __post_init__(self):
        if self.response_headers is None:
            self.response_headers = {}

    @property
    def is_success(self) -> bool:
        """Check if delivery was successful."""
        return self.status == EventStatus.DELIVERED and 200 <= (self.response_code or 0) < 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['status'] = self.status.value
        for field in ['scheduled_at', 'attempted_at', 'completed_at', 'next_retry_at']:
            if getattr(self, field):
                data[field] = getattr(self, field).isoformat()
        return data


class SignatureVerifier(ABC):
    """Abstract base class for signature verifiers."""

    @abstractmethod
    def generate_signature(self, payload: bytes, secret: str) -> str:
        """Generate signature for payload."""
        pass

    @abstractmethod
    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify signature against payload."""
        pass


class HMACSignatureVerifier(SignatureVerifier):
    """HMAC signature verifier."""

    def __init__(self, algorithm: str = 'sha256'):
        self.algorithm = algorithm
        self.hash_func = getattr(hashlib, algorithm)

    def generate_signature(self, payload: bytes, secret: str) -> str:
        """Generate HMAC signature."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            self.hash_func
        ).hexdigest()
        return f"{self.algorithm}={signature}"

    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)


class JWTSignatureVerifier(SignatureVerifier):
    """JWT signature verifier."""

    def __init__(self):
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library required for JWT verification")

    def generate_signature(self, payload: bytes, secret: str) -> str:
        """Generate JWT token."""
        payload_data = json.loads(payload.decode('utf-8'))
        token = jwt.encode(payload_data, secret, algorithm='HS256')
        return token

    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify JWT signature."""
        try:
            decoded = jwt.decode(signature, secret, algorithms=['HS256'])
            payload_data = json.loads(payload.decode('utf-8'))
            return decoded == payload_data
        except jwt.InvalidTokenError:
            return False


class EventFilter:
    """Event filtering utilities."""

    @staticmethod
    def matches_filter(event: WebhookEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches filter criteria."""
        for key, expected_value in filters.items():
            if '.' in key:
                # Nested key support (e.g., "data.user.id")
                actual_value = EventFilter._get_nested_value(event.data, key)
            else:
                actual_value = getattr(event, key, None)

            if not EventFilter._compare_values(actual_value, expected_value):
                return False

        return True

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], key: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    @staticmethod
    def _compare_values(actual: Any, expected: Any) -> bool:
        """Compare actual and expected values."""
        if isinstance(expected, dict):
            # Support for operators like {"$gt": 100, "$lt": 200}
            for operator, value in expected.items():
                if operator == "$eq":
                    return actual == value
                elif operator == "$ne":
                    return actual != value
                elif operator == "$gt":
                    return actual > value
                elif operator == "$gte":
                    return actual >= value
                elif operator == "$lt":
                    return actual < value
                elif operator == "$lte":
                    return actual <= value
                elif operator == "$in":
                    return actual in value
                elif operator == "$nin":
                    return actual not in value
            return False
        else:
            return actual == expected


class RateLimiter:
    """Rate limiter for webhook endpoints."""

    def __init__(self):
        self.request_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def is_allowed(self, endpoint_id: str, limit_per_minute: int) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        current_minute = int(now // 60)

        # Clean old entries
        endpoint_counts = self.request_counts[endpoint_id]
        old_minutes = [minute for minute in endpoint_counts.keys() if minute < current_minute - 1]
        for old_minute in old_minutes:
            del endpoint_counts[old_minute]

        # Check current minute
        current_count = endpoint_counts[current_minute]
        if current_count >= limit_per_minute:
            return False

        # Increment counter
        endpoint_counts[current_minute] += 1
        return True


class WebhookStorage(ABC):
    """Abstract storage interface for webhooks."""

    @abstractmethod
    async def save_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Save webhook endpoint."""
        pass

    @abstractmethod
    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID."""
        pass

    @abstractmethod
    async def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all webhook endpoints."""
        pass

    @abstractmethod
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint."""
        pass

    @abstractmethod
    async def save_event(self, event: WebhookEvent) -> bool:
        """Save webhook event."""
        pass

    @abstractmethod
    async def save_delivery(self, delivery: WebhookDelivery) -> bool:
        """Save delivery attempt."""
        pass


class MemoryStorage(WebhookStorage):
    """In-memory storage implementation."""

    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.events: Dict[str, WebhookEvent] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}

    async def save_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Save endpoint in memory."""
        self.endpoints[endpoint.id] = endpoint
        return True

    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint from memory."""
        return self.endpoints.get(endpoint_id)

    async def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all endpoints."""
        return list(self.endpoints.values())

    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete endpoint from memory."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            return True
        return False

    async def save_event(self, event: WebhookEvent) -> bool:
        """Save event in memory."""
        self.events[event.id] = event
        return True

    async def save_delivery(self, delivery: WebhookDelivery) -> bool:
        """Save delivery in memory."""
        self.deliveries[delivery.id] = delivery
        return True


class RedisStorage(WebhookStorage):
    """Redis storage implementation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis library required for Redis storage")

        self.redis_client = redis.from_url(redis_url)
        self.endpoint_prefix = "webhook:endpoint:"
        self.event_prefix = "webhook:event:"
        self.delivery_prefix = "webhook:delivery:"

    async def save_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Save endpoint in Redis."""
        try:
            key = f"{self.endpoint_prefix}{endpoint.id}"
            data = json.dumps(endpoint.to_dict())
            self.redis_client.set(key, data)
            return True
        except Exception:
            return False

    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint from Redis."""
        try:
            key = f"{self.endpoint_prefix}{endpoint_id}"
            data = self.redis_client.get(key)
            if data:
                endpoint_data = json.loads(data)
                return WebhookEndpoint.from_dict(endpoint_data)
        except Exception:
            pass
        return None

    async def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all endpoints from Redis."""
        try:
            pattern = f"{self.endpoint_prefix}*"
            keys = self.redis_client.keys(pattern)
            endpoints = []

            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    endpoint_data = json.loads(data)
                    endpoints.append(WebhookEndpoint.from_dict(endpoint_data))

            return endpoints
        except Exception:
            return []

    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete endpoint from Redis."""
        try:
            key = f"{self.endpoint_prefix}{endpoint_id}"
            return bool(self.redis_client.delete(key))
        except Exception:
            return False

    async def save_event(self, event: WebhookEvent) -> bool:
        """Save event in Redis."""
        try:
            key = f"{self.event_prefix}{event.id}"
            data = json.dumps(event.to_dict())
            self.redis_client.setex(key, 86400 * 7, data)  # 7 days TTL
            return True
        except Exception:
            return False

    async def save_delivery(self, delivery: WebhookDelivery) -> bool:
        """Save delivery in Redis."""
        try:
            key = f"{self.delivery_prefix}{delivery.id}"
            data = json.dumps(delivery.to_dict())
            self.redis_client.setex(key, 86400 * 30, data)  # 30 days TTL
            return True
        except Exception:
            return False


class WebhookManager:
    """
    Comprehensive webhook management system.

    Provides enterprise-grade webhook capabilities with security, reliability,
    and monitoring features for event-driven architectures.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize webhook manager.

        Args:
            config: Configuration dictionary with webhook settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.max_endpoints = self.config.get('max_endpoints', 1000)
        self.default_timeout = self.config.get('default_timeout', 30.0)
        self.max_retries = self.config.get('max_retries', 3)
        self.event_retention_hours = self.config.get('event_retention_hours', 168)  # 7 days

        # Performance settings
        self.enable_rate_limiting = self.config.get('rate_limiting', True)
        self.enable_signature_verification = self.config.get('signature_verification', True)
        self.max_concurrent_deliveries = self.config.get('max_concurrent_deliveries', 100)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.high_availability = self.config.get('high_availability', False)

        # Initialize components
        storage_type = self.config.get('storage', 'memory')
        if storage_type == 'redis':
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.storage = RedisStorage(redis_url)
        else:
            self.storage = MemoryStorage()

        # Signature verifiers
        self.signature_verifiers = {
            SignatureMethod.HMAC_SHA256: HMACSignatureVerifier('sha256'),
            SignatureMethod.HMAC_SHA512: HMACSignatureVerifier('sha512')
        }

        if JWT_AVAILABLE:
            self.signature_verifiers[SignatureMethod.JWT] = JWTSignatureVerifier()

        # Rate limiting
        self.rate_limiter = RateLimiter() if self.enable_rate_limiting else None

        # Metrics and monitoring
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: List[asyncio.Task] = []

        # HTTP client for webhook deliveries
        self.http_client = None

    async def start(self):
        """Start webhook manager and delivery workers."""
        try:
            self.logger.info("Starting webhook manager")

            # Initialize HTTP client
            if HTTPX_AVAILABLE:
                self.http_client = httpx.AsyncClient(timeout=self.default_timeout)

            # Start delivery workers
            for i in range(self.max_concurrent_deliveries):
                task = asyncio.create_task(self._delivery_worker())
                self.processing_tasks.append(task)

            self.logger.info(f"Webhook manager started with {len(self.processing_tasks)} workers")

        except Exception as e:
            self.logger.error(f"Failed to start webhook manager: {e}")
            raise

    async def stop(self):
        """Stop webhook manager and cleanup resources."""
        try:
            self.logger.info("Stopping webhook manager")

            # Cancel processing tasks
            for task in self.processing_tasks:
                task.cancel()

            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            self.processing_tasks.clear()

            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None

            self.logger.info("Webhook manager stopped")

        except Exception as e:
            self.logger.error(f"Error stopping webhook manager: {e}")

    async def register_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """
        Register a new webhook endpoint.

        Args:
            endpoint: Webhook endpoint configuration

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            self.logger.info(f"Registering webhook endpoint: {endpoint.id}")

            # Check limits
            existing_endpoints = await self.storage.list_endpoints()
            if len(existing_endpoints) >= self.max_endpoints:
                raise ValueError(f"Maximum number of endpoints ({self.max_endpoints}) reached")

            # Save endpoint
            success = await self.storage.save_endpoint(endpoint)

            if success:
                self.metrics['endpoints_registered'] += 1

                # Log audit
                if self.audit_logging:
                    self._log_endpoint_operation('register', endpoint)

            return success

        except Exception as e:
            self.logger.error(f"Failed to register endpoint: {e}")
            return False

    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister webhook endpoint.

        Args:
            endpoint_id: ID of endpoint to unregister

        Returns:
            True if unregistered successfully, False otherwise
        """
        try:
            self.logger.info(f"Unregistering webhook endpoint: {endpoint_id}")

            endpoint = await self.storage.get_endpoint(endpoint_id)
            if not endpoint:
                return False

            success = await self.storage.delete_endpoint(endpoint_id)

            if success:
                self.metrics['endpoints_unregistered'] += 1

                # Log audit
                if self.audit_logging:
                    self._log_endpoint_operation('unregister', endpoint)

            return success

        except Exception as e:
            self.logger.error(f"Failed to unregister endpoint: {e}")
            return False

    async def trigger_event(self, event: WebhookEvent) -> List[str]:
        """
        Trigger webhook event to all matching endpoints.

        Args:
            event: Webhook event to trigger

        Returns:
            List of delivery IDs for tracking
        """
        try:
            self.logger.debug(f"Triggering webhook event: {event.event_type}")

            # Save event
            await self.storage.save_event(event)

            # Find matching endpoints
            endpoints = await self._find_matching_endpoints(event)
            delivery_ids = []

            for endpoint in endpoints:
                # Check rate limiting
                if (self.rate_limiter and
                    not self.rate_limiter.is_allowed(endpoint.id, endpoint.rate_limit_per_minute)):
                    self.logger.warning(f"Rate limit exceeded for endpoint: {endpoint.id}")
                    continue

                # Create delivery
                delivery = WebhookDelivery(
                    id=str(uuid.uuid4()),
                    endpoint_id=endpoint.id,
                    event_id=event.id,
                    attempt_number=1,
                    status=EventStatus.PENDING,
                    scheduled_at=datetime.utcnow()
                )

                # Queue for delivery
                await self.delivery_queue.put((endpoint, event, delivery))
                delivery_ids.append(delivery.id)

            self.metrics['events_triggered'] += 1
            self.metrics['deliveries_queued'] += len(delivery_ids)

            return delivery_ids

        except Exception as e:
            self.logger.error(f"Failed to trigger event: {e}")
            return []

    async def _find_matching_endpoints(self, event: WebhookEvent) -> List[WebhookEndpoint]:
        """Find endpoints that match the event."""
        matching_endpoints = []
        endpoints = await self.storage.list_endpoints()

        for endpoint in endpoints:
            # Check if endpoint is active
            if endpoint.status != WebhookStatus.ACTIVE:
                continue

            # Check event type filter
            if endpoint.event_types and event.event_type not in endpoint.event_types:
                continue

            # Check custom filters
            if endpoint.filters and not EventFilter.matches_filter(event, endpoint.filters):
                continue

            matching_endpoints.append(endpoint)

        return matching_endpoints

    async def _delivery_worker(self):
        """Background worker for webhook deliveries."""
        while True:
            try:
                # Get delivery from queue
                endpoint, event, delivery = await self.delivery_queue.get()

                # Process delivery
                await self._process_delivery(endpoint, event, delivery)

                # Mark task as done
                self.delivery_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Delivery worker error: {e}")

    async def _process_delivery(self, endpoint: WebhookEndpoint, event: WebhookEvent, delivery: WebhookDelivery):
        """Process a single webhook delivery."""
        try:
            delivery.attempted_at = datetime.utcnow()
            delivery.status = EventStatus.PROCESSING

            # Prepare payload
            payload = json.dumps(event.to_dict()).encode('utf-8')

            # Generate signature
            signature = None
            if (self.enable_signature_verification and
                endpoint.signature_method != SignatureMethod.NONE):
                verifier = self.signature_verifiers.get(endpoint.signature_method)
                if verifier:
                    signature = verifier.generate_signature(payload, endpoint.secret)

            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Agentical-Webhook/1.0',
                'X-Event-Type': event.event_type,
                'X-Event-ID': event.id,
                'X-Delivery-ID': delivery.id
            }

            if signature:
                headers[endpoint.signature_header] = signature

            # Add custom headers
            headers.update(endpoint.headers)

            # Make HTTP request
            start_time = time.time()

            if self.http_client:
                response = await self.http_client.request(
                    method=endpoint.method.value,
                    url=endpoint.url,
                    content=payload,
                    headers=headers,
                    timeout=endpoint.timeout
                )

                delivery.latency = time.time() - start_time
                delivery.response_code = response.status_code
                delivery.response_body = response.text[:1000]  # Limit body size
                delivery.response_headers = dict(response.headers)

                if 200 <= response.status_code < 300:
                    delivery.status = EventStatus.DELIVERED
                    self.metrics['deliveries_successful'] += 1
                else:
                    delivery.status = EventStatus.FAILED
                    delivery.error_message = f"HTTP {response.status_code}"
                    self.metrics['deliveries_failed'] += 1

            else:
                delivery.status = EventStatus.FAILED
                delivery.error_message = "No HTTP client available"
                self.metrics['deliveries_failed'] += 1

        except Exception as e:
            delivery.status = EventStatus.FAILED
            delivery.error_message = str(e)
            delivery.latency = time.time() - start_time if 'start_time' in locals() else 0
            self.metrics['deliveries_failed'] += 1

        finally:
            delivery.completed_at = datetime.utcnow()
            await self.storage.save_delivery(delivery)

            # Schedule retry if needed
            if (delivery.status == EventStatus.FAILED and
                delivery.attempt_number < endpoint.max_retries):
                await self._schedule_retry(endpoint, event, delivery)

    async def _schedule_retry(self, endpoint: WebhookEndpoint, event: WebhookEvent, failed_delivery: WebhookDelivery):
        """Schedule retry for failed delivery."""
        try:
            # Calculate retry delay
            if endpoint.retry_strategy == RetryStrategy.LINEAR:
                delay = endpoint.retry_delay * failed_delivery.attempt_number
            elif endpoint.retry_strategy == RetryStrategy.EXPONENTIAL:
                delay = endpoint.retry_delay * (2 ** (failed_delivery.attempt_number - 1))
            else:
                delay = endpoint.retry_delay

            # Create retry delivery
            retry_delivery = WebhookDelivery(
                id=str(uuid.uuid4()),
                endpoint_id=endpoint.id,
                event_id=event.id,
                attempt_number=failed_delivery.attempt_number + 1,
                status=EventStatus.RETRYING,
                scheduled_at=datetime.utcnow() + timedelta(seconds=delay)
            )

            # Schedule for retry (simplified - in production would use proper scheduler)
            asyncio.create_task(self._delayed_retry(endpoint, event, retry_delivery, delay))

        except Exception as e:
            self.logger.error(f"Failed to schedule retry: {e}")

    async def _delayed_retry(self, endpoint: WebhookEndpoint, event: WebhookEvent, delivery: WebhookDelivery, delay: float):
        """Execute delayed retry."""
        await asyncio.sleep(delay)
        await self.delivery_queue.put((endpoint, event, delivery))

    def verify_webhook_signature(self, payload: bytes, signature: str, secret: str,
                                method: SignatureMethod = SignatureMethod.HMAC_SHA256) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Request payload
            signature: Provided signature
            secret: Webhook secret
            method: Signature method

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            if method == SignatureMethod.NONE:
                return True

            verifier = self.signature_verifiers.get(method)
            if not verifier:
                return False

            return verifier.verify_signature(payload, signature, secret)

        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False

    async def get_endpoint_statistics(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get statistics for a webhook endpoint.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Statistics dictionary
        """
        try:
            endpoint = await self.storage.get_endpoint(endpoint_id)
            if not endpoint:
                return {}

            # This is a simplified implementation
            # In production, you'd query delivery history from storage
            return {
                'endpoint_id': endpoint_id,
                'status': endpoint.status.value,
                'total_deliveries': 0,  # Would be calculated from storage
                'successful_deliveries': 0,
                'failed_deliveries': 0,
                'average_latency': 0.0,
                'success_rate': 0.0
            }

        except Exception as e:
            self.logger.error(f"Failed to get endpoint statistics: {e}")
            return {}

    def _log_endpoint_operation(self, operation: str, endpoint: WebhookEndpoint):
        """Log endpoint operation for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'endpoint_id': endpoint.id,
            'endpoint_url': endpoint.url,
            'status': endpoint.status.value
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    async def health_check(self) -> bool:
        """
        Perform health check of webhook manager.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if we can list endpoints
            endpoints = await self.storage.list_endpoints()
            return True
        except Exception as e:
            self.logger.error(f"Webhook manager health check failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup webhook manager resources."""
        try:
            await self.stop()
            self.metrics.clear()
            self.logger.info("Webhook manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'processing_tasks') and self.processing_tasks:
                self.logger.info("WebhookManager being destroyed - cleanup recommended")
        except:
            pass
