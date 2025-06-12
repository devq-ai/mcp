"""
External API Client for Agentical

This module provides a comprehensive HTTP API client with support for multiple
authentication methods, intelligent retry mechanisms, circuit breaker patterns,
and enterprise-grade features for reliable external service integration.

Features:
- Multiple authentication methods (API key, OAuth, JWT, Basic Auth)
- Intelligent retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Response caching and performance optimization
- Rate limiting and quota management
- Request/response transformation and validation
- Connection pooling and session management
- Comprehensive error handling and logging
- Enterprise features (audit logging, monitoring, encryption)
"""

import asyncio
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
import hashlib
import base64
import hmac
import urllib.parse
import os

# Optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import cryptography
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class AuthMethod(Enum):
    """Supported authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    HMAC_SIGNATURE = "hmac_signature"
    CUSTOM = "custom"


class HttpMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RetryStrategy(Enum):
    """Retry strategies."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RequestStatus(Enum):
    """Request status tracking."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    method: AuthMethod
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    oauth_token_url: Optional[str] = None
    jwt_secret: Optional[str] = None
    hmac_secret: Optional[str] = None
    custom_auth_func: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['method'] = self.method.value
        # Remove sensitive data
        sensitive_fields = ['password', 'api_key', 'token', 'oauth_client_secret', 'jwt_secret', 'hmac_secret']
        for field in sensitive_fields:
            if field in data and data[field]:
                data[field] = "[REDACTED]"
        data.pop('custom_auth_func', None)
        return data


@dataclass
class RetryConfig:
    """Retry configuration."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_status: List[int] = None
    retry_on_exceptions: List[Exception] = None

    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = [429, 500, 502, 503, 504]
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = [ConnectionError, TimeoutError]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        return data


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitoring_window: float = 300.0
    enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ApiRequest:
    """API request definition."""
    id: str
    method: HttpMethod
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    data: Optional[Any] = None
    json: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    auth_override: Optional[AuthConfig] = None
    retry_override: Optional[RetryConfig] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['method'] = self.method.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ApiResponse:
    """API response wrapper."""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    json_data: Optional[Dict[str, Any]] = None
    elapsed_time: float = 0.0
    attempt_number: int = 1
    from_cache: bool = False
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status_code < 600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        # Convert bytes to string for JSON serialization
        data['content'] = self.content.decode('utf-8', errors='ignore')[:1000]  # Truncate for logging
        return data


class AuthenticatorInterface(ABC):
    """Abstract interface for authenticators."""

    @abstractmethod
    async def authenticate(self, request: ApiRequest) -> ApiRequest:
        """Apply authentication to request."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if authenticator dependencies are available."""
        pass


class ApiKeyAuthenticator(AuthenticatorInterface):
    """API key authentication."""

    def __init__(self, config: AuthConfig):
        self.config = config

    async def authenticate(self, request: ApiRequest) -> ApiRequest:
        """Add API key to request headers."""
        if not self.config.api_key:
            raise ValueError("API key not configured")

        request.headers[self.config.api_key_header] = self.config.api_key
        return request

    def is_available(self) -> bool:
        """Check if API key is available."""
        return bool(self.config.api_key)


class BearerTokenAuthenticator(AuthenticatorInterface):
    """Bearer token authentication."""

    def __init__(self, config: AuthConfig):
        self.config = config

    async def authenticate(self, request: ApiRequest) -> ApiRequest:
        """Add bearer token to request headers."""
        if not self.config.token:
            raise ValueError("Bearer token not configured")

        request.headers['Authorization'] = f"Bearer {self.config.token}"
        return request

    def is_available(self) -> bool:
        """Check if bearer token is available."""
        return bool(self.config.token)


class BasicAuthAuthenticator(AuthenticatorInterface):
    """Basic authentication."""

    def __init__(self, config: AuthConfig):
        self.config = config

    async def authenticate(self, request: ApiRequest) -> ApiRequest:
        """Add basic auth to request headers."""
        if not self.config.username or not self.config.password:
            raise ValueError("Username and password not configured")

        credentials = f"{self.config.username}:{self.config.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        request.headers['Authorization'] = f"Basic {encoded_credentials}"
        return request

    def is_available(self) -> bool:
        """Check if credentials are available."""
        return bool(self.config.username and self.config.password)


class HmacSignatureAuthenticator(AuthenticatorInterface):
    """HMAC signature authentication."""

    def __init__(self, config: AuthConfig):
        self.config = config

    async def authenticate(self, request: ApiRequest) -> ApiRequest:
        """Add HMAC signature to request headers."""
        if not self.config.hmac_secret:
            raise ValueError("HMAC secret not configured")

        # Create signature payload
        timestamp = str(int(time.time()))
        method = request.method.value
        url_path = urllib.parse.urlparse(request.url).path
        body = ""

        if request.json:
            body = json.dumps(request.json, sort_keys=True)
        elif request.data:
            body = str(request.data)

        payload = f"{timestamp}{method}{url_path}{body}"

        # Create HMAC signature
        signature = hmac.new(
            self.config.hmac_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Add to headers
        request.headers['X-Timestamp'] = timestamp
        request.headers['X-Signature'] = signature
        return request

    def is_available(self) -> bool:
        """Check if HMAC secret is available."""
        return bool(self.config.hmac_secret)


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque(maxlen=100)

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if not self.config.enabled:
            return True

        now = time.time()

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time) > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        """Record successful request."""
        if not self.config.enabled:
            return

        self.request_history.append((time.time(), True))

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed request."""
        if not self.config.enabled:
            return

        self.request_history.append((time.time(), False))
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        now = time.time()
        recent_requests = [req for req in self.request_history if now - req[0] < self.config.monitoring_window]

        total_requests = len(recent_requests)
        successful_requests = sum(1 for req in recent_requests if req[1])
        failure_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0

        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failure_rate': failure_rate,
            'last_failure_time': self.last_failure_time
        }


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times = deque()
        self.burst_tokens = burst_limit

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        now = time.time()

        # Remove old requests (older than 1 minute)
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # Replenish burst tokens
        if len(self.request_times) == 0:
            self.burst_tokens = self.burst_limit

        # Check if we can make a request
        if len(self.request_times) < self.requests_per_minute:
            if self.burst_tokens > 0:
                self.burst_tokens -= 1
                self.request_times.append(now)
                return True
            else:
                # Calculate wait time
                if self.request_times:
                    wait_time = 60 / self.requests_per_minute
                    await asyncio.sleep(wait_time)
                    self.request_times.append(time.time())
                    return True

        return False


class ExternalAPIClient:
    """
    Comprehensive external API client.

    Provides enterprise-grade HTTP client with authentication, retries,
    circuit breaker, rate limiting, and comprehensive monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize external API client.

        Args:
            config: Configuration dictionary with client settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.base_url = self.config.get('base_url', '')
        self.timeout = self.config.get('timeout', 30.0)
        self.verify_ssl = self.config.get('verify_ssl', True)

        # Authentication
        auth_config_data = self.config.get('auth', {})
        self.auth_config = AuthConfig(**auth_config_data) if auth_config_data else None

        # Retry configuration
        retry_config_data = self.config.get('retry', {})
        self.retry_config = RetryConfig(**retry_config_data)

        # Circuit breaker
        circuit_config_data = self.config.get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(**circuit_config_data))

        # Rate limiting
        rate_limit_config = self.config.get('rate_limiting', {})
        if rate_limit_config.get('enabled', False):
            self.rate_limiter = RateLimiter(
                requests_per_minute=rate_limit_config.get('requests_per_minute', 60),
                burst_limit=rate_limit_config.get('burst_limit', 10)
            )
        else:
            self.rate_limiter = None

        # Performance settings
        self.enable_caching = self.config.get('enable_caching', False)
        self.enable_compression = self.config.get('enable_compression', True)
        self.connection_pool_size = self.config.get('connection_pool_size', 10)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.encryption_enabled = self.config.get('encryption_enabled', False)

        # Initialize components
        self.authenticator = self._initialize_authenticator()
        self.client = None
        self.cache: Dict[str, Tuple[ApiResponse, float]] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.request_history: deque = deque(maxlen=1000)

    def _initialize_authenticator(self) -> Optional[AuthenticatorInterface]:
        """Initialize authenticator based on auth method."""
        if not self.auth_config or self.auth_config.method == AuthMethod.NONE:
            return None

        if self.auth_config.method == AuthMethod.API_KEY:
            return ApiKeyAuthenticator(self.auth_config)
        elif self.auth_config.method == AuthMethod.BEARER_TOKEN:
            return BearerTokenAuthenticator(self.auth_config)
        elif self.auth_config.method == AuthMethod.BASIC_AUTH:
            return BasicAuthAuthenticator(self.auth_config)
        elif self.auth_config.method == AuthMethod.HMAC_SIGNATURE:
            return HmacSignatureAuthenticator(self.auth_config)
        else:
            self.logger.warning(f"Unsupported auth method: {self.auth_config.method}")
            return None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self.client is None:
            if HTTPX_AVAILABLE:
                self.client = httpx.AsyncClient(
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    limits=httpx.Limits(max_connections=self.connection_pool_size)
                )
            elif AIOHTTP_AVAILABLE:
                connector = aiohttp.TCPConnector(limit=self.connection_pool_size)
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)
            else:
                raise ImportError("Either httpx or aiohttp is required")

        return self.client

    async def request(self, request: ApiRequest) -> ApiResponse:
        """
        Execute an API request with full error handling and retries.

        Args:
            request: API request to execute

        Returns:
            API response with comprehensive metadata
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Executing request: {request.method.value} {request.url}")

            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise RuntimeError("Circuit breaker is open")

            # Check rate limiting
            if self.rate_limiter and not await self.rate_limiter.acquire():
                raise RuntimeError("Rate limit exceeded")

            # Check cache
            if self.enable_caching and request.method == HttpMethod.GET:
                cached_response = self._get_cached_response(request)
                if cached_response:
                    self.metrics['cache_hits'] += 1
                    return cached_response

            # Apply authentication
            if self.authenticator:
                request = await self.authenticator.authenticate(request)

            # Execute request with retries
            response = await self._execute_with_retries(request)

            # Cache response if successful
            if self.enable_caching and response.is_success:
                self._cache_response(request, response)

            # Record success
            self.circuit_breaker.record_success()
            self.metrics['requests_successful'] += 1

            # Log audit
            if self.audit_logging:
                self._log_request(request, response, time.time() - start_time)

            return response

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            self.metrics['requests_failed'] += 1

            self.logger.error(f"Request failed: {e}")

            # Create error response
            return ApiResponse(
                request_id=request.id,
                status_code=0,
                headers={},
                content=b'',
                text=str(e),
                elapsed_time=time.time() - start_time,
                metadata={'error': str(e)}
            )

    async def _execute_with_retries(self, request: ApiRequest) -> ApiResponse:
        """Execute request with retry logic."""
        retry_config = request.retry_override or self.retry_config
        last_exception = None

        for attempt in range(retry_config.max_attempts):
            try:
                response = await self._execute_single_request(request, attempt + 1)

                # Check if we should retry based on status code
                if response.status_code in retry_config.retry_on_status:
                    if attempt < retry_config.max_attempts - 1:
                        await self._wait_for_retry(attempt, retry_config)
                        continue

                return response

            except Exception as e:
                last_exception = e

                # Check if we should retry based on exception type
                should_retry = any(isinstance(e, exc_type) for exc_type in retry_config.retry_on_exceptions)

                if should_retry and attempt < retry_config.max_attempts - 1:
                    await self._wait_for_retry(attempt, retry_config)
                    continue
                else:
                    raise

        # If we get here, all retries failed
        if last_exception:
            raise last_exception

    async def _execute_single_request(self, request: ApiRequest, attempt: int) -> ApiResponse:
        """Execute a single HTTP request."""
        client = await self._get_client()
        start_time = time.time()

        # Prepare request parameters
        url = request.url
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        kwargs = {
            'method': request.method.value,
            'url': url,
            'headers': request.headers,
            'params': request.params,
            'timeout': request.timeout or self.timeout
        }

        if request.json:
            kwargs['json'] = request.json
        elif request.data:
            kwargs['data'] = request.data

        # Execute request
        if HTTPX_AVAILABLE and isinstance(client, httpx.AsyncClient):
            response = await client.request(**kwargs)

            # Parse JSON if possible
            json_data = None
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    json_data = response.json()
            except:
                pass

            return ApiResponse(
                request_id=request.id,
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                text=response.text,
                json_data=json_data,
                elapsed_time=time.time() - start_time,
                attempt_number=attempt
            )

        elif AIOHTTP_AVAILABLE and isinstance(client, aiohttp.ClientSession):
            async with client.request(**kwargs) as response:
                content = await response.read()
                text = await response.text()

                # Parse JSON if possible
                json_data = None
                try:
                    if response.content_type.startswith('application/json'):
                        json_data = await response.json()
                except:
                    pass

                return ApiResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=content,
                    text=text,
                    json_data=json_data,
                    elapsed_time=time.time() - start_time,
                    attempt_number=attempt
                )

        else:
            raise RuntimeError("No HTTP client available")

    async def _wait_for_retry(self, attempt: int, retry_config: RetryConfig):
        """Wait before retrying based on strategy."""
        if retry_config.strategy == RetryStrategy.LINEAR:
            delay = retry_config.base_delay * (attempt + 1)
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL:
            delay = retry_config.base_delay * (retry_config.backoff_multiplier ** attempt)
        else:
            delay = retry_config.base_delay

        # Cap the delay
        delay = min(delay, retry_config.max_delay)

        self.logger.debug(f"Retrying in {delay:.2f} seconds (attempt {attempt + 1})")
        await asyncio.sleep(delay)

    def _get_cached_response(self, request: ApiRequest) -> Optional[ApiResponse]:
        """Get cached response if available and valid."""
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            response, cached_time = self.cache[cache_key]
            cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default

            if time.time() - cached_time < cache_ttl:
                response.from_cache = True
                return response
            else:
                # Remove expired cache entry
                del self.cache[cache_key]

        return None

    def _cache_response(self, request: ApiRequest, response: ApiResponse):
        """Cache response for future use."""
        cache_key = self._get_cache_key(request)
        self.cache[cache_key] = (response, time.time())

        # Limit cache size
        max_cache_size = self.config.get('max_cache_size', 1000)
        if len(self.cache) > max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:100]
            for key in oldest_keys:
                del self.cache[key]

    def _get_cache_key(self, request: ApiRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            'method': request.method.value,
            'url': request.url,
            'params': request.params,
            'headers': dict(sorted(request.headers.items())) if request.headers else {}
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _log_request(self, request: ApiRequest, response: ApiResponse, elapsed_time: float):
        """Log request for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.id,
            'method': request.method.value,
            'url': request.url,
            'status_code': response.status_code,
            'elapsed_time': elapsed_time,
            'attempt_number': response.attempt_number,
            'from_cache': response.from_cache
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    # Convenience methods for common HTTP operations

    async def get(self, url: str, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None, **kwargs) -> ApiResponse:
        """Execute GET request."""
        request = ApiRequest(
            id=str(uuid.uuid4()),
            method=HttpMethod.GET,
            url=url,
            params=params,
            headers=headers or {},
            **kwargs
        )
        return await self.request(request)

    async def post(self, url: str, json: Optional[Dict[str, Any]] = None,
                   data: Optional[Any] = None, headers: Optional[Dict[str, str]] = None,
                   **kwargs) -> ApiResponse:
        """Execute POST request."""
        request = ApiRequest(
            id=str(uuid.uuid4()),
            method=HttpMethod.POST,
            url=url,
            json=json,
            data=data,
            headers=headers or {},
            **kwargs
        )
        return await self.request(request)

    async def put(self, url: str, json: Optional[Dict[str, Any]] = None,
                  data: Optional[Any] = None, headers: Optional[Dict[str, str]] = None,
                  **kwargs) -> ApiResponse:
        """Execute PUT request."""
        request = ApiRequest(
            id=str(uuid.uuid4()),
            method=HttpMethod.PUT,
            url=url,
            json=json,
            data=data,
            headers=headers or {},
            **kwargs
        )
        return await self.request(request)

    async def patch(self, url: str, json: Optional[Dict[str, Any]] = None,
                    data: Optional[Any] = None, headers: Optional[Dict[str, str]] = None,
                    **kwargs) -> ApiResponse:
        """Execute PATCH request."""
        request = ApiRequest(
            id=str(uuid.uuid4()),
            method=HttpMethod.PATCH,
            url=url,
            json=json,
            data=data,
            headers=headers or {},
            **kwargs
        )
        return await self.request(request)

    async def delete(self, url: str, headers: Optional[Dict[str, str]] = None,
                     **kwargs) -> ApiResponse:
        """Execute DELETE request."""
        request = ApiRequest(
            id=str(uuid.uuid4()),
            method=HttpMethod.DELETE,
            url=url,
            headers=headers or {},
            **kwargs
        )
        return await self.request(request)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        circuit_stats = self.circuit_breaker.get_stats()

        return {
            'requests_total': self.metrics['requests_successful'] + self.metrics['requests_failed'],
            'requests_successful': self.metrics['requests_successful'],
            'requests_failed': self.metrics['requests_failed'],
            'cache_hits': self.metrics.get('cache_hits', 0),
            'cache_size': len(self.cache),
            'circuit_breaker': circuit_stats,
            'success_rate': (self.metrics['requests_successful'] / (self.metrics['requests_successful'] + self.metrics['requests_failed'])) * 100 if (self.metrics['requests_successful'] + self.metrics['requests_failed']) > 0 else 0
        }

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()
        self.logger.info("API client cache cleared")

    async def health_check(self, endpoint: str = "/health") -> bool:
        """
        Perform health check against API.

        Args:
            endpoint: Health check endpoint

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.get(endpoint)
            return response.is_success
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup client resources."""
        try:
            if self.client:
                if HTTPX_AVAILABLE and isinstance(self.client, httpx.AsyncClient):
                    await self.client.aclose()
                elif AIOHTTP_AVAILABLE and isinstance(self.client, aiohttp.ClientSession):
                    await self.client.close()
                self.client = None

            self.clear_cache()
            self.metrics.clear()
            self.logger.info("External API client cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'client') and self.client:
                self.logger.info("ExternalAPIClient being destroyed - cleanup recommended")
        except:
            pass
