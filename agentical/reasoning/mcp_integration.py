"""
MCP Integration Component for Bayes-MCP Server Communication

This module provides integration with the bayes-mcp server for the Agentical
framework's Bayesian reasoning system, enabling seamless communication with
external Bayesian inference services through the Model Context Protocol.

Features:
- Direct integration with bayes-mcp server
- Asynchronous communication with connection pooling
- Request/response handling with retry logic
- Server health monitoring and failover
- Performance optimization and caching
- Comprehensive logging and observability
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import asyncio
import json
import aiohttp
from urllib.parse import urljoin

import logfire
from pydantic import BaseModel, Field, validator

from agentical.core.exceptions import (
    AgentError,
    ValidationError,
    ConfigurationError,
    ConnectionError
)
from agentical.core.structured_logging import (
    StructuredLogger,
    CorrelationContext,
    LogLevel,
    OperationType
)


class ServerStatus(str, Enum):
    """MCP server status states."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class InferenceType(str, Enum):
    """Types of inference requests supported by bayes-mcp server."""
    BELIEF_UPDATE = "belief_update"
    HYPOTHESIS_TEST = "hypothesis_test"
    PARAMETER_ESTIMATION = "parameter_estimation"
    MODEL_COMPARISON = "model_comparison"
    PREDICTION = "prediction"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"


class ResponseFormat(str, Enum):
    """Response formats for MCP communication."""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


@dataclass
class InferenceRequest:
    """Request structure for bayes-mcp server inference."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    inference_type: InferenceType = InferenceType.BELIEF_UPDATE

    # Core request data
    hypothesis_id: str = ""
    prior_belief: float = 0.5
    evidence_data: Dict[str, Any] = field(default_factory=dict)
    evidence_likelihood: float = 0.0
    evidence_reliability: float = 1.0

    # Request parameters
    method: str = "exact"
    confidence_threshold: float = 0.75
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6

    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_ms: int = 30000

    # Request tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class InferenceResponse:
    """Response structure from bayes-mcp server."""
    request_id: str = ""
    response_id: str = field(default_factory=lambda: str(uuid4()))

    # Response status
    success: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Core inference results
    posterior_belief: float = 0.0
    confidence_level: float = 0.0
    uncertainty_measure: float = 0.0

    # Additional results
    likelihood_ratio: float = 0.0
    bayes_factor: float = 0.0
    evidence_weight: float = 0.0

    # Computational metadata
    iterations_performed: int = 0
    convergence_achieved: bool = False
    computation_time_ms: float = 0.0
    method_used: str = ""

    # Extended results
    distribution_parameters: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)

    # Response metadata
    server_id: str = ""
    server_version: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MCPConfig(BaseModel):
    """Configuration for MCP server integration."""

    # Server connection
    server_url: str = Field(default="http://localhost:3000")
    api_version: str = Field(default="v1")
    timeout_seconds: float = Field(default=30.0, gt=0.0)

    # Connection pooling
    max_connections: int = Field(default=10, ge=1)
    connection_timeout: float = Field(default=5.0, gt=0.0)
    keep_alive_timeout: float = Field(default=60.0, gt=0.0)

    # Retry configuration
    max_retries: int = Field(default=3, ge=0)
    retry_delay_ms: int = Field(default=1000, ge=0)
    exponential_backoff: bool = True

    # Request settings
    default_format: ResponseFormat = ResponseFormat.JSON
    compression: bool = True
    batch_requests: bool = True
    max_batch_size: int = Field(default=10, ge=1)

    # Health monitoring
    health_check_interval: int = Field(default=30, ge=1)
    max_consecutive_failures: int = Field(default=5, ge=1)
    circuit_breaker_timeout: int = Field(default=60, ge=1)

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=0)
    enable_request_compression: bool = True

    # Security
    api_key: Optional[str] = None
    enable_tls: bool = True
    verify_ssl: bool = True

    # Logging
    log_requests: bool = True
    log_responses: bool = True
    detailed_logging: bool = False


class BayesMCPClient:
    """
    Client for communicating with bayes-mcp server.

    Provides comprehensive integration with bayes-mcp server including:
    - Asynchronous request/response handling
    - Connection pooling and retry logic
    - Health monitoring and circuit breaker pattern
    - Request batching and caching
    - Performance optimization and observability
    """

    def __init__(
        self,
        config: MCPConfig,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the MCP client.

        Args:
            config: Client configuration
            logger: Optional structured logger
        """
        self.config = config
        self.logger = logger or StructuredLogger("bayes_mcp_client")

        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self.server_status = ServerStatus.OFFLINE
        self.last_health_check = datetime.utcnow()

        # Request tracking
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.request_cache: Dict[str, InferenceResponse] = {}

        # Performance metrics
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.cache_hits = 0

        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time: Optional[datetime] = None

        # Batch processing
        self.batch_queue: List[InferenceRequest] = []
        self.batch_processing_task: Optional[asyncio.Task] = None

        logfire.info(
            "Bayes MCP client initialized",
            server_url=config.server_url,
            api_version=config.api_version
        )

    async def initialize(self) -> None:
        """Initialize the MCP client and establish server connection."""
        with logfire.span("Initialize MCP Client"):
            try:
                # Create HTTP session with connection pooling
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=self.config.keep_alive_timeout,
                    enable_cleanup_closed=True
                )

                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout_seconds,
                    connect=self.config.connection_timeout
                )

                headers = {
                    "User-Agent": "Agentical-BayesMCP-Client/1.0",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }

                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"

                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=headers
                )

                # Perform initial health check
                await self._health_check()

                # Start batch processing if enabled
                if self.config.batch_requests:
                    self.batch_processing_task = asyncio.create_task(
                        self._batch_processor()
                    )

                self.logger.log(
                    LogLevel.INFO,
                    "MCP client initialized successfully",
                    operation_type=OperationType.INITIALIZATION,
                    server_status=self.server_status.value
                )

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Failed to initialize MCP client: {str(e)}",
                    operation_type=OperationType.INITIALIZATION,
                    error=str(e)
                )
                raise ConfigurationError(f"MCP client initialization failed: {str(e)}")

    async def send_inference_request(
        self,
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        Send inference request to bayes-mcp server.

        Args:
            request: Inference request to send

        Returns:
            Inference response from server
        """
        start_time = datetime.utcnow()

        with logfire.span("Send Inference Request", request_id=request.request_id):
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open():
                    raise ConnectionError("Circuit breaker is open")

                # Check cache
                cache_key = self._generate_cache_key(request)
                if self.config.enable_caching and cache_key in self.request_cache:
                    cached_response = self.request_cache[cache_key]
                    if self._is_cache_valid(cached_response):
                        self.cache_hits += 1
                        return cached_response

                # Add to batch queue if batch processing is enabled
                if self.config.batch_requests and len(self.batch_queue) < self.config.max_batch_size:
                    self.batch_queue.append(request)
                    self.pending_requests[request.request_id] = request

                    # Wait for batch processing or timeout
                    timeout_time = start_time + timedelta(milliseconds=request.timeout_ms)
                    while (request.request_id in self.pending_requests and
                           datetime.utcnow() < timeout_time):
                        await asyncio.sleep(0.1)

                    if request.request_id in self.pending_requests:
                        # Request timed out in batch
                        del self.pending_requests[request.request_id]
                        raise TimeoutError("Batch request timeout")

                    # Request was processed, get result from cache
                    if cache_key in self.request_cache:
                        return self.request_cache[cache_key]

                # Send individual request
                response = await self._send_individual_request(request)

                # Cache successful response
                if self.config.enable_caching and response.success:
                    self.request_cache[cache_key] = response

                # Update metrics
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.total_response_time += response_time
                self.request_count += 1

                if response.success:
                    self.successful_requests += 1
                    self.consecutive_failures = 0
                else:
                    self.failed_requests += 1
                    self.consecutive_failures += 1
                    self._check_circuit_breaker()

                self.logger.log(
                    LogLevel.INFO,
                    f"Inference request completed",
                    operation_type=OperationType.REQUEST,
                    request_id=request.request_id,
                    success=response.success,
                    response_time_ms=response_time
                )

                return response

            except Exception as e:
                self.failed_requests += 1
                self.consecutive_failures += 1
                self._check_circuit_breaker()

                self.logger.log(
                    LogLevel.ERROR,
                    f"Inference request failed: {str(e)}",
                    operation_type=OperationType.REQUEST,
                    request_id=request.request_id,
                    error=str(e)
                )
                raise AgentError(f"Inference request failed: {str(e)}")

    async def batch_inference_requests(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        Send multiple inference requests as a batch.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses
        """
        with logfire.span("Batch Inference Requests", batch_size=len(requests)):
            if not requests:
                return []

            # Split into batches if necessary
            responses = []
            batch_size = self.config.max_batch_size

            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                batch_responses = await self._send_batch_request(batch)
                responses.extend(batch_responses)

            return responses

    async def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and health information."""
        with logfire.span("Get Server Status"):
            await self._health_check()

            avg_response_time = (
                self.total_response_time / max(self.request_count, 1)
                if self.request_count > 0 else 0.0
            )

            success_rate = (
                self.successful_requests / max(self.request_count, 1)
                if self.request_count > 0 else 0.0
            )

            cache_hit_rate = (
                self.cache_hits / max(self.request_count, 1)
                if self.config.enable_caching and self.request_count > 0 else 0.0
            )

            status = {
                "server_status": self.server_status.value,
                "circuit_breaker_open": self.circuit_breaker_open,
                "consecutive_failures": self.consecutive_failures,
                "last_health_check": self.last_health_check.isoformat(),
                "metrics": {
                    "request_count": self.request_count,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "success_rate": success_rate,
                    "average_response_time_ms": avg_response_time,
                    "cache_hits": self.cache_hits,
                    "cache_hit_rate": cache_hit_rate,
                    "pending_requests": len(self.pending_requests),
                    "cached_responses": len(self.request_cache)
                }
            }

            return status

    async def close(self) -> None:
        """Close the MCP client and clean up resources."""
        with logfire.span("Close MCP Client"):
            try:
                # Cancel batch processing task
                if self.batch_processing_task and not self.batch_processing_task.done():
                    self.batch_processing_task.cancel()
                    try:
                        await self.batch_processing_task
                    except asyncio.CancelledError:
                        pass

                # Close HTTP session
                if self.session and not self.session.closed:
                    await self.session.close()

                self.logger.log(
                    LogLevel.INFO,
                    "MCP client closed successfully",
                    operation_type=OperationType.CLEANUP
                )

            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Error closing MCP client: {str(e)}",
                    operation_type=OperationType.CLEANUP,
                    error=str(e)
                )

    # Private helper methods

    async def _send_individual_request(
        self,
        request: InferenceRequest
    ) -> InferenceResponse:
        """Send a single inference request to the server."""
        if not self.session:
            raise ConnectionError("MCP client not initialized")

        url = urljoin(self.config.server_url, f"/api/{self.config.api_version}/inference")

        request_data = {
            "request_id": request.request_id,
            "inference_type": request.inference_type.value,
            "hypothesis_id": request.hypothesis_id,
            "prior_belief": request.prior_belief,
            "evidence_data": request.evidence_data,
            "evidence_likelihood": request.evidence_likelihood,
            "evidence_reliability": request.evidence_reliability,
            "method": request.method,
            "confidence_threshold": request.confidence_threshold,
            "max_iterations": request.max_iterations,
            "convergence_tolerance": request.convergence_tolerance,
            "context": request.context,
            "priority": request.priority,
            "timeout_ms": request.timeout_ms
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, json=request_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return self._parse_response(response_data)
                    else:
                        error_text = await response.text()
                        raise ConnectionError(f"Server returned {response.status}: {error_text}")

            except asyncio.TimeoutError:
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_ms / 1000.0
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise TimeoutError("Request timeout after retries")

            except Exception as e:
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_ms / 1000.0
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _send_batch_request(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Send a batch of inference requests to the server."""
        if not self.session:
            raise ConnectionError("MCP client not initialized")

        url = urljoin(self.config.server_url, f"/api/{self.config.api_version}/inference/batch")

        batch_data = {
            "requests": [
                {
                    "request_id": req.request_id,
                    "inference_type": req.inference_type.value,
                    "hypothesis_id": req.hypothesis_id,
                    "prior_belief": req.prior_belief,
                    "evidence_data": req.evidence_data,
                    "evidence_likelihood": req.evidence_likelihood,
                    "evidence_reliability": req.evidence_reliability,
                    "method": req.method,
                    "confidence_threshold": req.confidence_threshold,
                    "max_iterations": req.max_iterations,
                    "convergence_tolerance": req.convergence_tolerance,
                    "context": req.context,
                    "priority": req.priority,
                    "timeout_ms": req.timeout_ms
                }
                for req in requests
            ]
        }

        async with self.session.post(url, json=batch_data) as response:
            if response.status == 200:
                response_data = await response.json()
                return [self._parse_response(resp) for resp in response_data.get("responses", [])]
            else:
                error_text = await response.text()
                raise ConnectionError(f"Batch request failed {response.status}: {error_text}")

    def _parse_response(self, response_data: Dict[str, Any]) -> InferenceResponse:
        """Parse server response into InferenceResponse object."""
        return InferenceResponse(
            request_id=response_data.get("request_id", ""),
            response_id=response_data.get("response_id", str(uuid4())),
            success=response_data.get("success", False),
            error_code=response_data.get("error_code"),
            error_message=response_data.get("error_message"),
            posterior_belief=response_data.get("posterior_belief", 0.0),
            confidence_level=response_data.get("confidence_level", 0.0),
            uncertainty_measure=response_data.get("uncertainty_measure", 0.0),
            likelihood_ratio=response_data.get("likelihood_ratio", 0.0),
            bayes_factor=response_data.get("bayes_factor", 0.0),
            evidence_weight=response_data.get("evidence_weight", 0.0),
            iterations_performed=response_data.get("iterations_performed", 0),
            convergence_achieved=response_data.get("convergence_achieved", False),
            computation_time_ms=response_data.get("computation_time_ms", 0.0),
            method_used=response_data.get("method_used", ""),
            distribution_parameters=response_data.get("distribution_parameters", {}),
            confidence_intervals=response_data.get("confidence_intervals", {}),
            diagnostic_info=response_data.get("diagnostic_info", {}),
            server_id=response_data.get("server_id", ""),
            server_version=response_data.get("server_version", "")
        )

    async def _health_check(self) -> None:
        """Perform health check on the bayes-mcp server."""
        if not self.session:
            self.server_status = ServerStatus.OFFLINE
            return

        try:
            url = urljoin(self.config.server_url, f"/api/{self.config.api_version}/health")

            async with self.session.get(url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    server_status = health_data.get("status", "unknown")

                    if server_status == "healthy":
                        self.server_status = ServerStatus.ONLINE
                    elif server_status == "degraded":
                        self.server_status = ServerStatus.DEGRADED
                    else:
                        self.server_status = ServerStatus.ERROR
                else:
                    self.server_status = ServerStatus.ERROR

        except Exception:
            self.server_status = ServerStatus.OFFLINE

        self.last_health_check = datetime.utcnow()

    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            "inference_type": request.inference_type.value,
            "hypothesis_id": request.hypothesis_id,
            "prior_belief": request.prior_belief,
            "evidence_likelihood": request.evidence_likelihood,
            "evidence_reliability": request.evidence_reliability,
            "method": request.method
        }
        return f"mcp_cache_{hash(json.dumps(key_data, sort_keys=True))}"

    def _is_cache_valid(self, response: InferenceResponse) -> bool:
        """Check if cached response is still valid."""
        if not self.config.enable_caching:
            return False

        age_seconds = (datetime.utcnow() - response.timestamp).total_seconds()
        return age_seconds < self.config.cache_ttl_seconds

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker_open:
            return False

        if (self.circuit_breaker_reset_time and
            datetime.utcnow() >= self.circuit_breaker_reset_time):
            self.circuit_breaker_open = False
            self.consecutive_failures = 0
            self.circuit_breaker_reset_time = None
            return False

        return True

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be triggered."""
        if (self.consecutive_failures >= self.config.max_consecutive_failures and
            not self.circuit_breaker_open):
            self.circuit_breaker_open = True
            self.circuit_breaker_reset_time = (
                datetime.utcnow() + timedelta(seconds=self.config.circuit_breaker_timeout)
            )

            self.logger.log(
                LogLevel.WARNING,
                "Circuit breaker opened due to consecutive failures",
                operation_type=OperationType.MONITORING,
                consecutive_failures=self.consecutive_failures,
                reset_time=self.circuit_breaker_reset_time.isoformat()
            )

    async def _batch_processor(self) -> None:
        """Background task for processing batched requests."""
        while True:
            try:
                if len(self.batch_queue) >= self.config.max_batch_size:
                    # Process full batch
                    batch = self.batch_queue[:self.config.max_batch_size]
                    self.batch_queue = self.batch_queue[self.config.max_batch_size:]

                    responses = await self._send_batch_request(batch)

                    # Cache responses and remove from pending
                    for response in responses:
                        if response.request_id in self.pending_requests:
                            request = self.pending_requests[response.request_id]
                            cache_key = self._generate_cache_key(request)
                            if self.config.enable_caching and response.success:
                                self.request_cache[cache_key] = response
                            del self.pending_requests[response.request_id]

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log(
                    LogLevel.ERROR,
                    f"Batch processor error: {str(e)}",
                    operation_type=OperationType.BATCH_PROCESSING,
                    error=str(e)
                )
                await asyncio.sleep(1.0)  # Wait before retrying
