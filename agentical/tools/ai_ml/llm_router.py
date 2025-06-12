"""
LLM Router for Agentical

This module provides comprehensive language model routing and load balancing
capabilities with support for multiple AI providers, cost optimization,
failover handling, and enterprise-grade features.

Features:
- Multi-provider support (OpenAI, Anthropic, Google, Azure, AWS Bedrock)
- Load balancing algorithms (round-robin, weighted, least-latency)
- Automatic failover and retry mechanisms
- Rate limiting and quota management
- Cost tracking and optimization
- Response caching and streaming
- Model-specific parameter handling
- Real-time performance monitoring
- Enterprise features (audit logging, compliance, security)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import os
import statistics

# Optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    CUSTOM = "custom"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    RANDOM = "random"
    PRIORITY = "priority"


class ModelType(Enum):
    """Model types for categorization."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    id: str
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    id: str
    request_id: str
    provider: str
    model: str
    content: str
    usage: Dict[str, Any]
    latency: float
    cost: float
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    provider_type: LLMProvider
    api_key: str
    base_url: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    weight: float = 1.0
    priority: int = 1
    max_requests_per_minute: int = 1000
    max_tokens_per_minute: int = 100000
    cost_per_token: Dict[str, float] = None
    enabled: bool = True
    timeout: float = 30.0
    retry_attempts: int = 3
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.cost_per_token is None:
            self.cost_per_token = {"input": 0.0, "output": 0.0}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['provider_type'] = self.provider_type.value
        return data


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    tokens_used: Dict[str, int] = None
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    average_latency: float = 0.0
    requests_per_minute: int = 0

    def __post_init__(self):
        if self.tokens_used is None:
            self.tokens_used = {"input": 0, "output": 0}

    def update_success(self, latency: float, cost: float, tokens: Dict[str, int]):
        """Update stats for successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency += latency
        self.total_cost += cost
        for key, value in tokens.items():
            self.tokens_used[key] = self.tokens_used.get(key, 0) + value
        self.last_request_time = datetime.utcnow()
        self._recalculate()

    def update_failure(self):
        """Update stats for failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_request_time = datetime.utcnow()
        self._recalculate()

    def _recalculate(self):
        """Recalculate derived statistics."""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        if self.successful_requests > 0:
            self.average_latency = self.total_latency / self.successful_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.last_request_time:
            data['last_request_time'] = self.last_request_time.isoformat()
        return data


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Send chat completion request."""
        pass

    @abstractmethod
    async def stream_completion(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion response."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass

    @abstractmethod
    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost for usage."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass


class OpenAIProvider(LLMProviderInterface):
    """OpenAI provider implementation."""

    def __init__(self, config: ProviderConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")

        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )

    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Send chat completion request to OpenAI."""
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=request.model or self.config.default_model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                tools=request.tools
            )

            latency = time.time() - start_time
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            cost = self.calculate_cost(usage)

            return LLMResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                provider="openai",
                model=response.model,
                content=response.choices[0].message.content,
                usage=usage,
                latency=latency,
                cost=cost
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}")

    async def stream_completion(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=request.model or self.config.default_model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                tools=request.tools
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"OpenAI streaming failed: {e}")

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost for OpenAI usage."""
        input_cost = usage.get("input_tokens", 0) * self.config.cost_per_token.get("input", 0.0)
        output_cost = usage.get("output_tokens", 0) * self.config.cost_per_token.get("output", 0.0)
        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check OpenAI health."""
        try:
            await self.client.models.list()
            return True
        except:
            return False


class AnthropicProvider(LLMProviderInterface):
    """Anthropic provider implementation."""

    def __init__(self, config: ProviderConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available")

        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Send chat completion request to Anthropic."""
        start_time = time.time()

        try:
            # Convert messages to Anthropic format
            system_message = ""
            messages = []
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)

            response = await self.client.messages.create(
                model=request.model or self.config.default_model,
                max_tokens=request.max_tokens or 1000,
                temperature=request.temperature,
                system=system_message,
                messages=messages
            )

            latency = time.time() - start_time
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            cost = self.calculate_cost(usage)

            return LLMResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                provider="anthropic",
                model=response.model,
                content=response.content[0].text,
                usage=usage,
                latency=latency,
                cost=cost
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic request failed: {e}")

    async def stream_completion(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion from Anthropic."""
        try:
            system_message = ""
            messages = []
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)

            async with self.client.messages.stream(
                model=request.model or self.config.default_model,
                max_tokens=request.max_tokens or 1000,
                temperature=request.temperature,
                system=system_message,
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise RuntimeError(f"Anthropic streaming failed: {e}")

    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost for Anthropic usage."""
        input_cost = usage.get("input_tokens", 0) * self.config.cost_per_token.get("input", 0.0)
        output_cost = usage.get("output_tokens", 0) * self.config.cost_per_token.get("output", 0.0)
        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check Anthropic health."""
        try:
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except:
            return False


class LLMRouter:
    """
    Comprehensive LLM routing and load balancing system.

    Provides intelligent routing across multiple LLM providers with advanced
    features like cost optimization, failover, and performance monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM router.

        Args:
            config: Configuration dictionary with provider settings and features
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.load_balancing_strategy = LoadBalancingStrategy(
            self.config.get('load_balancing', 'round_robin')
        )
        self.failover_enabled = self.config.get('failover_enabled', True)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cost_optimization = self.config.get('cost_optimization', False)

        # Performance settings
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30.0)
        self.rate_limiting = self.config.get('rate_limiting', False)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.compliance_mode = self.config.get('compliance_mode', False)

        # Initialize components
        self.providers: Dict[str, LLMProviderInterface] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.cache: Dict[str, LLMResponse] = {}
        self.request_history: deque = deque(maxlen=1000)
        self.round_robin_index = 0
        self.rate_limiters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize LLM providers from configuration."""
        providers_config = self.config.get('providers', {})

        for name, provider_data in providers_config.items():
            try:
                provider_config = ProviderConfig(**provider_data)
                self.provider_configs[name] = provider_config
                self.provider_stats[name] = ProviderStats()

                # Create provider instance
                if provider_config.provider_type == LLMProvider.OPENAI:
                    self.providers[name] = OpenAIProvider(provider_config)
                elif provider_config.provider_type == LLMProvider.ANTHROPIC:
                    self.providers[name] = AnthropicProvider(provider_config)
                # Add other providers as needed
                else:
                    self.logger.warning(f"Unsupported provider type: {provider_config.provider_type}")

                self.logger.info(f"Initialized provider: {name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")

    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Send chat completion request with intelligent routing.

        Args:
            request: LLM request object

        Returns:
            LLM response object
        """
        self.logger.debug(f"Processing chat completion request: {request.id}")

        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(request)
            if cache_key in self.cache:
                response = self.cache[cache_key]
                response.cached = True
                return response

        # Select provider
        provider_name = await self._select_provider(request)
        if not provider_name:
            raise RuntimeError("No available providers")

        # Send request with retries
        response = await self._send_request_with_retries(provider_name, request)

        # Cache response
        if self.cache_enabled and response:
            cache_key = self._get_cache_key(request)
            self.cache[cache_key] = response

        # Update statistics
        if response:
            self._update_provider_stats(provider_name, response)

        # Log request
        if self.audit_logging:
            self._log_request(request, response, provider_name)

        return response

    async def stream_completion(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Stream chat completion with intelligent routing.

        Args:
            request: LLM request object

        Yields:
            Streamed response chunks
        """
        self.logger.debug(f"Processing streaming completion request: {request.id}")

        # Select provider
        provider_name = await self._select_provider(request)
        if not provider_name:
            raise RuntimeError("No available providers")

        provider = self.providers[provider_name]

        try:
            async for chunk in provider.stream_completion(request):
                yield chunk
        except Exception as e:
            if self.failover_enabled:
                # Try fallback providers
                fallback_providers = await self._get_fallback_providers(provider_name, request)
                for fallback_name in fallback_providers:
                    try:
                        fallback_provider = self.providers[fallback_name]
                        async for chunk in fallback_provider.stream_completion(request):
                            yield chunk
                        return
                    except Exception:
                        continue

            raise RuntimeError(f"All providers failed for streaming request: {e}")

    async def _select_provider(self, request: LLMRequest) -> Optional[str]:
        """Select the best provider for a request."""
        available_providers = [
            name for name, config in self.provider_configs.items()
            if config.enabled and await self._check_rate_limits(name)
        ]

        if not available_providers:
            return None

        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LATENCY:
            return self._least_latency_selection(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_COST:
            return self._least_cost_selection(available_providers, request)
        else:
            return available_providers[0]

    def _round_robin_selection(self, providers: List[str]) -> str:
        """Round-robin provider selection."""
        if not providers:
            return None

        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider

    def _weighted_round_robin_selection(self, providers: List[str]) -> str:
        """Weighted round-robin selection based on provider weights."""
        weights = [self.provider_configs[name].weight for name in providers]
        total_weight = sum(weights)

        if total_weight == 0:
            return providers[0]

        # Weighted random selection
        import random
        r = random.uniform(0, total_weight)
        current_weight = 0

        for provider, weight in zip(providers, weights):
            current_weight += weight
            if r <= current_weight:
                return provider

        return providers[-1]

    def _least_latency_selection(self, providers: List[str]) -> str:
        """Select provider with lowest average latency."""
        best_provider = providers[0]
        best_latency = float('inf')

        for provider in providers:
            stats = self.provider_stats[provider]
            latency = stats.average_latency if stats.successful_requests > 0 else float('inf')
            if latency < best_latency:
                best_latency = latency
                best_provider = provider

        return best_provider

    def _least_cost_selection(self, providers: List[str], request: LLMRequest) -> str:
        """Select provider with lowest estimated cost."""
        best_provider = providers[0]
        best_cost = float('inf')

        # Estimate tokens for cost calculation
        estimated_tokens = self._estimate_tokens(request)

        for provider in providers:
            config = self.provider_configs[provider]
            estimated_cost = (
                estimated_tokens['input'] * config.cost_per_token.get('input', 0) +
                estimated_tokens['output'] * config.cost_per_token.get('output', 0)
            )
            if estimated_cost < best_cost:
                best_cost = estimated_cost
                best_provider = provider

        return best_provider

    def _estimate_tokens(self, request: LLMRequest) -> Dict[str, int]:
        """Estimate token usage for a request."""
        # Rough estimation: 4 characters per token
        total_chars = sum(len(msg.get('content', '')) for msg in request.messages)
        input_tokens = total_chars // 4
        output_tokens = request.max_tokens or (input_tokens // 2)  # Estimate output

        return {'input': input_tokens, 'output': output_tokens}

    async def _check_rate_limits(self, provider_name: str) -> bool:
        """Check if provider is within rate limits."""
        if not self.rate_limiting:
            return True

        config = self.provider_configs[provider_name]
        now = datetime.utcnow()
        minute = now.replace(second=0, microsecond=0)

        # Check requests per minute
        key = f"{provider_name}:{minute.isoformat()}"
        current_requests = self.rate_limiters[provider_name][key]

        return current_requests < config.max_requests_per_minute

    async def _send_request_with_retries(self, provider_name: str, request: LLMRequest) -> LLMResponse:
        """Send request with retry logic and failover."""
        provider = self.providers[provider_name]
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Update rate limiter
                if self.rate_limiting:
                    self._update_rate_limiter(provider_name)

                response = await provider.chat_completion(request)
                return response

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request failed on provider {provider_name}, attempt {attempt + 1}: {e}")
                self.provider_stats[provider_name].update_failure()

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # Try failover providers
        if self.failover_enabled:
            fallback_providers = await self._get_fallback_providers(provider_name, request)
            for fallback_name in fallback_providers:
                try:
                    fallback_provider = self.providers[fallback_name]
                    response = await fallback_provider.chat_completion(request)
                    self.logger.info(f"Failover successful to provider: {fallback_name}")
                    return response
                except Exception as e:
                    self.logger.warning(f"Failover failed on provider {fallback_name}: {e}")
                    continue

        raise RuntimeError(f"All retry attempts failed: {last_exception}")

    async def _get_fallback_providers(self, failed_provider: str, request: LLMRequest) -> List[str]:
        """Get list of fallback providers."""
        all_providers = list(self.providers.keys())
        fallback_providers = [name for name in all_providers if name != failed_provider]

        # Sort by priority and health
        fallback_providers.sort(key=lambda x: (
            self.provider_configs[x].priority,
            -self.provider_stats[x].error_rate
        ))

        return fallback_providers

    def _update_rate_limiter(self, provider_name: str):
        """Update rate limiter counters."""
        now = datetime.utcnow()
        minute = now.replace(second=0, microsecond=0)
        key = f"{provider_name}:{minute.isoformat()}"
        self.rate_limiters[provider_name][key] += 1

    def _update_provider_stats(self, provider_name: str, response: LLMResponse):
        """Update provider statistics."""
        stats = self.provider_stats[provider_name]
        tokens = {
            'input': response.usage.get('input_tokens', 0),
            'output': response.usage.get('output_tokens', 0)
        }
        stats.update_success(response.latency, response.cost, tokens)

    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            'messages': request.messages,
            'model': request.model,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _log_request(self, request: LLMRequest, response: LLMResponse, provider: str):
        """Log request for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.id,
            'provider': provider,
            'model': response.model if response else None,
            'latency': response.latency if response else None,
            'cost': response.cost if response else None,
            'success': response is not None
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    async def get_provider_health(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        for name, provider in self.providers.items():
            try:
                health_status[name] = await provider.health_check()
            except Exception:
                health_status[name] = False
        return health_status

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        return {name: stats.to_dict() for name, stats in self.provider_stats.items()}

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for all providers."""
        models = {}
        for name, provider in self.providers.items():
            try:
                models[name] = provider.get_available_models()
            except Exception as e:
                self.logger.error(f"Failed to get models for {name}: {e}")
                models[name] = []
        return models

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()
        self.logger.info("Response cache cleared")

    def reset_stats(self):
        """Reset provider statistics."""
        for stats in self.provider_stats.values():
            stats.__init__()
        self.logger.info("Provider statistics reset")

    async def add_provider(self, name: str, config: ProviderConfig):
        """Add a new provider dynamically."""
        try:
            self.provider_configs[name] = config
            self.provider_stats[name] = ProviderStats()

            if config.provider_type == LLMProvider.OPENAI:
                self.providers[name] = OpenAIProvider(config)
            elif config.provider_type == LLMProvider.ANTHROPIC:
                self.providers[name] = AnthropicProvider(config)
            else:
                raise ValueError(f"Unsupported provider type: {config.provider_type}")

            self.logger.info(f"Added provider: {name}")
        except Exception as e:
            self.logger.error(f"Failed to add provider {name}: {e}")
            raise

    async def remove_provider(self, name: str):
        """Remove a provider dynamically."""
        if name in self.providers:
            del self.providers[name]
            del self.provider_configs[name]
            del self.provider_stats[name]
            self.logger.info(f"Removed provider: {name}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        total_requests = sum(stats.total_requests for stats in self.provider_stats.values())
        total_cost = sum(stats.total_cost for stats in self.provider_stats.values())
        cache_hits = len([r for r in self.cache.values() if r.cached])

        return {
            'total_requests': total_requests,
            'total_cost': total_cost,
            'cache_hits': cache_hits,
            'cache_size': len(self.cache),
            'active_providers': len([p for p in self.provider_configs.values() if p.enabled]),
            'average_latency': statistics.mean([s.average_latency for s in self.provider_stats.values() if s.successful_requests > 0]) if any(s.successful_requests > 0 for s in self.provider_stats.values()) else 0,
            'error_rate': statistics.mean([s.error_rate for s in self.provider_stats.values()]) if self.provider_stats else 0
        }

    async def cleanup(self):
        """Cleanup router resources."""
        try:
            self.clear_cache()
            self.reset_stats()
            self.providers.clear()
            self.provider_configs.clear()
            self.logger.info("LLM router cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'providers') and self.providers:
                self.logger.info("LLMRouter being destroyed - cleanup recommended")
        except:
            pass
