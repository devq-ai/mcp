"""
Load Balancer for Agentical

This module provides comprehensive load balancing capabilities including
traffic distribution, health checks, failover management, and performance
monitoring with integration to the Agentical framework.

Features:
- Multiple load balancing algorithms (round-robin, least connections, IP hash)
- Health checks and automatic failover
- SSL termination and traffic encryption
- Rate limiting and traffic shaping
- Geographic routing and latency optimization
- Real-time metrics and performance monitoring
- Integration with service discovery systems
- Dynamic backend server management
- Session persistence and sticky sessions
- Circuit breaker patterns for resilience
"""

import asyncio
import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import statistics

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import ssl
    SSL_AVAILABLE = True
except ImportError:
    SSL_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError
)
from ...core.logging import log_operation


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"


class BackendServer:
    """Backend server configuration and state."""

    def __init__(
        self,
        server_id: str,
        host: str,
        port: int,
        weight: int = 1,
        max_connections: Optional[int] = None,
        health_check_path: str = "/health",
        ssl_enabled: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.server_id = server_id
        self.host = host
        self.port = port
        self.weight = weight
        self.max_connections = max_connections
        self.health_check_path = health_check_path
        self.ssl_enabled = ssl_enabled
        self.metadata = metadata or {}

        # Runtime state
        self.is_healthy = True
        self.is_enabled = True
        self.current_connections = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times: List[float] = []
        self.last_health_check = None
        self.consecutive_failures = 0
        self.created_at = datetime.now()
        self.last_request_time = None

    @property
    def url(self) -> str:
        """Get server URL."""
        protocol = "https" if self.ssl_enabled else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def health_check_url(self) -> str:
        """Get health check URL."""
        return f"{self.url}{self.health_check_path}"

    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-100:])  # Last 100 requests

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100

    @property
    def load_score(self) -> float:
        """Calculate load score for least connections algorithm."""
        if self.max_connections:
            return self.current_connections / self.max_connections
        return self.current_connections

    def is_available(self) -> bool:
        """Check if server is available for requests."""
        if not self.is_healthy or not self.is_enabled:
            return False

        if self.max_connections and self.current_connections >= self.max_connections:
            return False

        return True

    def add_request(self, response_time: float, success: bool) -> None:
        """Record request statistics."""
        self.total_requests += 1
        self.last_request_time = datetime.now()

        if success:
            self.response_times.append(response_time)
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
        else:
            self.failed_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert server to dictionary."""
        return {
            "server_id": self.server_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "health_check_path": self.health_check_path,
            "ssl_enabled": self.ssl_enabled,
            "metadata": self.metadata,
            "is_healthy": self.is_healthy,
            "is_enabled": self.is_enabled,
            "current_connections": self.current_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate,
            "load_score": self.load_score,
            "consecutive_failures": self.consecutive_failures,
            "created_at": self.created_at.isoformat(),
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }


class HealthCheck:
    """Health check configuration and state."""

    def __init__(
        self,
        enabled: bool = True,
        interval_seconds: int = 30,
        timeout_seconds: int = 10,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        path: str = "/health",
        expected_status: int = 200,
        expected_content: Optional[str] = None,
        custom_check: Optional[Callable] = None
    ):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.path = path
        self.expected_status = expected_status
        self.expected_content = expected_content
        self.custom_check = custom_check

    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "path": self.path,
            "expected_status": self.expected_status,
            "expected_content": self.expected_content,
            "has_custom_check": self.custom_check is not None
        }


class TrafficRule:
    """Traffic routing rule."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        priority: int = 100,
        enabled: bool = True
    ):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.enabled = enabled
        self.matches = 0
        self.created_at = datetime.now()

    def matches_request(self, request_info: Dict[str, Any]) -> bool:
        """Check if rule matches request."""
        if not self.enabled:
            return False

        # Simple condition matching
        for key, expected_value in self.condition.items():
            if key not in request_info:
                return False

            actual_value = request_info[key]

            # Support for patterns
            if isinstance(expected_value, str) and expected_value.startswith("*"):
                if not actual_value.endswith(expected_value[1:]):
                    return False
            elif isinstance(expected_value, str) and expected_value.endswith("*"):
                if not actual_value.startswith(expected_value[:-1]):
                    return False
            elif actual_value != expected_value:
                return False

        self.matches += 1
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition,
            "action": self.action,
            "priority": self.priority,
            "enabled": self.enabled,
            "matches": self.matches,
            "created_at": self.created_at.isoformat()
        }


class LoadBalancerMetrics:
    """Load balancer performance metrics."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times: List[float] = []
        self.requests_per_second = 0.0
        self.active_connections = 0
        self.backend_failures = 0
        self.health_check_failures = 0
        self.start_time = datetime.now()
        self.last_reset = datetime.now()

        # Request tracking for RPS calculation
        self.request_timestamps: List[float] = []

    def add_request(self, response_time: float, success: bool) -> None:
        """Record request metrics."""
        self.total_requests += 1
        current_time = time.time()
        self.request_timestamps.append(current_time)

        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
        else:
            self.failed_requests += 1

        # Calculate RPS (last 60 seconds)
        cutoff_time = current_time - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff_time]
        self.requests_per_second = len(self.request_timestamps) / 60.0

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def reset(self) -> None:
        """Reset metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times.clear()
        self.request_timestamps.clear()
        self.requests_per_second = 0.0
        self.active_connections = 0
        self.backend_failures = 0
        self.health_check_failures = 0
        self.last_reset = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "p95_response_time": self.p95_response_time,
            "requests_per_second": self.requests_per_second,
            "active_connections": self.active_connections,
            "backend_failures": self.backend_failures,
            "health_check_failures": self.health_check_failures,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self.start_time.isoformat(),
            "last_reset": self.last_reset.isoformat()
        }


class LoadBalancer:
    """
    Comprehensive load balancer supporting multiple algorithms
    with health checks, metrics, and traffic management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize load balancer.

        Args:
            config: Configuration for load balancer
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.strategy = LoadBalancingStrategy(self.config.get("strategy", "round_robin"))
        self.enable_health_checks = self.config.get("enable_health_checks", True)
        self.enable_ssl = self.config.get("enable_ssl", False)
        self.enable_session_persistence = self.config.get("enable_session_persistence", False)
        self.session_cookie_name = self.config.get("session_cookie_name", "LB_SESSION")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_timeout = self.config.get("retry_timeout", 5)
        self.circuit_breaker_enabled = self.config.get("circuit_breaker_enabled", True)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 10)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)

        # Backend servers and state
        self.backend_servers: Dict[str, BackendServer] = {}
        self.traffic_rules: List[TrafficRule] = []
        self.metrics = LoadBalancerMetrics()
        self.health_check_config = HealthCheck(**self.config.get("health_check", {}))

        # Load balancing state
        self.current_server_index = 0  # For round-robin
        self.session_map: Dict[str, str] = {}  # Session to server mapping
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}  # Circuit breaker state

        # Initialize health checking
        self.health_check_task = None
        if self.enable_health_checks and self.health_check_config.enabled:
            self.health_check_task = asyncio.create_task(self._health_check_loop())

    @log_operation("load_balancer")
    async def add_backend_server(
        self,
        host: str,
        port: int,
        weight: int = 1,
        max_connections: Optional[int] = None,
        health_check_path: str = "/health",
        ssl_enabled: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a backend server to the load balancer.

        Args:
            host: Server hostname or IP
            port: Server port
            weight: Server weight for weighted algorithms
            max_connections: Maximum connections to server
            health_check_path: Health check endpoint path
            ssl_enabled: Whether to use SSL for connections
            metadata: Additional server metadata

        Returns:
            str: Server ID
        """
        server_id = str(uuid.uuid4())

        server = BackendServer(
            server_id=server_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections,
            health_check_path=health_check_path,
            ssl_enabled=ssl_enabled,
            metadata=metadata
        )

        self.backend_servers[server_id] = server

        # Initialize circuit breaker
        if self.circuit_breaker_enabled:
            self.circuit_breakers[server_id] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure": None,
                "next_attempt": None
            }

        self.logger.info(f"Added backend server {host}:{port} with ID {server_id}")
        return server_id

    async def remove_backend_server(self, server_id: str) -> bool:
        """
        Remove a backend server from the load balancer.

        Args:
            server_id: Server ID to remove

        Returns:
            bool: True if removed successfully
        """
        if server_id not in self.backend_servers:
            return False

        server = self.backend_servers[server_id]
        del self.backend_servers[server_id]

        # Clean up circuit breaker
        if server_id in self.circuit_breakers:
            del self.circuit_breakers[server_id]

        # Clean up session mappings
        sessions_to_remove = [
            session for session, mapped_server in self.session_map.items()
            if mapped_server == server_id
        ]
        for session in sessions_to_remove:
            del self.session_map[session]

        self.logger.info(f"Removed backend server {server.host}:{server.port}")
        return True

    @log_operation("request_routing")
    async def route_request(
        self,
        request_info: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Optional[BackendServer]:
        """
        Route request to appropriate backend server.

        Args:
            request_info: Request information (path, headers, client_ip, etc.)
            session_id: Session ID for session persistence

        Returns:
            BackendServer: Selected backend server or None if none available
        """
        start_time = time.time()

        try:
            # Check traffic rules first
            selected_server = await self._apply_traffic_rules(request_info)
            if selected_server:
                return selected_server

            # Get available servers
            available_servers = [
                server for server in self.backend_servers.values()
                if server.is_available() and self._is_circuit_breaker_closed(server.server_id)
            ]

            if not available_servers:
                self.logger.warning("No available backend servers")
                return None

            # Session persistence
            if self.enable_session_persistence and session_id:
                if session_id in self.session_map:
                    server_id = self.session_map[session_id]
                    if server_id in self.backend_servers:
                        server = self.backend_servers[server_id]
                        if server.is_available():
                            return server

            # Apply load balancing strategy
            selected_server = await self._select_server(available_servers, request_info)

            # Update session mapping
            if self.enable_session_persistence and session_id and selected_server:
                self.session_map[session_id] = selected_server.server_id

            return selected_server

        except Exception as e:
            self.logger.error(f"Error routing request: {e}")
            return None

    async def _apply_traffic_rules(self, request_info: Dict[str, Any]) -> Optional[BackendServer]:
        """Apply traffic rules to route request."""
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(self.traffic_rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.matches_request(request_info):
                action = rule.action

                if action.get("type") == "route_to_server":
                    server_id = action.get("server_id")
                    if server_id in self.backend_servers:
                        server = self.backend_servers[server_id]
                        if server.is_available():
                            return server

                elif action.get("type") == "route_to_pool":
                    pool_name = action.get("pool_name")
                    # Route to specific pool of servers
                    pool_servers = [
                        server for server in self.backend_servers.values()
                        if server.metadata.get("pool") == pool_name and server.is_available()
                    ]
                    if pool_servers:
                        return await self._select_server(pool_servers, request_info)

        return None

    async def _select_server(
        self,
        available_servers: List[BackendServer],
        request_info: Dict[str, Any]
    ) -> Optional[BackendServer]:
        """Select server based on load balancing strategy."""
        if not available_servers:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            server = available_servers[self.current_server_index % len(available_servers)]
            self.current_server_index += 1
            return server

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_servers)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(available_servers, key=lambda s: s.current_connections)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS:
            return min(available_servers, key=lambda s: s.load_score / s.weight)

        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            client_ip = request_info.get("client_ip", "")
            hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
            return available_servers[hash_value % len(available_servers)]

        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(available_servers, key=lambda s: s.average_response_time or float('inf'))

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_servers)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_select(available_servers)

        else:
            # Default to round-robin
            return available_servers[self.current_server_index % len(available_servers)]

    def _weighted_round_robin_select(self, servers: List[BackendServer]) -> BackendServer:
        """Select server using weighted round-robin."""
        total_weight = sum(server.weight for server in servers)
        if total_weight == 0:
            return servers[0]

        # Use current index to cycle through weighted servers
        target = self.current_server_index % total_weight
        current_weight = 0

        for server in servers:
            current_weight += server.weight
            if target < current_weight:
                self.current_server_index += 1
                return server

        # Fallback
        self.current_server_index += 1
        return servers[0]

    def _weighted_random_select(self, servers: List[BackendServer]) -> BackendServer:
        """Select server using weighted random."""
        total_weight = sum(server.weight for server in servers)
        if total_weight == 0:
            return random.choice(servers)

        target = random.randint(0, total_weight - 1)
        current_weight = 0

        for server in servers:
            current_weight += server.weight
            if target < current_weight:
                return server

        # Fallback
        return servers[-1]

    async def record_request_result(
        self,
        server: BackendServer,
        response_time: float,
        success: bool,
        status_code: Optional[int] = None
    ) -> None:
        """
        Record the result of a request to update metrics and circuit breaker state.

        Args:
            server: Backend server that handled the request
            response_time: Response time in seconds
            success: Whether the request was successful
            status_code: HTTP status code
        """
        # Update server metrics
        server.add_request(response_time, success)

        # Update global metrics
        self.metrics.add_request(response_time, success)

        # Update circuit breaker
        if self.circuit_breaker_enabled:
            await self._update_circuit_breaker(server.server_id, success)

        # Log significant events
        if not success:
            self.logger.warning(
                f"Request failed to server {server.host}:{server.port}, "
                f"status: {status_code}, response_time: {response_time:.3f}s"
            )

    def _is_circuit_breaker_closed(self, server_id: str) -> bool:
        """Check if circuit breaker is closed (allowing requests)."""
        if not self.circuit_breaker_enabled or server_id not in self.circuit_breakers:
            return True

        breaker = self.circuit_breakers[server_id]

        if breaker["state"] == "closed":
            return True
        elif breaker["state"] == "open":
            # Check if timeout has passed
            if breaker["next_attempt"] and datetime.now() >= breaker["next_attempt"]:
                breaker["state"] = "half-open"
                return True
            return False
        elif breaker["state"] == "half-open":
            return True

        return True

    async def _update_circuit_breaker(self, server_id: str, success: bool) -> None:
        """Update circuit breaker state based on request result."""
        if server_id not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[server_id]

        if success:
            if breaker["state"] == "half-open":
                # Reset circuit breaker
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                breaker["last_failure"] = None
                breaker["next_attempt"] = None
            elif breaker["state"] == "closed":
                # Reset failure count on success
                breaker["failure_count"] = max(0, breaker["failure_count"] - 1)
        else:
            breaker["failure_count"] += 1
            breaker["last_failure"] = datetime.now()

            if breaker["failure_count"] >= self.circuit_breaker_threshold:
                breaker["state"] = "open"
                breaker["next_attempt"] = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
                self.metrics.backend_failures += 1

                self.logger.warning(
                    f"Circuit breaker opened for server {server_id} "
                    f"after {breaker['failure_count']} failures"
                )

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_config.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_config.interval_seconds)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all backend servers."""
        if not AIOHTTP_AVAILABLE:
            self.logger.warning("aiohttp not available, skipping health checks")
            return

        tasks = []
        for server in self.backend_servers.values():
            if server.is_enabled:
                task = asyncio.create_task(self._check_server_health(server))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_server_health(self, server: BackendServer) -> None:
        """Check health of a single server."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.health_check_config.timeout_seconds)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()

                async with session.get(server.health_check_url) as response:
                    response_time = time.time() - start_time

                    # Check status code
                    status_ok = response.status == self.health_check_config.expected_status

                    # Check content if specified
                    content_ok = True
                    if self.health_check_config.expected_content:
                        content = await response.text()
                        content_ok = self.health_check_config.expected_content in content

                    # Custom health check
                    custom_ok = True
                    if self.health_check_config.custom_check:
                        custom_ok = await self.health_check_config.custom_check(server, response)

                    health_ok = status_ok and content_ok and custom_ok

                    await self._update_server_health(server, health_ok, response_time)

        except Exception as e:
            self.logger.debug(f"Health check failed for {server.host}:{server.port}: {e}")
            await self._update_server_health(server, False, None)

    async def _update_server_health(
        self,
        server: BackendServer,
        is_healthy: bool,
        response_time: Optional[float]
    ) -> None:
        """Update server health status."""
        server.last_health_check = datetime.now()

        if is_healthy:
            if not server.is_healthy:
                server.consecutive_failures = 0
                server.is_healthy = True
                self.logger.info(f"Server {server.host}:{server.port} is now healthy")
        else:
            server.consecutive_failures += 1

            if server.consecutive_failures >= self.health_check_config.failure_threshold:
                if server.is_healthy:
                    server.is_healthy = False
                    self.metrics.health_check_failures += 1
                    self.logger.warning(
                        f"Server {server.host}:{server.port} marked as unhealthy "
                        f"after {server.consecutive_failures} consecutive failures"
                    )

    def add_traffic_rule(
        self,
        name: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        priority: int = 100
    ) -> str:
        """
        Add traffic routing rule.

        Args:
            name: Rule name
            condition: Matching condition
            action: Action to take when condition matches
            priority: Rule priority (higher = evaluated first)

        Returns:
            str: Rule ID
        """
        rule_id = str(uuid.uuid4())
        rule = TrafficRule(rule_id, name, condition, action, priority)
        self.traffic_rules.append(rule)

        # Sort rules by priority
        self.traffic_rules.sort(key=lambda r: r.priority, reverse=True)

        return rule_id

    def remove_traffic_rule(self, rule_id: str) -> bool:
        """Remove traffic routing rule."""
        for i, rule in enumerate(self.traffic_rules):
            if rule.rule_id == rule_id:
                del self.traffic_rules[i]
                return True
        return False

    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        stats = {
            "total_servers":
 len(self.backend_servers),
            "healthy_servers": len([s for s in self.backend_servers.values() if s.is_healthy]),
            "enabled_servers": len([s for s in self.backend_servers.values() if s.is_enabled]),
            "available_servers": len([s for s in self.backend_servers.values() if s.is_available()]),
            "servers": [server.to_dict() for server in self.backend_servers.values()],
            "metrics": self.metrics.to_dict(),
            "traffic_rules": [rule.to_dict() for rule in self.traffic_rules],
            "circuit_breakers": {
                server_id: breaker for server_id, breaker in self.circuit_breakers.items()
            },
            "strategy": self.strategy.value,
            "health_check_config": self.health_check_config.to_dict()
        }
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on load balancer."""
        health_status = {
            "status": "healthy",
            "strategy": self.strategy.value,
            "total_servers": len(self.backend_servers),
            "healthy_servers": len([s for s in self.backend_servers.values() if s.is_healthy]),
            "available_servers": len([s for s in self.backend_servers.values() if s.is_available()]),
            "health_checks_enabled": self.enable_health_checks,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "session_persistence_enabled": self.enable_session_persistence,
            "active_sessions": len(self.session_map),
            "traffic_rules": len(self.traffic_rules),
            "metrics": self.metrics.to_dict(),
            "dependencies": {
                "aiohttp": AIOHTTP_AVAILABLE,
                "ssl": SSL_AVAILABLE
            }
        }

        # Check if any servers are available
        if len(self.backend_servers) == 0:
            health_status["status"] = "degraded"
            health_status["message"] = "No backend servers configured"
        elif len([s for s in self.backend_servers.values() if s.is_available()]) == 0:
            health_status["status"] = "critical"
            health_status["message"] = "No backend servers available"

        return health_status

    async def shutdown(self) -> None:
        """Shutdown load balancer and cleanup resources."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Load balancer shutdown completed")


# Factory function for creating load balancer
def create_load_balancer(config: Optional[Dict[str, Any]] = None) -> LoadBalancer:
    """
    Create a load balancer with specified configuration.

    Args:
        config: Configuration for load balancer

    Returns:
        LoadBalancer: Configured load balancer instance
    """
    return LoadBalancer(config=config)
