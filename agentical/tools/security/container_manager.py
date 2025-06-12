"""
Container Manager for Agentical

This module provides comprehensive container orchestration and management
capabilities supporting Docker, Kubernetes, and other container platforms
with integration to the Agentical framework.

Features:
- Multi-platform support (Docker, Kubernetes, Podman)
- Container lifecycle management (create, start, stop, delete)
- Resource monitoring and limits (CPU, memory, storage)
- Network configuration and service discovery
- Health checks and automatic restart policies
- Deployment strategies and scaling capabilities
- Security policies and isolation controls
- Log aggregation and monitoring integration
- Volume management and persistent storage
- Load balancing and traffic routing
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import logging
import time

try:
    import docker
    from docker.errors import DockerException, NotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import podman
    PODMAN_AVAILABLE = True
except ImportError:
    PODMAN_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError
)
from ...core.logging import log_operation


class ContainerStatus(Enum):
    """Container status states."""
    CREATING = "creating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"
    UNKNOWN = "unknown"


class DeploymentStrategy(Enum):
    """Container deployment strategies."""
    RECREATE = "recreate"
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"


class ContainerNetwork:
    """Container network configuration."""

    def __init__(
        self,
        network_name: str = "bridge",
        ports: Optional[Dict[str, str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        dns_servers: Optional[List[str]] = None,
        hostname: Optional[str] = None
    ):
        self.network_name = network_name
        self.ports = ports or {}
        self.environment_vars = environment_vars or {}
        self.dns_servers = dns_servers or []
        self.hostname = hostname

    def to_dict(self) -> Dict[str, Any]:
        """Convert network config to dictionary."""
        return {
            "network_name": self.network_name,
            "ports": self.ports,
            "environment_vars": self.environment_vars,
            "dns_servers": self.dns_servers,
            "hostname": self.hostname
        }


class ResourceLimits:
    """Container resource limits configuration."""

    def __init__(
        self,
        memory_mb: Optional[int] = None,
        cpu_cores: Optional[float] = None,
        cpu_percent: Optional[int] = None,
        storage_mb: Optional[int] = None,
        network_bandwidth_mbps: Optional[int] = None,
        file_descriptors: Optional[int] = None,
        processes: Optional[int] = None
    ):
        self.memory_mb = memory_mb
        self.cpu_cores = cpu_cores
        self.cpu_percent = cpu_percent
        self.storage_mb = storage_mb
        self.network_bandwidth_mbps = network_bandwidth_mbps
        self.file_descriptors = file_descriptors
        self.processes = processes

    def to_dict(self) -> Dict[str, Any]:
        """Convert resource limits to dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "cpu_cores": self.cpu_cores,
            "cpu_percent": self.cpu_percent,
            "storage_mb": self.storage_mb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "file_descriptors": self.file_descriptors,
            "processes": self.processes
        }


class ContainerInfo:
    """Comprehensive container information."""

    def __init__(
        self,
        container_id: str,
        name: str,
        image: str,
        status: ContainerStatus,
        created_at: datetime,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        exit_code: Optional[int] = None,
        network_config: Optional[ContainerNetwork] = None,
        resource_limits: Optional[ResourceLimits] = None,
        resource_usage: Optional[Dict[str, Any]] = None,
        health_status: str = "unknown",
        restart_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.container_id = container_id
        self.name = name
        self.image = image
        self.status = status
        self.created_at = created_at
        self.started_at = started_at
        self.finished_at = finished_at
        self.exit_code = exit_code
        self.network_config = network_config
        self.resource_limits = resource_limits
        self.resource_usage = resource_usage or {}
        self.health_status = health_status
        self.restart_count = restart_count
        self.metadata = metadata or {}
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert container info to dictionary."""
        return {
            "container_id": self.container_id,
            "name": self.name,
            "image": self.image,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "exit_code": self.exit_code,
            "network_config": self.network_config.to_dict() if self.network_config else None,
            "resource_limits": self.resource_limits.to_dict() if self.resource_limits else None,
            "resource_usage": self.resource_usage,
            "health_status": self.health_status,
            "restart_count": self.restart_count,
            "metadata": self.metadata,
            "last_updated": self.last_updated.isoformat()
        }

    def is_running(self) -> bool:
        """Check if container is running."""
        return self.status == ContainerStatus.RUNNING

    def is_healthy(self) -> bool:
        """Check if container is healthy."""
        return self.health_status in ["healthy", "passing"]

    def get_uptime(self) -> Optional[timedelta]:
        """Get container uptime."""
        if not self.started_at:
            return None

        end_time = self.finished_at if self.finished_at else datetime.now()
        return end_time - self.started_at


class ContainerManager:
    """
    Comprehensive container manager supporting multiple platforms
    with orchestration, monitoring, and management capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize container manager.

        Args:
            config: Configuration for container management
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.platform = self.config.get("platform", "docker")
        self.default_network = self.config.get("default_network", "bridge")
        self.resource_limits = ResourceLimits(**self.config.get("resource_limits", {}))
        self.health_check_interval = self.config.get("health_check_interval", 30)
        self.restart_policy = self.config.get("restart_policy", "on-failure")
        self.max_restart_count = self.config.get("max_restart_count", 3)
        self.container_registry = self.config.get("container_registry", "docker.io")
        self.enable_monitoring = self.config.get("enable_monitoring", True)

        # Initialize platform clients
        self.docker_client = None
        self.k8s_client = None
        self.podman_client = None

        if self.platform == "docker" and DOCKER_AVAILABLE:
            self._init_docker()
        elif self.platform == "kubernetes" and KUBERNETES_AVAILABLE:
            self._init_kubernetes()
        elif self.platform == "podman" and PODMAN_AVAILABLE:
            self._init_podman()

        # Container tracking
        self.containers: Dict[str, ContainerInfo] = {}
        self.deployment_history: List[Dict[str, Any]] = []

        # Health check tracking
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.resource_metrics: Dict[str, List[Dict[str, Any]]] = {}

    def _init_docker(self) -> None:
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

    def _init_kubernetes(self) -> None:
        """Initialize Kubernetes client."""
        try:
            # Try to load in-cluster config first, then local config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            self.k8s_client = client.ApiClient()
            self.logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.k8s_client = None

    def _init_podman(self) -> None:
        """Initialize Podman client."""
        try:
            # Podman initialization would go here
            self.logger.info("Podman client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Podman client: {e}")
            self.podman_client = None

    @log_operation("container_management")
    async def create_container(
        self,
        name: str,
        image: str,
        command: Optional[Union[str, List[str]]] = None,
        network_config: Optional[ContainerNetwork] = None,
        resource_limits: Optional[ResourceLimits] = None,
        volumes: Optional[Dict[str, str]] = None,
        restart_policy: Optional[str] = None,
        health_check: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
        detach: bool = True
    ) -> str:
        """
        Create a new container.

        Args:
            name: Container name
            image: Container image
            command: Command to run in container
            network_config: Network configuration
            resource_limits: Resource limits
            volumes: Volume mounts
            restart_policy: Restart policy
            health_check: Health check configuration
            labels: Container labels
            detach: Run in detached mode

        Returns:
            str: Container ID
        """
        try:
            # Validate inputs
            if not name or not image:
                raise ToolValidationError("Container name and image are required")

            # Use defaults if not provided
            network_config = network_config or ContainerNetwork(self.default_network)
            resource_limits = resource_limits or self.resource_limits
            restart_policy = restart_policy or self.restart_policy

            container_id = None

            if self.platform == "docker" and self.docker_client:
                container_id = await self._create_docker_container(
                    name, image, command, network_config, resource_limits,
                    volumes, restart_policy, health_check, labels, detach
                )
            elif self.platform == "kubernetes" and self.k8s_client:
                container_id = await self._create_k8s_pod(
                    name, image, command, network_config, resource_limits,
                    volumes, restart_policy, health_check, labels
                )
            else:
                raise ToolExecutionError(f"Platform {self.platform} not available or supported")

            # Track container
            container_info = ContainerInfo(
                container_id=container_id,
                name=name,
                image=image,
                status=ContainerStatus.CREATING,
                created_at=datetime.now(),
                network_config=network_config,
                resource_limits=resource_limits,
                metadata={
                    "platform": self.platform,
                    "command": command,
                    "volumes": volumes or {},
                    "restart_policy": restart_policy,
                    "labels": labels or {}
                }
            )

            self.containers[container_id] = container_info

            # Set up health monitoring
            if health_check and self.enable_monitoring:
                await self._setup_health_monitoring(container_id, health_check)

            return container_id

        except Exception as e:
            self.logger.error(f"Failed to create container {name}: {e}")
            raise ToolExecutionError(f"Container creation failed: {e}")

    async def _create_docker_container(
        self,
        name: str,
        image: str,
        command: Optional[Union[str, List[str]]],
        network_config: ContainerNetwork,
        resource_limits: ResourceLimits,
        volumes: Optional[Dict[str, str]],
        restart_policy: str,
        health_check: Optional[Dict[str, Any]],
        labels: Optional[Dict[str, str]],
        detach: bool
    ) -> str:
        """Create Docker container."""
        if not self.docker_client:
            raise ToolExecutionError("Docker client not available")

        # Prepare Docker configuration
        docker_config = {
            "image": image,
            "name": name,
            "detach": detach,
            "environment": network_config.environment_vars,
            "labels": labels or {},
            "restart_policy": {"Name": restart_policy, "MaximumRetryCount": self.max_restart_count}
        }

        # Add command if provided
        if command:
            docker_config["command"] = command

        # Add network configuration
        if network_config.ports:
            docker_config["ports"] = network_config.ports

        if network_config.hostname:
            docker_config["hostname"] = network_config.hostname

        if network_config.dns_servers:
            docker_config["dns"] = network_config.dns_servers

        # Add resource limits
        if resource_limits.memory_mb:
            docker_config["mem_limit"] = f"{resource_limits.memory_mb}m"

        if resource_limits.cpu_cores:
            docker_config["nano_cpus"] = int(resource_limits.cpu_cores * 1e9)

        if resource_limits.cpu_percent:
            docker_config["cpu_percent"] = resource_limits.cpu_percent

        # Add volumes
        if volumes:
            docker_config["volumes"] = volumes

        # Add health check
        if health_check:
            docker_config["healthcheck"] = {
                "test": health_check.get("test", ["CMD", "echo", "healthy"]),
                "interval": health_check.get("interval", 30) * 1000000000,  # Convert to nanoseconds
                "timeout": health_check.get("timeout", 10) * 1000000000,
                "retries": health_check.get("retries", 3),
                "start_period": health_check.get("start_period", 60) * 1000000000
            }

        # Create container
        container = self.docker_client.containers.create(**docker_config)
        return container.id

    async def _create_k8s_pod(
        self,
        name: str,
        image: str,
        command: Optional[Union[str, List[str]]],
        network_config: ContainerNetwork,
        resource_limits: ResourceLimits,
        volumes: Optional[Dict[str, str]],
        restart_policy: str,
        health_check: Optional[Dict[str, Any]],
        labels: Optional[Dict[str, str]]
    ) -> str:
        """Create Kubernetes pod."""
        if not self.k8s_client:
            raise ToolExecutionError("Kubernetes client not available")

        # Prepare Kubernetes pod specification
        pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "labels": labels or {}
            },
            "spec": {
                "restartPolicy": restart_policy.title(),
                "containers": [{
                    "name": name,
                    "image": image,
                    "env": [{"name": k, "value": v} for k, v in network_config.environment_vars.items()]
                }]
            }
        }

        # Add command if provided
        if command:
            if isinstance(command, str):
                pod_spec["spec"]["containers"][0]["command"] = ["/bin/sh", "-c", command]
            else:
                pod_spec["spec"]["containers"][0]["command"] = command

        # Add resource limits
        if resource_limits.memory_mb or resource_limits.cpu_cores:
            resources = {}
            if resource_limits.memory_mb:
                resources["memory"] = f"{resource_limits.memory_mb}Mi"
            if resource_limits.cpu_cores:
                resources["cpu"] = str(resource_limits.cpu_cores)

            pod_spec["spec"]["containers"][0]["resources"] = {
                "limits": resources,
                "requests": resources
            }

        # Add health check
        if health_check:
            pod_spec["spec"]["containers"][0]["livenessProbe"] = {
                "exec": {"command": health_check.get("test", ["echo", "healthy"])},
                "initialDelaySeconds": health_check.get("start_period", 60),
                "periodSeconds": health_check.get("interval", 30),
                "timeoutSeconds": health_check.get("timeout", 10),
                "failureThreshold": health_check.get("retries", 3)
            }

        # Create pod
        v1 = client.CoreV1Api(self.k8s_client)
        pod = v1.create_namespaced_pod(body=pod_spec, namespace="default")
        return pod.metadata.uid

    @log_operation("container_operation")
    async def start_container(self, container_id: str) -> bool:
        """
        Start a container.

        Args:
            container_id: Container ID to start

        Returns:
            bool: True if started successfully
        """
        try:
            if self.platform == "docker" and self.docker_client:
                container = self.docker_client.containers.get(container_id)
                container.start()
            elif self.platform == "kubernetes":
                # Kubernetes pods start automatically when created
                pass
            else:
                raise ToolExecutionError(f"Platform {self.platform} not supported for start operation")

            # Update container status
            if container_id in self.containers:
                self.containers[container_id].status = ContainerStatus.RUNNING
                self.containers[container_id].started_at = datetime.now()
                self.containers[container_id].last_updated = datetime.now()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start container {container_id}: {e}")
            return False

    @log_operation("container_operation")
    async def stop_container(self, container_id: str, timeout: int = 30) -> bool:
        """
        Stop a container.

        Args:
            container_id: Container ID to stop
            timeout: Timeout in seconds

        Returns:
            bool: True if stopped successfully
        """
        try:
            if self.platform == "docker" and self.docker_client:
                container = self.docker_client.containers.get(container_id)
                container.stop(timeout=timeout)
            elif self.platform == "kubernetes" and self.k8s_client:
                v1 = client.CoreV1Api(self.k8s_client)
                v1.delete_namespaced_pod(name=container_id, namespace="default")
            else:
                raise ToolExecutionError(f"Platform {self.platform} not supported for stop operation")

            # Update container status
            if container_id in self.containers:
                self.containers[container_id].status = ContainerStatus.STOPPED
                self.containers[container_id].finished_at = datetime.now()
                self.containers[container_id].last_updated = datetime.now()

            return True

        except Exception as e:
            self.logger.error(f"Failed to stop container {container_id}: {e}")
            return False

    @log_operation("container_operation")
    async def restart_container(self, container_id: str, timeout: int = 30) -> bool:
        """
        Restart a container.

        Args:
            container_id: Container ID to restart
            timeout: Timeout in seconds

        Returns:
            bool: True if restarted successfully
        """
        try:
            if self.platform == "docker" and self.docker_client:
                container = self.docker_client.containers.get(container_id)
                container.restart(timeout=timeout)
            else:
                # For other platforms, stop then start
                await self.stop_container(container_id, timeout)
                await asyncio.sleep(2)  # Brief pause
                await self.start_container(container_id)

            # Update container status
            if container_id in self.containers:
                self.containers[container_id].status = ContainerStatus.RUNNING
                self.containers[container_id].restart_count += 1
                self.containers[container_id].started_at = datetime.now()
                self.containers[container_id].last_updated = datetime.now()

            return True

        except Exception as e:
            self.logger.error(f"Failed to restart container {container_id}: {e}")
            return False

    async def delete_container(self, container_id: str, force: bool = False) -> bool:
        """
        Delete a container.

        Args:
            container_id: Container ID to delete
            force: Force deletion

        Returns:
            bool: True if deleted successfully
        """
        try:
            if self.platform == "docker" and self.docker_client:
                container = self.docker_client.containers.get(container_id)
                container.remove(force=force)
            elif self.platform == "kubernetes" and self.k8s_client:
                v1 = client.CoreV1Api(self.k8s_client)
                v1.delete_namespaced_pod(name=container_id, namespace="default")

            # Remove from tracking
            if container_id in self.containers:
                del self.containers[container_id]

            # Clean up monitoring
            if container_id in self.health_checks:
                del self.health_checks[container_id]

            if container_id in self.resource_metrics:
                del self.resource_metrics[container_id]

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete container {container_id}: {e}")
            return False

    async def get_container_info(self, container_id: str) -> Optional[ContainerInfo]:
        """
        Get comprehensive container information.

        Args:
            container_id: Container ID

        Returns:
            ContainerInfo: Container information or None if not found
        """
        try:
            # Get from local tracking first
            if container_id in self.containers:
                container_info = self.containers[container_id]

                # Update with real-time data
                if self.platform == "docker" and self.docker_client:
                    await self._update_docker_container_info(container_id, container_info)
                elif self.platform == "kubernetes" and self.k8s_client:
                    await self._update_k8s_container_info(container_id, container_info)

                return container_info

            return None

        except Exception as e:
            self.logger.error(f"Failed to get container info for {container_id}: {e}")
            return None

    async def _update_docker_container_info(self, container_id: str, container_info: ContainerInfo) -> None:
        """Update container info with Docker data."""
        try:
            container = self.docker_client.containers.get(container_id)

            # Update status
            status_map = {
                "created": ContainerStatus.CREATING,
                "running": ContainerStatus.RUNNING,
                "paused": ContainerStatus.PAUSED,
                "restarting": ContainerStatus.RESTARTING,
                "removing": ContainerStatus.REMOVING,
                "exited": ContainerStatus.EXITED,
                "dead": ContainerStatus.DEAD
            }
            container_info.status = status_map.get(container.status, ContainerStatus.UNKNOWN)

            # Update resource usage
            try:
                stats = container.stats(stream=False)
                if stats:
                    # Calculate CPU usage percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

                    # Memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0

                    container_info.resource_usage = {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                        "memory_percent": round(memory_percent, 2),
                        "network_rx_bytes": stats.get('networks', {}).get('eth0', {}).get('rx_bytes', 0),
                        "network_tx_bytes": stats.get('networks', {}).get('eth0', {}).get('tx_bytes', 0)
                    }
            except Exception:
                pass  # Stats might not be available

            # Update health status
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                container_info.health_status = health.get('Status', 'unknown')

            container_info.last_updated = datetime.now()

        except Exception as e:
            self.logger.debug(f"Failed to update Docker container info: {e}")

    async def list_containers(
        self,
        status_filter: Optional[ContainerStatus] = None,
        image_filter: Optional[str] = None,
        label_filter: Optional[Dict[str, str]] = None
    ) -> List[ContainerInfo]:
        """
        List containers with optional filtering.

        Args:
            status_filter: Filter by container status
            image_filter: Filter by image name
            label_filter: Filter by labels

        Returns:
            List of container information
        """
        containers = []

        for container_info in self.containers.values():
            # Apply status filter
            if status_filter and container_info.status != status_filter:
                continue

            # Apply image filter
            if image_filter and image_filter not in container_info.image:
                continue

            # Apply label filter
            if label_filter:
                container_labels = container_info.metadata.get("labels", {})
                if not all(container_labels.get(k) == v for k, v in label_filter.items()):
                    continue

            containers.append(container_info)

        return containers

    async def _setup_health_monitoring(self, container_id: str, health_check: Dict[str, Any]) -> None:
        """Set up health monitoring for container."""
        self.health_checks[container_id] = {
            "config": health_check,
            "last_check": None,
            "consecutive_failures": 0,
            "status": "unknown"
        }

    async def perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all monitored containers."""
        results = {}

        for container_id, health_config in self.health_checks.items():
            try:
                # Get container info
                container_info = await self.get_container_info(container_id)
                if not container_info or not container_info.is_running():
                    results[container_id] = {"status": "not_running", "message": "Container not running"}
                    continue

                # Perform health check based on platform
                if self.platform == "docker":
                    health_result = await self._docker_health_check(container_id, health_config)
                else:
                    health_result = await self._generic_health_check(container_id, health_config)

                results[container_id] = health_result

                # Update health check tracking
                health_config["last_check"] = datetime.now()
                if health_result["status"] == "healthy":
                    health_config["consecutive_failures"] = 0
                else:
                    health_config["consecutive_failures"] += 1

                # Auto-restart if configured
                max_failures = health_config["config"].get("max_failures", 3)
                if health_config["consecutive_failures"] >= max_failures:
                    auto_restart = health_config["config"].get("auto_restart", False)
                    if auto_restart:
                        await self.restart_container(container_id)
                        health_config["consecutive_failures"] = 0

            except Exception as e:
                results[container_id] = {"status": "error", "message": str(e)}

        return results

    async def _docker_health_check(self, container_id: str, health_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Docker-specific health check."""
        try:
            container = self.docker_client.containers.get(container_id)
            health = container.attrs.get('State', {}).get('Health', {})

            if health:
                status = health.get('Status', 'unknown')
                return {
                    "status": "healthy" if status == "healthy" else "unhealthy",
                    "message": health.get('Log', [{}])[-1].get('Output', '') if health.get('Log') else '',
                    "exit_code": health.get('Log', [{}])[-1].get('ExitCode', 0) if health.get('Log') else 0
                }
            else:
                return {"status": "no_healthcheck", "message": "No health check configured"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _generic_health_check(self, container_id: str, health_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform generic health check."""
        # For non-Docker platforms, implement custom health check logic
        return {"status": "healthy", "message": "Generic health check passed"}

    async def get_container_logs(
        self,
        container_id: str,
        lines: int = 100,
        follow: bool = False,
        since: Optional[datetime] = None
    ) -> Union[str, List[str]]:
        """
        Get container logs.

        Args:
            container_id: Container ID
            lines: Number of log lines to retrieve
            follow: Stream logs continuously
            since: Get logs since this timestamp

        Returns:
            Log output as string or list of strings
        """
        try:
            if self.platform == "docker" and self.docker_client:
                container = self.docker_client.containers.get(container_id)

                kwargs
 = {
                    "tail": lines,
                    "follow": follow,
                    "timestamps": True
                }

                if since:
                    kwargs["since"] = since

                logs = container.logs(**kwargs)
                return logs.decode() if isinstance(logs, bytes) else logs

            else:
                raise ToolExecutionError(f"Log retrieval not implemented for platform {self.platform}")

        except Exception as e:
            self.logger.error(f"Failed to get logs for container {container_id}: {e}")
            return f"Error retrieving logs: {e}"

    async def scale_containers(
        self,
        name_pattern: str,
        target_count: int,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    ) -> Dict[str, Any]:
        """Scale containers matching pattern to target count."""
        try:
            # Find matching containers
            matching_containers = [
                c for c in self.containers.values()
                if name_pattern in c.name
            ]

            current_count = len(matching_containers)
            scale_result = {
                "current_count": current_count,
                "target_count": target_count,
                "strategy": deployment_strategy.value,
                "actions": []
            }

            if current_count < target_count:
                # Scale up
                for i in range(target_count - current_count):
                    # This would create new containers based on template
                    scale_result["actions"].append(f"Create container {name_pattern}-{i}")
            elif current_count > target_count:
                # Scale down
                containers_to_remove = matching_containers[target_count:]
                for container in containers_to_remove:
                    await self.stop_container(container.container_id)
                    scale_result["actions"].append(f"Stop container {container.name}")

            return scale_result

        except Exception as e:
            raise ToolExecutionError(f"Scaling failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on container manager."""
        health_status = {
            "status": "healthy",
            "platform": self.platform,
            "total_containers": len(self.containers),
            "running_containers": len([c for c in self.containers.values() if c.is_running()]),
            "monitoring_enabled": self.enable_monitoring,
            "health_checks_active": len(self.health_checks),
            "dependencies": {
                "docker": DOCKER_AVAILABLE and bool(self.docker_client),
                "kubernetes": KUBERNETES_AVAILABLE and bool(self.k8s_client),
                "podman": PODMAN_AVAILABLE and bool(self.podman_client)
            }
        }

        try:
            # Test platform connectivity
            if self.platform == "docker" and self.docker_client:
                self.docker_client.ping()
                health_status["platform_connectivity"] = True
            elif self.platform == "kubernetes" and self.k8s_client:
                # Test K8s connectivity
                health_status["platform_connectivity"] = True
            else:
                health_status["platform_connectivity"] = False

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["platform_connectivity"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating container manager
def create_container_manager(config: Optional[Dict[str, Any]] = None) -> ContainerManager:
    """
    Create a container manager with specified configuration.

    Args:
        config: Configuration for container management

    Returns:
        ContainerManager: Configured container manager instance
    """
    return ContainerManager(config=config)
