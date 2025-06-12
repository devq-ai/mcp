"""
Workflow Registry for Agentical

This module provides the WorkflowRegistry class that manages workflow
discovery, registration, and lifecycle management within the Agentical framework.

Features:
- Workflow type registration and discovery
- Dynamic workflow loading and validation
- Workflow template management
- Performance monitoring and metrics
- Integration with database and caching
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Type, Union, Callable
from enum import Enum
from pathlib import Path
import importlib
import inspect

import logfire
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import (
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowValidationError,
    ConfigurationError
)
from ..core.logging import log_operation
from ..db.models.workflow import (
    Workflow,
    WorkflowType,
    WorkflowStatus
)
from ..db.repositories.workflow import AsyncWorkflowRepository


class WorkflowDiscoveryMode(Enum):
    """Workflow discovery modes."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


class WorkflowRegistryEntry:
    """Registry entry for a workflow type."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        handler_class: Type,
        handler_factory: Callable,
        metadata: Dict[str, Any],
        config: Dict[str, Any]
    ):
        self.workflow_type = workflow_type
        self.handler_class = handler_class
        self.handler_factory = handler_factory
        self.metadata = metadata
        self.config = config
        self.registered_at = datetime.utcnow()
        self.usage_count = 0
        self.last_used = None
        self.is_enabled = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry entry to dictionary."""
        return {
            "workflow_type": self.workflow_type.value,
            "handler_class": self.handler_class.__name__,
            "metadata": self.metadata,
            "config": self.config,
            "registered_at": self.registered_at.isoformat(),
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_enabled": self.is_enabled
        }


class WorkflowRegistry:
    """
    Central registry for workflow types and handlers.

    Manages workflow discovery, registration, and provides access to
    workflow handlers and templates for the workflow engine.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        discovery_mode: WorkflowDiscoveryMode = WorkflowDiscoveryMode.HYBRID,
        auto_discover_paths: Optional[List[str]] = None,
        enable_caching: bool = True
    ):
        """Initialize the workflow registry."""
        self.db_session = db_session
        self.discovery_mode = discovery_mode
        self.auto_discover_paths = auto_discover_paths or []
        self.enable_caching = enable_caching

        # Registry storage
        self._registry: Dict[WorkflowType, WorkflowRegistryEntry] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, WorkflowType] = {}

        # Repository for database operations
        self.workflow_repo = AsyncWorkflowRepository(db_session)

        # Performance tracking
        self._metrics = {
            "registrations": 0,
            "lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "discovery_runs": 0,
            "errors": 0
        }

        # Cache for frequently accessed workflows
        self._workflow_cache: Dict[int, Workflow] = {}
        self._cache_expiry: Dict[int, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)

        logfire.info(
            "Workflow registry initialized",
            discovery_mode=discovery_mode.value,
            auto_discover_paths=len(self.auto_discover_paths),
            caching_enabled=enable_caching
        )

    async def initialize(self) -> None:
        """Initialize the registry with auto-discovery."""
        with logfire.span("Initialize workflow registry"):
            # Register built-in workflow types
            await self._register_builtin_workflows()

            # Auto-discover workflows if enabled
            if self.discovery_mode in [WorkflowDiscoveryMode.AUTOMATIC, WorkflowDiscoveryMode.HYBRID]:
                await self._auto_discover_workflows()

            # Load workflow templates
            await self._load_workflow_templates()

            logfire.info(
                "Workflow registry initialized",
                registered_types=len(self._registry),
                templates_loaded=len(self._templates)
            )

    async def register_workflow_type(
        self,
        workflow_type: WorkflowType,
        handler_class: Type,
        handler_factory: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Register a workflow type with its handler.

        Args:
            workflow_type: The workflow type to register
            handler_class: The handler class for this workflow type
            handler_factory: Optional factory function for creating handlers
            metadata: Optional metadata about the workflow type
            config: Optional configuration for the workflow type
            force: Whether to overwrite existing registration

        Returns:
            bool: True if registration was successful

        Raises:
            WorkflowValidationError: If registration validation fails
        """
        with logfire.span(
            "Register workflow type",
            workflow_type=workflow_type.value,
            handler_class=handler_class.__name__
        ):
            # Check if already registered
            if workflow_type in self._registry and not force:
                raise WorkflowValidationError(
                    f"Workflow type {workflow_type.value} is already registered"
                )

            # Validate handler class
            if not self._validate_handler_class(handler_class):
                raise WorkflowValidationError(
                    f"Handler class {handler_class.__name__} is not valid"
                )

            # Create default factory if not provided
            if not handler_factory:
                handler_factory = lambda *args, **kwargs: handler_class(*args, **kwargs)

            # Create registry entry
            entry = WorkflowRegistryEntry(
                workflow_type=workflow_type,
                handler_class=handler_class,
                handler_factory=handler_factory,
                metadata=metadata or {},
                config=config or {}
            )

            # Register the entry
            self._registry[workflow_type] = entry
            self._metrics["registrations"] += 1

            logfire.info(
                "Workflow type registered",
                workflow_type=workflow_type.value,
                handler_class=handler_class.__name__,
                metadata_keys=list((metadata or {}).keys())
            )

            return True

    async def unregister_workflow_type(self, workflow_type: WorkflowType) -> bool:
        """Unregister a workflow type."""
        if workflow_type in self._registry:
            del self._registry[workflow_type]
            logfire.info("Workflow type unregistered", workflow_type=workflow_type.value)
            return True
        return False

    async def get_workflow_handler(
        self,
        workflow_type: WorkflowType,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get a workflow handler instance for the given type.

        Args:
            workflow_type: The workflow type
            config: Optional configuration for the handler

        Returns:
            Handler instance or None if not found
        """
        with logfire.span("Get workflow handler", workflow_type=workflow_type.value):
            self._metrics["lookups"] += 1

            entry = self._registry.get(workflow_type)
            if not entry:
                self._metrics["cache_misses"] += 1
                logfire.warning("Workflow type not found", workflow_type=workflow_type.value)
                return None

            if not entry.is_enabled:
                logfire.warning("Workflow type is disabled", workflow_type=workflow_type.value)
                return None

            self._metrics["cache_hits"] += 1

            try:
                # Create handler instance
                handler_config = {**entry.config, **(config or {})}
                handler = entry.handler_factory(config=handler_config)

                # Update usage tracking
                entry.usage_count += 1
                entry.last_used = datetime.utcnow()

                logfire.debug(
                    "Workflow handler created",
                    workflow_type=workflow_type.value,
                    handler_class=entry.handler_class.__name__
                )

                return handler

            except Exception as e:
                self._metrics["errors"] += 1
                logfire.error(
                    "Failed to create workflow handler",
                    workflow_type=workflow_type.value,
                    error=str(e)
                )
                raise WorkflowError(f"Failed to create handler for {workflow_type.value}: {str(e)}")

    async def get_registered_types(self) -> List[WorkflowType]:
        """Get all registered workflow types."""
        return [wt for wt, entry in self._registry.items() if entry.is_enabled]

    async def get_workflow_info(self, workflow_type: WorkflowType) -> Optional[Dict[str, Any]]:
        """Get information about a registered workflow type."""
        entry = self._registry.get(workflow_type)
        return entry.to_dict() if entry else None

    async def enable_workflow_type(self, workflow_type: WorkflowType) -> bool:
        """Enable a workflow type."""
        entry = self._registry.get(workflow_type)
        if entry:
            entry.is_enabled = True
            logfire.info("Workflow type enabled", workflow_type=workflow_type.value)
            return True
        return False

    async def disable_workflow_type(self, workflow_type: WorkflowType) -> bool:
        """Disable a workflow type."""
        entry = self._registry.get(workflow_type)
        if entry:
            entry.is_enabled = False
            logfire.info("Workflow type disabled", workflow_type=workflow_type.value)
            return True
        return False

    async def register_workflow_template(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        force: bool = False
    ) -> bool:
        """Register a workflow template."""
        if template_name in self._templates and not force:
            raise WorkflowValidationError(f"Template {template_name} already exists")

        # Validate template data
        if not self._validate_template_data(template_data):
            raise WorkflowValidationError(f"Invalid template data for {template_name}")

        self._templates[template_name] = template_data
        logfire.info("Workflow template registered", template_name=template_name)
        return True

    async def get_workflow_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow template by name."""
        return self._templates.get(template_name)

    async def get_available_templates(self) -> List[str]:
        """Get list of available workflow templates."""
        return list(self._templates.keys())

    async def create_workflow_from_template(
        self,
        template_name: str,
        workflow_name: str,
        parameters: Dict[str, Any]
    ) -> Optional[Workflow]:
        """Create a workflow instance from a template."""
        template = await self.get_workflow_template(template_name)
        if not template:
            raise WorkflowNotFoundError(f"Template {template_name} not found")

        with logfire.span("Create workflow from template", template_name=template_name, workflow_name=workflow_name):
            # TODO: Implement template instantiation logic
            # This would create a workflow instance with steps based on the template

            # For now, create a basic workflow
            workflow = Workflow(
                name=workflow_name,
                workflow_type=WorkflowType(template.get("type", "sequential")),
                description=template.get("description", f"Workflow created from template {template_name}"),
                status=WorkflowStatus.DRAFT,
                configuration=parameters
            )

            # Save to database
            self.db_session.add(workflow)
            await self.db_session.commit()
            await self.db_session.refresh(workflow)

            logfire.info(
                "Workflow created from template",
                template_name=template_name,
                workflow_name=workflow_name,
                workflow_id=workflow.id
            )

            return workflow

    async def search_workflows(
        self,
        query: str,
        workflow_type: Optional[WorkflowType] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50
    ) -> List[Workflow]:
        """Search for workflows in the database."""
        return await self.workflow_repo.search_workflows(
            query=query,
            workflow_type=workflow_type,
            status=status,
            limit=limit
        )

    async def get_workflow_cached(self, workflow_id: int) -> Optional[Workflow]:
        """Get a workflow with caching."""
        if not self.enable_caching:
            return await self.workflow_repo.get(workflow_id)

        # Check cache
        if workflow_id in self._workflow_cache:
            expiry = self._cache_expiry.get(workflow_id)
            if expiry and datetime.utcnow() < expiry:
                self._metrics["cache_hits"] += 1
                return self._workflow_cache[workflow_id]

        # Cache miss - fetch from database
        self._metrics["cache_misses"] += 1
        workflow = await self.workflow_repo.get(workflow_id)

        if workflow and self.enable_caching:
            self._workflow_cache[workflow_id] = workflow
            self._cache_expiry[workflow_id] = datetime.utcnow() + self._cache_ttl

        return workflow

    async def invalidate_workflow_cache(self, workflow_id: int) -> None:
        """Invalidate cached workflow data."""
        self._workflow_cache.pop(workflow_id, None)
        self._cache_expiry.pop(workflow_id, None)

    async def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry performance metrics."""
        return {
            **self._metrics,
            "registered_types": len(self._registry),
            "enabled_types": len([e for e in self._registry.values() if e.is_enabled]),
            "templates_count": len(self._templates),
            "cache_size": len(self._workflow_cache),
            "cache_hit_rate": (
                self._metrics["cache_hits"] / max(self._metrics["lookups"], 1)
            ) * 100
        }

    async def _register_builtin_workflows(self) -> None:
        """Register built-in workflow types."""
        # These will be implemented in the standard and graph workflow modules
        builtin_types = [
            WorkflowType.SEQUENTIAL,
            WorkflowType.PARALLEL,
            WorkflowType.CONDITIONAL,
            WorkflowType.LOOP,
            WorkflowType.PIPELINE
        ]

        for workflow_type in builtin_types:
            # For now, register with placeholder handlers
            # These will be replaced when we implement the actual handlers
            class PlaceholderHandler:
                def __init__(self, config=None):
                    self.config = config or {}

            await self.register_workflow_type(
                workflow_type=workflow_type,
                handler_class=PlaceholderHandler,
                metadata={"builtin": True, "status": "placeholder"},
                config={"placeholder": True}
            )

    async def _auto_discover_workflows(self) -> None:
        """Auto-discover workflow implementations."""
        self._metrics["discovery_runs"] += 1

        for discovery_path in self.auto_discover_paths:
            try:
                await self._discover_workflows_in_path(discovery_path)
            except Exception as e:
                logfire.error(
                    "Workflow discovery failed",
                    path=discovery_path,
                    error=str(e)
                )

    async def _discover_workflows_in_path(self, path: str) -> None:
        """Discover workflows in a specific path."""
        # TODO: Implement actual workflow discovery
        # This would scan Python modules for workflow implementations
        logfire.debug("Discovering workflows", path=path)

    async def _load_workflow_templates(self) -> None:
        """Load workflow templates from configuration."""
        # Load default templates
        default_templates = {
            "simple_sequential": {
                "type": "sequential",
                "description": "Simple sequential workflow template",
                "steps": [
                    {"type": "agent_task", "name": "step1"},
                    {"type": "tool_execution", "name": "step2"},
                    {"type": "data_transform", "name": "step3"}
                ]
            },
            "parallel_processing": {
                "type": "parallel",
                "description": "Parallel processing workflow template",
                "steps": [
                    {"type": "parallel", "name": "parallel_step", "tasks": [
                        {"type": "agent_task", "name": "task1"},
                        {"type": "agent_task", "name": "task2"},
                        {"type": "tool_execution", "name": "task3"}
                    ]}
                ]
            },
            "conditional_routing": {
                "type": "conditional",
                "description": "Conditional routing workflow template",
                "steps": [
                    {"type": "condition", "name": "decision_point"},
                    {"type": "agent_task", "name": "true_path"},
                    {"type": "tool_execution", "name": "false_path"}
                ]
            }
        }

        for template_name, template_data in default_templates.items():
            await self.register_workflow_template(template_name, template_data)

    def _validate_handler_class(self, handler_class: Type) -> bool:
        """Validate that a handler class meets requirements."""
        # Check if class has required methods
        required_methods = ["__init__"]  # Add more as needed

        for method in required_methods:
            if not hasattr(handler_class, method):
                return False

        return True

    def _validate_template_data(self, template_data: Dict[str, Any]) -> bool:
        """Validate workflow template data."""
        required_fields = ["type", "description"]

        for field in required_fields:
            if field not in template_data:
                return False

        # Validate workflow type
        try:
            WorkflowType(template_data["type"])
        except ValueError:
            return False

        return True

    async def cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        if not self.enable_caching:
            return

        now = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._cache_expiry.items()
            if expiry < now
        ]

        for key in expired_keys:
            self._workflow_cache.pop(key, None)
            self._cache_expiry.pop(key, None)

        if expired_keys:
            logfire.debug("Cache cleanup completed", expired_entries=len(expired_keys))

    async def shutdown(self) -> None:
        """Shutdown the registry and cleanup resources."""
        logfire.info("Workflow registry shutting down")

        # Clear caches
        self._workflow_cache.clear()
        self._cache_expiry.clear()

        # Clear registry
        self._registry.clear()
        self._templates.clear()

        logfire.info("Workflow registry shutdown complete")

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"WorkflowRegistry(types={len(self._registry)}, "
            f"templates={len(self._templates)}, "
            f"discovery_mode={self.discovery_mode.value})"
        )
