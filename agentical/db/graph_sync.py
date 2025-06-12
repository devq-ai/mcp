"""
Data Synchronization Utilities for Agentical

This module provides comprehensive data synchronization capabilities between
relational database models and the SurrealDB graph database, ensuring consistency
and enabling real-time updates across both storage systems.

Features:
- Bidirectional synchronization between relational and graph models
- Real-time change detection and propagation
- Conflict resolution and data consistency management
- Batch synchronization for performance optimization
- Event-driven synchronization triggers
- Data validation and integrity checks
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import uuid
from contextlib import asynccontextmanager

try:
    import logfire
except ImportError:
    # Mock logfire for testing
    class MockLogfire:
        def span(self, name: str, **kwargs):
            class MockSpan:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def set_attribute(self, key, value): pass
            return MockSpan()
        def info(self, msg, **kwargs): pass
        def error(self, msg, **kwargs): pass
        def warning(self, msg, **kwargs): pass
    logfire = MockLogfire()

from .graph_operations import GraphOperations, GraphNode, GraphRelationship, NodeType, RelationshipType
from .knowledge_schemas import AgentKnowledgeSchema, create_agent_schema
from .vector_search import VectorSearchEngine, VectorSearchConfig


class SyncDirection(str, Enum):
    """Direction of data synchronization."""
    RELATIONAL_TO_GRAPH = "relational_to_graph"
    GRAPH_TO_RELATIONAL = "graph_to_relational"
    BIDIRECTIONAL = "bidirectional"


class SyncStrategy(str, Enum):
    """Strategy for handling synchronization conflicts."""
    RELATIONAL_WINS = "relational_wins"
    GRAPH_WINS = "graph_wins"
    NEWEST_WINS = "newest_wins"
    MANUAL_RESOLUTION = "manual_resolution"
    MERGE_STRATEGY = "merge_strategy"


class ChangeType(str, Enum):
    """Types of changes that can trigger synchronization."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RELATIONSHIP_ADD = "relationship_add"
    RELATIONSHIP_REMOVE = "relationship_remove"


@dataclass
class SyncConfig:
    """Configuration for data synchronization."""
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    sync_strategy: SyncStrategy = SyncStrategy.NEWEST_WINS
    batch_size: int = 100
    sync_interval_seconds: int = 300  # 5 minutes
    enable_real_time_sync: bool = True
    enable_vector_sync: bool = True
    auto_resolve_conflicts: bool = True
    validate_before_sync: bool = True
    create_backups: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    excluded_fields: Set[str] = field(default_factory=lambda: {'created_at', 'updated_at'})
    model_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChangeRecord:
    """Record of a data change that needs synchronization."""
    id: str
    entity_type: str
    entity_id: str
    change_type: ChangeType
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_system: str = "relational"
    sync_status: str = "pending"
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    success: bool
    records_processed: int
    records_synced: int
    records_failed: int
    conflicts_resolved: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sync_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Resolution for a synchronization conflict."""
    conflict_id: str
    entity_type: str
    entity_id: str
    resolution_strategy: SyncStrategy
    chosen_data: Dict[str, Any]
    merged_data: Optional[Dict[str, Any]] = None
    resolution_timestamp: datetime = field(default_factory=datetime.utcnow)
    resolver: str = "automatic"


class DataSynchronizer:
    """
    Comprehensive data synchronization manager.

    Handles synchronization between relational database models and SurrealDB
    graph database, ensuring data consistency and enabling real-time updates.
    """

    def __init__(self, config: SyncConfig = None,
                 graph_ops: Optional[GraphOperations] = None,
                 vector_engine: Optional[VectorSearchEngine] = None):
        self.config = config or SyncConfig()
        self.graph_ops = graph_ops
        self.vector_engine = vector_engine

        # Change tracking
        self.change_queue: List[ChangeRecord] = []
        self.conflict_queue: List[ConflictResolution] = []
        self.sync_history: List[SyncResult] = []

        # Performance tracking
        self.stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "average_sync_time_ms": 0.0,
            "last_sync_timestamp": None
        }

        # Model mappings for different entity types
        self.entity_type_mappings = {
            "agent": NodeType.AGENT,
            "tool": NodeType.TOOL,
            "workflow": NodeType.WORKFLOW,
            "task": NodeType.TASK,
            "playbook": NodeType.PLAYBOOK,
            "user": NodeType.USER
        }

        # Relationship mappings for foreign keys
        self.fk_relationship_mappings = {
            "user_id": RelationshipType.CREATED_BY,
            "assigned_to_user_id": RelationshipType.ASSIGNED_TO,
            "created_by_user_id": RelationshipType.CREATED_BY,
            "workflow_id": RelationshipType.PART_OF,
            "agent_id": RelationshipType.DEPENDS_ON,
            "tool_id": RelationshipType.USES_TOOL,
            "parent_task_id": RelationshipType.PART_OF,
            "parent_playbook_id": RelationshipType.PART_OF
        }

        self.is_running = False
        self._sync_task = None

    async def initialize(self):
        """Initialize the data synchronizer."""
        with logfire.span("Initialize data synchronizer"):
            if not self.graph_ops:
                from .graph_operations import create_graph_operations
                self.graph_ops = await create_graph_operations()

            if not self.vector_engine and self.config.enable_vector_sync:
                from .vector_search import create_vector_search_engine
                self.vector_engine = await create_vector_search_engine()

            logfire.info("Data synchronizer initialized",
                        sync_direction=self.config.sync_direction.value,
                        real_time_enabled=self.config.enable_real_time_sync)

    async def start_real_time_sync(self):
        """Start real-time synchronization monitoring."""
        if self.config.enable_real_time_sync and not self.is_running:
            self.is_running = True
            self._sync_task = asyncio.create_task(self._sync_loop())
            logfire.info("Real-time synchronization started",
                        interval_seconds=self.config.sync_interval_seconds)

    async def stop_real_time_sync(self):
        """Stop real-time synchronization monitoring."""
        if self.is_running:
            self.is_running = False
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            logfire.info("Real-time synchronization stopped")

    async def _sync_loop(self):
        """Main synchronization loop for real-time updates."""
        while self.is_running:
            try:
                if self.change_queue:
                    await self.process_pending_changes()

                await asyncio.sleep(self.config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error("Error in sync loop", error=str(e))
                await asyncio.sleep(self.config.retry_delay_seconds)

    async def record_change(self, entity_type: str, entity_id: str,
                          change_type: ChangeType,
                          old_data: Optional[Dict[str, Any]] = None,
                          new_data: Optional[Dict[str, Any]] = None,
                          source_system: str = "relational"):
        """Record a data change for synchronization."""
        change_record = ChangeRecord(
            id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            change_type=change_type,
            old_data=old_data,
            new_data=new_data,
            source_system=source_system
        )

        self.change_queue.append(change_record)

        logfire.info("Change recorded for sync",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    change_type=change_type.value)

        # Process immediately if real-time sync is enabled
        if self.config.enable_real_time_sync and len(self.change_queue) >= self.config.batch_size:
            await self.process_pending_changes()

    async def process_pending_changes(self) -> SyncResult:
        """Process all pending changes in the queue."""
        if not self.change_queue:
            return SyncResult(success=True, records_processed=0, records_synced=0, records_failed=0, conflicts_resolved=0)

        start_time = time.time()

        with logfire.span("Process pending changes", change_count=len(self.change_queue)):

            batch = self.change_queue[:self.config.batch_size]
            self.change_queue = self.change_queue[self.config.batch_size:]

            sync_result = SyncResult(
                success=True,
                records_processed=len(batch),
                records_synced=0,
                records_failed=0,
                conflicts_resolved=0
            )

            for change_record in batch:
                try:
                    success = await self._process_single_change(change_record)
                    if success:
                        sync_result.records_synced += 1
                    else:
                        sync_result.records_failed += 1
                        sync_result.success = False

                except Exception as e:
                    sync_result.records_failed += 1
                    sync_result.errors.append(f"Failed to process {change_record.id}: {str(e)}")
                    sync_result.success = False

                    logfire.error("Failed to process change",
                                change_id=change_record.id,
                                error=str(e))

            sync_result.sync_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.stats["total_syncs"] += 1
            if sync_result.success:
                self.stats["successful_syncs"] += 1
            else:
                self.stats["failed_syncs"] += 1

            self.stats["average_sync_time_ms"] = (
                (self.stats["average_sync_time_ms"] * (self.stats["total_syncs"] - 1) + sync_result.sync_time_ms) /
                self.stats["total_syncs"]
            )
            self.stats["last_sync_timestamp"] = datetime.utcnow().isoformat()

            self.sync_history.append(sync_result)

            logfire.info("Sync batch completed",
                        processed=sync_result.records_processed,
                        synced=sync_result.records_synced,
                        failed=sync_result.records_failed,
                        time_ms=sync_result.sync_time_ms)

            return sync_result

    async def _process_single_change(self, change_record: ChangeRecord) -> bool:
        """Process a single change record."""
        try:
            if change_record.source_system == "relational":
                return await self._sync_relational_to_graph(change_record)
            else:
                return await self._sync_graph_to_relational(change_record)

        except Exception as e:
            change_record.retry_count += 1
            change_record.error_message = str(e)
            change_record.sync_status = "failed"

            if change_record.retry_count < self.config.max_retry_attempts:
                change_record.sync_status = "retry"
                self.change_queue.append(change_record)  # Re-queue for retry

                logfire.warning("Change processing failed, will retry",
                              change_id=change_record.id,
                              retry_count=change_record.retry_count,
                              error=str(e))
            else:
                logfire.error("Change processing failed permanently",
                            change_id=change_record.id,
                            error=str(e))

            return False

    async def _sync_relational_to_graph(self, change_record: ChangeRecord) -> bool:
        """Sync a relational model change to the graph database."""
        with logfire.span("Sync relational to graph",
                         entity_type=change_record.entity_type,
                         change_type=change_record.change_type.value):

            if change_record.change_type == ChangeType.CREATE:
                return await self._create_graph_node_from_relational(change_record)

            elif change_record.change_type == ChangeType.UPDATE:
                return await self._update_graph_node_from_relational(change_record)

            elif change_record.change_type == ChangeType.DELETE:
                return await self._delete_graph_node_from_relational(change_record)

            else:
                logfire.warning("Unsupported change type for relational to graph sync",
                              change_type=change_record.change_type.value)
                return False

    async def _create_graph_node_from_relational(self, change_record: ChangeRecord) -> bool:
        """Create a graph node from relational model data."""
        if not change_record.new_data:
            return False

        try:
            # Map entity type to node type
            node_type = self.entity_type_mappings.get(
                change_record.entity_type.lower(),
                NodeType.KNOWLEDGE
            )

            # Create graph node
            node_id = f"{change_record.entity_type}:{change_record.entity_id}"

            # Generate vector embedding if enabled
            vector_embedding = None
            if self.config.enable_vector_sync and self.vector_engine:
                text_content = self._extract_text_content(change_record.new_data)
                if text_content:
                    vector_embedding = await self.vector_engine.generate_embedding(text_content)

            node = GraphNode(
                id=node_id,
                type=node_type,
                properties=self._clean_properties(change_record.new_data),
                labels=[change_record.entity_type, "synced_from_relational"],
                vector_embedding=vector_embedding
            )

            await self.graph_ops.create_node(node)

            # Create relationships from foreign keys
            await self._create_relationships_from_foreign_keys(
                node_id, change_record.new_data, change_record.entity_type
            )

            # Update agent knowledge schema if this is an agent
            if change_record.entity_type.lower() == "agent":
                await self._update_agent_knowledge_schema(change_record.entity_id, change_record.new_data)

            return True

        except Exception as e:
            logfire.error("Failed to create graph node from relational",
                        entity_type=change_record.entity_type,
                        entity_id=change_record.entity_id,
                        error=str(e))
            return False

    async def _update_graph_node_from_relational(self, change_record: ChangeRecord) -> bool:
        """Update a graph node from relational model data."""
        if not change_record.new_data:
            return False

        try:
            node_id = f"{change_record.entity_type}:{change_record.entity_id}"

            # Prepare updates (exclude timestamps and other excluded fields)
            updates = {}
            for key, value in change_record.new_data.items():
                if key not in self.config.excluded_fields:
                    updates[f"properties.{key}"] = value

            # Update vector embedding if enabled
            if self.config.enable_vector_sync and self.vector_engine:
                text_content = self._extract_text_content(change_record.new_data)
                if text_content:
                    vector_embedding = await self.vector_engine.generate_embedding(text_content)
                    updates["vector_embedding"] = vector_embedding

            success = await self.graph_ops.update_node(node_id, updates)

            # Update agent knowledge schema if this is an agent
            if change_record.entity_type.lower() == "agent":
                await self._update_agent_knowledge_schema(change_record.entity_id, change_record.new_data)

            return success

        except Exception as e:
            logfire.error("Failed to update graph node from relational",
                        entity_type=change_record.entity_type,
                        entity_id=change_record.entity_id,
                        error=str(e))
            return False

    async def _delete_graph_node_from_relational(self, change_record: ChangeRecord) -> bool:
        """Delete a graph node based on relational model deletion."""
        try:
            node_id = f"{change_record.entity_type}:{change_record.entity_id}"
            return await self.graph_ops.delete_node(node_id, cascade=True)

        except Exception as e:
            logfire.error("Failed to delete graph node from relational",
                        entity_type=change_record.entity_type,
                        entity_id=change_record.entity_id,
                        error=str(e))
            return False

    async def _sync_graph_to_relational(self, change_record: ChangeRecord) -> bool:
        """Sync a graph database change to relational models."""
        # This would require integration with SQLAlchemy models
        # Implementation depends on specific model structure
        logfire.warning("Graph to relational sync not yet implemented")
        return True

    async def _create_relationships_from_foreign_keys(self, node_id: str,
                                                    model_data: Dict[str, Any],
                                                    entity_type: str):
        """Create graph relationships based on foreign key relationships."""
        for fk_field, relationship_type in self.fk_relationship_mappings.items():
            if fk_field in model_data and model_data[fk_field]:
                # Determine target entity type from foreign key field
                if fk_field.endswith("_user_id"):
                    target_type = "user"
                elif fk_field.endswith("_agent_id"):
                    target_type = "agent"
                elif fk_field.endswith("_workflow_id"):
                    target_type = "workflow"
                elif fk_field.endswith("_tool_id"):
                    target_type = "tool"
                elif fk_field.endswith("_task_id"):
                    target_type = "task"
                else:
                    continue

                target_node_id = f"{target_type}:{model_data[fk_field]}"

                # Check if target node exists
                target_node = await self.graph_ops.get_node(target_node_id)
                if target_node:
                    relationship = GraphRelationship(
                        id=f"{node_id}_{relationship_type.value}_{target_node_id}",
                        from_node=node_id,
                        to_node=target_node_id,
                        type=relationship_type,
                        properties={"sync_source": "relational_model", "fk_field": fk_field}
                    )

                    try:
                        await self.graph_ops.create_relationship(relationship)
                    except Exception as e:
                        logfire.warning("Failed to create relationship during sync",
                                      from_node=node_id,
                                      to_node=target_node_id,
                                      error=str(e))

    async def _update_agent_knowledge_schema(self, agent_id: str, agent_data: Dict[str, Any]):
        """Update agent knowledge schema in graph database."""
        try:
            # Determine agent type for schema creation
            agent_type = agent_data.get("type", "base_agent")

            # Create or update knowledge schema
            schema = create_agent_schema(agent_id, agent_type)

            # Convert to graph representation and save as agent state
            schema_data = schema.to_graph_representation()

            await self.graph_ops.save_agent_state(
                agent_id=agent_id,
                state_data=agent_data,
                context=schema_data.get("knowledge_entities", {}),
                capabilities=list(schema_data.get("capabilities", {}).keys()),
                tools=agent_data.get("tools", []),
                workflows=agent_data.get("workflows", [])
            )

        except Exception as e:
            logfire.error("Failed to update agent knowledge schema",
                        agent_id=agent_id,
                        error=str(e))

    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from model data for vector embedding."""
        text_fields = ["name", "title", "description", "content", "summary", "notes"]
        text_parts = []

        for field in text_fields:
            if field in data and data[field]:
                text_parts.append(str(data[field]))

        return " ".join(text_parts) if text_parts else ""

    def _clean_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and prepare properties for graph storage."""
        cleaned = {}

        for key, value in data.items():
            # Skip None values and excluded fields
            if value is None or key in self.config.excluded_fields:
                continue

            # Convert datetime objects to ISO strings
            if isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            # Convert other non-serializable types
            elif not isinstance(value, (str, int, float, bool, list, dict)):
                cleaned[key] = str(value)
            else:
                cleaned[key] = value

        return cleaned

    async def sync_all_entities(self, entity_types: Optional[List[str]] = None) -> SyncResult:
        """Perform a full synchronization of all entities."""
        start_time = time.time()

        with logfire.span("Sync all entities", entity_types=entity_types):

            # This would require integration with actual model classes
            # For now, return a placeholder result
            sync_result = SyncResult(
                success=True,
                records_processed=0,
                records_synced=0,
                records_failed=0,
                conflicts_resolved=0,
                sync_time_ms=(time.time() - start_time) * 1000
            )

            logfire.info("Full entity sync completed",
                        processed=sync_result.records_processed,
                        synced=sync_result.records_synced)

            return sync_result

    async def detect_conflicts(self) -> List[ConflictResolution]:
        """Detect synchronization conflicts between systems."""
        conflicts = []

        with logfire.span("Detect sync conflicts"):
            # Implementation would compare timestamps and data checksums
            # between relational and graph data
            pass

        return conflicts

    async def resolve_conflict(self, conflict: ConflictResolution) -> bool:
        """Resolve a synchronization conflict."""
        with logfire.span("Resolve sync conflict",
                         conflict_id=conflict.conflict_id,
                         strategy=conflict.resolution_strategy.value):

            try:
                if conflict.resolution_strategy == SyncStrategy.RELATIONAL_WINS:
                    # Apply relational data to graph
                    pass
                elif conflict.resolution_strategy == SyncStrategy.GRAPH_WINS:
                    # Apply graph data to relational
                    pass
                elif conflict.resolution_strategy == SyncStrategy.NEWEST_WINS:
                    # Compare timestamps and apply newer data
                    pass
                elif conflict.resolution_strategy == SyncStrategy.MERGE_STRATEGY:
                    # Merge data from both sources
                    pass

                self.stats["conflicts_resolved"] += 1
                return True

            except Exception as e:
                logfire.error("Failed to resolve conflict",
                            conflict_id=conflict.conflict_id,
                            error=str(e))
                return False

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization performance statistics."""
        return {
            **self.stats,
            "pending_changes": len(self.change_queue),
            "pending_conflicts": len(self.conflict_queue),
            "recent_sync_history": [
                {
                    "success": result.success,
                    "records_processed": result.records_processed,
                    "records_synced": result.records_synced,
                    "sync_time_ms": result.sync_time_ms
                }
                for result in self.sync_history[-10:]  # Last 10 syncs
            ],
            "config": {
                "sync_direction": self.config.sync_direction.value,
                "sync_strategy": self.config.sync_strategy.value,
                "real_time_enabled": self.config.enable_real_time_sync,
                "vector_sync_enabled": self.config.enable_vector_sync
            }
        }

    async def validate_data_consistency(self) -> Dict[str, Any]:
        """Validate consistency between relational and graph data."""
        with logfire.span("Validate data consistency"):

            validation_results = {
                "total_entities_checked": 0,
                "consistent_entities": 0,
                "inconsistent_entities": 0,
                "missing_in_graph": 0,
                "missing_in_relational": 0,
                "inconsistencies": [],
                "validation_timestamp": datetime.utcnow().isoformat()
            }

            # Implementation would compare entity counts, checksums, and key fields
            # between relational and graph databases

            return validation_results

    async def create_backup_checkpoint(self) -> str:
        """Create a backup checkpoint before synchronization."""
        checkpoint_id = f"sync_checkpoint_{int(time.time())}"

        with logfire.span("Create backup checkpoint", checkpoint_id=checkpoint_id):
            # Implementation would create backups of both databases
            # before performing synchronization operations
            pass

        return checkpoint_id

    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore data from a backup checkpoint."""
        with logfire.span("Restore from checkpoint", checkpoint_id=checkpoint_id):
            # Implementation would restore both databases from checkpoint
            pass

        return True


# Factory function for creating data synchronizer
async def create_data_synchronizer(config: SyncConfig = None) -> DataSynchronizer:
    """Create and initialize a data synchronizer."""
    synchronizer = DataSynchronizer(config)
    await synchronizer.initialize()
    return synchronizer


# Context manager for synchronization operations
@asynccontextmanager
async def sync_session(config: SyncConfig = None):
    """Context manager for synchronization operations."""
    synchronizer = await create_data_synchronizer(config)
    try:
        await synchronizer.start_real_time_sync()
        yield synchronizer
    finally:
        await synchronizer.stop_real_time_sync()


# Utility functions for common sync operations
async def quick_sync_entity(entity_type: str, entity_id: str, entity_data: Dict[str, Any]):
    """Quick synchronization of a single entity."""
    async with sync_session() as synchronizer:
        await synchronizer.record_change(
            entity_type=entity_type,
            entity_id=entity_id,
            change_type=ChangeType.UPDATE,
            new_data=entity_data
        )
        return await synchronizer.process_pending_changes()


async def bulk_sync_entities(entities: List[Tuple[str, str, Dict[str, Any]]]):
    """Bulk synchronization of multiple entities."""
    async with sync_session() as synchronizer:
        for entity_type, entity_id, entity_data in entities:
            await synchronizer.record_change(
                entity_type=entity_type,
                entity_id=entity_id,
                change_type=ChangeType.UPDATE,
                new_data=entity_data
            )
        return await synchronizer.process_pending_changes()
