"""
SurrealDB Graph Operations for Agentical

This module provides comprehensive graph database operations for SurrealDB,
including knowledge graph management, agent state persistence, vector search,
and graph traversal capabilities.

Features:
- Knowledge graph operations and relationship management
- Agent state persistence and retrieval
- Graph queries and traversal operations
- Vector search capabilities for semantic search
- Data synchronization and consistency management
- Graph analytics and metrics
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid
import numpy as np
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

from .surrealdb_client import SurrealDBManager, get_surrealdb_manager


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    ASSIGNED_TO = "assigned_to"
    USES_TOOL = "uses_tool"
    EXECUTES_WORKFLOW = "executes_workflow"
    HAS_CAPABILITY = "has_capability"
    KNOWS_ABOUT = "knows_about"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    TRIGGERS = "triggers"
    CONTAINS = "contains"
    REFERENCES = "references"


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    TASK = "task"
    PLAYBOOK = "playbook"
    USER = "user"
    KNOWLEDGE = "knowledge"
    CAPABILITY = "capability"
    EXECUTION = "execution"
    RESULT = "result"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: NodeType
    properties: Dict[str, Any]
    labels: List[str] = None
    vector_embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class GraphRelationship:
    """Represents a relationship/edge in the knowledge graph."""
    id: str
    from_node: str
    to_node: str
    type: RelationshipType
    properties: Dict[str, Any] = None
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class GraphTraversalPath:
    """Represents a path through the graph during traversal."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    total_weight: float = 0.0
    path_length: int = 0

    def __post_init__(self):
        self.path_length = len(self.nodes)
        self.total_weight = sum(rel.weight for rel in self.relationships)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    node: GraphNode
    similarity_score: float
    distance: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GraphOperations:
    """
    Comprehensive graph operations for SurrealDB.

    Provides high-level graph database operations including:
    - Node and relationship management
    - Graph traversal and path finding
    - Vector search and similarity operations
    - Agent state persistence
    - Knowledge graph analytics
    """

    def __init__(self, surrealdb_manager: Optional[SurrealDBManager] = None):
        self.surrealdb_manager = surrealdb_manager
        self._vector_index_cache = {}

    async def initialize(self):
        """Initialize the graph operations system."""
        if not self.surrealdb_manager:
            self.surrealdb_manager = await get_surrealdb_manager()

        await self._create_graph_schema()
        await self._setup_indexes()

        logfire.info("Graph operations initialized")

    async def _create_graph_schema(self):
        """Create the graph database schema."""
        with logfire.span("Create graph schema"):
            # Define node table
            await self.surrealdb_manager.execute_query("""
                DEFINE TABLE graph_node SCHEMAFULL;
                DEFINE FIELD id ON TABLE graph_node TYPE string;
                DEFINE FIELD type ON TABLE graph_node TYPE string;
                DEFINE FIELD properties ON TABLE graph_node TYPE object;
                DEFINE FIELD labels ON TABLE graph_node TYPE array;
                DEFINE FIELD vector_embedding ON TABLE graph_node TYPE array;
                DEFINE FIELD created_at ON TABLE graph_node TYPE datetime;
                DEFINE FIELD updated_at ON TABLE graph_node TYPE datetime;
            """)

            # Define relationship table
            await self.surrealdb_manager.execute_query("""
                DEFINE TABLE graph_relationship SCHEMAFULL;
                DEFINE FIELD id ON TABLE graph_relationship TYPE string;
                DEFINE FIELD from_node ON TABLE graph_relationship TYPE string;
                DEFINE FIELD to_node ON TABLE graph_relationship TYPE string;
                DEFINE FIELD type ON TABLE graph_relationship TYPE string;
                DEFINE FIELD properties ON TABLE graph_relationship TYPE object;
                DEFINE FIELD weight ON TABLE graph_relationship TYPE number;
                DEFINE FIELD confidence ON TABLE graph_relationship TYPE number;
                DEFINE FIELD created_at ON TABLE graph_relationship TYPE datetime;
            """)

            # Define agent state table
            await self.surrealdb_manager.execute_query("""
                DEFINE TABLE agent_state SCHEMAFULL;
                DEFINE FIELD agent_id ON TABLE agent_state TYPE string;
                DEFINE FIELD state_data ON TABLE agent_state TYPE object;
                DEFINE FIELD context ON TABLE agent_state TYPE object;
                DEFINE FIELD capabilities ON TABLE agent_state TYPE array;
                DEFINE FIELD tools ON TABLE agent_state TYPE array;
                DEFINE FIELD workflows ON TABLE agent_state TYPE array;
                DEFINE FIELD knowledge_base ON TABLE agent_state TYPE object;
                DEFINE FIELD performance_metrics ON TABLE agent_state TYPE object;
                DEFINE FIELD last_execution ON TABLE agent_state TYPE datetime;
                DEFINE FIELD created_at ON TABLE agent_state TYPE datetime;
                DEFINE FIELD updated_at ON TABLE agent_state TYPE datetime;
            """)

    async def _setup_indexes(self):
        """Setup performance indexes for graph operations."""
        with logfire.span("Setup graph indexes"):
            # Node indexes
            await self.surrealdb_manager.execute_query("""
                DEFINE INDEX node_id_idx ON TABLE graph_node COLUMNS id UNIQUE;
                DEFINE INDEX node_type_idx ON TABLE graph_node COLUMNS type;
                DEFINE INDEX node_labels_idx ON TABLE graph_node COLUMNS labels;
            """)

            # Relationship indexes
            await self.surrealdb_manager.execute_query("""
                DEFINE INDEX rel_from_idx ON TABLE graph_relationship COLUMNS from_node;
                DEFINE INDEX rel_to_idx ON TABLE graph_relationship COLUMNS to_node;
                DEFINE INDEX rel_type_idx ON TABLE graph_relationship COLUMNS type;
                DEFINE INDEX rel_composite_idx ON TABLE graph_relationship COLUMNS from_node, to_node, type;
            """)

            # Agent state indexes
            await self.surrealdb_manager.execute_query("""
                DEFINE INDEX agent_state_id_idx ON TABLE agent_state COLUMNS agent_id UNIQUE;
                DEFINE INDEX agent_state_updated_idx ON TABLE agent_state COLUMNS updated_at;
            """)

    # Node Operations

    async def create_node(self, node: GraphNode) -> str:
        """Create a new node in the graph."""
        with logfire.span("Create graph node", node_type=node.type, node_id=node.id):
            query = """
                CREATE graph_node CONTENT {
                    id: $id,
                    type: $type,
                    properties: $properties,
                    labels: $labels,
                    vector_embedding: $vector_embedding,
                    created_at: $created_at,
                    updated_at: $updated_at
                }
            """

            result = await self.surrealdb_manager.execute_query(query, {
                'id': node.id,
                'type': node.type.value,
                'properties': node.properties,
                'labels': node.labels,
                'vector_embedding': node.vector_embedding,
                'created_at': node.created_at.isoformat(),
                'updated_at': node.updated_at.isoformat()
            })

            logfire.info("Graph node created", node_id=node.id, node_type=node.type)
            return result[0]['id'] if result else node.id

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        with logfire.span("Get graph node", node_id=node_id):
            query = "SELECT * FROM graph_node WHERE id = $node_id"
            result = await self.surrealdb_manager.execute_query(query, {'node_id': node_id})

            if result:
                data = result[0]
                return GraphNode(
                    id=data['id'],
                    type=NodeType(data['type']),
                    properties=data['properties'],
                    labels=data['labels'],
                    vector_embedding=data.get('vector_embedding'),
                    created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                    updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
                )
            return None

    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update a node's properties."""
        with logfire.span("Update graph node", node_id=node_id):
            updates['updated_at'] = datetime.utcnow().isoformat()

            # Build SET clause dynamically
            set_clauses = []
            params = {'node_id': node_id}

            for key, value in updates.items():
                param_key = f"update_{key}"
                set_clauses.append(f"{key} = ${param_key}")
                params[param_key] = value

            query = f"UPDATE graph_node SET {', '.join(set_clauses)} WHERE id = $node_id"

            result = await self.surrealdb_manager.execute_query(query, params)
            success = bool(result)

            logfire.info("Graph node updated", node_id=node_id, success=success)
            return success

    async def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """Delete a node and optionally its relationships."""
        with logfire.span("Delete graph node", node_id=node_id, cascade=cascade):
            if cascade:
                # Delete all relationships involving this node
                await self.surrealdb_manager.execute_query(
                    "DELETE FROM graph_relationship WHERE from_node = $node_id OR to_node = $node_id",
                    {'node_id': node_id}
                )

            # Delete the node
            result = await self.surrealdb_manager.execute_query(
                "DELETE FROM graph_node WHERE id = $node_id",
                {'node_id': node_id}
            )

            success = bool(result)
            logfire.info("Graph node deleted", node_id=node_id, cascade=cascade, success=success)
            return success

    # Relationship Operations

    async def create_relationship(self, relationship: GraphRelationship) -> str:
        """Create a new relationship between nodes."""
        with logfire.span("Create graph relationship",
                         from_node=relationship.from_node,
                         to_node=relationship.to_node,
                         rel_type=relationship.type):

            # Verify nodes exist
            from_exists = await self.get_node(relationship.from_node)
            to_exists = await self.get_node(relationship.to_node)

            if not from_exists or not to_exists:
                raise ValueError(f"Source or target node does not exist: {relationship.from_node} -> {relationship.to_node}")

            query = """
                CREATE graph_relationship CONTENT {
                    id: $id,
                    from_node: $from_node,
                    to_node: $to_node,
                    type: $type,
                    properties: $properties,
                    weight: $weight,
                    confidence: $confidence,
                    created_at: $created_at
                }
            """

            result = await self.surrealdb_manager.execute_query(query, {
                'id': relationship.id,
                'from_node': relationship.from_node,
                'to_node': relationship.to_node,
                'type': relationship.type.value,
                'properties': relationship.properties,
                'weight': relationship.weight,
                'confidence': relationship.confidence,
                'created_at': relationship.created_at.isoformat()
            })

            logfire.info("Graph relationship created",
                        relationship_id=relationship.id,
                        from_node=relationship.from_node,
                        to_node=relationship.to_node)

            return result[0]['id'] if result else relationship.id

    async def get_relationships(self, node_id: str, direction: str = "both") -> List[GraphRelationship]:
        """Get all relationships for a node."""
        with logfire.span("Get node relationships", node_id=node_id, direction=direction):
            if direction == "outgoing":
                query = "SELECT * FROM graph_relationship WHERE from_node = $node_id"
            elif direction == "incoming":
                query = "SELECT * FROM graph_relationship WHERE to_node = $node_id"
            else:  # both
                query = "SELECT * FROM graph_relationship WHERE from_node = $node_id OR to_node = $node_id"

            result = await self.surrealdb_manager.execute_query(query, {'node_id': node_id})

            relationships = []
            for data in result:
                relationships.append(GraphRelationship(
                    id=data['id'],
                    from_node=data['from_node'],
                    to_node=data['to_node'],
                    type=RelationshipType(data['type']),
                    properties=data['properties'],
                    weight=data['weight'],
                    confidence=data['confidence'],
                    created_at=datetime.fromisoformat(data['created_at'])
                ))

            return relationships

    # Graph Traversal Operations

    async def find_shortest_path(self, from_node_id: str, to_node_id: str,
                                max_depth: int = 10) -> Optional[GraphTraversalPath]:
        """Find the shortest path between two nodes using BFS."""
        with logfire.span("Find shortest path", from_node=from_node_id, to_node=to_node_id):

            # Use SurrealDB's graph traversal capabilities
            query = """
                SELECT VALUE ->graph_relationship->(graph_node WHERE id = $to_node)
                FROM graph_node:$from_node
                GRAPH(1, $max_depth)
            """

            result = await self.surrealdb_manager.execute_query(query, {
                'from_node': from_node_id,
                'to_node': to_node_id,
                'max_depth': max_depth
            })

            if not result:
                return None

            # Convert result to GraphTraversalPath
            # This is a simplified implementation - would need to parse the actual path
            nodes = []
            relationships = []

            # For now, return a basic path structure
            from_node = await self.get_node(from_node_id)
            to_node = await self.get_node(to_node_id)

            if from_node and to_node:
                nodes = [from_node, to_node]
                # Find direct relationship if exists
                direct_rels = await self.get_relationships(from_node_id, "outgoing")
                for rel in direct_rels:
                    if rel.to_node == to_node_id:
                        relationships = [rel]
                        break

            return GraphTraversalPath(nodes=nodes, relationships=relationships)

    async def find_neighbors(self, node_id: str, depth: int = 1,
                           relationship_types: Optional[List[RelationshipType]] = None) -> List[GraphNode]:
        """Find neighboring nodes within specified depth."""
        with logfire.span("Find neighbors", node_id=node_id, depth=depth):

            type_filter = ""
            params = {'node_id': node_id, 'depth': depth}

            if relationship_types:
                type_list = [rt.value for rt in relationship_types]
                type_filter = f"WHERE type IN {type_list}"

            query = f"""
                SELECT VALUE ->graph_relationship{type_filter}->(graph_node)
                FROM graph_node:$node_id
                GRAPH(1, $depth)
            """

            result = await self.surrealdb_manager.execute_query(query, params)

            neighbors = []
            for node_data in result:
                if isinstance(node_data, dict) and 'id' in node_data:
                    node = GraphNode(
                        id=node_data['id'],
                        type=NodeType(node_data['type']),
                        properties=node_data['properties'],
                        labels=node_data.get('labels', []),
                        vector_embedding=node_data.get('vector_embedding'),
                        created_at=datetime.fromisoformat(node_data['created_at']) if node_data.get('created_at') else None,
                        updated_at=datetime.fromisoformat(node_data['updated_at']) if node_data.get('updated_at') else None
                    )
                    neighbors.append(node)

            return neighbors

    # Vector Search Operations

    async def add_vector_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """Add or update vector embedding for a node."""
        with logfire.span("Add vector embedding", node_id=node_id):
            return await self.update_node(node_id, {'vector_embedding': embedding})

    async def vector_similarity_search(self, query_vector: List[float],
                                     top_k: int = 10,
                                     node_type: Optional[NodeType] = None,
                                     threshold: float = 0.0) -> List[VectorSearchResult]:
        """Perform vector similarity search."""
        with logfire.span("Vector similarity search", top_k=top_k, node_type=node_type):

            # For now, implement a basic similarity search
            # In production, you'd want to use a proper vector database or SurrealDB's vector capabilities
            type_filter = f"WHERE type = '{node_type.value}'" if node_type else ""

            query = f"""
                SELECT *, vector::similarity::cosine(vector_embedding, $query_vector) AS similarity_score
                FROM graph_node {type_filter}
                WHERE vector_embedding IS NOT NULL
                ORDER BY similarity_score DESC
                LIMIT $top_k
            """

            result = await self.surrealdb_manager.execute_query(query, {
                'query_vector': query_vector,
                'top_k': top_k
            })

            search_results = []
            for data in result:
                if data.get('similarity_score', 0) >= threshold:
                    node = GraphNode(
                        id=data['id'],
                        type=NodeType(data['type']),
                        properties=data['properties'],
                        labels=data.get('labels', []),
                        vector_embedding=data.get('vector_embedding'),
                        created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                        updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
                    )

                    search_results.append(VectorSearchResult(
                        node=node,
                        similarity_score=data['similarity_score'],
                        distance=1.0 - data['similarity_score'],  # Convert similarity to distance
                        metadata={'query_time': datetime.utcnow().isoformat()}
                    ))

            return search_results

    # Agent State Persistence

    async def save_agent_state(self, agent_id: str, state_data: Dict[str, Any],
                             context: Optional[Dict[str, Any]] = None,
                             capabilities: Optional[List[str]] = None,
                             tools: Optional[List[str]] = None,
                             workflows: Optional[List[str]] = None) -> bool:
        """Save agent state to the graph database."""
        with logfire.span("Save agent state", agent_id=agent_id):

            now = datetime.utcnow().isoformat()

            # Upsert agent state
            query = """
                UPSERT agent_state:$agent_id CONTENT {
                    agent_id: $agent_id,
                    state_data: $state_data,
                    context: $context,
                    capabilities: $capabilities,
                    tools: $tools,
                    workflows: $workflows,
                    knowledge_base: $knowledge_base,
                    performance_metrics: $performance_metrics,
                    last_execution: $last_execution,
                    updated_at: $updated_at
                }
            """

            params = {
                'agent_id': agent_id,
                'state_data': state_data,
                'context': context or {},
                'capabilities': capabilities or [],
                'tools': tools or [],
                'workflows': workflows or [],
                'knowledge_base': {},  # Will be populated by knowledge operations
                'performance_metrics': {},  # Will be updated by execution tracking
                'last_execution': now,
                'updated_at': now
            }

            result = await self.surrealdb_manager.execute_query(query, params)
            success = bool(result)

            logfire.info("Agent state saved", agent_id=agent_id, success=success)
            return success

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state from the graph database."""
        with logfire.span("Load agent state", agent_id=agent_id):

            query = "SELECT * FROM agent_state WHERE agent_id = $agent_id"
            result = await self.surrealdb_manager.execute_query(query, {'agent_id': agent_id})

            if result:
                return result[0]
            return None

    async def update_agent_performance_metrics(self, agent_id: str, metrics: Dict[str, Any]) -> bool:
        """Update agent performance metrics."""
        with logfire.span("Update agent performance metrics", agent_id=agent_id):

            query = """
                UPDATE agent_state SET
                    performance_metrics = $metrics,
                    updated_at = $updated_at
                WHERE agent_id = $agent_id
            """

            result = await self.surrealdb_manager.execute_query(query, {
                'agent_id': agent_id,
                'metrics': metrics,
                'updated_at': datetime.utcnow().isoformat()
            })

            return bool(result)

    # Knowledge Graph Analytics

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        with logfire.span("Get graph statistics"):

            # Node count by type
            node_counts = await self.surrealdb_manager.execute_query("""
                SELECT type, count() AS count
                FROM graph_node
                GROUP BY type
            """)

            # Relationship count by type
            rel_counts = await self.surrealdb_manager.execute_query("""
                SELECT type, count() AS count
                FROM graph_relationship
                GROUP BY type
            """)

            # Total counts
            total_nodes = await self.surrealdb_manager.execute_query("SELECT count() FROM graph_node")
            total_rels = await self.surrealdb_manager.execute_query("SELECT count() FROM graph_relationship")

            # Agent states
            active_agents = await self.surrealdb_manager.execute_query("""
                SELECT count() FROM agent_state
                WHERE last_execution > $threshold
            """, {'threshold': (datetime.utcnow() - timedelta(hours=24)).isoformat()})

            return {
                'total_nodes': total_nodes[0]['count'] if total_nodes else 0,
                'total_relationships': total_rels[0]['count'] if total_rels else 0,
                'nodes_by_type': {item['type']: item['count'] for item in node_counts},
                'relationships_by_type': {item['type']: item['count'] for item in rel_counts},
                'active_agents_24h': active_agents[0]['count'] if active_agents else 0,
                'generated_at': datetime.utcnow().isoformat()
            }

    async def find_central_nodes(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Find the most connected nodes (highest degree centrality)."""
        with logfire.span("Find central nodes", limit=limit):

            query = """
                SELECT from_node AS node_id, count() AS degree
                FROM graph_relationship
                GROUP BY from_node
                UNION ALL
                SELECT to_node AS node_id, count() AS degree
                FROM graph_relationship
                GROUP BY to_node
                ORDER BY degree DESC
                LIMIT $limit
            """

            result = await self.surrealdb_manager.execute_query(query, {'limit': limit})

            return [(item['node_id'], item['degree']) for item in result]

    # Data Synchronization

    async def sync_from_relational_models(self, model_type: str, model_data: Dict[str, Any]) -> bool:
        """Synchronize data from relational models to graph format."""
        with logfire.span("Sync from relational models", model_type=model_type):

            try:
                # Create node from relational model
                node_id = f"{model_type}:{model_data.get('id', str(uuid.uuid4()))}"

                node = GraphNode(
                    id=node_id,
                    type=NodeType(model_type.lower()),
                    properties=model_data,
                    labels=[model_type, "synced_from_relational"]
                )

                await self.create_node(node)

                # Create relationships based on foreign keys
                await self._create_relationships_from_foreign_keys(node_id, model_data, model_type)

                logfire.info("Model synced to graph", model_type=model_type, node_id=node_id)
                return True

            except Exception as e:
                logfire.error("Failed to sync model to graph",
                            model_type=model_type,
                            error=str(e))
                return False

    async def _create_relationships_from_foreign_keys(self, node_id: str,
                                                    model_data: Dict[str, Any],
                                                    model_type: str):
        """Create relationships based on foreign key relationships."""

        # Common foreign key patterns
        fk_patterns = {
            'user_id': ('user', RelationshipType.CREATED_BY),
            'assigned_to_user_id': ('user', RelationshipType.ASSIGNED_TO),
            'created_by_user_id': ('user', RelationshipType.CREATED_BY),
            'workflow_id': ('workflow', RelationshipType.PART_OF),
            'agent_id': ('agent', RelationshipType.DEPENDS_ON),
            'tool_id': ('tool', RelationshipType.USES_TOOL),
            'parent_task_id': ('task', RelationshipType.PART_OF)
        }

        for fk_field, (target_type, rel_type) in fk_patterns.items():
            if fk_field in model_data and model_data[fk_field]:
                target_node_id = f"{target_type}:{model_data[fk_field]}"

                relationship = GraphRelationship(
                    id=f"{node_id}_{rel_type.value}_{target_node_id}",
                    from_node=node_id,
                    to_node=target_node_id,
                    type=rel_type,
                    properties={'sync_source': 'relational_model'}
                )

                try:
                    await self.create_relationship(relationship)
                except Exception as e:
                    logfire.warning("Failed to create relationship during sync",
                                  from_node=node_id,
                                  to_node=target_node_id,
                                  error=str(e))

    async def cleanup_orphaned_nodes(self) -> int:
        """Clean up nodes that have no relationships."""
        with logfire.span("Cleanup orphaned nodes"):

            query = """
                DELETE FROM graph_node
                WHERE id NOT IN (
                    SELECT from_node FROM graph_relationship
                    UNION ALL
                    SELECT to_node FROM graph_relationship
                )
                AND type NOT IN ['user', 'agent']  -- Preserve important node types
            """

            result = await self.surrealdb_manager.execute_query(query)
            count = len(result) if result else 0

            logfire.info("Orphaned nodes cleaned up", count=count)
            return count


# Convenience factory function
async def create_graph_operations() -> GraphOperations:
    """Create and initialize a GraphOperations instance."""
    graph_ops = GraphOperations()
    await graph_ops.initialize()
    return graph_ops


# Context manager for graph transactions
@asynccontextmanager
async def graph_transaction():
    """Context manager for graph database transactions."""
    graph_ops = await create_graph_operations()
    try:
        # Begin transaction
        await graph_ops.surrealdb_manager.execute_query("BEGIN TRANSACTION")
        yield graph_ops
        # Commit transaction
        await graph_ops.surrealdb_manager.execute_query("COMMIT TRANSACTION")
    except Exception as e:
        # Rollback on error
        await graph_ops.surrealdb_manager.execute_query("CANCEL TRANSACTION")
        logfire.error("Graph transaction rolled back", error=str(e))
        raise
