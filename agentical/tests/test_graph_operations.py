"""
Tests for SurrealDB Graph Operations

This module contains comprehensive tests for the graph operations functionality,
including node management, relationship operations, graph traversal, vector search,
agent state persistence, and data synchronization.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import uuid

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.graph_operations import (
    GraphOperations, GraphNode, GraphRelationship, GraphTraversalPath,
    VectorSearchResult, NodeType, RelationshipType, create_graph_operations
)
from db.knowledge_schemas import (
    AgentKnowledgeSchema, CodeAgentSchema, DataScienceAgentSchema, SuperAgentSchema,
    KnowledgeEntity, AgentCapability, ToolUsagePattern, LearningRecord,
    AgentDomain, KnowledgeType, CapabilityLevel, create_agent_schema
)
from db.vector_search import (
    VectorSearchEngine, VectorSearchConfig, VectorSearchResult,
    EmbeddingModel, SimilarityMetric, create_vector_search_engine
)
from db.graph_sync import (
    DataSynchronizer, SyncConfig, ChangeRecord, SyncResult,
    SyncDirection, SyncStrategy, ChangeType, create_data_synchronizer
)


class TestGraphOperations:
    """Test cases for basic graph operations."""

    @pytest.fixture
    async def mock_surrealdb_manager(self):
        """Mock SurrealDB manager for testing."""
        manager = Mock()
        manager.execute_query = AsyncMock()
        return manager

    @pytest.fixture
    async def graph_ops(self, mock_surrealdb_manager):
        """Create GraphOperations instance with mocked dependencies."""
        ops = GraphOperations(mock_surrealdb_manager)
        await ops.initialize()
        return ops

    @pytest.fixture
    def sample_node(self):
        """Create a sample graph node for testing."""
        return GraphNode(
            id="test_agent_001",
            type=NodeType.AGENT,
            properties={
                "name": "Test Agent",
                "description": "A test agent for unit testing",
                "capabilities": ["coding", "testing"]
            },
            labels=["agent", "test"],
            vector_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )

    @pytest.fixture
    def sample_relationship(self):
        """Create a sample graph relationship for testing."""
        return GraphRelationship(
            id="rel_001",
            from_node="agent:001",
            to_node="tool:001",
            type=RelationshipType.USES_TOOL,
            properties={"frequency": "high", "success_rate": 0.95},
            weight=0.8,
            confidence=0.9
        )

    async def test_node_creation(self, graph_ops, sample_node, mock_surrealdb_manager):
        """Test creating a graph node."""
        # Mock successful creation
        mock_surrealdb_manager.execute_query.return_value = [{"id": "test_agent_001"}]

        result = await graph_ops.create_node(sample_node)

        assert result == "test_agent_001"
        mock_surrealdb_manager.execute_query.assert_called_once()

        # Verify the query was called with correct parameters
        call_args = mock_surrealdb_manager.execute_query.call_args
        assert "CREATE graph_node CONTENT" in call_args[0][0]
        assert call_args[0][1]["id"] == "test_agent_001"
        assert call_args[0][1]["type"] == "agent"

    async def test_node_retrieval(self, graph_ops, mock_surrealdb_manager):
        """Test retrieving a graph node."""
        # Mock node data
        mock_data = {
            "id": "test_agent_001",
            "type": "agent",
            "properties": {"name": "Test Agent"},
            "labels": ["agent", "test"],
            "vector_embedding": [0.1, 0.2, 0.3],
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-01-15T10:00:00Z"
        }
        mock_surrealdb_manager.execute_query.return_value = [mock_data]

        node = await graph_ops.get_node("test_agent_001")

        assert node is not None
        assert node.id == "test_agent_001"
        assert node.type == NodeType.AGENT
        assert node.properties["name"] == "Test Agent"
        assert len(node.vector_embedding) == 3

    async def test_node_update(self, graph_ops, mock_surrealdb_manager):
        """Test updating a graph node."""
        mock_surrealdb_manager.execute_query.return_value = [{"updated": True}]

        updates = {"properties.name": "Updated Agent", "properties.version": "2.0"}
        result = await graph_ops.update_node("test_agent_001", updates)

        assert result is True
        mock_surrealdb_manager.execute_query.assert_called_once()

    async def test_node_deletion(self, graph_ops, mock_surrealdb_manager):
        """Test deleting a graph node."""
        mock_surrealdb_manager.execute_query.return_value = [{"deleted": True}]

        result = await graph_ops.delete_node("test_agent_001", cascade=True)

        assert result is True
        # Should call execute_query twice (once for relationships, once for node)
        assert mock_surrealdb_manager.execute_query.call_count == 2

    async def test_relationship_creation(self, graph_ops, sample_relationship, mock_surrealdb_manager):
        """Test creating a graph relationship."""
        # Mock that both nodes exist
        mock_surrealdb_manager.execute_query.side_effect = [
            [{"id": "agent:001"}],  # from_node exists
            [{"id": "tool:001"}],   # to_node exists
            [{"id": "rel_001"}]     # relationship created
        ]

        result = await graph_ops.create_relationship(sample_relationship)

        assert result == "rel_001"
        assert mock_surrealdb_manager.execute_query.call_count == 3

    async def test_relationship_retrieval(self, graph_ops, mock_surrealdb_manager):
        """Test retrieving node relationships."""
        mock_relationships = [
            {
                "id": "rel_001",
                "from_node": "agent:001",
                "to_node": "tool:001",
                "type": "uses_tool",
                "properties": {"frequency": "high"},
                "weight": 0.8,
                "confidence": 0.9,
                "created_at": "2025-01-15T10:00:00Z"
            }
        ]
        mock_surrealdb_manager.execute_query.return_value = mock_relationships

        relationships = await graph_ops.get_relationships("agent:001", "outgoing")

        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.id == "rel_001"
        assert rel.type == RelationshipType.USES_TOOL
        assert rel.weight == 0.8

    async def test_shortest_path_finding(self, graph_ops, mock_surrealdb_manager):
        """Test finding shortest path between nodes."""
        # Mock path data
        mock_surrealdb_manager.execute_query.return_value = [{"path": "found"}]

        # Mock node retrieval for path construction
        mock_nodes = [
            {"id": "agent:001", "type": "agent", "properties": {}, "labels": []},
            {"id": "tool:001", "type": "tool", "properties": {}, "labels": []}
        ]
        mock_surrealdb_manager.execute_query.side_effect = [
            [{"path": "found"}],  # path query
            mock_nodes[0:1],      # from_node
            mock_nodes[1:2]       # to_node
        ]

        path = await graph_ops.find_shortest_path("agent:001", "tool:001")

        assert path is not None
        assert len(path.nodes) == 2
        assert path.nodes[0].id == "agent:001"
        assert path.nodes[1].id == "tool:001"

    async def test_neighbor_finding(self, graph_ops, mock_surrealdb_manager):
        """Test finding neighboring nodes."""
        mock_neighbors = [
            {
                "id": "tool:001",
                "type": "tool",
                "properties": {"name": "Test Tool"},
                "labels": ["tool"],
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z"
            }
        ]
        mock_surrealdb_manager.execute_query.return_value = mock_neighbors

        neighbors = await graph_ops.find_neighbors("agent:001", depth=1)

        assert len(neighbors) == 1
        neighbor = neighbors[0]
        assert neighbor.id == "tool:001"
        assert neighbor.type == NodeType.TOOL

    async def test_agent_state_persistence(self, graph_ops, mock_surrealdb_manager):
        """Test saving and loading agent state."""
        # Test saving
        mock_surrealdb_manager.execute_query.return_value = [{"saved": True}]

        state_data = {
            "current_task": "coding",
            "context": {"project": "agentical"},
            "performance": {"success_rate": 0.95}
        }

        result = await graph_ops.save_agent_state(
            agent_id="agent_001",
            state_data=state_data,
            capabilities=["coding", "testing"],
            tools=["vscode", "pytest"]
        )

        assert result is True

        # Test loading
        mock_agent_state = {
            "agent_id": "agent_001",
            "state_data": state_data,
            "capabilities": ["coding", "testing"],
            "tools": ["vscode", "pytest"],
            "updated_at": "2025-01-15T10:00:00Z"
        }
        mock_surrealdb_manager.execute_query.return_value = [mock_agent_state]

        loaded_state = await graph_ops.load_agent_state("agent_001")

        assert loaded_state is not None
        assert loaded_state["agent_id"] == "agent_001"
        assert loaded_state["state_data"]["current_task"] == "coding"

    async def test_graph_statistics(self, graph_ops, mock_surrealdb_manager):
        """Test getting graph statistics."""
        # Mock statistics data
        mock_surrealdb_manager.execute_query.side_effect = [
            [{"type": "agent", "count": 5}, {"type": "tool", "count": 10}],  # node counts
            [{"type": "uses_tool", "count": 15}],  # relationship counts
            [{"count": 15}],  # total nodes
            [{"count": 15}],  # total relationships
            [{"count": 3}]    # active agents
        ]

        stats = await graph_ops.get_graph_statistics()

        assert stats["total_nodes"] == 15
        assert stats["total_relationships"] == 15
        assert stats["nodes_by_type"]["agent"] == 5
        assert stats["nodes_by_type"]["tool"] == 10
        assert stats["active_agents_24h"] == 3

    async def test_vector_search_integration(self, graph_ops, mock_surrealdb_manager):
        """Test vector similarity search."""
        # Mock vector search results
        mock_results = [
            {
                "id": "knowledge:001",
                "type": "knowledge",
                "properties": {"title": "Python Best Practices"},
                "labels": ["knowledge"],
                "vector_embedding": [0.8, 0.6, 0.4],
                "similarity_score": 0.95,
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z"
            }
        ]
        mock_surrealdb_manager.execute_query.return_value = mock_results

        query_vector = [0.9, 0.7, 0.5]
        results = await graph_ops.vector_similarity_search(
            query_vector=query_vector,
            top_k=5,
            threshold=0.8
        )

        assert len(results) == 1
        result = results[0]
        assert result.similarity_score == 0.95
        assert result.node.id == "knowledge:001"


class TestKnowledgeSchemas:
    """Test cases for agent knowledge schemas."""

    def test_base_knowledge_schema_creation(self):
        """Test creating a base knowledge schema."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        assert schema.agent_id == "agent_001"
        assert schema.domain == AgentDomain.CODE_DEVELOPMENT
        assert len(schema.knowledge_entities) == 0
        assert len(schema.capabilities) == 0

    def test_knowledge_entity_addition(self):
        """Test adding knowledge entities to schema."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        entity = KnowledgeEntity(
            id="python_basics",
            title="Python Basics",
            description="Basic Python programming concepts",
            knowledge_type=KnowledgeType.DECLARATIVE,
            domain=AgentDomain.CODE_DEVELOPMENT,
            content={"concepts": ["variables", "functions", "classes"]},
            tags=["python", "basics"]
        )

        entity_id = schema.add_knowledge_entity(entity)

        assert entity_id == "python_basics"
        assert "python_basics" in schema.knowledge_entities
        assert schema.knowledge_entities["python_basics"].title == "Python Basics"

    def test_capability_addition(self):
        """Test adding capabilities to schema."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        capability = AgentCapability(
            name="code_generation",
            description="Generate code in various languages",
            domain=AgentDomain.CODE_DEVELOPMENT,
            level=CapabilityLevel.ADVANCED,
            required_tools=["editor", "compiler"],
            knowledge_requirements=["programming_languages"]
        )

        cap_name = schema.add_capability(capability)

        assert cap_name == "code_generation"
        assert "code_generation" in schema.capabilities
        assert schema.capabilities["code_generation"].level == CapabilityLevel.ADVANCED

    def test_tool_usage_recording(self):
        """Test recording tool usage patterns."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        # Record successful tool usage
        schema.record_tool_usage(
            tool_name="vscode",
            success=True,
            parameters={"language": "python", "file_type": ".py"},
            context="coding_session"
        )

        assert "vscode" in schema.tool_patterns
        pattern = schema.tool_patterns["vscode"]
        assert pattern.usage_frequency == 1.0
        assert pattern.success_rate == 1.0
        assert "coding_session" in pattern.typical_contexts

    def test_relevant_knowledge_retrieval(self):
        """Test retrieving relevant knowledge entities."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        # Add some knowledge entities
        entities = [
            KnowledgeEntity(
                id="python_basics",
                title="Python Basics",
                description="Basic Python concepts",
                knowledge_type=KnowledgeType.DECLARATIVE,
                domain=AgentDomain.CODE_DEVELOPMENT,
                content={},
                tags=["python", "basics"]
            ),
            KnowledgeEntity(
                id="testing_practices",
                title="Testing Best Practices",
                description="Unit testing and TDD practices",
                knowledge_type=KnowledgeType.PROCEDURAL,
                domain=AgentDomain.CODE_DEVELOPMENT,
                content={},
                tags=["testing", "tdd"]
            )
        ]

        for entity in entities:
            schema.add_knowledge_entity(entity)

        # Search for Python-related knowledge
        relevant = schema.get_relevant_knowledge("python", limit=5)

        assert len(relevant) >= 1
        assert any(entity.id == "python_basics" for entity in relevant)

    def test_code_agent_schema(self):
        """Test CodeAgentSchema with pre-initialized capabilities."""
        schema = CodeAgentSchema("code_agent_001")

        assert schema.domain == AgentDomain.CODE_DEVELOPMENT
        assert len(schema.capabilities) > 0
        assert len(schema.knowledge_entities) > 0

        # Check for expected capabilities
        assert "code_generation" in schema.capabilities
        assert "code_review" in schema.capabilities
        assert "testing" in schema.capabilities

        # Check for expected knowledge
        knowledge_ids = list(schema.knowledge_entities.keys())
        assert any("python" in kid.lower() for kid in knowledge_ids)

    def test_super_agent_schema(self):
        """Test SuperAgentSchema with coordination capabilities."""
        schema = SuperAgentSchema("super_agent_001")

        assert schema.domain == AgentDomain.META_COORDINATION
        assert len(schema.capabilities) > 0

        # Check for coordination capabilities
        assert "agent_orchestration" in schema.capabilities
        assert "resource_allocation" in schema.capabilities

    def test_schema_factory_function(self):
        """Test the schema factory function."""
        # Test code agent creation
        code_schema = create_agent_schema("agent_001", "code_agent")
        assert isinstance(code_schema, CodeAgentSchema)

        # Test super agent creation
        super_schema = create_agent_schema("agent_002", "super_agent")
        assert isinstance(super_schema, SuperAgentSchema)

        # Test generic agent creation
        generic_schema = create_agent_schema("agent_003", "unknown_type")
        assert isinstance(generic_schema, AgentKnowledgeSchema)

    def test_schema_serialization(self):
        """Test converting schema to graph representation."""
        schema = AgentKnowledgeSchema("agent_001", AgentDomain.CODE_DEVELOPMENT)

        # Add some data
        entity = KnowledgeEntity(
            id="test_entity",
            title="Test Entity",
            description="Test description",
            knowledge_type=KnowledgeType.DECLARATIVE,
            domain=AgentDomain.CODE_DEVELOPMENT,
            content={"test": "data"}
        )
        schema.add_knowledge_entity(entity)

        graph_repr = schema.to_graph_representation()

        assert graph_repr["agent_id"] == "agent_001"
        assert graph_repr["domain"] == "code_development"
        assert "knowledge_entities" in graph_repr
        assert "test_entity" in graph_repr["knowledge_entities"]


class TestVectorSearch:
    """Test cases for vector search functionality."""

    @pytest.fixture
    def vector_config(self):
        """Create test vector search configuration."""
        return VectorSearchConfig(
            embedding_model=EmbeddingModel.SENTENCE_BERT,
            similarity_metric=SimilarityMetric.COSINE,
            cache_embeddings=True,
            batch_size=5,
            embedding_dimension=384
        )

    @pytest.fixture
    async def vector_engine(self, vector_config):
        """Create vector search engine for testing."""
        engine = VectorSearchEngine(vector_config)
        await engine.initialize()
        return engine

    async def test_vector_engine_initialization(self, vector_engine):
        """Test vector search engine initialization."""
        assert vector_engine.is_initialized
        assert vector_engine.config.embedding_dimension == 384
        assert vector_engine.embedding_model is not None

    async def test_embedding_generation(self, vector_engine):
        """Test generating embeddings for text."""
        text = "This is a test sentence for embedding generation."
        embedding = await vector_engine.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == vector_engine.config.embedding_dimension
        assert all(isinstance(val, (int, float)) for val in embedding)

    async def test_batch_embedding_generation(self, vector_engine):
        """Test generating embeddings for multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence with different content."
        ]

        embeddings = await vector_engine.generate_batch_embeddings(texts)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == vector_engine.config.embedding_dimension

    async def test_similarity_calculation(self, vector_engine):
        """Test calculating similarity between embeddings."""
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Identical
        embedding3 = [0.9, 0.8, 0.7, 0.6, 0.5]  # Different

        # Test identical embeddings
        similarity_identical = vector_engine.calculate_similarity(embedding1, embedding2)
        assert abs(similarity_identical - 1.0) < 0.001

        # Test different embeddings
        similarity_different = vector_engine.calculate_similarity(embedding1, embedding3)
        assert 0.0 <= similarity_different <= 1.0
        assert similarity_different < similarity_identical

    async def test_similarity_search(self, vector_engine):
        """Test vector similarity search."""
        query_text = "Python programming tutorial"
        candidate_texts = [
            "Learn Python programming basics",
            "JavaScript fundamentals guide",
            "Python advanced concepts",
            "HTML and CSS tutorial",
            "Python for data science"
        ]
        candidate_ids = [f"doc_{i}" for i in range(len(candidate_texts))]

        results = await vector_engine.search_similar(
            query_text=query_text,
            candidate_texts=candidate_texts,
            candidate_ids=candidate_ids,
            top_k=3,
            threshold=0.0
        )

        assert len(results) <= 3
        assert all(isinstance(result, VectorSearchResult) for result in results)

        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score

        # Python-related documents should rank higher
        python_results = [r for r in results if "python" in r.content.lower()]
        assert len(python_results) > 0

    async def test_embedding_caching(self, vector_engine):
        """Test embedding caching functionality."""
        text = "Test text for caching"

        # First generation should miss cache
        start_time = time.time()
        embedding1 = await vector_engine.generate_embedding(text)
        first_time = time.time() - start_time

        # Second generation should hit cache (faster)
        start_time = time.time()
        embedding2 = await vector_engine.generate_embedding(text)
        second_time = time.time() - start_time

        assert embedding1 == embedding2
        assert vector_engine.stats["cache_hits"] > 0

    async def test_clustering(self, vector_engine):
        """Test embedding clustering functionality."""
        texts = [
            "Python programming language",
            "JavaScript development",
            "Python web frameworks",
            "Java programming basics",
            "Python data analysis",
            "C++ programming guide"
        ]
        entity_ids = [f"text_{i}" for i in range(len(texts))]

        clusters = await vector_engine.cluster_embeddings(
            texts=texts,
            entity_ids=entity_ids,
            n_clusters=3
        )

        if clusters:  # Only test if clustering is available
            assert len(clusters) == 3

            # Each cluster should have at least one entity
            for cluster_info in clusters.values():
                assert cluster_info.size > 0
                assert len(cluster_info.entity_ids) == cluster_info.size
                assert len(cluster_info.centroid) == vector_engine.config.embedding_dimension

    async def test_statistics_tracking(self, vector_engine):
        """Test performance statistics tracking."""
        # Generate some embeddings and searches
        texts = ["test text 1", "test text 2", "test text 3"]
        await vector_engine.generate_batch_embeddings(texts)

        await vector_engine.search_similar(
            query_text="test query",
            candidate_texts=texts,
            top_k=2
        )

        stats = vector_engine.get_statistics()

        assert "embeddings_generated" in stats
        assert "searches_performed" in stats
        assert "cache_hit_ratio" in stats
        assert "average_search_time_ms" in stats
        assert stats["embeddings_generated"] > 0
        assert stats["searches_performed"] > 0


class TestDataSynchronization:
    """Test cases for data synchronization functionality."""

    @pytest.fixture
    def sync_config(self):
        """Create test synchronization configuration."""
        return SyncConfig(
            sync_direction=SyncDirection.BIDIRECTIONAL,
            sync_strategy=SyncStrategy.NEWEST_WINS,
            batch_size=10,
            enable_real_time_sync=False,  # Disable for testing
            enable_vector_sync=True
        )

    @pytest.fixture
    async def mock_graph_ops(self):
        """Mock graph operations for testing."""
        graph_ops = Mock(spec=GraphOperations)
        graph_ops.create_node = AsyncMock(return_value="node_id")
        graph_ops.get_node = AsyncMock(return_value=None)
        graph_ops.update_node = AsyncMock(return_value=True)
        graph_ops.delete_node = AsyncMock(return_value=True)
        graph_ops.create_relationship = AsyncMock(return_value="rel_id")
        graph_ops.save_agent_state = AsyncMock(return_value=True)
        return graph_ops

    @pytest.fixture
    async def mock_vector_engine(self):
        """Mock vector search engine for testing."""
        engine = Mock(spec=VectorSearchEngine)
        engine.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return engine

    @pytest.fixture
    async def data_sync(self, sync_config, mock_graph_ops, mock_vector_engine):
        """Create data synchronizer for testing."""
        sync = DataSynchronizer(sync_config, mock_graph_ops, mock_vector_engine)
        await sync.initialize()
        return sync

    async def test_synchronizer_initialization(self, data_sync):
        """Test data synchronizer initialization."""
        assert data_sync.config.sync_direction == SyncDirection.BIDIRECTIONAL
        assert data_sync.graph_ops is not None
        assert data_sync.vector_engine is not None

    async def test_change_recording(self, data_sync):
        """Test recording data changes."""
        entity_data = {
            "id": "agent_001",
            "name": "Test Agent",
            "type": "code_agent",
            "description": "Test agent for synchronization"
        }

        await data_sync.record_change(
            entity_type="agent",
            entity_id="agent_001",
            change_type=ChangeType.CREATE,
            new_data=entity_data
        )

        assert len(data_sync.change_queue) == 1
        change = data_sync.change_queue[0]
        assert change.entity_type == "agent"
        assert change.entity_id == "agent_001"
        assert change.change_type == ChangeType.CREATE
        assert change.new_data == entity_data

    async def test_change_processing(self, data_sync, mock_graph_ops):
        """Test processing pending changes."""
        # Record a change
        entity_data = {
            "id": "agent_001",
            "name": "Test Agent",
            "type": "code_agent"
        }

        await data_sync.record_change(
            entity_type="agent",
            entity_id="agent_001",
            change_type=ChangeType.CREATE,
            new_data=entity_data
        )

        # Process changes
        result = await data_sync.process_pending_changes()

        assert result.success
        assert result.records_processed == 1
        assert result.records_synced == 1
        assert result.records_failed == 0

        # Verify graph operations were called
        mock_graph_ops.create_node.assert_called_once()

    async def test_agent_state_synchronization(self, data_sync, mock_graph_ops):
        """Test synchronizing agent state to graph."""
        agent_data = {
            "id": "agent_001",
            "name": "Test Agent",
            "type": "code_agent",
            "capabilities": ["coding", "testing"],
            "tools": ["vscode", "pytest"]
        }

        await data_sync.record_change(
            entity_type="agent",
            entity_id="agent_001",
            change_type=ChangeType.UPDATE,
            new_data=agent_data
        )

        await data_sync.process_pending_changes()

        # Verify agent state was saved
        mock_graph_ops.save_agent_state.assert_called_once()
        call_args = mock_graph_ops.save_agent_state.call_args
        assert call_args[1]["agent_id"] == "agent_001"

    async def test_vector_embedding_sync(self, data_sync, mock_vector_engine):
        """Test synchronizing with vector embeddings."""
        entity_data = {
            "id": "knowledge_001",
            "title": "Python Best Practices",
            "description": "Guidelines for writing clean Python code",
            "content": "Use meaningful variable names and follow PEP 8"
        }

        await data_sync.record_change(
            entity_type="knowledge",
            entity_id="knowledge_001",
            change_type=ChangeType.CREATE,
            new_data=entity_data
        )

        await data_sync.process_pending_changes()

        # Verify embedding was generated
        mock_vector_engine.generate_embedding.assert_called_once()

    async def test_relationship_creation_from_foreign_keys(self, data_sync, mock_graph_ops):
        """Test creating relationships from foreign key data."""
        # Mock that target node exists
        mock_graph_ops.get_node.return_value = Mock()

        task_data = {
            "id": "task_001",
            "name": "Test Task",
            "assigned_to_user_id": "user_001",
            "workflow_id": "workflow_001"
        }

        await data_sync.record_change(
            entity_type="task",
            entity_id="task_001",
            change_type=ChangeType.CREATE,
            new_data=task_data
        )

        await data_sync.process_pending_changes()

        # Verify relationships were created
        assert mock_graph_ops.create_relationship.call_count >= 1

    async def test_sync_statistics(self, data_sync):
        """Test synchronization statistics tracking."""
        # Process some changes
        for i in range(3):
            await data_sync.record_change(
                entity_type="test",
                entity_id=f"test_{i}",
                change_type=ChangeType.CREATE,
                new_data={"id": f"test_{i}", "name": f"Test {i}"}
            )

        await data_sync.process_pending_changes()

        stats = data_sync.get_sync_statistics()

        assert "total_syncs" in stats
        assert "successful_syncs" in stats
        assert "pending_changes" in stats
        assert stats["total_syncs"] > 0

    async def test_retry_mechanism(self, data_sync, mock_graph_ops):
        """Test retry mechanism for failed synchronizations."""
        # Make graph operations fail initially
        mock_graph_ops.create_node.side_effect = [
            Exception("Network error"),  # First attempt fails
            Exception("Network error"),  # Second attempt fails
            "node_id"  # Third attempt succeeds
        ]

        await data_sync.record_change(
            entity_type="test",
            entity_id="test_retry",
            change_type=ChangeType.CREATE,
            new_data={"id": "test_retry", "name": "Test Retry"}
        )

        # Process multiple times to trigger retries
        for _ in range(3):
            await data_sync.process_pending_changes()

        # Should eventually succeed
        assert mock_graph_ops.create_node.call_count == 3


class TestIntegration:
    """Integration tests for all graph operations components."""

    @pytest.fixture
    async def full_system(self):
        """Set up complete system for integration testing."""
        # Mock dependencies
        mock_surrealdb = Mock()
        mock_surrealdb.execute_query = AsyncMock()

        # Create components
        graph_ops = GraphOperations(mock_surrealdb)
        await graph_ops.initialize()

        vector_config = VectorSearchConfig(
            embedding_model=EmbeddingModel.SENTENCE_BERT,
            cache_embeddings=True
        )
        vector_engine = VectorSearchEngine(vector_config)
        await vector_engine.initialize()

        sync_config = SyncConfig(
            enable_real_time_sync=False,
            enable_vector_sync=True
        )
        data_sync = DataSynchronizer(sync_config, graph_ops, vector_engine)
        await data_sync.initialize()

        return {
            "graph_ops": graph_ops,
            "vector_engine": vector_engine,
            "data_sync": data_sync,
            "mock_surrealdb": mock_surrealdb
        }

    async def test_end_to_end_agent_workflow(self, full_system):
        """Test complete agent workflow from creation to knowledge management."""
        graph_ops = full_system["graph_ops"]
        vector_engine = full_system["vector_engine"]
        data_sync = full_system["data_sync"]
        mock_surrealdb = full_system["mock_surrealdb"]

        # Mock successful operations
        mock_surrealdb.execute_query.return_value = [{"id": "agent_001"}]

        # 1. Create agent
        agent_data = {
            "id": "agent_001",
            "name": "Test Code Agent",
            "type": "code_agent",
            "description": "An agent for testing code operations",
            "capabilities": ["coding", "testing", "debugging"]
        }

        # 2. Sync to graph database
        await data_sync.record_change(
            entity_type="agent",
            entity_id="agent_001",
            change_type=ChangeType.CREATE,
            new_data=agent_data
        )

        result = await data_sync.process_pending_changes()
        assert result.success

        # 3. Create knowledge schema
        schema = create_agent_schema("agent_001", "code_agent")
        assert isinstance(schema, CodeAgentSchema)
        assert len(schema.capabilities) > 0

        # 4. Generate embeddings for agent description
        embedding = await vector_engine.generate_embedding(agent_data["description"])
        assert len(embedding) == vector_engine.config.embedding_dimension

        # 5. Record tool usage
        schema.record_tool_usage("vscode", True, {"language": "python"})
        assert "vscode" in schema.tool_patterns

    async def test_knowledge_discovery_workflow(self, full_system):
        """Test knowledge discovery and similarity search workflow."""
        vector_engine = full_system["vector_engine"]

        # Create knowledge entities
        knowledge_texts = [
            "Python best practices for clean code",
            "JavaScript async/await patterns",
            "Python testing with pytest framework",
            "React hooks and state management",
            "Python data structures and algorithms"
        ]

        # Generate embeddings
        embeddings = await vector_engine.generate_batch_embeddings(knowledge_texts)
        assert len(embeddings) == len(knowledge_texts)

        # Search for Python-related knowledge
        query = "Python programming guidelines"
        results = await vector_engine.search_similar(
            query_text=query,
            candidate_texts=knowledge_texts,
            candidate_ids=[f"knowledge_{i}" for i in range(len(knowledge_texts))],
            top_k=3
        )

        # Python-related items should rank higher
        python_results = [r for r in results if "python" in r.content.lower()]
        assert len(python_results) >= 2

    async def test_relationship_mapping_workflow(self, full_system):
        """Test relationship creation and traversal workflow."""
        graph_ops = full_system["graph_ops"]
        mock_surrealdb = full_system["mock_surrealdb"]

        # Mock node existence checks
        mock_surrealdb.execute_query.side_effect = [
            [{"id": "agent:001"}],  # agent exists
            [{"id": "tool:001"}],   # tool exists
            [{"id": "rel_001"}],    # relationship created
            [{"id": "workflow:001"}], # workflow exists
            [{"id": "rel_002"}]     # second relationship created
        ]

        # Create relationships
        agent_tool_rel = GraphRelationship(
            id="rel_001",
            from_node="agent:001",
            to_node="tool:001",
            type=RelationshipType.USES_TOOL,
            properties={"frequency": "daily"},
            confidence=0.9
        )

        agent_workflow_rel = GraphRelationship(
            id="rel_002",
            from_node="agent:001",
            to_node="workflow:001",
            type=RelationshipType.EXECUTES_WORKFLOW,
            properties={"success_rate": 0.95},
            confidence=0.8
        )

        # Create relationships
        rel1_id = await graph_ops.create_relationship(agent_tool_rel)
        rel2_id = await graph_ops.create_relationship(agent_workflow_rel)

        assert rel1_id == "rel_001"
        assert rel2_id == "rel_002"

    async def test_performance_metrics_tracking(self, full_system):
        """Test performance metrics across all components."""
        vector_engine = full_system["vector_engine"]
        data_sync = full_system["data_sync"]

        # Generate some activity
        texts = [f"Test document {i}" for i in range(10)]
        await vector_engine.generate_batch_embeddings(texts)

        for i in range(5):
            await data_sync.record_change(
                entity_type="test",
                entity_id=f"test_{i}",
                change_type=ChangeType.CREATE,
                new_data={"id": f"test_{i}", "name": f"Test {i}"}
            )

        await data_sync.process_pending_changes()

        # Check vector engine stats
        vector_stats = vector_engine.get_statistics()
        assert vector_stats["embeddings_generated"] >= 10

        # Check sync stats
        sync_stats = data_sync.get_sync_statistics()
        assert sync_stats["total_syncs"] >= 1

    async def test_error_handling_and_recovery(self, full_system):
        """Test error handling and recovery mechanisms."""
        graph_ops = full_system["graph_ops"]
        data_sync = full_system["data_sync"]
        mock_surrealdb = full_system["mock_surrealdb"]

        # Test graph operation errors
        mock_surrealdb.execute_query.side_effect = Exception("Database connection error")

        # This should handle the error gracefully
        try:
            await graph_ops.create_node(GraphNode(
                id="test_node",
                type=NodeType.AGENT,
                properties={"name": "Test"}
            ))
        except Exception:
            pass  # Expected to fail

        # Test sync error handling
        await data_sync.record_change(
            entity_type="test",
            entity_id="error_test",
            change_type=ChangeType.CREATE,
            new_data={"id": "error_test"}
        )

        result = await data_sync.process_pending_changes()
        assert not result.success
        assert result.records_failed > 0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    async def test_multi_agent_collaboration_graph(self):
        """Test building a graph for multi-agent collaboration."""
        # Create multiple agent schemas
        code_agent = create_agent_schema("code_agent_001", "code_agent")
        data_agent = create_agent_schema("data_agent_001", "data_science_agent")
        super_agent = create_agent_schema("super_agent_001", "super_agent")

        # Verify different capabilities
        assert "code_generation" in code_agent.capabilities
        assert "data_analysis" in data_agent.capabilities
        assert "agent_orchestration" in super_agent.capabilities

        # Test collaboration patterns
        schemas = [code_agent, data_agent, super_agent]
        merged_knowledge = merge_knowledge_schemas(schemas)

        assert len(merged_knowledge["agents"]) == 3
        assert "combined_knowledge" in merged_knowledge
        assert "shared_capabilities" in merged_knowledge

    async def test_knowledge_evolution_tracking(self):
        """Test tracking knowledge evolution over time."""
        schema = AgentKnowledgeSchema("evolving_agent", AgentDomain.CODE_DEVELOPMENT)

        # Add initial knowledge
        initial_knowledge = KnowledgeEntity(
            id="python_v1",
            title="Python Basics",
            description="Basic Python concepts",
            knowledge_type=KnowledgeType.DECLARATIVE,
            domain=AgentDomain.CODE_DEVELOPMENT,
            content={"version": "1.0", "concepts": ["variables", "functions"]},
            confidence=0.8
        )
        schema.add_knowledge_entity(initial_knowledge)

        # Record learning event
        learning_record = LearningRecord(
            id="learn_001",
            agent_id="evolving_agent",
            knowledge_entity_id="python_v1",
            learning_event="completed_tutorial",
            outcome="improved_understanding",
            confidence_change=0.1,
            performance_impact=0.15
        )
        schema.learning_history.append(learning_record)

        # Update knowledge based on learning
        updated_knowledge = schema.knowledge_entities["python_v1"]
        updated_knowledge.confidence += learning_record.confidence_change
        updated_knowledge.access_count += 1

        assert updated_knowledge.confidence == 0.9
        assert len(schema.learning_history) == 1

    async def test_large_scale_vector_operations(self):
        """Test vector operations at scale."""
        config = VectorSearchConfig(
            batch_size=50,
            cache_embeddings=True,
            max_cache_size=1000
        )

        engine = VectorSearchEngine(config)
        await engine.initialize()

        # Generate large batch of texts
        large_batch = [f"Document {i} with unique content about topic {i % 10}"
                      for i in range(100)]

        # Test batch processing
        embeddings = await engine.generate_batch_embeddings(large_batch)
        assert len(embeddings) == 100

        # Test clustering on large dataset
        clusters = await engine.cluster_embeddings(
            texts=large_batch,
            entity_ids=[f"doc_{i}" for i in range(100)],
            n_clusters=10
        )

        if clusters:  # Only test if clustering is available
            assert len(clusters) == 10
            total_entities = sum(cluster.size for cluster in clusters.values())
            assert total_entities == 100

    async def test_data_consistency_validation(self):
        """Test data consistency validation across systems."""
        # Mock scenario where relational and graph data might diverge
        sync_config = SyncConfig(
            validate_before_sync=True,
            auto_resolve_conflicts=True
        )

        # Mock components
        mock_graph_ops = Mock(spec=GraphOperations)
        mock_vector_engine = Mock(spec=VectorSearchEngine)

        synchronizer = DataSynchronizer(sync_config, mock_graph_ops, mock_vector_engine)
        await synchronizer.initialize()

        # Test validation
        validation_results = await synchronizer.validate_data_consistency()

        assert "total_entities_checked" in validation_results
        assert "consistent_entities" in validation_results
        assert "validation_timestamp" in validation_results


# Performance and load testing
class TestPerformance:
    """Performance tests for graph operations."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent graph operations."""
        # Mock SurrealDB manager
        mock_manager = Mock()
        mock_manager.execute_query = AsyncMock(return_value=[{"id": "test"}])

        graph_ops = GraphOperations(mock_manager)
        await graph_ops.initialize()

        # Create multiple concurrent operations
        tasks = []
        for i in range(10):
            node = GraphNode(
                id=f"concurrent_node_{i}",
                type=NodeType.AGENT,
                properties={"name": f"Agent {i}"}
            )
            tasks.append(graph_ops.create_node(node))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (or at least not raise exceptions)
        assert len(results) == 10
        assert not any(isinstance(result, Exception) for result in results)

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of data."""
        config = VectorSearchConfig(batch_size=100)
        engine = VectorSearchEngine(config)
        await engine.initialize()

        # Generate large dataset
        texts = [f"Large dataset item {i} with content" for i in range(500)]

        start_time = time.time()
        embeddings = await engine.generate_batch_embeddings(texts)
        processing_time = time.time() - start_time

        assert len(embeddings) == 500
        # Should complete in reasonable time (adjust threshold as needed)
        assert processing_time < 30.0  # 30 seconds max

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        config = VectorSearchConfig(
            max_cache_size=100,  # Limited cache size
            cache_embeddings=True
        )
        engine = VectorSearchEngine(config)
        await engine.initialize()

        # Generate more embeddings than cache can hold
        for i in range(200):
            text = f"Memory test text {i}"
            await engine.generate_embedding(text)

        # Cache should not exceed max size
        assert len(engine.embedding_cache) <= config.max_cache_size

        # Should still be functional
        stats = engine.get_statistics()
        assert stats["embeddings_generated"] == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
