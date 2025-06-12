"""
Vector Search Utilities for Agentical

This module provides comprehensive vector search capabilities for semantic search,
similarity matching, and knowledge retrieval in the SurrealDB graph database.

Features:
- Vector embedding generation and management
- Semantic similarity search
- Knowledge entity clustering
- Vector index optimization
- Multi-modal embedding support
- Similarity caching and performance optimization
"""

import asyncio
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import pickle
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

try:
    # Optional dependencies for enhanced embedding capabilities
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    CUSTOM = "custom"


class SimilarityMetric(str, Enum):
    """Similarity metrics for vector comparison."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations."""
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_BERT
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    cache_embeddings: bool = True
    cache_similarity_results: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 10000
    batch_size: int = 32
    embedding_dimension: Optional[int] = None
    normalize_vectors: bool = True
    enable_clustering: bool = True
    cluster_cache_ttl: int = 7200


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    entity_id: str
    content: str
    similarity_score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    search_time_ms: float = 0.0


@dataclass
class EmbeddingCache:
    """Cache entry for embeddings."""
    embedding: List[float]
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClusterInfo:
    """Information about vector clusters."""
    cluster_id: int
    centroid: List[float]
    entity_ids: List[str]
    intra_cluster_similarity: float
    created_at: datetime
    size: int = 0


class VectorSearchEngine:
    """
    High-performance vector search engine for semantic similarity.

    Provides embedding generation, similarity search, clustering, and caching
    capabilities for knowledge entities in the graph database.
    """

    def __init__(self, config: VectorSearchConfig = None):
        self.config = config or VectorSearchConfig()
        self.embedding_cache: Dict[str, EmbeddingCache] = {}
        self.similarity_cache: Dict[str, Tuple[List[VectorSearchResult], datetime]] = {}
        self.clusters: Dict[int, ClusterInfo] = {}
        self.embedding_model = None
        self.is_initialized = False

        # Performance metrics
        self.stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "searches_performed": 0,
            "average_search_time_ms": 0.0,
            "total_search_time_ms": 0.0
        }

    async def initialize(self):
        """Initialize the vector search engine."""
        with logfire.span("Initialize vector search engine"):
            try:
                if self.config.embedding_model == EmbeddingModel.SENTENCE_BERT:
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                        self.config.embedding_dimension = 384
                    else:
                        logfire.warning("SentenceTransformers not available, using simple embedding")
                        self.embedding_model = self._simple_text_embedding
                        self.config.embedding_dimension = 300

                elif self.config.embedding_model == EmbeddingModel.SENTENCE_BERT_LARGE:
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                        self.config.embedding_dimension = 768
                    else:
                        logfire.warning("SentenceTransformers not available, using simple embedding")
                        self.embedding_model = self._simple_text_embedding
                        self.config.embedding_dimension = 300

                else:
                    # For OpenAI or custom models, use API-based embedding
                    self.embedding_model = self._api_embedding
                    self.config.embedding_dimension = 1536  # Default for OpenAI models

                self.is_initialized = True
                logfire.info("Vector search engine initialized",
                           model=self.config.embedding_model.value,
                           dimension=self.config.embedding_dimension)

            except Exception as e:
                logfire.error("Failed to initialize vector search engine", error=str(e))
                raise

    def _simple_text_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Simple text embedding fallback when advanced models aren't available."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Simple character-based embedding
            text_lower = text.lower()
            embedding = np.zeros(self.config.embedding_dimension)

            # Character frequency features
            for i, char in enumerate(text_lower[:self.config.embedding_dimension]):
                if char.isalnum():
                    embedding[i] = ord(char) / 255.0

            # Add some text statistics
            if len(embedding) > 10:
                embedding[-10] = len(text) / 1000.0  # Text length
                embedding[-9] = text.count(' ') / len(text) if len(text) > 0 else 0  # Word density
                embedding[-8] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0  # Uppercase ratio
                embedding[-7] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0  # Digit ratio
                embedding[-6] = len(set(text.lower())) / len(text) if len(text) > 0 else 0  # Character diversity

            if self.config.normalize_vectors:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding.tolist())

        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def _api_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using API-based models (OpenAI, etc.)."""
        # Placeholder for API-based embedding
        # In production, implement actual API calls to OpenAI or other services
        logfire.warning("API-based embedding not implemented, using fallback")
        return self._simple_text_embedding(texts)

    async def generate_embedding(self, text: str, force_regenerate: bool = False) -> List[float]:
        """Generate or retrieve cached embedding for text."""
        if not self.is_initialized:
            await self.initialize()

        # Create cache key
        cache_key = hashlib.md5(f"{text}_{self.config.embedding_model.value}".encode()).hexdigest()

        # Check cache first
        if not force_regenerate and self.config.cache_embeddings and cache_key in self.embedding_cache:
            cache_entry = self.embedding_cache[cache_key]

            # Check if cache is still valid
            if datetime.utcnow() - cache_entry.created_at < timedelta(seconds=self.config.cache_ttl_seconds):
                cache_entry.access_count += 1
                cache_entry.last_accessed = datetime.utcnow()
                self.stats["cache_hits"] += 1
                return cache_entry.embedding

        # Generate new embedding
        start_time = time.time()

        with logfire.span("Generate embedding", text_length=len(text)):
            try:
                if hasattr(self.embedding_model, 'encode'):
                    # SentenceTransformers model
                    embedding = self.embedding_model.encode([text])[0].tolist()
                else:
                    # Custom embedding function
                    embedding = self.embedding_model(text)

                if self.config.normalize_vectors and isinstance(embedding, list):
                    embedding_array = np.array(embedding)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding = (embedding_array / norm).tolist()

                # Cache the embedding
                if self.config.cache_embeddings:
                    self.embedding_cache[cache_key] = EmbeddingCache(
                        embedding=embedding,
                        created_at=datetime.utcnow()
                    )

                    # Manage cache size
                    if len(self.embedding_cache) > self.config.max_cache_size:
                        await self._cleanup_embedding_cache()

                self.stats["embeddings_generated"] += 1
                self.stats["cache_misses"] += 1

                generation_time = (time.time() - start_time) * 1000
                logfire.info("Embedding generated",
                           cache_key=cache_key,
                           dimension=len(embedding),
                           generation_time_ms=generation_time)

                return embedding

            except Exception as e:
                logfire.error("Failed to generate embedding", text=text[:100], error=str(e))
                raise

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not self.is_initialized:
            await self.initialize()

        with logfire.span("Generate batch embeddings", batch_size=len(texts)):
            embeddings = []

            # Process in batches for memory efficiency
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]

                # Check cache for each text in batch
                batch_embeddings = []
                texts_to_generate = []
                cache_keys = []

                for text in batch:
                    cache_key = hashlib.md5(f"{text}_{self.config.embedding_model.value}".encode()).hexdigest()
                    cache_keys.append(cache_key)

                    if (self.config.cache_embeddings and
                        cache_key in self.embedding_cache and
                        datetime.utcnow() - self.embedding_cache[cache_key].created_at <
                        timedelta(seconds=self.config.cache_ttl_seconds)):

                        batch_embeddings.append(self.embedding_cache[cache_key].embedding)
                        self.stats["cache_hits"] += 1
                    else:
                        batch_embeddings.append(None)
                        texts_to_generate.append((len(batch_embeddings) - 1, text, cache_key))

                # Generate embeddings for uncached texts
                if texts_to_generate:
                    texts_only = [item[1] for item in texts_to_generate]

                    if hasattr(self.embedding_model, 'encode'):
                        # SentenceTransformers batch processing
                        generated = self.embedding_model.encode(texts_only)
                    else:
                        # Custom embedding function
                        generated = [self.embedding_model(text) for text in texts_only]

                    # Update batch_embeddings and cache
                    for (idx, text, cache_key), embedding in zip(texts_to_generate, generated):
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()

                        if self.config.normalize_vectors:
                            embedding_array = np.array(embedding)
                            norm = np.linalg.norm(embedding_array)
                            if norm > 0:
                                embedding = (embedding_array / norm).tolist()

                        batch_embeddings[idx] = embedding

                        # Cache the embedding
                        if self.config.cache_embeddings:
                            self.embedding_cache[cache_key] = EmbeddingCache(
                                embedding=embedding,
                                created_at=datetime.utcnow()
                            )

                    self.stats["embeddings_generated"] += len(texts_to_generate)
                    self.stats["cache_misses"] += len(texts_to_generate)

                embeddings.extend(batch_embeddings)

            return embeddings

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate similarity between two embeddings."""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        if self.config.similarity_metric == SimilarityMetric.COSINE:
            # Cosine similarity
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        elif self.config.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(arr1 - arr2)
            return 1.0 / (1.0 + distance)

        elif self.config.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product
            return np.dot(arr1, arr2)

        elif self.config.similarity_metric == SimilarityMetric.MANHATTAN:
            # Manhattan distance (converted to similarity)
            distance = np.sum(np.abs(arr1 - arr2))
            return 1.0 / (1.0 + distance)

        else:
            raise ValueError(f"Unsupported similarity metric: {self.config.similarity_metric}")

    async def search_similar(self, query_text: str,
                           candidate_texts: List[str],
                           candidate_ids: List[str] = None,
                           top_k: int = 10,
                           threshold: float = 0.0,
                           metadata: List[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar texts using vector embeddings."""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        with logfire.span("Vector similarity search",
                         query_length=len(query_text),
                         candidate_count=len(candidate_texts),
                         top_k=top_k):

            # Generate cache key for this search
            search_key = hashlib.md5(
                f"{query_text}_{len(candidate_texts)}_{top_k}_{threshold}_{self.config.similarity_metric.value}".encode()
            ).hexdigest()

            # Check similarity cache
            if (self.config.cache_similarity_results and
                search_key in self.similarity_cache):

                cached_results, cache_time = self.similarity_cache[search_key]
                if datetime.utcnow() - cache_time < timedelta(seconds=self.config.cache_ttl_seconds):
                    logfire.info("Returning cached similarity results", search_key=search_key)
                    return cached_results

            try:
                # Generate query embedding
                query_embedding = await self.generate_embedding(query_text)

                # Generate candidate embeddings
                candidate_embeddings = await self.generate_batch_embeddings(candidate_texts)

                # Calculate similarities
                results = []
                for i, candidate_embedding in enumerate(candidate_embeddings):
                    similarity = self.calculate_similarity(query_embedding, candidate_embedding)
                    distance = 1.0 - similarity if self.config.similarity_metric == SimilarityMetric.COSINE else similarity

                    if similarity >= threshold:
                        result = VectorSearchResult(
                            entity_id=candidate_ids[i] if candidate_ids else str(i),
                            content=candidate_texts[i],
                            similarity_score=similarity,
                            distance=distance,
                            metadata=metadata[i] if metadata else {},
                            embedding=candidate_embedding if len(candidate_embedding) <= 100 else None  # Include only for small embeddings
                        )
                        results.append(result)

                # Sort by similarity score and take top_k
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                results = results[:top_k]

                # Add search time to results
                search_time_ms = (time.time() - start_time) * 1000
                for result in results:
                    result.search_time_ms = search_time_ms

                # Cache results
                if self.config.cache_similarity_results:
                    self.similarity_cache[search_key] = (results, datetime.utcnow())

                    # Manage cache size
                    if len(self.similarity_cache) > self.config.max_cache_size:
                        await self._cleanup_similarity_cache()

                # Update statistics
                self.stats["searches_performed"] += 1
                self.stats["total_search_time_ms"] += search_time_ms
                self.stats["average_search_time_ms"] = (
                    self.stats["total_search_time_ms"] / self.stats["searches_performed"]
                )

                logfire.info("Similarity search completed",
                           results_count=len(results),
                           search_time_ms=search_time_ms,
                           top_similarity=results[0].similarity_score if results else 0.0)

                return results

            except Exception as e:
                logfire.error("Similarity search failed",
                            query=query_text[:100],
                            error=str(e))
                raise

    async def cluster_embeddings(self, texts: List[str],
                               entity_ids: List[str] = None,
                               n_clusters: int = None) -> Dict[int, ClusterInfo]:
        """Cluster embeddings using K-means clustering."""
        if not SKLEARN_AVAILABLE:
            logfire.warning("scikit-learn not available, skipping clustering")
            return {}

        if not self.is_initialized:
            await self.initialize()

        with logfire.span("Cluster embeddings",
                         text_count=len(texts),
                         n_clusters=n_clusters):

            try:
                # Generate embeddings for all texts
                embeddings = await self.generate_batch_embeddings(texts)
                embeddings_array = np.array(embeddings)

                # Determine optimal number of clusters if not specified
                if n_clusters is None:
                    n_clusters = min(max(2, len(texts) // 10), 20)

                # Perform K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_array)

                # Create cluster information
                clusters = {}
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_embeddings = embeddings_array[cluster_mask]
                    cluster_entity_ids = [entity_ids[i] if entity_ids else str(i)
                                        for i, mask in enumerate(cluster_mask) if mask]

                    # Calculate intra-cluster similarity
                    if len(cluster_embeddings) > 1:
                        similarities = []
                        for i in range(len(cluster_embeddings)):
                            for j in range(i + 1, len(cluster_embeddings)):
                                sim = self.calculate_similarity(
                                    cluster_embeddings[i].tolist(),
                                    cluster_embeddings[j].tolist()
                                )
                                similarities.append(sim)
                        intra_similarity = np.mean(similarities) if similarities else 0.0
                    else:
                        intra_similarity = 1.0

                    clusters[cluster_id] = ClusterInfo(
                        cluster_id=cluster_id,
                        centroid=kmeans.cluster_centers_[cluster_id].tolist(),
                        entity_ids=cluster_entity_ids,
                        intra_cluster_similarity=intra_similarity,
                        created_at=datetime.utcnow(),
                        size=len(cluster_entity_ids)
                    )

                # Cache clusters
                if self.config.enable_clustering:
                    self.clusters.update(clusters)

                logfire.info("Clustering completed",
                           n_clusters=n_clusters,
                           average_cluster_size=np.mean([info.size for info in clusters.values()]))

                return clusters

            except Exception as e:
                logfire.error("Clustering failed", error=str(e))
                return {}

    async def find_cluster_for_text(self, text: str) -> Optional[int]:
        """Find the most appropriate cluster for a given text."""
        if not self.clusters:
            return None

        with logfire.span("Find cluster for text"):
            try:
                # Generate embedding for the text
                text_embedding = await self.generate_embedding(text)

                # Find closest cluster centroid
                best_cluster_id = None
                best_similarity = -1.0

                for cluster_id, cluster_info in self.clusters.items():
                    similarity = self.calculate_similarity(text_embedding, cluster_info.centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster_id = cluster_id

                logfire.info("Cluster found for text",
                           cluster_id=best_cluster_id,
                           similarity=best_similarity)

                return best_cluster_id

            except Exception as e:
                logfire.error("Failed to find cluster for text", error=str(e))
                return None

    async def _cleanup_embedding_cache(self):
        """Clean up old entries from embedding cache."""
        # Remove oldest entries based on last access time
        sorted_entries = sorted(
            self.embedding_cache.items(),
            key=lambda x: x[1].last_accessed
        )

        # Remove oldest 20% of entries
        remove_count = len(sorted_entries) // 5
        for cache_key, _ in sorted_entries[:remove_count]:
            del self.embedding_cache[cache_key]

        logfire.info("Embedding cache cleaned", removed_entries=remove_count)

    async def _cleanup_similarity_cache(self):
        """Clean up old entries from similarity cache."""
        # Remove oldest entries
        sorted_entries = sorted(
            self.similarity_cache.items(),
            key=lambda x: x[1][1]  # Sort by cache time
        )

        # Remove oldest 20% of entries
        remove_count = len(sorted_entries) // 5
        for cache_key, _ in sorted_entries[:remove_count]:
            del self.similarity_cache[cache_key]

        logfire.info("Similarity cache cleaned", removed_entries=remove_count)

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the vector search engine."""
        return {
            **self.stats,
            "cache_hit_ratio": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0.0
            ),
            "embedding_cache_size": len(self.embedding_cache),
            "similarity_cache_size": len(self.similarity_cache),
            "cluster_count": len(self.clusters),
            "config": {
                "embedding_model": self.config.embedding_model.value,
                "similarity_metric": self.config.similarity_metric.value,
                "embedding_dimension": self.config.embedding_dimension,
                "cache_enabled": self.config.cache_embeddings
            }
        }

    async def clear_caches(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.similarity_cache.clear()
        self.clusters.clear()
        logfire.info("All caches cleared")

    async def export_embeddings(self, format: str = "json") -> Union[str, bytes]:
        """Export cached embeddings for backup or analysis."""
        with logfire.span("Export embeddings", format=format):
            export_data = {
                "config": {
                    "embedding_model": self.config.embedding_model.value,
                    "embedding_dimension": self.config.embedding_dimension,
                    "similarity_metric": self.config.similarity_metric.value
                },
                "embeddings": {
                    cache_key: {
                        "embedding": cache_entry.embedding,
                        "created_at": cache_entry.created_at.isoformat(),
                        "access_count": cache_entry.access_count
                    }
                    for cache_key, cache_entry in self.embedding_cache.items()
                },
                "clusters": {
                    str(cluster_id): {
                        "centroid": cluster_info.centroid,
                        "entity_ids": cluster_info.entity_ids,
                        "size": cluster_info.size,
                        "intra_cluster_similarity": cluster_info.intra_cluster_similarity,
                        "created_at": cluster_info.created_at.isoformat()
                    }
                    for cluster_id, cluster_info in self.clusters.items()
                },
                "statistics": self.get_statistics(),
                "export_timestamp": datetime.utcnow().isoformat()
            }

            if format == "json":
                return json.dumps(export_data, indent=2)
            elif format == "pickle":
                return pickle.dumps(export_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")


# Factory function for creating vector search engine
async def create_vector_search_engine(config: VectorSearchConfig = None) -> VectorSearchEngine:
    """Create and initialize a vector search engine."""
    engine = VectorSearchEngine(config)
    await engine.initialize()
    return engine


# Context manager for vector search operations
@asynccontextmanager
async def vector_search_session(config: VectorSearchConfig = None):
    """Context manager for vector search operations."""
    engine = await create_vector_search_engine(config)
    try:
        yield engine
    finally:
        # Cleanup if needed
        pass


# Utility functions for common vector operations
async def quick_similarity_search(query: str,
                                candidates: List[str],
                                top_k: int = 5) -> List[VectorSearchResult]:
    """Quick similarity search with default configuration."""
    async with vector_search_session() as engine:
        return await engine.search_similar(query, candidates, top_k=top_k)


async def batch_similarity_matrix(texts: List[str]) -> np.ndarray:
    """Generate similarity matrix for a list of texts."""
    async with vector_search_session() as engine:
        embeddings = await engine.generate_batch_embeddings(texts)

        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                similarity = engine.calculate_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix


def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Fast cosine similarity calculation using NumPy."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product != 0 else 0.0
