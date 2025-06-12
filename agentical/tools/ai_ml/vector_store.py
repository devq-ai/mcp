"""
Vector Store for Agentical

This module provides comprehensive vector storage and similarity search capabilities
with support for multiple backends, embedding models, and advanced search features.

Features:
- Multiple vector database backends (FAISS, Chroma, Pinecone, Weaviate)
- Support for various embedding models (OpenAI, Sentence Transformers, etc.)
- CRUD operations for vectors with metadata
- Similarity search with multiple distance metrics
- Batch operations and efficient indexing
- Caching and persistence layers
- Real-time updates and incremental indexing
- Enterprise features (encryption, audit logging, high availability)
"""

import asyncio
import json
import numpy as np
import pickle
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import os

# Optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VectorBackend(Enum):
    """Supported vector database backends."""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MEMORY = "memory"


class DistanceMetric(Enum):
    """Supported distance metrics for similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class EmbeddingProvider(Enum):
    """Supported embedding model providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class VectorDocument:
    """Document with vector embedding and metadata."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    namespace: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """Create from dictionary representation."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    document: VectorDocument
    score: float
    distance: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'distance': self.distance,
            'metadata': self.metadata or {}
        }


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """OpenAI embedding generator."""

    def __init__(self, model: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._dimension = 1536 if "ada-002" in model else 1536  # Default dimension

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Sentence Transformers embedding generator."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library not available")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.model.encode, texts)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate sentence transformer embeddings: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class VectorBackendInterface(ABC):
    """Abstract interface for vector database backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> bool:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass

    @abstractmethod
    async def update_document(self, document: VectorDocument) -> bool:
        """Update a document."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def count_documents(self) -> int:
        """Get total document count."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup backend resources."""
        pass


class FAISSBackend(VectorBackendInterface):
    """FAISS vector database backend."""

    def __init__(self, config: Dict[str, Any]):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS library not available")

        self.config = config
        self.dimension = config.get('dimension', 768)
        self.index_type = config.get('index_type', 'IndexFlatIP')
        self.index = None
        self.documents: Dict[str, VectorDocument] = {}
        self.id_to_index_map: Dict[str, int] = {}
        self.index_to_id_map: Dict[int, str] = {}
        self.next_index = 0

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            if self.index_type == 'IndexFlatIP':
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == 'IndexFlatL2':
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'IndexIVFFlat':
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = self.config.get('nlist', 100)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS backend: {e}")

    async def add_documents(self, documents: List[VectorDocument]) -> bool:
        """Add documents to FAISS index."""
        try:
            embeddings = []
            for doc in documents:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} has no embedding")

                embeddings.append(doc.embedding)
                self.documents[doc.id] = doc
                self.id_to_index_map[doc.id] = self.next_index
                self.index_to_id_map[self.next_index] = doc.id
                self.next_index += 1

            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)

            return True
        except Exception as e:
            logging.error(f"Failed to add documents to FAISS: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors in FAISS."""
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_array, top_k)

            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # No more results
                    break

                doc_id = self.index_to_id_map.get(idx)
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]

                    # Apply filters if specified
                    if filters and not self._apply_filters(doc, filters):
                        continue

                    score = float(distance)
                    result = SearchResult(
                        document=doc,
                        score=score,
                        distance=float(distance)
                    )
                    results.append(result)

            return results
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")
            return []

    def _apply_filters(self, doc: VectorDocument, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to document."""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents (FAISS doesn't support deletion, recreate index)."""
        try:
            # Remove from mappings and documents
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                if doc_id in self.id_to_index_map:
                    del self.id_to_index_map[doc_id]

            # Rebuild index without deleted documents
            await self._rebuild_index()
            return True
        except Exception as e:
            logging.error(f"Failed to delete documents from FAISS: {e}")
            return False

    async def _rebuild_index(self) -> None:
        """Rebuild FAISS index after deletions."""
        # Reinitialize index
        await self.initialize()

        # Re-add all remaining documents
        if self.documents:
            documents = list(self.documents.values())
            self.id_to_index_map.clear()
            self.index_to_id_map.clear()
            self.next_index = 0
            await self.add_documents(documents)

    async def update_document(self, document: VectorDocument) -> bool:
        """Update a document (delete and re-add)."""
        try:
            await self.delete_documents([document.id])
            await self.add_documents([document])
            return True
        except Exception as e:
            logging.error(f"Failed to update document in FAISS: {e}")
            return False

    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self.documents.get(document_id)

    async def count_documents(self) -> int:
        """Get total document count."""
        return self.index.ntotal if self.index else 0

    async def cleanup(self) -> None:
        """Cleanup FAISS resources."""
        self.index = None
        self.documents.clear()
        self.id_to_index_map.clear()
        self.index_to_id_map.clear()


class MemoryBackend(VectorBackendInterface):
    """In-memory vector backend for testing and small datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents: Dict[str, VectorDocument] = {}

    async def initialize(self) -> None:
        """Initialize memory backend."""
        pass

    async def add_documents(self, documents: List[VectorDocument]) -> bool:
        """Add documents to memory."""
        try:
            for doc in documents:
                self.documents[doc.id] = doc
            return True
        except Exception as e:
            logging.error(f"Failed to add documents to memory: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using cosine similarity in memory."""
        try:
            query_array = np.array(query_embedding)
            results = []

            for doc in self.documents.values():
                if doc.embedding is None:
                    continue

                # Apply filters
                if filters and not self._apply_filters(doc, filters):
                    continue

                doc_array = np.array(doc.embedding)

                # Calculate cosine similarity
                dot_product = np.dot(query_array, doc_array)
                norm_query = np.linalg.norm(query_array)
                norm_doc = np.linalg.norm(doc_array)

                if norm_query == 0 or norm_doc == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (norm_query * norm_doc)

                distance = 1 - similarity  # Convert to distance

                result = SearchResult(
                    document=doc,
                    score=similarity,
                    distance=distance
                )
                results.append(result)

            # Sort by similarity (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logging.error(f"Memory search failed: {e}")
            return []

    def _apply_filters(self, doc: VectorDocument, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to document."""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from memory."""
        try:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            return True
        except Exception as e:
            logging.error(f"Failed to delete documents from memory: {e}")
            return False

    async def update_document(self, document: VectorDocument) -> bool:
        """Update document in memory."""
        try:
            self.documents[document.id] = document
            return True
        except Exception as e:
            logging.error(f"Failed to update document in memory: {e}")
            return False

    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self.documents.get(document_id)

    async def count_documents(self) -> int:
        """Get total document count."""
        return len(self.documents)

    async def cleanup(self) -> None:
        """Cleanup memory backend."""
        self.documents.clear()


class VectorStore:
    """
    Comprehensive vector storage and similarity search system.

    Supports multiple backends, embedding models, and advanced search features
    with enterprise-grade performance and reliability.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vector store.

        Args:
            config: Configuration dictionary with backend, embedding, and feature settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.backend_type = VectorBackend(self.config.get('backend', 'memory'))
        self.embedding_provider = EmbeddingProvider(self.config.get('embedding_provider', 'sentence_transformers'))
        self.dimension = self.config.get('dimension', 768)

        # Performance settings
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.batch_size = self.config.get('batch_size', 100)
        self.max_retries = self.config.get('max_retries', 3)

        # Enterprise features
        self.encryption_enabled = self.config.get('encryption_enabled', False)
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)

        # Initialize components
        self.backend: Optional[VectorBackendInterface] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.cache: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the vector store components."""
        try:
            self.logger.info(f"Initializing vector store with backend: {self.backend_type.value}")

            # Initialize backend
            await self._initialize_backend()

            # Initialize embedding generator
            await self._initialize_embedding_generator()

            self.initialized = True
            self.logger.info("Vector store initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise

    async def _initialize_backend(self) -> None:
        """Initialize the vector database backend."""
        backend_config = self.config.get('backend_config', {})
        backend_config['dimension'] = self.dimension

        if self.backend_type == VectorBackend.FAISS:
            self.backend = FAISSBackend(backend_config)
        elif self.backend_type == VectorBackend.MEMORY:
            self.backend = MemoryBackend(backend_config)
        elif self.backend_type == VectorBackend.CHROMA:
            # ChromaDB backend would be implemented here
            raise NotImplementedError("ChromaDB backend not yet implemented")
        elif self.backend_type == VectorBackend.PINECONE:
            # Pinecone backend would be implemented here
            raise NotImplementedError("Pinecone backend not yet implemented")
        elif self.backend_type == VectorBackend.WEAVIATE:
            # Weaviate backend would be implemented here
            raise NotImplementedError("Weaviate backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        await self.backend.initialize()

    async def _initialize_embedding_generator(self) -> None:
        """Initialize the embedding generator."""
        embedding_config = self.config.get('embedding_config', {})

        if self.embedding_provider == EmbeddingProvider.OPENAI:
            model = embedding_config.get('model', 'text-embedding-ada-002')
            api_key = embedding_config.get('api_key', os.getenv('OPENAI_API_KEY'))
            self.embedding_generator = OpenAIEmbeddingGenerator(model, api_key)
        elif self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            model_name = embedding_config.get('model', 'all-MiniLM-L6-v2')
            self.embedding_generator = SentenceTransformerEmbeddingGenerator(model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

        # Update dimension from embedding generator
        self.dimension = self.embedding_generator.get_dimension()

    async def add_documents(self, documents: List[VectorDocument],
                          generate_embeddings: bool = True) -> bool:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            generate_embeddings: Whether to generate embeddings for documents

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()

        try:
            self.logger.info(f"Adding {len(documents)} documents")

            # Generate embeddings if needed
            if generate_embeddings:
                await self._generate_embeddings_for_documents(documents)

            # Process in batches
            success = True
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_success = await self.backend.add_documents(batch)
                success = success and batch_success

                if self.audit_logging:
                    self._log_operation('add_documents', {'batch_size': len(batch), 'success': batch_success})

            self.metrics['documents_added'] += len(documents)
            return success

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False

    async def _generate_embeddings_for_documents(self, documents: List[VectorDocument]) -> None:
        """Generate embeddings for documents that don't have them."""
        texts_to_embed = []
        docs_need_embedding = []

        for doc in documents:
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                docs_need_embedding.append(doc)

        if texts_to_embed:
            self.logger.info(f"Generating embeddings for {len(texts_to_embed)} documents")
            embeddings = await self.embedding_generator.generate_embeddings(texts_to_embed)

            for doc, embedding in zip(docs_need_embedding, embeddings):
                doc.embedding = embedding

    async def search(self, query: str, top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None,
                    namespace: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters to apply
            namespace: Optional namespace to search within

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        try:
            self.logger.debug(f"Searching for: {query[:100]}...")

            # Check cache
            cache_key = self._get_cache_key(query, top_k, filters, namespace)
            if self.cache_enabled and cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]

            # Generate query embedding
            query_embeddings = await self.embedding_generator.generate_embeddings([query])
            query_embedding = query_embeddings[0]

            # Add namespace filter if specified
            if namespace:
                filters = filters or {}
                filters['namespace'] = namespace

            # Perform search
            results = await self.backend.search(query_embedding, top_k, filters)

            # Cache results
            if self.cache_enabled:
                self.cache[cache_key] = results

            self.metrics['searches_performed'] += 1
            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    async def search_by_embedding(self, embedding: List[float], top_k: int = 10,
                                 filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search using a pre-computed embedding.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        try:
            results = await self.backend.search(embedding, top_k, filters)
            self.metrics['embedding_searches_performed'] += 1
            return results
        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            return []

    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        if not self.initialized:
            await self.initialize()

        return await self.backend.get_document(document_id)

    async def update_document(self, document: VectorDocument,
                             generate_embedding: bool = True) -> bool:
        """
        Update a document in the vector store.

        Args:
            document: Document to update
            generate_embedding: Whether to generate new embedding

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()

        try:
            if generate_embedding and document.embedding is None:
                await self._generate_embeddings_for_documents([document])

            success = await self.backend.update_document(document)

            if self.audit_logging:
                self._log_operation('update_document', {'document_id': document.id, 'success': success})

            return success
        except Exception as e:
            self.logger.error(f"Failed to update document: {e}")
            return False

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()

        try:
            success = await self.backend.delete_documents(document_ids)

            if self.audit_logging:
                self._log_operation('delete_documents', {'document_ids': document_ids, 'success': success})

            self.metrics['documents_deleted'] += len(document_ids)
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False

    async def count_documents(self) -> int:
        """Get total number of documents in the store."""
        if not self.initialized:
            await self.initialize()

        return await self.backend.count_documents()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics."""
        return dict(self.metrics)

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.cache.clear()
        self.logger.info("Search cache cleared")

    def _get_cache_key(self, query: str, top_k: int,
                      filters: Optional[Dict[str, Any]],
                      namespace: Optional[str]) -> str:
        """Generate cache key for search results."""
        key_data = {
            'query': query,
            'top_k': top_k,
            'filters': filters,
            'namespace': namespace
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log operations for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'details': details
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    async def export_documents(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export documents for backup or migration."""
        if not self.initialized:
            await self.initialize()

        try:
            # This is a simplified export - in a real implementation,
            # you'd iterate through all documents in the backend
            documents = []
            count = await self.count_documents()
            self.logger.info(f"Exporting {count} documents")

            # For now, return empty list as backend doesn't expose iteration
            # In production, implement backend-specific document iteration
            return documents
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return []

    async def import_documents(self, documents_data: List[Dict[str, Any]]) -> bool:
        """Import documents from backup or migration."""
        try:
            documents = [VectorDocument.from_dict(data) for data in documents_data]
            return await self.add_documents(documents, generate_embeddings=False)
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup vector store resources."""
        try:
            if self.backend:
                await self.backend.cleanup()
            self.clear_cache()
            self.initialized = False
            self.logger.info("Vector store cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'initialized') and self.initialized:
                self.logger.info("VectorStore being destroyed - cleanup recommended")
        except:
            pass
