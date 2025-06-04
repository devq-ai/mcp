#!/usr/bin/env python3
"""
Ptolemies Knowledge Base Integration for Crawl4AI

This module provides integration between Crawl4AI and the Ptolemies Knowledge Base
for storing crawled content and managing knowledge items.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from surrealdb import Surreal

logger = logging.getLogger("crawl4ai.ptolemies_integration")

class PtolemiesIntegration:
    """Integration with Ptolemies Knowledge Base."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Ptolemies integration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.db_client = None
        self.logger = logger
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "ptolemies-integration/config.json"
            )
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            
            # Use default config as fallback
            return {
                "surrealdb": {
                    "host": "localhost",
                    "port": 8000,
                    "user": "root",
                    "password": "root",
                    "namespace": "ptolemies",
                    "database": "knowledge"
                },
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "dimensions": 1536
                }
            }
    
    async def connect(self) -> None:
        """Connect to SurrealDB."""
        surrealdb_config = self.config.get("surrealdb", {})
        
        host = surrealdb_config.get("host", "localhost")
        port = surrealdb_config.get("port", 8000)
        user = surrealdb_config.get("user", "root")
        password = surrealdb_config.get("password", "root")
        namespace = surrealdb_config.get("namespace", "ptolemies")
        database = surrealdb_config.get("database", "knowledge")
        
        url = f"http://{host}:{port}"
        
        try:
            # Create a SurrealDB client instance
            self.db_client = Surreal(url)
            
            # Connect to the database
            await self.db_client.connect()
            
            # Authenticate with credentials
            await self.db_client.signin({"user": user, "pass": password})
            
            # Use the specified namespace and database
            await self.db_client.use(namespace, database)
            
            self.logger.info(f"Connected to SurrealDB at {url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to SurrealDB: {str(e)}")
            raise ConnectionError(f"Failed to connect to SurrealDB: {str(e)}")
    
    async def ensure_connected(self) -> None:
        """Ensure connection to SurrealDB is established."""
        if not self.db_client:
            await self.connect()
    
    async def store_knowledge_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """Store knowledge items in the Ptolemies Knowledge Base.
        
        Args:
            items: List of knowledge items to store
            
        Returns:
            List of created item IDs
        """
        await self.ensure_connected()
        
        item_ids = []
        
        for item in items:
            try:
                # Format the item for SurrealDB
                knowledge_item = {
                    "title": item.get("title", "Untitled"),
                    "content": item.get("content", ""),
                    "source": item.get("source", ""),
                    "source_type": item.get("source_type", "web"),
                    "content_type": item.get("content_type", "text/html"),
                    "tags": item.get("tags", []),
                    "category": item.get("category"),
                    "metadata": item.get("metadata", {}),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                # Store in SurrealDB
                result = await self.db_client.create("knowledge_item", knowledge_item)
                
                # Extract the ID (format: knowledge_item:uuid)
                if result and isinstance(result, list) and len(result) > 0:
                    item_id = result[0].get("id")
                    if item_id:
                        item_ids.append(item_id)
                        self.logger.debug(f"Created knowledge item: {item_id}")
                
            except Exception as e:
                self.logger.error(f"Error storing knowledge item: {e}")
        
        self.logger.info(f"Stored {len(item_ids)} knowledge items in Ptolemies")
        return item_ids
    
    async def generate_embeddings(self, item_ids: List[str]) -> None:
        """Generate embeddings for knowledge items.
        
        This function triggers the embedding generation process in Ptolemies.
        
        Args:
            item_ids: List of knowledge item IDs
        """
        if not item_ids:
            self.logger.info("No knowledge items to generate embeddings for")
            return
            
        await self.ensure_connected()
        
        embedding_config = self.config.get("embedding", {})
        provider = embedding_config.get("provider", "openai")
        model = embedding_config.get("model", "text-embedding-3-small")
        
        try:
            # Execute SurrealQL query to trigger embedding generation
            query = """
            BEGIN TRANSACTION;
            
            LET $items = SELECT * FROM knowledge_item 
                WHERE id IN $item_ids 
                AND (embedding IS NONE OR embedding_model != $model);
                
            FOR $item IN $items {
                UPDATE $item SET 
                    embedding_requested = true,
                    embedding_model = $model,
                    embedding_provider = $provider,
                    embedding_requested_at = time::now()
                WHERE id = $item.id;
            }
            
            COMMIT TRANSACTION;
            """
            
            params = {
                "item_ids": item_ids,
                "model": model,
                "provider": provider
            }
            
            await self.db_client.query(query, params)
            
            self.logger.info(f"Requested embeddings for {len(item_ids)} knowledge items")
            
        except Exception as e:
            self.logger.error(f"Error requesting embeddings: {e}")
    
    async def get_knowledge_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a knowledge item by ID.
        
        Args:
            item_id: The ID of the knowledge item
            
        Returns:
            The knowledge item or None if not found
        """
        await self.ensure_connected()
        
        try:
            result = await self.db_client.select(item_id)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0]
            return None
        except Exception as e:
            self.logger.error(f"Error getting knowledge item {item_id}: {e}")
            return None