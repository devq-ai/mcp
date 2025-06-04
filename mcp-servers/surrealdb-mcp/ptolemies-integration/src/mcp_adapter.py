#!/usr/bin/env python3
"""
MCP adapter for SurrealDB integration with Ptolemies
"""

import json
import logging
from typing import Dict, List, Any, Optional

class SurrealDBMCPAdapter:
    """Adapter for exposing SurrealDB operations via MCP."""
    
    def __init__(self, client):
        """Initialize the MCP adapter."""
        self.client = client
        self.logger = logging.getLogger("surrealdb_mcp_adapter")
        
    async def handle_request(self, request: Dict) -> Dict:
        """Handle an MCP request."""
        tool = request.get("tool")
        operation = request.get("operation")
        parameters = request.get("parameters", {})
        
        if tool != "surrealdb":
            return {"error": {"message": "Invalid tool", "code": "invalid_tool"}}
        
        if operation == "query":
            return await self._handle_query(parameters)
        elif operation == "store":
            return await self._handle_store(parameters)
        elif operation == "retrieve":
            return await self._handle_retrieve(parameters)
        else:
            return {"error": {"message": "Invalid operation", "code": "invalid_operation"}}
    
    async def _handle_query(self, parameters: Dict) -> Dict:
        """Handle a query operation."""
        query = parameters.get("query")
        params = parameters.get("params", {})
        
        if not query:
            return {"error": {"message": "Query is required", "code": "missing_parameter"}}
        
        try:
            result = await self.client.execute_query(query, params)
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {"error": {"message": str(e), "code": "query_error"}}
    
    async def _handle_store(self, parameters: Dict) -> Dict:
        """Handle a store operation."""
        item = parameters.get("item")
        
        if not item:
            return {"error": {"message": "Item is required", "code": "missing_parameter"}}
        
        try:
            result = await self.client.store_knowledge_item(item)
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error storing knowledge item: {e}")
            return {"error": {"message": str(e), "code": "store_error"}}
    
    async def _handle_retrieve(self, parameters: Dict) -> Dict:
        """Handle a retrieve operation."""
        item_id = parameters.get("id")
        
        if not item_id:
            return {"error": {"message": "Item ID is required", "code": "missing_parameter"}}
        
        try:
            result = await self.client.retrieve_knowledge_item(item_id)
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge item: {e}")
            return {"error": {"message": str(e), "code": "retrieve_error"}}
