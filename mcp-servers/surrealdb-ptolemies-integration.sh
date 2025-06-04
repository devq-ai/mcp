#!/bin/bash
# Integration script for SurrealDB MCP and Ptolemies Knowledge Base

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== SurrealDB MCP + Ptolemies Integration =====${NC}"
echo ""

# Base directories
MCP_BASE="/Users/dionedge/devqai/mcp"
PTOLEMIES_BASE="/Users/dionedge/devqai/ptolemies"
SURREALDB_MCP_DIR="${MCP_BASE}/mcp-servers/surrealdb-mcp"

# Check if directories exist
if [ ! -d "$SURREALDB_MCP_DIR" ]; then
  echo -e "${RED}SurrealDB MCP directory not found!${NC}"
  exit 1
fi

if [ ! -d "$PTOLEMIES_BASE" ]; then
  echo -e "${RED}Ptolemies directory not found!${NC}"
  exit 1
fi

echo -e "${YELLOW}Setting up SurrealDB MCP and Ptolemies integration...${NC}"

# Create integration directory
INTEGRATION_DIR="${SURREALDB_MCP_DIR}/ptolemies-integration"
mkdir -p "$INTEGRATION_DIR"

# Create symbolic link to Ptolemies
ln -sf "$PTOLEMIES_BASE" "${INTEGRATION_DIR}/ptolemies"

# Create integration README
cat > "${INTEGRATION_DIR}/README.md" << EOL
# SurrealDB MCP + Ptolemies Integration

This directory contains integration code for connecting the SurrealDB MCP server with the Ptolemies Knowledge Base system.

## Components

1. **SurrealDB MCP Server** - Provides SurrealDB database operations via MCP
2. **Ptolemies Knowledge Base** - Knowledge storage and retrieval system

## Integration Setup

### Prerequisites
- SurrealDB installed and running
- Ptolemies Knowledge Base configured
- MCP environment activated

### Configuration

Update the \`config.json\` file with the SurrealDB connection details:

\`\`\`json
{
  "surrealdb": {
    "host": "localhost",
    "port": 8000,
    "user": "root",
    "password": "root",
    "namespace": "ptolemies",
    "database": "knowledge"
  },
  "vector_db": {
    "type": "qdrant",
    "host": "localhost",
    "port": 6333
  }
}
\`\`\`

## Usage

Once configured, the SurrealDB MCP server will provide the following capabilities to Ptolemies:

1. **Graph Data Storage** - Store knowledge items and their relationships
2. **Real-time Updates** - Push updates to agents when knowledge changes
3. **Query Language** - Execute SurrealQL queries for complex data retrieval
4. **ACID Transactions** - Ensure data consistency across operations

## Architecture

\`\`\`
┌─────────────┐     ┌───────────────┐     ┌───────────────┐
│             │     │               │     │               │
│    Agent    │────▶│  MCP Client   │────▶│  SurrealDB    │
│             │     │               │     │  MCP Server   │
└─────────────┘     └───────────────┘     └───────┬───────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌───────────────┐     ┌───────────────┐
│             │     │               │     │               │
│  Knowledge  │◀───▶│   Ptolemies   │◀───▶│   SurrealDB   │
│     API     │     │  Knowledge    │     │   Database    │
│             │     │     Base      │     │               │
└─────────────┘     └───────────────┘     └───────────────┘
\`\`\`
EOL

# Create example configuration
cat > "${INTEGRATION_DIR}/config.json" << EOL
{
  "surrealdb": {
    "host": "localhost",
    "port": 8000,
    "user": "root",
    "password": "root",
    "namespace": "ptolemies",
    "database": "knowledge"
  },
  "vector_db": {
    "type": "qdrant",
    "host": "localhost",
    "port": 6333
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimensions": 1536
  },
  "mcp": {
    "host": "localhost",
    "port": 8080,
    "api_key": "your_api_key_here"
  }
}
EOL

# Create integration helper script
cat > "${INTEGRATION_DIR}/setup-integration.py" << EOL
#!/usr/bin/env python3
"""
SurrealDB MCP + Ptolemies Integration Setup
"""

import os
import json
import argparse
from pathlib import Path

def setup_integration(config_path=None, ptolemies_path=None):
    """Set up the integration between SurrealDB MCP and Ptolemies."""
    print("Setting up SurrealDB MCP + Ptolemies integration...")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Config file not found!")
            return False
    
    # Determine Ptolemies path
    if not ptolemies_path:
        ptolemies_path = os.path.join(os.path.dirname(__file__), "ptolemies")
        if os.path.islink(ptolemies_path):
            ptolemies_path = os.readlink(ptolemies_path)
    
    if not os.path.exists(ptolemies_path):
        print(f"Ptolemies path not found: {ptolemies_path}")
        return False
    
    print(f"Using Ptolemies at: {ptolemies_path}")
    
    # Create example knowledge item
    knowledge_item = {
        "title": "SurrealDB MCP Integration",
        "content": "This is a test knowledge item for the SurrealDB MCP + Ptolemies integration.",
        "tags": ["mcp", "surrealdb", "integration", "test"],
        "metadata": {
            "source": "integration_setup",
            "timestamp": "2025-06-01T12:00:00Z"
        }
    }
    
    # Save example knowledge item
    example_path = os.path.join(os.path.dirname(__file__), "example_knowledge.json")
    with open(example_path, 'w') as f:
        json.dump(knowledge_item, f, indent=2)
    
    print(f"Created example knowledge item at: {example_path}")
    
    # Create symlink to integration in Ptolemies (if it doesn't exist)
    ptolemies_integrations_dir = os.path.join(ptolemies_path, "integrations")
    if not os.path.exists(ptolemies_integrations_dir):
        os.makedirs(ptolemies_integrations_dir, exist_ok=True)
    
    integration_symlink = os.path.join(ptolemies_integrations_dir, "surrealdb-mcp")
    if not os.path.exists(integration_symlink):
        os.symlink(os.path.dirname(__file__), integration_symlink)
        print(f"Created symlink in Ptolemies integrations directory")
    
    print("Integration setup complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up SurrealDB MCP + Ptolemies integration")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--ptolemies", help="Path to Ptolemies directory")
    args = parser.parse_args()
    
    setup_integration(args.config, args.ptolemies)
EOL

chmod +x "${INTEGRATION_DIR}/setup-integration.py"

# Create example integration code
mkdir -p "${INTEGRATION_DIR}/src"

cat > "${INTEGRATION_DIR}/src/surrealdb_client.py" << EOL
#!/usr/bin/env python3
"""
SurrealDB client for Ptolemies integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional

class SurrealDBClient:
    """Client for interacting with SurrealDB."""
    
    def __init__(self, host: str, port: int, user: str, password: str, 
                 namespace: str, database: str):
        """Initialize the SurrealDB client."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.namespace = namespace
        self.database = database
        self.logger = logging.getLogger("surrealdb_client")
        
    async def connect(self) -> bool:
        """Connect to the SurrealDB server."""
        self.logger.info(f"Connecting to SurrealDB at {self.host}:{self.port}")
        # In a real implementation, this would establish a connection
        return True
        
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Dict:
        """Execute a SurrealQL query."""
        self.logger.info(f"Executing query: {query}")
        # In a real implementation, this would execute the query
        return {"result": "success", "data": []}
        
    async def store_knowledge_item(self, item: Dict) -> Dict:
        """Store a knowledge item in SurrealDB."""
        self.logger.info(f"Storing knowledge item: {item.get('title', 'Untitled')}")
        # In a real implementation, this would store the item
        return {"id": "item:123", "result": "success"}
        
    async def retrieve_knowledge_item(self, item_id: str) -> Dict:
        """Retrieve a knowledge item from SurrealDB."""
        self.logger.info(f"Retrieving knowledge item: {item_id}")
        # In a real implementation, this would retrieve the item
        return {"id": item_id, "title": "Example Item", "content": "Example content"}
        
    async def close(self):
        """Close the connection to SurrealDB."""
        self.logger.info("Closing SurrealDB connection")
        # In a real implementation, this would close the connection
EOL

cat > "${INTEGRATION_DIR}/src/mcp_adapter.py" << EOL
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
EOL

echo -e "${GREEN}✅ SurrealDB + Ptolemies integration setup complete${NC}"
echo ""
echo -e "${YELLOW}Integration components:${NC}"
echo "1. Integration README with architecture diagram"
echo "2. Configuration template for SurrealDB and vector database"
echo "3. Integration setup script with Ptolemies linking"
echo "4. Example SurrealDB client implementation"
echo "5. MCP adapter for SurrealDB operations"
echo ""
echo -e "${BLUE}Next steps for SurrealDB + Ptolemies integration:${NC}"
echo "1. Install and configure SurrealDB (refer to README)"
echo "2. Run the integration setup script"
echo "3. Update the configuration with actual connection details"
echo "4. Complete the client implementation with actual SurrealDB operations"
echo ""