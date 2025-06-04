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

Update the `config.json` file with the SurrealDB connection details:

```json
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
```

## Usage

Once configured, the SurrealDB MCP server will provide the following capabilities to Ptolemies:

1. **Graph Data Storage** - Store knowledge items and their relationships
2. **Real-time Updates** - Push updates to agents when knowledge changes
3. **Query Language** - Execute SurrealQL queries for complex data retrieval
4. **ACID Transactions** - Ensure data consistency across operations

## Architecture

```
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
```
