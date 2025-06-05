#!/bin/bash

# Logfire MCP Server Installation Script

set -e

echo "Installing Logfire MCP Server..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install logfire mcp python-dotenv

# Create __init__.py files
touch "$SCRIPT_DIR/src/__init__.py"

# Make the main script executable
chmod +x "$SCRIPT_DIR/src/logfire-fastapi.py"

# Create a .env.example file if it doesn't exist
if [ ! -f "$SCRIPT_DIR/.env.example" ]; then
    cat > "$SCRIPT_DIR/.env.example" << EOF
# Logfire Configuration
LOGFIRE_TOKEN=your-logfire-token-here
LOGFIRE_PROJECT_NAME=mcp-server
LOGFIRE_SERVICE_NAME=mcp-logfire
LOGFIRE_ENVIRONMENT=development
EOF
fi

# Create README if it doesn't exist
if [ ! -f "$SCRIPT_DIR/README.md" ]; then
    cat > "$SCRIPT_DIR/README.md" << 'EOF'
# Logfire MCP Server

This MCP server provides integration with Pydantic Logfire for observability, monitoring, and debugging.

## Features

- **Structured Logging**: Send logs with different levels and structured data
- **Tracing**: Create spans to trace operations and measure performance
- **Metrics**: Record custom metrics with units and tags
- **Exception Tracking**: Log exceptions with full context
- **Framework Integration**: Get instrumentation code for FastAPI, Django, Flask

## Installation

```bash
./install.sh
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your Logfire token:
   ```
   LOGFIRE_TOKEN=your-logfire-token-here
   ```

## Usage

The server can be used with any MCP client. Add to your MCP configuration:

```json
{
  "logfire": {
    "command": "python",
    "args": ["-m", "logfire_mcp.src.logfire-fastapi"],
    "cwd": "/path/to/logfire-mcp",
    "env": {
      "LOGFIRE_TOKEN": "${LOGFIRE_TOKEN}"
    }
  }
}
```

## Available Tools

- `logfire_log`: Send structured logs
- `logfire_span`: Create tracing spans
- `logfire_metric`: Record metrics
- `logfire_exception`: Log exceptions
- `logfire_instrument_code`: Get framework instrumentation examples

## Example Usage

```python
# Send a log
await client.call_tool("logfire_log", {
    "level": "info",
    "message": "User logged in",
    "data": {"user_id": 123, "ip": "192.168.1.1"}
})

# Create a span
await client.call_tool("logfire_span", {
    "name": "database_query",
    "attributes": {"query": "SELECT * FROM users", "table": "users"},
    "duration_ms": 45
})

# Record a metric
await client.call_tool("logfire_metric", {
    "name": "api.response_time",
    "value": 123.45,
    "unit": "ms",
    "tags": {"endpoint": "/api/users", "method": "GET"}
})
```
EOF
fi

echo ""
echo "Logfire MCP Server installed successfully!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your Logfire token"
echo "2. Add the server to your MCP configuration"
echo "3. Start using Logfire observability tools!"
echo ""