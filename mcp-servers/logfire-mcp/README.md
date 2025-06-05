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