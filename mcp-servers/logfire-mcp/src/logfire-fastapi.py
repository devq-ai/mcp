#!/usr/bin/env python3
"""
Logfire MCP Server

This MCP server provides integration with Pydantic Logfire for observability,
monitoring, and debugging of applications through the Model Context Protocol.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add MCP SDK to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    CallToolRequest,
    CallToolResult,
    ErrorData,
    TextContent,
    ServerCapabilities,
    ToolsCapability
)

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None
    print("Warning: logfire package not installed. Install with: pip install logfire", file=sys.stderr)

# Initialize the MCP server
app = Server("logfire-mcp")

# Global Logfire configuration
logfire_config = {
    "project_name": os.getenv("LOGFIRE_PROJECT_NAME", "mcp-server"),
    "token": os.getenv("LOGFIRE_TOKEN"),
    "service_name": os.getenv("LOGFIRE_SERVICE_NAME", "mcp-logfire"),
    "environment": os.getenv("LOGFIRE_ENVIRONMENT", "development")
}

# Initialize Logfire if available and configured
if LOGFIRE_AVAILABLE and logfire_config["token"]:
    logfire.configure(
        token=logfire_config["token"],
        project_name=logfire_config["project_name"],
        service_name=logfire_config["service_name"],
        environment=logfire_config["environment"]
    )
    logfire_initialized = True
else:
    logfire_initialized = False

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Logfire tools."""
    tools = []
    
    if not logfire_initialized:
        return [
            Tool(
                name="logfire_not_configured",
                description="Logfire is not configured. Set LOGFIRE_TOKEN environment variable.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    tools.extend([
        Tool(
            name="logfire_log",
            description="Send a log message to Logfire with optional structured data",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["debug", "info", "warning", "error", "critical"],
                        "description": "Log level"
                    },
                    "message": {
                        "type": "string",
                        "description": "Log message"
                    },
                    "data": {
                        "type": "object",
                        "description": "Optional structured data to include with the log"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for the log entry"
                    }
                },
                "required": ["level", "message"]
            }
        ),
        Tool(
            name="logfire_span",
            description="Create a Logfire span for tracing operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the span/operation"
                    },
                    "attributes": {
                        "type": "object",
                        "description": "Attributes to attach to the span"
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration of the operation in milliseconds (for completed spans)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="logfire_metric",
            description="Record a metric in Logfire",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Metric name"
                    },
                    "value": {
                        "type": "number",
                        "description": "Metric value"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g., 'ms', 'bytes', 'count')"
                    },
                    "tags": {
                        "type": "object",
                        "description": "Tags to associate with the metric"
                    }
                },
                "required": ["name", "value"]
            }
        ),
        Tool(
            name="logfire_exception",
            description="Log an exception to Logfire",
            inputSchema={
                "type": "object",
                "properties": {
                    "exception_type": {
                        "type": "string",
                        "description": "Type of the exception"
                    },
                    "message": {
                        "type": "string",
                        "description": "Exception message"
                    },
                    "traceback": {
                        "type": "string",
                        "description": "Exception traceback"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context about the exception"
                    }
                },
                "required": ["exception_type", "message"]
            }
        ),
        Tool(
            name="logfire_instrument_code",
            description="Get instrumentation code for FastAPI or other frameworks",
            inputSchema={
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": ["fastapi", "django", "flask", "general"],
                        "description": "Framework to instrument"
                    },
                    "options": {
                        "type": "object",
                        "description": "Framework-specific instrumentation options"
                    }
                },
                "required": ["framework"]
            }
        )
    ])
    
    return tools

@app.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Execute a Logfire tool."""
    
    if not logfire_initialized:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Error: Logfire is not configured. Please set the LOGFIRE_TOKEN environment variable."
            )],
            isError=True
        )
    
    tool_name = request.params.name
    arguments = request.params.arguments or {}
    
    try:
        if tool_name == "logfire_log":
            level = arguments.get("level", "info")
            message = arguments["message"]
            data = arguments.get("data", {})
            tags = arguments.get("tags", [])
            
            # Create log entry with appropriate level
            log_func = getattr(logfire, level, logfire.info)
            log_func(message, **data, tags=tags)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Logged {level.upper()}: {message}"
                )]
            )
            
        elif tool_name == "logfire_span":
            name = arguments["name"]
            attributes = arguments.get("attributes", {})
            duration_ms = arguments.get("duration_ms")
            
            # Create a span
            with logfire.span(name, **attributes) as span:
                if duration_ms:
                    # Simulate the operation duration
                    import time
                    time.sleep(duration_ms / 1000.0)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Created span: {name}" + (f" (duration: {duration_ms}ms)" if duration_ms else "")
                )]
            )
            
        elif tool_name == "logfire_metric":
            name = arguments["name"]
            value = arguments["value"]
            unit = arguments.get("unit", "count")
            tags = arguments.get("tags", {})
            
            # Record metric
            logfire.metric(name, value, unit=unit, **tags)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Recorded metric: {name}={value} {unit}"
                )]
            )
            
        elif tool_name == "logfire_exception":
            exception_type = arguments["exception_type"]
            message = arguments["message"]
            traceback = arguments.get("traceback", "")
            context = arguments.get("context", {})
            
            # Log exception
            logfire.exception(
                f"{exception_type}: {message}",
                exc_info=(exception_type, message, traceback),
                **context
            )
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Logged exception: {exception_type}: {message}"
                )]
            )
            
        elif tool_name == "logfire_instrument_code":
            framework = arguments["framework"]
            options = arguments.get("options", {})
            
            code_examples = {
                "fastapi": """
# FastAPI Instrumentation with Logfire

```python
import logfire
from fastapi import FastAPI
from logfire.integrations.fastapi import instrument_fastapi

# Configure Logfire
logfire.configure(
    token="your-logfire-token",
    project_name="your-project",
    service_name="your-service"
)

# Create FastAPI app
app = FastAPI()

# Instrument FastAPI with Logfire
instrument_fastapi(app)

# Your routes will now be automatically traced
@app.get("/")
async def root():
    logfire.info("Root endpoint called")
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    with logfire.span("fetch_item", item_id=item_id):
        # Your business logic here
        logfire.info(f"Fetching item {item_id}")
        return {"item_id": item_id}
```
""",
                "django": """
# Django Instrumentation with Logfire

```python
# In your Django settings.py
import logfire

# Configure Logfire
logfire.configure(
    token="your-logfire-token",
    project_name="your-project",
    service_name="your-django-service"
)

# In your middleware or views
from django.http import HttpResponse
import logfire

def my_view(request):
    with logfire.span("my_view", path=request.path):
        logfire.info("Processing request", method=request.method)
        # Your view logic
        return HttpResponse("Hello, World!")

# For middleware
class LogfireMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        with logfire.span("http_request", method=request.method, path=request.path):
            response = self.get_response(request)
            logfire.info("Request completed", status=response.status_code)
            return response
```
""",
                "flask": """
# Flask Instrumentation with Logfire

```python
import logfire
from flask import Flask, request

# Configure Logfire
logfire.configure(
    token="your-logfire-token",
    project_name="your-project",
    service_name="your-flask-service"
)

app = Flask(__name__)

# Before request hook
@app.before_request
def before_request():
    logfire.info("Request started", method=request.method, path=request.path)

# After request hook
@app.after_request
def after_request(response):
    logfire.info("Request completed", status=response.status_code)
    return response

@app.route('/')
def hello():
    with logfire.span("hello_route"):
        logfire.info("Hello route called")
        return "Hello, World!"

@app.route('/api/data/<int:data_id>')
def get_data(data_id):
    with logfire.span("get_data", data_id=data_id):
        # Your business logic
        logfire.info(f"Fetching data {data_id}")
        return {"data_id": data_id}
```
""",
                "general": """
# General Python Instrumentation with Logfire

```python
import logfire
import time
from contextlib import contextmanager

# Configure Logfire
logfire.configure(
    token="your-logfire-token",
    project_name="your-project",
    service_name="your-service"
)

# Function decorator for automatic tracing
def trace_function(func):
    def wrapper(*args, **kwargs):
        with logfire.span(f"function.{func.__name__}"):
            logfire.info(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            logfire.info(f"Completed {func.__name__}")
            return result
    return wrapper

# Context manager for operations
@contextmanager
def trace_operation(name, **attributes):
    with logfire.span(name, **attributes) as span:
        start_time = time.time()
        logfire.info(f"Starting {name}")
        try:
            yield span
        except Exception as e:
            logfire.exception(f"Error in {name}: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            logfire.info(f"Completed {name}", duration_ms=duration * 1000)

# Example usage
@trace_function
def process_data(data):
    with trace_operation("data_processing", data_size=len(data)):
        # Process your data
        logfire.metric("data.processed", len(data), unit="items")
        return data

# Logging examples
def example_logging():
    # Different log levels
    logfire.debug("Debug information", user_id=123)
    logfire.info("User logged in", user_id=123, ip="192.168.1.1")
    logfire.warning("High memory usage", usage_percent=85)
    logfire.error("Failed to connect to database", error_code="DB_001")
    
    # Structured logging with metrics
    logfire.metric("api.response_time", 145.3, unit="ms")
    logfire.metric("queue.size", 42, unit="items")
    
    # Exception handling
    try:
        risky_operation()
    except Exception as e:
        logfire.exception("Operation failed", operation="risky_operation", context={"retry_count": 3})
```
"""
            }
            
            code = code_examples.get(framework, code_examples["general"])
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Instrumentation code for {framework}:\n{code}"
                )]
            )
            
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Unknown tool: {tool_name}"
                )],
                isError=True
            )
            
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Error executing tool {tool_name}: {str(e)}"
            )],
            isError=True
        )

async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="logfire-mcp",
                server_version="0.1.0",
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(list_tools=True)
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())