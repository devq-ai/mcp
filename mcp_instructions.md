# MCP Integration Instructions for Zed and Pydantic AI

## Overview

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. This guide provides comprehensive instructions for integrating MCP with both the Zed editor and Pydantic AI applications.

## Zed Editor Integration

### Configuration

To use MCP servers in Zed, you need to add a context servers section in the Zed config (preferences > settings):

```json
{
  "context_servers": {
    "your-mcp": {
      "settings": {},
      "command": {
        "path": "node",
        "args": ["/path/to/mcp/index.js"],
        "env": {
          "API_KEY": "if-needed"
        }
      }
    }
  }
}
```

### Current Limitations and Features

**Prompts Only**: Zed will happily report it has detected tools in the logs (cmd-shift-p > "zed: open logs"), but it doesn't actually expose them in the assistant panel. Only prompts are supported for now, and those show up under the slash (`/`) command.

**Single Argument Limitation**: Only one prompt argument is supported. If you define more than one argument, it'll ignore the prompt definition without notice.

**Live Diffs**: To enable experimental live diffs, add this to your assistant configuration:

```json
{
  "assistant": {
    "default_model": {
      "provider": "zed.dev",
      "model": "claude-3-5-sonnet-latest"
    },
    "enable_experimental_live_diffs": true,
    "version": "2"
  }
}
```

### Available Extensions

Browse all available MCP extensions either on Zed's website or directly in Zed via the zed: extensions action in the Command Palette. Some examples include:
- PostgreSQL context server for database schema access
- Various data source integrations

### Creating Custom Extensions

If there's an existing MCP server you'd like to bring to Zed, check out the context server extension docs for how to make it available as an extension.

## Pydantic AI Integration

### Installation

You need to either install pydantic-ai, or pydantic-ai-slim with the mcp optional group:

```bash
pip install pydantic-ai[mcp]
# or
pip install pydantic-ai-slim[mcp]
```

**Note**: MCP integration requires Python 3.10 or higher.

### Connection Methods

Pydantic AI supports two transport methods for connecting to MCP servers:

#### 1. HTTP Transport (MCPServerHTTP)

MCPServerHTTP connects over HTTP using the Streamable HTTP transport to a server:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

server = MCPServerHTTP('http://localhost:3001/sse')
agent = Agent('openai:gpt-4o', mcp_servers=[server])

async def main():
    async with agent.run_mcp_servers():
        # Your agent logic here
        pass
```

**Important**: MCPServerHTTP requires an MCP server to be running and accepting HTTP connections before calling agent.run_mcp_servers().

#### 2. Stdio Transport (MCPServerStdio)

The stdio transport where the server is run as a subprocess and communicates with the client over stdin and stdout:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    'deno',
    args=[
        'run', '-N', '-R=node_modules', '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ]
)

agent = Agent('openai:gpt-4o', mcp_servers=[server])

async def main():
    async with agent.run_mcp_servers():
        r = await agent.run('Calculate the fibonacci sequence up to 10')
        print(r.data)
```

### Managing Multiple Servers with Tool Prefixes

When connecting to multiple MCP servers that might provide tools with the same name, you can use the tool_prefix parameter to avoid naming conflicts:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

python_server = MCPServerStdio(
    'deno',
    args=[
        'run', '-N',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ],
    tool_prefix='py'  # Tools will be prefixed with 'py_'
)

js_server = MCPServerStdio(
    'node',
    args=[
        'run', 'mcp-js-server.js',
        'stdio',
    ],
    tool_prefix='js'  # Tools will be prefixed with 'js_'
)

agent = Agent('openai:gpt-4o', mcp_servers=[python_server, js_server])
```

### Using Pydantic AI as MCP Server

PydanticAI models can also be used within MCP Servers. Here's how to create a simple MCP server using Pydantic AI:

```python
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def client():
    server_params = StdioServerParameters(
        command='uv',
        args=['run', 'mcp_server.py', 'server'],
        env=os.environ
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
```

## Best Practices

### 1. Tool Naming and Organization

- Use descriptive tool names that clearly indicate their purpose
- When using multiple servers, leverage tool prefixes to avoid conflicts
- Providing context about the types of tasks for which specific MCP server tools are best suited helps the LLM make informed decisions about tool usage

### 2. Data Structure Design

When defining dependencies and result structures using Pydantic models for agents interacting with MCP servers, it is crucial to be specific and precise in defining the data types and constraints. Using clear and descriptive field names and descriptions enhances documentation and understanding.

### 3. Error Handling

Always implement proper error handling for MCP server connections:

```python
async def main():
    try:
        async with agent.run_mcp_servers():
            result = await agent.run('Your query here')
            return result
    except Exception as e:
        print(f"MCP server error: {e}")
        # Handle fallback logic
```

### 4. Security Considerations

- Be cautious when exposing sensitive systems through MCP servers
- Use appropriate authentication mechanisms when available
- Validate all data exchanged between MCP servers and clients

## Available MCP Servers

### Official Servers

- **mcp-run-python**: Execute Python code in a secure environment
- **Logfire MCP server**: Search and analyze logs, traces, and metrics
- **PostgreSQL server**: Access database schemas and run queries

### Community Servers

An expanding ecosystem of pre-built MCP servers for various domains and applications is available at the [MCP servers repository](https://github.com/modelcontextprotocol/servers).

## Debugging and Monitoring

### Zed Debugging

Zed will happily report it has detected tools in the logs (cmd-shift-p > "zed: open logs") - use this for troubleshooting MCP server connections.

### Pydantic AI with Logfire

Seamlessly integrates with Pydantic Logfire for real-time debugging, performance monitoring, and behavior tracking of your LLM-powered applications:

```python
# Add three lines to instrument with logfire
import logfire

logfire.configure()
logfire.instrument_pydantic()
```

## Future Considerations

Enhanced Security: More sophisticated security models for managing AI access to sensitive systems. Cross-agent Collaboration: Frameworks for enabling multiple AI agents to collaborate using shared MCP servers. Domain-specific Extensions: Extensions to the MCP protocol to better support specific domains like healthcare, finance, and legal.

## Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Zed MCP Documentation](https://zed.dev/docs/ai/mcp/)
- [Pydantic AI MCP Documentation](https://ai.pydantic.dev/mcp/)
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [Anthropic MCP Announcement](https://anthropic.com/news/model-context-protocol)