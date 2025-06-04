# Claude 4 Sonnet Tool & Capability Analysis

Complete documentation and tools for analyzing Claude 4 Sonnet's capabilities, built-in tools, and custom tool integration using Pydantic AI.

## Directory Structure

```
claude4/
├── docs/           # Comprehensive documentation
├── examples/       # Working code examples
├── tools/          # Tool inspection utilities
├── results/        # Output files and inventories
└── README.md       # This file
```

## Quick Start

### 1. Install Dependencies
```bash
cd mcp
python -m venv pydantic_ai_env
source pydantic_ai_env/bin/activate  # Linux/Mac
pip install pydantic-ai
```

### 2. Run Tool Inspection
```bash
python claude4/examples/testmodel_demo.py
```

### 3. View Results
Check `claude4/results/complete_inventory.json` for extracted tool schemas.

## Documentation

### Core Guides
- **[Complete Capabilities](docs/complete_capabilities.md)** - Full reference of all Claude 4 Sonnet features
- **[Complete Summary](docs/complete_summary.md)** - Comprehensive tool and capability listing
- **[TestModel Guide](docs/testmodel_guide.md)** - Zero-cost tool inspection methodology
- **[Pydantic AI Tools Guide](docs/pydantic_ai_tools_guide.md)** - Tool development best practices
- **[Model Differences](docs/model_differences.md)** - Comparison across different LLMs

## Examples

### Working Code
- **[Implementation Examples](examples/implementation_examples.py)** - Complete technical examples
- **[TestModel Demo](examples/testmodel_demo.py)** - Working tool extraction demo

## Tools

### Inspection Utilities
- **[Inspector](tools/inspector.py)** - Production-ready tool inspector
- **[Simulator](tools/simulator.py)** - Tool simulation and testing

## Results

### Generated Inventories
- **[Complete Inventory](results/complete_inventory.json)** - Full tool extraction results
- **[Simulation Inventory](results/simulation_inventory.json)** - Simulated tool data

## Key Features

### Built-in Tools (4 Total)
1. **Code Execution** - Native Python execution
2. **Web Search** - Real-time search during extended thinking
3. **File API Access** - Local file operations and persistent memory
4. **MCP Connector** - Model Context Protocol integration

### Advanced Capabilities
- **Extended Thinking Mode** - Deep reasoning with tool access
- **Parallel Tool Execution** - Multiple tools simultaneously
- **Memory Capabilities** - Persistent context across sessions
- **Enhanced Function Calling** - Custom tool registration

### TestModel Benefits
- **Zero Cost** - No API calls required for tool inspection
- **Instant Results** - Immediate tool schema extraction
- **Complete Schemas** - Full JSON schemas with types and validation
- **CI/CD Ready** - Perfect for automated testing and validation

## Usage Examples

### Basic Tool Inspection
```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Create agent with TestModel
test_model = TestModel()
agent = Agent(test_model)

# Register tools
@agent.tool_plain
def my_tool(param: str) -> str:
    return f"Processed: {param}"

# Extract tool inventory
result = agent.run_sync("test")
tools = test_model.last_model_request_parameters.function_tools
print(f"Extracted {len(tools)} tools")
```

### Production Agent
```python
# With actual Claude 4 Sonnet (requires API key)
agent = Agent('anthropic:claude-4-sonnet-20250522')

@agent.tool_plain
def production_tool(data: str) -> dict:
    return {"processed": data, "timestamp": datetime.now().isoformat()}
```

## Development Workflow

1. **Design Tools** - Define tool interfaces and schemas
2. **Test with TestModel** - Validate tool extraction (free)
3. **Simulate Responses** - Test tool behavior without API costs
4. **Deploy to Production** - Use with actual Claude 4 Sonnet
5. **Monitor & Optimize** - Track usage and performance

## Best Practices

### Tool Design
- Use clear, descriptive names and documentation
- Implement proper type hints for automatic schema generation
- Handle errors gracefully with ModelRetry for recoverable issues
- Design for both simple and context-aware use cases

### Testing Strategy
- Use TestModel for development and CI/CD validation
- Test tool schemas before production deployment
- Validate parameter types and return structures
- Monitor tool usage and performance in production

### Security Considerations
- Implement permission-based tool access
- Validate and sanitize all tool inputs
- Use safe execution environments for code tools
- Audit tool usage and maintain access logs

## Contributing

When adding new tools or examples:
1. Update relevant documentation
2. Add comprehensive examples
3. Test with TestModel first
4. Include error handling patterns
5. Document security considerations

## License

This analysis and tooling is provided for educational and development purposes. Claude 4 Sonnet is a product of Anthropic.