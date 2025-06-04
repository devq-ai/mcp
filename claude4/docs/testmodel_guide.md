# TestModel Tool Inventory Guide for Claude 4 Sonnet

## Overview

TestModel is Pydantic AI's built-in testing utility that allows you to inspect and extract detailed information about tools available to Claude 4 Sonnet without making actual API calls. This guide shows you how to use TestModel to get a comprehensive inventory of all available tools.

## What TestModel Reveals

When you use TestModel with a Pydantic AI agent, you can extract:

- **Tool Definitions**: Complete JSON schemas for each tool
- **Parameter Details**: Types, descriptions, defaults, and requirements
- **Tool Metadata**: Names, descriptions, and configuration
- **Model Parameters**: Settings sent to Claude 4 Sonnet
- **Function Call Support**: What tools are actually available to the model

## Basic Usage Pattern

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Create your agent with tools
agent = Agent('anthropic:claude-4-sonnet-20250522')

@agent.tool_plain
def my_tool(param: str) -> str:
    """Example tool."""
    return f"Processed: {param}"

# Extract tool inventory using TestModel
test_model = TestModel()

with agent.override(model=test_model):
    # Make a dummy request to trigger tool schema generation
    result = agent.run_sync("What tools are available?")
    
    # Extract the function tools
    function_tools = test_model.last_model_request_parameters.function_tools
    
    # Analyze each tool
    for tool_def in function_tools:
        print(f"Tool: {tool_def.name}")
        print(f"Description: {tool_def.description}")
        print(f"Schema: {tool_def.parameters_json_schema}")
        print(f"Strict mode: {getattr(tool_def, 'strict', None)}")
```

## Complete Tool Inventory Function

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
import json
from datetime import datetime
from typing import Dict, List, Any

def get_comprehensive_tool_inventory(agent: Agent) -> Dict[str, Any]:
    """Extract comprehensive tool inventory using TestModel."""
    
    test_model = TestModel()
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {},
        "tools": [],
        "statistics": {
            "total_tools": 0,
            "simple_tools": 0,
            "context_tools": 0,
            "total_parameters": 0,
            "required_parameters": 0,
            "optional_parameters": 0
        }
    }
    
    with agent.override(model=test_model):
        try:
            # Trigger tool schema generation
            result = agent.run_sync("List available tools")
            
            # Extract function tools
            function_tools = test_model.last_model_request_parameters.function_tools
            inventory["statistics"]["total_tools"] = len(function_tools)
            
            # Analyze each tool
            for tool_def in function_tools:
                tool_info = analyze_tool_definition(tool_def)
                inventory["tools"].append(tool_info)
                
                # Update statistics
                if tool_info["tool_type"] == "simple":
                    inventory["statistics"]["simple_tools"] += 1
                else:
                    inventory["statistics"]["context_tools"] += 1
                
                inventory["statistics"]["total_parameters"] += tool_info["parameter_count"]
                inventory["statistics"]["required_parameters"] += len(tool_info["required_params"])
                inventory["statistics"]["optional_parameters"] += len(tool_info["optional_params"])
            
            # Add model information
            inventory["model_info"] = {
                "system": getattr(test_model, 'system', 'unknown'),
                "name": str(test_model),
                "function_tools_count": len(function_tools),
                "text_output_allowed": test_model.last_model_request_parameters.allow_text_output
            }
            
        except Exception as e:
            inventory["error"] = str(e)
    
    return inventory

def analyze_tool_definition(tool_def: ToolDefinition) -> Dict[str, Any]:
    """Analyze a single tool definition."""
    schema = tool_def.parameters_json_schema or {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Determine if tool is context-aware
    is_context_aware = any(
        "RunContext" in str(prop) for prop in properties.values()
    )
    
    return {
        "name": tool_def.name,
        "description": tool_def.description or "No description",
        "tool_type": "context_aware" if is_context_aware else "simple",
        "strict_mode": getattr(tool_def, 'strict', None),
        "parameter_count": len(properties),
        "required_params": required,
        "optional_params": [p for p in properties.keys() if p not in required],
        "full_schema": schema,
        "parameter_details": {
            name: {
                "type": prop.get("type", "unknown"),
                "description": prop.get("description", ""),
                "default": prop.get("default"),
                "required": name in required
            }
            for name, prop in properties.items()
        }
    }
```

## Claude 4 Sonnet Specific Features

When using TestModel with Claude 4 Sonnet, you can detect these specific capabilities:

### Built-in Tools Detection

```python
def detect_claude_4_features(inventory: Dict[str, Any]) -> Dict[str, bool]:
    """Detect Claude 4 Sonnet specific features."""
    
    tool_names = [tool["name"] for tool in inventory["tools"]]
    
    return {
        "code_execution": "code_execution" in tool_names,
        "web_search": "web_search" in tool_names or "search_web" in tool_names,
        "file_access": any("file" in name for name in tool_names),
        "parallel_tools": inventory["statistics"]["total_tools"] > 1,
        "context_injection": inventory["statistics"]["context_tools"] > 0,
        "structured_outputs": any(
            tool.get("parameter_details") for tool in inventory["tools"]
        ),
        "strict_mode_support": any(
            tool.get("strict_mode") for tool in inventory["tools"]
        )
    }
```

### Extended Thinking Mode Detection

```python
def check_extended_thinking_support(agent: Agent) -> bool:
    """Check if agent supports extended thinking mode."""
    
    test_model = TestModel()
    
    with agent.override(model=test_model):
        try:
            # Try a complex query that would benefit from extended thinking
            result = agent.run_sync(
                "Analyze this complex problem requiring multiple steps and tool usage"
            )
            
            # Check if model parameters indicate extended thinking support
            params = test_model.last_model_request_parameters
            return hasattr(params, 'thinking_mode') or 'thinking' in str(params)
            
        except Exception:
            return False
```

## Real-World Example: Customer Support Agent

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional

@dataclass
class SupportContext:
    customer_id: str
    database_url: str
    api_key: str

class SupportResponse(BaseModel):
    message: str = Field(description="Response to customer")
    escalate: bool = Field(description="Whether to escalate")
    confidence: float = Field(description="Response confidence", ge=0, le=1)

# Create support agent
support_agent = Agent(
    'anthropic:claude-4-sonnet-20250522',
    deps_type=SupportContext,
    output_type=SupportResponse,
    system_prompt="You are a helpful customer support agent."
)

@support_agent.tool_plain
def search_knowledge_base(query: str, category: str = "general") -> List[str]:
    """Search the knowledge base for answers."""
    return [f"KB result for '{query}' in {category}"]

@support_agent.tool
def get_customer_orders(ctx: RunContext[SupportContext], limit: int = 10) -> List[Dict]:
    """Get customer's recent orders."""
    return [{"order_id": f"ORD-{i}", "customer_id": ctx.deps.customer_id} for i in range(limit)]

@support_agent.tool
def create_ticket(ctx: RunContext[SupportContext], subject: str, priority: str = "normal") -> str:
    """Create a support ticket."""
    return f"Ticket created: {subject} (Priority: {priority})"

# Extract tool inventory
inventory = get_comprehensive_tool_inventory(support_agent)

# Print summary
print("=== CUSTOMER SUPPORT AGENT TOOL INVENTORY ===")
print(f"Total tools: {inventory['statistics']['total_tools']}")
print(f"Context-aware tools: {inventory['statistics']['context_tools']}")
print(f"Simple tools: {inventory['statistics']['simple_tools']}")

for tool in inventory["tools"]:
    print(f"\nTool: {tool['name']} ({tool['tool_type']})")
    print(f"  Description: {tool['description']}")
    print(f"  Parameters: {tool['parameter_count']}")
    if tool['required_params']:
        print(f"  Required: {tool['required_params']}")
    if tool['optional_params']:
        print(f"  Optional: {tool['optional_params']}")
```

## Automated Tool Documentation

```python
def generate_tool_documentation(inventory: Dict[str, Any]) -> str:
    """Generate markdown documentation from tool inventory."""
    
    doc = f"""# Tool Inventory Report

Generated: {inventory['timestamp']}
Total Tools: {inventory['statistics']['total_tools']}

## Summary Statistics

- Simple Tools: {inventory['statistics']['simple_tools']}
- Context-Aware Tools: {inventory['statistics']['context_tools']}
- Total Parameters: {inventory['statistics']['total_parameters']}
- Required Parameters: {inventory['statistics']['required_parameters']}
- Optional Parameters: {inventory['statistics']['optional_parameters']}

## Tool Details

"""
    
    for i, tool in enumerate(inventory['tools'], 1):
        doc += f"""### {i}. {tool['name']} ({tool['tool_type'].title()})

**Description:** {tool['description']}

**Parameters:** {tool['parameter_count']} total

"""
        if tool['required_params']:
            doc += f"**Required:** {', '.join(tool['required_params'])}\n\n"
        
        if tool['optional_params']:
            doc += f"**Optional:** {', '.join(tool['optional_params'])}\n\n"
        
        if tool['parameter_details']:
            doc += "**Parameter Details:**\n\n"
            for param, details in tool['parameter_details'].items():
                req_status = "Required" if details['required'] else "Optional"
                default_info = f" (default: `{details['default']}`)" if details['default'] is not None else ""
                doc += f"- `{param}`: {details['type']} ({req_status}){default_info}\n"
                if details['description']:
                    doc += f"  - {details['description']}\n"
            doc += "\n"
        
        doc += f"**Strict Mode:** {tool['strict_mode'] or 'Not set'}\n\n"
        doc += "---\n\n"
    
    return doc

# Usage
inventory = get_comprehensive_tool_inventory(agent)
documentation = generate_tool_documentation(inventory)

with open("tool_inventory_report.md", "w") as f:
    f.write(documentation)
```

## Testing Tool Availability

```python
def test_tool_availability(agent: Agent, tool_name: str) -> Dict[str, Any]:
    """Test if a specific tool is available and working."""
    
    test_model = TestModel()
    
    with agent.override(model=test_model):
        try:
            # Make a request that would use the specific tool
            result = agent.run_sync(f"Use the {tool_name} tool")
            
            # Check if the tool was actually available
            function_tools = test_model.last_model_request_parameters.function_tools
            available_tools = [tool.name for tool in function_tools]
            
            return {
                "tool_available": tool_name in available_tools,
                "all_available_tools": available_tools,
                "test_result": result.output if hasattr(result, 'output') else str(result),
                "error": None
            }
            
        except Exception as e:
            return {
                "tool_available": False,
                "all_available_tools": [],
                "test_result": None,
                "error": str(e)
            }

# Usage
result = test_tool_availability(agent, "search_knowledge_base")
print(f"Tool available: {result['tool_available']}")
print(f"All tools: {result['all_available_tools']}")
```

## Best Practices

### 1. Always Use TestModel for Inspection

```python
# ✅ Good: Use TestModel for inspection
def inspect_tools(agent):
    test_model = TestModel()
    with agent.override(model=test_model):
        agent.run_sync("test")
        return test_model.last_model_request_parameters.function_tools

# ❌ Bad: Making actual API calls for inspection
def inspect_tools_bad(agent):
    result = agent.run_sync("What tools do you have?")  # Costs money!
    return result.output
```

### 2. Handle Dependencies Properly

```python
# When your agent has dependencies, provide them during testing
@dataclass
class TestDeps:
    user_id: str = "test_user"
    api_key: str = "test_key"

agent = Agent('anthropic:claude-4-sonnet-20250522', deps_type=TestDeps)

# Provide test dependencies
test_deps = TestDeps()
with agent.override(model=TestModel()):
    result = agent.run_sync("test", deps=test_deps)
```

### 3. Save and Version Tool Inventories

```python
def save_versioned_inventory(inventory: Dict[str, Any], version: str):
    """Save tool inventory with version for tracking changes."""
    
    filename = f"tool_inventory_v{version}_{datetime.now().strftime('%Y%m%d')}.json"
    
    with open(filename, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    
    return filename

# Track tool changes over time
inventory_v1 = get_comprehensive_tool_inventory(agent_v1)
inventory_v2 = get_comprehensive_tool_inventory(agent_v2)

save_versioned_inventory(inventory_v1, "1.0")
save_versioned_inventory(inventory_v2, "1.1")
```

## Troubleshooting

### Common Issues

1. **ImportError**: Install pydantic-ai with `pip install pydantic-ai`

2. **Empty function_tools**: Ensure you've registered tools with the agent before testing

3. **Missing dependencies**: Provide test dependencies when your tools use RunContext

4. **Tool not appearing**: Check that tool decorators are applied correctly

### Debugging Tool Registration

```python
def debug_tool_registration(agent: Agent):
    """Debug tool registration issues."""
    
    test_model = TestModel()
    
    print("Debugging tool registration...")
    
    with agent.override(model=test_model):
        try:
            result = agent.run_sync("debug")
            tools = test_model.last_model_request_parameters.function_tools
            
            print(f"✅ Found {len(tools)} tools")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Check:")
            print("  1. Tools are properly decorated")
            print("  2. Dependencies are provided")
            print("  3. Agent is properly initialized")

# Usage
debug_tool_registration(agent)
```

## Integration Examples

### CI/CD Tool Validation

```python
def validate_agent_tools_in_ci():
    """Validate agent tools in CI/CD pipeline."""
    
    agent = create_production_agent()
    inventory = get_comprehensive_tool_inventory(agent)
    
    # Validate expected tools are present
    expected_tools = ["search_knowledge_base", "create_ticket", "get_customer_orders"]
    actual_tools = [tool["name"] for tool in inventory["tools"]]
    
    missing_tools = set(expected_tools) - set(actual_tools)
    
    if missing_tools:
        raise AssertionError(f"Missing required tools: {missing_tools}")
    
    # Validate tool parameters
    for tool in inventory["tools"]:
        if not tool["description"]:
            raise AssertionError(f"Tool {tool['name']} missing description")
    
    print("✅ All tool validations passed")
    return True

# Run in CI
if __name__ == "__main__":
    validate_agent_tools_in_ci()
```

This guide provides comprehensive coverage of using TestModel to extract tool inventories from Claude 4 Sonnet agents, enabling better development, testing, and documentation workflows.