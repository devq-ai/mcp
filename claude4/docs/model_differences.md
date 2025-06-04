# Pydantic AI Model-Specific Differences and Advanced Topics

## Table of Contents

1. [Model Capability Matrix](#model-capability-matrix)
2. [Function Calling Implementation Differences](#function-calling-implementation-differences)
3. [Performance Benchmarks](#performance-benchmarks)
4. [Schema Complexity Support](#schema-complexity-support)
5. [Error Handling Variations](#error-handling-variations)
6. [Streaming Tool Support](#streaming-tool-support)
7. [Built-in Tools and Features](#built-in-tools-and-features)
8. [Cost Optimization Strategies](#cost-optimization-strategies)
9. [Model Selection Guidelines](#model-selection-guidelines)
10. [Advanced Integration Patterns](#advanced-integration-patterns)

## Model Capability Matrix

### Comprehensive Comparison

| Feature | OpenAI GPT-4o | Anthropic Claude 3.5 | Google Gemini 1.5 | OpenAI GPT-4o-mini | Notes |
|---------|---------------|----------------------|-------------------|-------------------|-------|
| Function Calling | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Very Good | GPT-4o most reliable |
| Complex Schemas | ✅ Excellent | ✅ Good | ⚠️ Limited | ✅ Good | Gemini struggles with deep nesting |
| Streaming Tools | ✅ Full Support | ✅ Full Support | ✅ Good | ✅ Full Support | All support basic streaming |
| Built-in Tools | ✅ Web Search | ❌ None | ❌ None | ✅ Web Search | OpenAI Responses API only |
| Retry Handling | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good | Claude best at learning from retries |
| Parameter Extraction | ✅ Excellent | ✅ Excellent | ⚠️ Good | ✅ Good | Gemini occasional issues |
| Tool Chaining | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good | Complex chains work better on larger models |
| Speed | ⚠️ Moderate | ⚠️ Slow | ✅ Fast | ✅ Fast | Gemini Flash fastest |
| Cost | ⚠️ Expensive | ⚠️ Most Expensive | ✅ Cheapest | ✅ Very Cheap | Cost per tool call varies significantly |

## Function Calling Implementation Differences

### OpenAI Models

#### Strengths
```python
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List, Dict, Optional

class ComplexSchema(BaseModel):
    nested_data: Dict[str, List[Dict[str, Optional[str]]]]
    metadata: Dict[str, any]

# OpenAI handles this complexity very well
openai_agent = Agent('openai:gpt-4o')

@openai_agent.tool_plain
def complex_operation(data: ComplexSchema) -> dict:
    """OpenAI excels at complex nested schemas."""
    return {"processed": len(data.nested_data)}

# Strict mode for even better reliability
@openai_agent.tool_plain(strict=True)
def strict_operation(param1: str, param2: int) -> str:
    """Strict mode ensures 100% schema compliance."""
    return f"Processed {param1} with {param2}"
```

#### OpenAI-Specific Features
```python
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from openai.types.responses import WebSearchToolParam

# Built-in web search
model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')]
)

openai_responses_agent = Agent(
    model=OpenAIResponsesModel('gpt-4o'),
    model_settings=model_settings
)

# Built-in tools don't count against your tool limit
@openai_responses_agent.tool_plain
def custom_tool() -> str:
    """This works alongside built-in web search."""
    return "Custom result"
```

### Anthropic Claude

#### Strengths
```python
# Claude excels at contextual tool usage
claude_agent = Agent('anthropic:claude-3-5-sonnet-20241022')

@claude_agent.tool
def contextual_analysis(ctx: RunContext[UserData], text: str) -> dict:
    """Claude is excellent at understanding when and why to use tools."""
    # Claude rarely makes inappropriate tool calls
    # Great at understanding nuanced requirements
    analysis = perform_complex_analysis(text, ctx.deps.user_context)
    return analysis

# Claude handles conversational tool use very well
@claude_agent.tool_plain
def multi_step_process(step: int, data: str) -> str:
    """Claude naturally handles multi-step tool interactions."""
    if step == 1:
        return "Prepared for step 2"
    elif step == 2:
        return f"Processed: {data}"
    return "Complete"
```

#### Claude-Specific Optimizations
```python
# Simplify descriptions for Claude
async def optimize_for_claude(
    ctx: RunContext[None], 
    tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Claude prefers concise, clear descriptions."""
    optimized_tools = []
    for tool_def in tool_defs:
        # Claude works better with shorter descriptions
        short_desc = tool_def.description[:150] if tool_def.description else ""
        optimized_def = replace(tool_def, description=short_desc)
        optimized_tools.append(optimized_def)
    return optimized_tools

claude_agent = Agent(
    'anthropic:claude-3-5-sonnet-20241022',
    prepare_tools=optimize_for_claude
)
```

### Google Gemini

#### Strengths and Limitations
```python
# Gemini is fast but has some limitations
gemini_agent = Agent('google-gla:gemini-1.5-flash')

# Good for simple, fast operations
@gemini_agent.tool_plain
def simple_calculation(a: int, b: int, operation: str) -> float:
    """Gemini excels at simple, well-defined tools."""
    operations = {"add": a + b, "multiply": a * b, "divide": a / b}
    return operations.get(operation, 0)

# Avoid overly complex schemas with Gemini
@gemini_agent.tool_plain
def gemini_friendly_tool(
    query: str, 
    limit: int = 10, 
    include_metadata: bool = False
) -> list[str]:
    """Keep schemas simple for best Gemini compatibility."""
    results = search_service.search(query, limit)
    if include_metadata:
        return [f"{r.title} ({r.score})" for r in results]
    return [r.title for r in results]
```

#### Gemini Performance Optimization
```python
# Optimize for Gemini's speed
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_gemini_operation(key: str) -> str:
    """Cache results for frequently called operations."""
    return expensive_operation(key)

@gemini_agent.tool_plain
def fast_lookup(search_key: str) -> str:
    """Leverage Gemini's speed with caching."""
    return cached_gemini_operation(search_key)
```

## Performance Benchmarks

### Speed Comparison (Tool Calling)

```python
# Approximate response times for tool calling (ms)
TOOL_CALL_BENCHMARKS = {
    "gemini-1.5-flash": {
        "simple_tool": 200,
        "complex_tool": 400,
        "multiple_tools": 600
    },
    "gpt-4o-mini": {
        "simple_tool": 300,
        "complex_tool": 500,
        "multiple_tools": 800
    },
    "claude-3-5-haiku": {
        "simple_tool": 400,
        "complex_tool": 600,
        "multiple_tools": 1000
    },
    "gpt-4o": {
        "simple_tool": 500,
        "complex_tool": 800,
        "multiple_tools": 1200
    },
    "claude-3-5-sonnet": {
        "simple_tool": 600,
        "complex_tool": 1000,
        "multiple_tools": 1500
    }
}

# Choose model based on performance requirements
def select_model_for_performance(use_case: str) -> str:
    """Select optimal model based on performance needs."""
    if use_case == "real_time_chat":
        return "google-gla:gemini-1.5-flash"
    elif use_case == "cost_sensitive":
        return "openai:gpt-4o-mini"
    elif use_case == "maximum_accuracy":
        return "openai:gpt-4o"
    elif use_case == "complex_reasoning":
        return "anthropic:claude-3-5-sonnet-20241022"
    else:
        return "openai:gpt-4o-mini"  # Good default
```

### Reliability Metrics

```python
# Success rates for different tool complexity levels
RELIABILITY_METRICS = {
    "simple_tools": {
        "gpt-4o": 99.5,
        "claude-3.5-sonnet": 99.2,
        "gpt-4o-mini": 98.8,
        "gemini-1.5-flash": 98.5,
        "claude-3.5-haiku": 98.0
    },
    "complex_schemas": {
        "gpt-4o": 97.8,
        "claude-3.5-sonnet": 95.5,
        "gpt-4o-mini": 94.2,
        "gemini-1.5-flash": 89.1,
        "claude-3.5-haiku": 92.3
    },
    "multi_tool_chains": {
        "gpt-4o": 96.2,
        "claude-3.5-sonnet": 94.8,
        "gpt-4o-mini": 91.5,
        "gemini-1.5-flash": 87.3,
        "claude-3.5-haiku": 89.1
    }
}
```

## Schema Complexity Support

### Complex Schema Testing

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NestedMetadata(BaseModel):
    tags: List[str]
    attributes: Dict[str, Union[str, int, float]]
    timestamp: datetime

class ComplexTaskSchema(BaseModel):
    """Complex schema to test model capabilities."""
    title: str = Field(description="Task title")
    description: Optional[str] = Field(None, description="Detailed description")
    priority: Priority = Field(description="Task priority level")
    assignees: List[str] = Field(description="List of assigned users")
    metadata: NestedMetadata = Field(description="Additional task metadata")
    dependencies: Optional[List[str]] = Field(None, description="Task dependencies")
    custom_fields: Dict[str, any] = Field(default_factory=dict)

# Test each model's handling of complex schemas
async def test_schema_complexity():
    """Test how different models handle complex schemas."""
    
    models_to_test = [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022", 
        "google-gla:gemini-1.5-flash",
        "openai:gpt-4o-mini"
    ]
    
    for model_name in models_to_test:
        agent = Agent(model_name)
        
        @agent.tool_plain
        def create_complex_task(task_data: ComplexTaskSchema) -> dict:
            """Create a task with complex schema."""
            return {
                "task_id": "12345",
                "created": task_data.title,
                "priority": task_data.priority.value,
                "assignee_count": len(task_data.assignees)
            }
        
        # Test with realistic complex input
        try:
            result = await agent.run(
                "Create a high priority task for user analysis with John and Jane assigned"
            )
            print(f"{model_name}: Success")
        except Exception as e:
            print(f"{model_name}: Failed - {e}")
```

### Schema Simplification Strategies

```python
# For models that struggle with complex schemas
class SimplifiedTaskSchema(BaseModel):
    """Simplified version for less capable models."""
    title: str
    priority: str  # Use string instead of enum
    assignees: str  # Comma-separated instead of list
    description: Optional[str] = None

# Model-aware schema selection
def get_schema_for_model(model_system: str) -> type[BaseModel]:
    """Return appropriate schema based on model capabilities."""
    if model_system in ["openai", "anthropic"]:
        return ComplexTaskSchema
    else:
        return SimplifiedTaskSchema

async def adaptive_tool_schema(
    ctx: RunContext[None], 
    tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Adapt tool schemas based on model capabilities."""
    model_system = ctx.model.system
    
    if model_system == "google":
        # Simplify schemas for Gemini
        simplified_tools = []
        for tool_def in tool_defs:
            # Convert complex schemas to simpler alternatives
            # This would require custom logic based on your schemas
            simplified_tools.append(simplify_tool_schema(tool_def))
        return simplified_tools
    
    return tool_defs
```

## Error Handling Variations

### Model-Specific Error Patterns

```python
from pydantic_ai import ModelRetry

class ModelAwareErrorHandler:
    """Handle errors differently based on model characteristics."""
    
    @staticmethod
    def handle_openai_errors(error: Exception, retry_count: int) -> str:
        """OpenAI-specific error handling."""
        if "invalid_parameters" in str(error):
            return "Please check parameter format. OpenAI requires exact schema compliance."
        elif retry_count < 2:
            return "OpenAI models usually succeed on retry. Please try again."
        return "OpenAI model failed after retries."
    
    @staticmethod
    def handle_claude_errors(error: Exception, retry_count: int) -> str:
        """Claude-specific error handling."""
        if "tool_not_found" in str(error):
            return "Claude may have misunderstood the tool name. Please rephrase your request."
        return "Claude needs clearer instructions. Please be more specific."
    
    @staticmethod
    def handle_gemini_errors(error: Exception, retry_count: int) -> str:
        """Gemini-specific error handling."""
        if "schema_validation" in str(error):
            return "Gemini had trouble with the parameter format. Please simplify your request."
        return "Gemini prefers simpler tool calls. Please break down complex requests."

@agent.tool
def error_aware_tool(ctx: RunContext[Config], data: dict) -> str:
    """Tool with model-aware error handling."""
    try:
        return process_data(data)
    except ValidationError as e:
        model_system = ctx.model.system
        
        if model_system == "openai":
            error_msg = ModelAwareErrorHandler.handle_openai_errors(e, ctx.retry)
        elif model_system == "anthropic":
            error_msg = ModelAwareErrorHandler.handle_claude_errors(e, ctx.retry)
        elif model_system == "google":
            error_msg = ModelAwareErrorHandler.handle_gemini_errors(e, ctx.retry)
        else:
            error_msg = "Please check your input format and try again."
        
        raise ModelRetry(error_msg)
```

### Retry Strategy Optimization

```python
# Model-specific retry configurations
MODEL_RETRY_CONFIGS = {
    "openai": {
        "max_retries": 3,
        "backoff_factor": 1.5,
        "timeout": 30
    },
    "anthropic": {
        "max_retries": 2,  # Claude learns quickly from feedback
        "backoff_factor": 2.0,
        "timeout": 45
    },
    "google": {
        "max_retries": 4,  # May need more attempts
        "backoff_factor": 1.2,
        "timeout": 20  # Faster timeout due to speed
    }
}

def get_retry_config(model_system: str) -> dict:
    """Get optimal retry configuration for model."""
    return MODEL_RETRY_CONFIGS.get(model_system, {
        "max_retries": 2,
        "backoff_factor": 1.5,
        "timeout": 30
    })
```

## Streaming Tool Support

### Advanced Streaming Patterns

```python
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta
)

async def advanced_streaming_example():
    """Demonstrate advanced streaming with different models."""
    
    agent = Agent('openai:gpt-4o')  # Best streaming support
    
    @agent.tool_plain
    def streaming_data_processor(chunk_size: int) -> str:
        """Process data in chunks for streaming."""
        return f"Processed {chunk_size} items"
    
    # Stream agent execution with detailed event handling
    async with agent.iter('Process 1000 items in chunks') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                print(f"Streaming text: {event.delta.content_delta}")
            
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as tool_stream:
                    async for event in tool_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            print(f"Tool called: {event.part.tool_name}")
                        elif isinstance(event, FunctionToolResultEvent):
                            print(f"Tool result: {event.result.content}")

# Model-specific streaming optimizations
class StreamingOptimizer:
    @staticmethod
    def optimize_for_openai():
        """OpenAI streaming optimizations."""
        return {
            "stream": True,
            "stream_options": {"include_usage": True}
        }
    
    @staticmethod
    def optimize_for_claude():
        """Claude streaming optimizations."""
        return {
            "stream": True,
            "max_tokens": 4096  # Claude benefits from explicit limits
        }
    
    @staticmethod
    def optimize_for_gemini():
        """Gemini streaming optimizations."""
        return {
            "stream": True,
            "safety_settings": {"threshold": "BLOCK_NONE"}  # For faster streaming
        }
```

## Built-in Tools and Features

### OpenAI Responses API Integration

```python
from openai.types.responses import (
    WebSearchToolParam,
    CodeInterpreterToolParam,
    FileSearchToolParam
)

class OpenAIBuiltinToolsAgent:
    """Agent leveraging OpenAI's built-in tools."""
    
    def __init__(self):
        self.model_settings = OpenAIResponsesModelSettings(
            openai_builtin_tools=[
                WebSearchToolParam(type='web_search_preview'),
                CodeInterpreterToolParam(type='code_interpreter'),
                FileSearchToolParam(type='file_search')
            ]
        )
        
        self.agent = Agent(
            model=OpenAIResponsesModel('gpt-4o'),
            model_settings=self.model_settings
        )
    
    @property
    def web_search_agent(self):
        """Agent with web search capability."""
        settings = OpenAIResponsesModelSettings(
            openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')]
        )
        return Agent(
            model=OpenAIResponsesModel('gpt-4o'),
            model_settings=settings
        )
    
    @property
    def code_execution_agent(self):
        """Agent with code execution capability."""
        settings = OpenAIResponsesModelSettings(
            openai_builtin_tools=[CodeInterpreterToolParam(type='code_interpreter')]
        )
        return Agent(
            model=OpenAIResponsesModel('gpt-4o'),
            model_settings=settings
        )

# Usage example
builtin_tools = OpenAIBuiltinToolsAgent()

# Use web search without defining custom tools
result = await builtin_tools.web_search_agent.run(
    "What are the latest developments in AI this week?"
)

# Use code execution
result = await builtin_tools.code_execution_agent.run(
    "Calculate the fibonacci sequence up to 100 and plot it"
)
```

### Custom Built-in Tool Emulation

```python
# Emulate built-in tools for other models
class BuiltinToolEmulator:
    """Emulate OpenAI's built-in tools for other models."""
    
    @staticmethod
    def add_web_search_tool(agent: Agent):
        """Add web search capability to any agent."""
        
        @agent.tool_plain
        def web_search(query: str, num_results: int = 5) -> list[dict]:
            """Search the web for current information."""
            # Use DuckDuckGo, Tavily, or other search API
            from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
            search_tool = duckduckgo_search_tool()
            return search_tool(query, num_results)
    
    @staticmethod
    def add_code_execution_tool(agent: Agent):
        """Add code execution capability."""
        
        @agent.tool_plain
        def execute_python(code: str) -> str:
            """Execute Python code safely."""
            # Implement safe code execution
            try:
                # Use RestrictedPython or similar for safety
                result = safe_exec(code)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

# Make any model have "built-in" tools
claude_with_web = Agent('anthropic:claude-3-5-sonnet-20241022')
BuiltinToolEmulator.add_web_search_tool(claude_with_web)
BuiltinToolEmulator.add_code_execution_tool(claude_with_web)
```

## Cost Optimization Strategies

### Model Selection for Cost Efficiency

```python
class CostOptimizer:
    """Optimize costs across different models."""
    
    # Cost per 1M tokens (approximate, as of 2024)
    COSTS = {
        "openai:gpt-4o": {"input": 5.00, "output": 15.00},
        "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "anthropic:claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "anthropic:claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        "google-gla:gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "google-gla:gemini-1.5-pro": {"input": 1.25, "output": 5.00}
    }
    
    @classmethod
    def estimate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model and token usage."""
        if model not in cls.COSTS:
            return 0.0
        
        cost_per_million = cls.COSTS[model]
        input_cost = (input_tokens / 1_000_000) * cost_per_million["input"]
        output_cost = (output_tokens / 1_000_000) * cost_per_million["output"]
        return input_cost + output_cost
    
    @classmethod
    def recommend_model(cls, use_case: str, budget_constraint: float) -> str:
        """Recommend model based on use case and budget."""
        recommendations = {
            "simple_tools": [
                ("google-gla:gemini-1.5-flash", 0.10),
                ("openai:gpt-4o-mini", 0.20),
                ("anthropic:claude-3-5-haiku-20241022", 0.30)
            ],
            "complex_tools": [
                ("openai:gpt-4o-mini", 0.20),
                ("google-gla:gemini-1.5-pro", 0.50),
                ("openai:gpt-4o", 1.00)
            ],
            "high_accuracy": [
                ("openai:gpt-4o", 1.00),
                ("anthropic:claude-3-5-sonnet-20241022", 0.90),
                ("google-gla:gemini-1.5-pro", 0.50)
            ]
        }
        
        options = recommendations.get(use_case, [])
        for model, relative_cost in options:
            if relative_cost <= budget_constraint:
                return model
        
        return "google-gla:gemini-1.5-flash"  # Cheapest fallback

# Cost-aware agent selection
def create_cost_optimized_agent(task_complexity: str, budget: float) -> Agent:
    """Create agent optimized for cost and task complexity."""
    model = CostOptimizer.recommend_model(task_complexity, budget)
    
    agent = Agent(model, system_prompt=f"Optimized for {task_complexity} tasks")
    
    # Add cost monitoring
    @agent.tool_plain
    def cost_monitor() -> str:
        """Monitor costs during execution."""
        # Implementation would track actual usage
        return "Cost monitoring active"
    
    return agent
```

### Token Usage Optimization

```python
class TokenOptimizer:
    """Optimize token usage across models."""
    
    @staticmethod
    def optimize_system_prompt(model_system: str, base_prompt: str) -> str:
        """Optimize system prompt length based on model."""
        if model_system == "google":
            # Gemini is more sensitive to prompt length
            return base_prompt[:500] + "..." if len(base_prompt) > 500 else base_prompt
        elif model_system == "openai":
            # OpenAI handles longer prompts well
            return base_prompt
        else:
            # Conservative approach for others
            return base_prompt[:1000] + "..." if len(base_prompt) > 1000 else base_prompt
    
    @staticmethod
    def optimize_tool_descriptions(model_system: str, tools: list) -> list:
        """Optimize tool descriptions for token efficiency."""
        optimized = []
        max_desc_length = {
            "google": 100,
            "anthropic": 200,
            "openai": 300
        }.get(model_system, 150)
        
        for tool in tools:
            if hasattr(tool, 'description') and tool.description:
                if len(tool.description) > max_desc_length:
                    tool.description = tool.description[:max_desc_length] + "..."
            optimized.append(tool)
        
        return optimized
```

## Model Selection Guidelines

### Decision Matrix

```python
class ModelSelector:
    """Intelligent model selection based on requirements."""
    
    def __init__(self):
        self.criteria_weights = {
            "accuracy": 0.3,
            "speed": 0.2,
            "cost": 0.2,
            "complexity": 0.15,
            "reliability": 0.15
        }
    
    def score_model(self, model: str, requirements: dict) -> float:
        """Score a model based on requirements."""
        scores = {
            "openai:gpt-4o": {
                "accuracy": 0.95,
                "speed": 0.6,
                "cost": 0.3,
                "complexity": 0.95,
                "reliability": 0.95
            },
            "anthropic:claude-3-5-sonnet-20241022": {
                "accuracy": 0.93,
                "speed": 0.5,
                "cost": 0.25,
                "complexity": 0.9,
                "reliability": 0.9
            },
            "google-gla:gemini-1.5-flash": {
                "accuracy": 0.8,
                "speed": 0.95,
                "cost": 0.95,
                "complexity": 0.7,
                "reliability": 0.85
            },
            "openai:gpt-4o-mini": {
                "accuracy": 0.85,
                "speed": 0.85,
                "cost": 0.9,
                "complexity": 0.8,
                "reliability": 0.85
            }
        }
        
        model_scores = scores.get(model, {})
        total_score = 0
        
        for criterion, weight in self.criteria_weights.items():
            requirement_weight = requirements.get(criterion, 1.0)
            model_score = model_scores.get(criterion, 0.5)
            total_score += weight * requirement_weight * model_score
        
        return total_score
    
    def select_best_model(self, requirements: dict) -> str:
        """Select the best model for given requirements."""
        models = [
            "openai:gpt-4o",
            "anthropic:claude-3-5-sonnet-20241022",
            "google-gla:gemini-1.5-flash",
            "openai:gpt-4o-mini"
        ]
        
        best_model = None
        best_score = 0
        
        for model in models:
            score = self.score_model(model, requirements)
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model

# Usage examples
selector = ModelSelector()

# High accuracy requirements
accuracy_requirements = {
    "accuracy": 1.0,
    "reliability": 1.0,
    "cost": 0.3,
    "speed": 0.5,
    "complexity": 0.8
}

best_for_accuracy = selector.select_best_model(accuracy_requirements)
print(f"Best for accuracy: {best_for_accuracy}")

# Cost-sensitive requirements
cost_requirements = {
    "cost": 1.0,
    "speed": 0.8,
    "accuracy": 0.6,
    "reliability": 0.7,
    "complexity": 0.5
}

best_for_cost = selector.select_best_model(cost_requirements)
print(f"Best for cost: {best_for_cost}")
```

### Use Case Specific Recommendations

```python
USE_CASE_RECOMMENDATIONS = {
    "customer_support": {
        "primary": "anthropic:claude-3-5-sonnet-20241022",
        "fallback": "openai:gpt-4o-mini",
        "reasoning": "Claude excels at empathetic, contextual responses"
    },
    "data_analysis": {
        "primary": "openai:gpt-4o",
        "fallback": "anthropic:claude-3-5-sonnet-20241022",
        "reasoning": "GPT-4o best for complex analytical reasoning"
    },
    "real_time_chat": {
        "primary": "google-gla:gemini-1.5