# Pydantic AI Tools Guide

## Table of Contents

1. [Introduction to Tools](#introduction-to-tools)
2. [How Tools Work](#how-tools-work)
3. [Tool Registration Methods](#tool-registration-methods)
4. [Type Safety and Validation](#type-safety-and-validation)
5. [Dependency Injection with RunContext](#dependency-injection-with-runcontext)
6. [Dynamic Tool Control](#dynamic-tool-control)
7. [Error Handling and Retries](#error-handling-and-retries)
8. [Advanced Tool Types](#advanced-tool-types)
9. [Model-Specific Differences](#model-specific-differences)
10. [Best Practices](#best-practices)
11. [Real-World Examples](#real-world-examples)

## Introduction to Tools

Tools in Pydantic AI are functions that LLMs can call during conversations to perform specific tasks, retrieve information, or execute actions. They bridge the gap between the LLM's text generation capabilities and real-world functionality, enabling agents to:

- Access external APIs and databases
- Perform calculations and data processing
- Retrieve real-time information
- Execute system operations
- Integrate with other services

### Core Concept

When an LLM needs to perform a task it can't handle with pure text generation (like getting current time, accessing databases, or performing calculations), it can call these tools and use their outputs to generate responses.

```python
from pydantic_ai import Agent, RunContext
import random

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful assistant that can roll dice.'
)

@agent.tool_plain
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

# When you ask "roll a die", the LLM will call this tool
result = agent.run_sync('Please roll a die for me')
print(result.output)  # The LLM uses the tool result in its response
```

## How Tools Work

### Tool Execution Flow

The interaction follows this sequence:

1. User sends a message to the agent
2. Agent sends system prompt + user message to LLM
3. LLM decides it needs to use a tool
4. LLM generates a tool call with appropriate arguments
5. Agent validates arguments and executes the tool function
6. Tool returns result to agent
7. Agent sends tool result back to LLM
8. LLM incorporates tool result into final response
9. Agent returns final response to user

### Tool Schema Generation

Pydantic AI automatically generates JSON schemas for your tools based on function signatures and type hints:

```python
@agent.tool_plain
def search_products(query: str, max_results: int = 10) -> list[str]:
    """Search for products in the catalog."""
    # Implementation here
    pass

# Generated schema:
# {
#   "name": "search_products",
#   "description": "Search for products in the catalog.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "query": {"type": "string"},
#       "max_results": {"type": "integer", "default": 10}
#     },
#     "required": ["query"]
#   }
# }
```

## Tool Registration Methods

### Using Decorators

#### @agent.tool_plain - For Simple Functions

```python
@agent.tool_plain
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().isoformat()
```

#### @agent.tool - For Context-Aware Functions

```python
@agent.tool
def get_user_data(ctx: RunContext[DatabaseConnection], user_id: int) -> dict:
    """Get user data from database."""
    return ctx.deps.get_user(user_id)
```

### Via Agent Constructor

```python
def my_tool() -> str:
    return "Tool result"

def another_tool(ctx: RunContext[str]) -> str:
    return f"Hello {ctx.deps}"

agent = Agent(
    'openai:gpt-4o',
    tools=[
        my_tool,  # Simple function
        Tool(another_tool, takes_ctx=True)  # Explicit control
    ]
)
```

### Using Tool Class

```python
from pydantic_ai.tools import Tool

def greet(name: str) -> str:
    return f"Hello {name}!"

greet_tool = Tool(
    greet,
    takes_ctx=False,
    description="Custom greeting function"
)

agent = Agent('openai:gpt-4o', tools=[greet_tool])
```

## Type Safety and Validation

### Automatic Schema Generation

Pydantic AI automatically generates JSON schemas and validates arguments:

```python
from pydantic import BaseModel
from typing import Optional

class SearchParams(BaseModel):
    """Parameters for searching"""
    query: str
    category: Optional[str] = None
    max_results: int = 10
    include_inactive: bool = False

@agent.tool_plain
def search_items(params: SearchParams) -> list[dict]:
    """Search for items with filtering options."""
    # Pydantic automatically validates the params
    return perform_search(
        params.query, 
        params.category, 
        params.max_results,
        params.include_inactive
    )
```

### Complex Type Support

```python
from typing import List, Dict, Union
from datetime import datetime

@agent.tool_plain
def process_data(
    items: List[str],
    metadata: Dict[str, Union[str, int]],
    timestamp: datetime
) -> Dict[str, any]:
    """Process data with complex types."""
    return {
        "processed_items": len(items),
        "metadata_keys": list(metadata.keys()),
        "processed_at": timestamp.isoformat()
    }
```

## Dependency Injection with RunContext

### Basic Dependency Injection

```python
from dataclasses import dataclass

@dataclass
class AppDependencies:
    database: DatabaseConnection
    api_client: APIClient
    config: AppConfig

agent = Agent(
    'openai:gpt-4o',
    deps_type=AppDependencies
)

@agent.tool
def get_user_orders(ctx: RunContext[AppDependencies], user_id: int) -> list[dict]:
    """Get user's order history."""
    # Access injected dependencies
    orders = ctx.deps.database.get_orders(user_id)
    return [order.to_dict() for order in orders]

# Run with dependencies
deps = AppDependencies(
    database=db_connection,
    api_client=api_client,
    config=app_config
)

result = agent.run_sync('Get orders for user 123', deps=deps)
```

### Dynamic System Prompts with Dependencies

```python
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[AppDependencies]) -> str:
    user_role = await ctx.deps.database.get_user_role(ctx.deps.current_user_id)
    return f"You are a {user_role} assistant. Respond appropriately for this role."
```

## Dynamic Tool Control

### Conditional Tool Availability

```python
from typing import Union
from pydantic_ai.tools import ToolDefinition

async def filter_tools_by_permission(
    ctx: RunContext[UserSession], 
    tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    """Filter tools based on user permissions."""
    user_permissions = ctx.deps.user_permissions
    
    filtered_tools = []
    for tool_def in tool_defs:
        if tool_def.name in user_permissions:
            filtered_tools.append(tool_def)
    
    return filtered_tools

agent = Agent(
    'openai:gpt-4o',
    tools=[admin_tool, user_tool, public_tool],
    prepare_tools=filter_tools_by_permission,
    deps_type=UserSession
)
```

### Model-Specific Tool Configuration

```python
from dataclasses import replace

async def configure_tools_for_model(
    ctx: RunContext[None], 
    tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Configure tools based on the model being used."""
    
    if ctx.model.system == 'openai':
        # Enable strict mode for OpenAI models
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    
    elif ctx.model.system == 'anthropic':
        # Simplify descriptions for Claude
        simplified_tools = []
        for tool_def in tool_defs:
            simplified_def = replace(
                tool_def,
                description=tool_def.description[:100]  # Truncate long descriptions
            )
            simplified_tools.append(simplified_def)
        return simplified_tools
    
    return tool_defs

agent = Agent(
    'openai:gpt-4o',
    prepare_tools=configure_tools_for_model
)
```

## Error Handling and Retries

### Using ModelRetry

```python
from pydantic_ai import ModelRetry

@agent.tool_plain
def external_api_call(endpoint: str, params: dict) -> dict:
    """Call external API with error handling."""
    try:
        response = api_client.get(endpoint, params=params)
        if response.status_code == 400:
            raise ModelRetry(
                f"Invalid parameters for {endpoint}. "
                f"Please check the parameter format and try again."
            )
        return response.json()
    except ConnectionError:
        raise ModelRetry(
            f"Cannot connect to {endpoint}. Please try again later."
        )
```

### Tool-Level Retry Configuration

```python
@agent.tool(retries=3)
def unreliable_service(ctx: RunContext[Config], query: str) -> str:
    """Call an unreliable external service."""
    if ctx.retry < 2:  # Fail first two attempts
        raise ModelRetry("Service temporarily unavailable. Please retry.")
    return f"Success on attempt {ctx.retry + 1}: {query}"
```

### Graceful Degradation

```python
@agent.tool_plain
def get_weather_with_fallback(location: str) -> str:
    """Get weather with multiple fallback sources."""
    try:
        return primary_weather_api.get_weather(location)
    except Exception:
        try:
            return secondary_weather_api.get_weather(location)
        except Exception:
            return f"Weather data unavailable for {location}"
```

## Advanced Tool Types

### Returning Structured Data

```python
from pydantic import BaseModel
from datetime import datetime

class WeatherReport(BaseModel):
    location: str
    temperature: float
    humidity: int
    conditions: str
    timestamp: datetime

@agent.tool_plain
def get_detailed_weather(location: str) -> WeatherReport:
    """Get detailed weather information."""
    return WeatherReport(
        location=location,
        temperature=22.5,
        humidity=65,
        conditions="Partly cloudy",
        timestamp=datetime.now()
    )
```

### Multi-Modal Returns

```python
from pydantic_ai import ImageUrl, DocumentUrl

@agent.tool_plain
def generate_chart(data_type: str) -> ImageUrl:
    """Generate a chart image."""
    chart_url = chart_service.create_chart(data_type)
    return ImageUrl(url=chart_url)

@agent.tool_plain
def get_report(report_id: str) -> DocumentUrl:
    """Get a PDF report."""
    report_url = report_service.get_report_url(report_id)
    return DocumentUrl(url=report_url)
```

### Async Tools

```python
@agent.tool
async def async_database_query(
    ctx: RunContext[AsyncDatabase], 
    query: str
) -> list[dict]:
    """Execute async database query."""
    results = await ctx.deps.execute_query(query)
    return [row.to_dict() for row in results]
```

## Model-Specific Differences

### Function Calling Support Tiers

#### Tier 1: Full Function Calling Support
- **OpenAI models** (GPT-4, GPT-3.5-turbo): Native, robust function calling
- **Anthropic Claude**: Excellent function calling via "tool use"
- **Google Gemini**: Good function calling support

#### Tier 2: Limited Support
- **Older models**: May not support function calling
- **Some open-source models**: Limited or inconsistent function calling

### Reliability Comparison

```python
# Model reliability ranking for tool calling:

# 1. OpenAI GPT-4: Very reliable
#    - Rarely calls wrong tools
#    - Excellent parameter extraction
#    - Handles complex schemas well

# 2. Anthropic Claude 3.5: Very reliable
#    - Excellent at understanding when to use tools
#    - Good parameter handling
#    - Great contextual understanding

# 3. Google Gemini 1.5: Generally reliable
#    - Good tool selection
#    - Occasional parameter issues with complex schemas
#    - Fast execution

# 4. Smaller/older models: Less reliable
#    - May hallucinate tool calls
#    - Poor parameter extraction
#    - Limited schema support
```

### Performance Characteristics

```python
# Speed ranking (fastest to slowest for tool calling):
# 1. Gemini Flash - Very fast, good for simple tools
# 2. GPT-4o-mini - Fast and cost-effective
# 3. Claude 3.5 Haiku - Fast with good reliability
# 4. GPT-4o - Slower but most reliable
# 5. Claude 3.5 Sonnet - Slower but excellent quality

# Cost considerations:
# - Gemini: Often cheapest for tool calls
# - OpenAI: Mid-range pricing, good value
# - Claude: More expensive but high quality
```

### Built-in Tools Support

```python
# OpenAI Responses API with built-in tools
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from openai.types.responses import WebSearchToolParam

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)

agent = Agent(
    model=OpenAIResponsesModel('gpt-4o'),
    model_settings=model_settings
)
```

## Best Practices

### 1. Tool Design Principles

```python
# ✅ Good: Clear, specific tool with good documentation
@agent.tool_plain
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        Dictionary with price, change, and timestamp
    """
    return stock_api.get_price(symbol.upper())

# ❌ Bad: Vague tool with poor documentation
@agent.tool_plain
def get_data(thing: str) -> str:
    """Get some data."""
    return database.query(thing)
```

### 2. Error Handling Best Practices

```python
@agent.tool_plain
def robust_api_call(endpoint: str, data: dict) -> dict:
    """Make API call with comprehensive error handling."""
    try:
        response = requests.post(endpoint, json=data, timeout=30)
        
        if response.status_code == 400:
            raise ModelRetry(
                f"Invalid request data. Please check: {response.json().get('error', 'Unknown error')}"
            )
        elif response.status_code == 401:
            raise ModelRetry("Authentication failed. Please check credentials.")
        elif response.status_code >= 500:
            raise ModelRetry("Server error. Please try again later.")
        
        return response.json()
        
    except requests.Timeout:
        raise ModelRetry("Request timeout. Please try again.")
    except requests.ConnectionError:
        raise ModelRetry("Connection error. Please check network and try again.")
```

### 3. Type Safety Guidelines

```python
from typing import Literal, Optional
from enum import Enum

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"

@agent.tool_plain
def update_order_status(
    order_id: str,
    status: OrderStatus,
    tracking_number: Optional[str] = None
) -> dict:
    """Update order status with type-safe enum."""
    return order_service.update_status(order_id, status.value, tracking_number)
```

### 4. Performance Optimization

```python
from functools import lru_cache
import asyncio

# Cache expensive operations
@lru_cache(maxsize=100)
def get_expensive_data(key: str) -> dict:
    """Cached expensive operation."""
    return expensive_api_call(key)

@agent.tool_plain
def cached_lookup(search_key: str) -> dict:
    """Lookup with caching."""
    return get_expensive_data(search_key)

# Batch operations when possible
@agent.tool_plain
def get_multiple_users(user_ids: list[str]) -> list[dict]:
    """Get multiple users in one call instead of individual calls."""
    return database.get_users_batch(user_ids)
```

### 5. Security Considerations

```python
import re
from typing import Literal

@agent.tool
def safe_database_query(
    ctx: RunContext[DatabaseConnection],
    table: Literal["users", "orders", "products"],
    filters: dict
) -> list[dict]:
    """Safe database query with restricted tables and sanitized input."""
    
    # Validate table name (already restricted by Literal type)
    allowed_tables = ["users", "orders", "products"]
    if table not in allowed_tables:
        raise ValueError(f"Table {table} not allowed")
    
    # Sanitize filter values
    sanitized_filters = {}
    for key, value in filters.items():
        if isinstance(value, str):
            # Basic SQL injection prevention
            if re.search(r'[;\'"\\]', value):
                raise ModelRetry(f"Invalid characters in filter value: {value}")
        sanitized_filters[key] = value
    
    return ctx.deps.safe_query(table, sanitized_filters)
```

### 6. Testing Tools

```python
from pydantic_ai.models.test import TestModel
import pytest

@pytest.mark.asyncio
async def test_user_tool():
    """Test user-related tools."""
    test_model = TestModel()
    
    with agent.override(model=test_model):
        result = await agent.run('Get user data for ID 123', deps=test_deps)
        
        # Verify tool was called
        assert len(test_model.last_model_request_parameters.function_tools) > 0
        
        # Verify response
        assert "user" in result.output.lower()
```

## Real-World Examples

### Customer Support Agent

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional

@dataclass
class SupportDependencies:
    customer_id: int
    database: CustomerDatabase
    ticket_system: TicketSystem

class SupportResponse(BaseModel):
    response: str = Field(description="Response to customer")
    escalate: bool = Field(description="Whether to escalate to human")
    urgency: int = Field(description="Urgency level 1-5", ge=1, le=5)

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    output_type=SupportResponse,
    system_prompt="You are a helpful customer support agent."
)

@support_agent.tool
async def get_order_status(
    ctx: RunContext[SupportDependencies], 
    order_id: str
) -> dict:
    """Get order status and tracking information."""
    order = await ctx.deps.database.get_order(order_id)
    if not order or order.customer_id != ctx.deps.customer_id:
        raise ModelRetry("Order not found or not accessible to this customer.")
    
    return {
        "order_id": order.id,
        "status": order.status,
        "tracking_number": order.tracking_number,
        "estimated_delivery": order.estimated_delivery.isoformat()
    }

@support_agent.tool
async def create_support_ticket(
    ctx: RunContext[SupportDependencies],
    subject: str,
    description: str,
    category: Literal["billing", "technical", "shipping", "general"]
) -> str:
    """Create a support ticket for the customer."""
    ticket = await ctx.deps.ticket_system.create_ticket(
        customer_id=ctx.deps.customer_id,
        subject=subject,
        description=description,
        category=category
    )
    return f"Support ticket #{ticket.id} created successfully."
```

### Data Analysis Agent

```python
import pandas as pd
from typing import Union

@dataclass
class AnalyticsDependencies:
    data_warehouse: DataWarehouse
    visualization_service: VisualizationService

analytics_agent = Agent(
    'openai:gpt-4o',
    deps_type=AnalyticsDependencies,
    system_prompt="You are a data analyst. Help users analyze their data."
)

@analytics_agent.tool
async def query_sales_data(
    ctx: RunContext[AnalyticsDependencies],
    start_date: str,
    end_date: str,
    product_category: Optional[str] = None
) -> dict:
    """Query sales data for a date range."""
    data = await ctx.deps.data_warehouse.get_sales_data(
        start_date=start_date,
        end_date=end_date,
        category=product_category
    )
    
    df = pd.DataFrame(data)
    
    return {
        "total_sales": float(df['amount'].sum()),
        "transaction_count": len(df),
        "average_order_value": float(df['amount'].mean()),
        "top_products": df.groupby('product')['amount'].sum().head(5).to_dict()
    }

@analytics_agent.tool
async def create_chart(
    ctx: RunContext[AnalyticsDependencies],
    chart_type: Literal["bar", "line", "pie"],
    data_query: str,
    title: str
) -> ImageUrl:
    """Create a chart from data query."""
    chart_url = await ctx.deps.visualization_service.create_chart(
        chart_type=chart_type,
        query=data_query,
        title=title
    )
    return ImageUrl(url=chart_url)
```

### DevOps Assistant

```python
@dataclass
class DevOpsDependencies:
    kubernetes_client: KubernetesClient
    monitoring_service: MonitoringService
    deployment_service: DeploymentService

devops_agent = Agent(
    'anthropic:claude-3-5-sonnet-20241022',
    deps_type=DevOpsDependencies,
    system_prompt="You are a DevOps assistant. Help with deployments and monitoring."
)

@devops_agent.tool
async def get_pod_status(
    ctx: RunContext[DevOpsDependencies],
    namespace: str,
    app_name: Optional[str] = None
) -> list[dict]:
    """Get status of pods in a namespace."""
    pods = await ctx.deps.kubernetes_client.get_pods(
        namespace=namespace,
        label_selector=f"app={app_name}" if app_name else None
    )
    
    return [
        {
            "name": pod.name,
            "status": pod.status.phase,
            "ready": pod.status.ready,
            "restarts": pod.status.restart_count,
            "age": (datetime.now() - pod.metadata.creation_timestamp).days
        }
        for pod in pods
    ]

@devops_agent.tool
async def scale_deployment(
    ctx: RunContext[DevOpsDependencies],
    deployment_name: str,
    namespace: str,
    replicas: int
) -> str:
    """Scale a deployment to specified number of replicas."""
    if replicas < 0 or replicas > 50:
        raise ModelRetry("Replica count must be between 0 and 50.")
    
    await ctx.deps.kubernetes_client.scale_deployment(
        name=deployment_name,
        namespace=namespace,
        replicas=replicas
    )
    
    return f"Deployment {deployment_name} scaled to {replicas} replicas."

@devops_agent.tool
async def get_service_metrics(
    ctx: RunContext[DevOpsDependencies],
    service_name: str,
    time_range: Literal["1h", "6h", "24h", "7d"] = "1h"
) -> dict:
    """Get metrics for a service."""
    metrics = await ctx.deps.monitoring_service.get_metrics(
        service=service_name,
        time_range=time_range
    )
    
    return {
        "cpu_usage": metrics.avg_cpu_percent,
        "memory_usage": metrics.avg_memory_percent,
        "request_rate": metrics.requests_per_second,
        "error_rate": metrics.error_percentage,
        "response_time_p95": metrics.response_time_p95_ms
    }
```

This comprehensive guide covers all aspects of working with tools in Pydantic AI, from basic concepts to advanced patterns and real-world applications. The key is to start simple and gradually build more sophisticated tool chains as your understanding grows.