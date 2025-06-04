# Claude 4 Sonnet Complete Tool & Capability Summary

## Model Overview
- **Model**: Claude 4 Sonnet (anthropic:claude-4-sonnet-20250522)
- **Release**: May 22, 2025
- **Context Window**: 200,000 tokens
- **Max Output**: 64,000 tokens
- **Pricing**: $3/M input tokens, $15/M output tokens

## Built-in Tools (Anthropic-Provided)

### 1. Code Execution Tool ⚡
- **Type**: Native Python execution
- **Features**: Secure sandboxed environment, real-time debugging
- **Parameters**:
  - `code` (string, required): Python code to execute
  - `timeout` (integer, optional): Execution timeout (default: 30s)
  - `isolated` (boolean, optional): Isolated environment (default: true)

### 2. Web Search Tool 🌐
- **Type**: Built-in web search during extended thinking
- **Features**: Real-time information retrieval, context-aware queries
- **Activation**: Automatic during extended thinking mode
- **Integration**: Results incorporated into reasoning process

### 3. File API Access 📁
- **Type**: Local file system operations
- **Features**: Read/write files, persistent memory, knowledge storage
- **Capabilities**: Extract key facts, maintain session continuity
- **Requirements**: Developer-granted permissions

### 4. MCP Connector 🔌
- **Type**: Model Context Protocol integration
- **Features**: External data sources, standardized communication
- **Benefits**: Streamlined tool integration, extensible architecture

## Advanced Capabilities

### Extended Thinking Mode 🧠
- **Feature**: Deep reasoning with tool access
- **Modes**: Fast responses vs. extended thinking
- **Tool Integration**: Use tools during reasoning process
- **Benefits**: Complex problem solving, multi-step analysis

### Parallel Tool Execution 🔀
- **Feature**: Multiple tools simultaneously
- **Benefits**: Reduced response time, efficient workflows
- **Use Cases**: Complex orchestration, batch operations

### Memory Capabilities 💾
- **Feature**: Persistent context across sessions
- **Implementation**: File-based storage
- **Benefits**: Continuity, accumulated knowledge, personalization

### Enhanced Function Calling 🛠️
- **Feature**: Advanced custom tool support
- **Validation**: JSON schema, parameter type checking
- **Error Handling**: Retries, structured error responses
- **Output Types**: Structured data, multi-modal content

## Custom Tool Types (Pydantic AI)

### Simple Tools 📋
**No context required - standalone functions**

Examples:
```python
@agent.tool_plain
def calculate_math(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

@agent.tool_plain
def format_text(text: str, operation: str = "lowercase") -> str:
    """Format text with operations."""
    return getattr(text, operation)()

@agent.tool_plain
def get_timestamp(timezone: str = "UTC") -> str:
    """Get current timestamp."""
    return datetime.now().isoformat()
```

Common Use Cases:
- Mathematical calculations
- Text processing and formatting
- Data validation (email, URLs)
- UUID generation
- Time/date operations
- File format conversions

### Context-Aware Tools 🔧
**Require RunContext for dependency injection**

Examples:
```python
@agent.tool
def get_user_data(ctx: RunContext[AppContext], user_id: str) -> dict:
    """Get user data with context."""
    return ctx.deps.database.get_user(user_id)

@agent.tool
def execute_query(ctx: RunContext[AppContext], query: str) -> dict:
    """Execute database query with permissions."""
    if "db_access" not in ctx.deps.user.permissions:
        raise ModelRetry("Access denied")
    return ctx.deps.database.execute(query)
```

Common Use Cases:
- Database operations
- User-specific actions
- Authentication-required operations
- Session management
- Permission-based access
- Multi-tenant operations

## Tool Configuration Options

### Strict Mode 🔒
```python
@agent.tool_plain(strict=True)
def strict_tool(param: str) -> str:
    """100% schema compliance enforced."""
    return param.upper()
```

### Retry Configuration 🔄
```python
@agent.tool(retries=3)
def unreliable_service(ctx: RunContext[Context]) -> dict:
    """Automatic retry on failures."""
    if ctx.retry < 2:
        raise ModelRetry("Service unavailable, retrying...")
    return {"success": True, "attempt": ctx.retry + 1}
```

### Conditional Tool Availability 🎛️
```python
async def filter_tools(ctx: RunContext[Context], tools: List[ToolDefinition]) -> List[ToolDefinition]:
    """Dynamic tool filtering based on context."""
    user_permissions = ctx.deps.user.permissions
    return [tool for tool in tools if tool.name in user_permissions]

agent = Agent(model, prepare_tools=filter_tools)
```

## Supported Parameter Types

### Primitive Types
- `str`: Text strings
- `int`: Integers
- `float`: Floating point numbers
- `bool`: Boolean values

### Complex Types
- `List[T]`: Arrays of typed elements
- `Dict[str, T]`: Object mappings
- `Optional[T]`: Nullable parameters
- `Union[T1, T2]`: Multiple possible types
- `Literal["a", "b"]`: Enumerated values

### Structured Data
```python
class UserProfile(BaseModel):
    name: str
    email: str
    age: Optional[int] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

@agent.tool_plain
def create_user(profile: UserProfile) -> str:
    """Create user with validated data."""
    return f"Created user: {profile.name}"
```

## Return Types

### Simple Returns
- Primitive values (string, number, boolean)
- Dictionaries and lists
- JSON-serializable objects

### Structured Returns
```python
class WeatherReport(BaseModel):
    temperature: float
    conditions: str
    humidity: int

@agent.tool_plain
def get_weather(location: str) -> WeatherReport:
    """Return structured weather data."""
    return WeatherReport(temperature=22.5, conditions="Sunny", humidity=65)
```

### Multi-modal Returns
```python
from pydantic_ai import ImageUrl, DocumentUrl

@agent.tool_plain
def generate_chart() -> ImageUrl:
    """Return image URL."""
    return ImageUrl(url="https://example.com/chart.png")

@agent.tool_plain
def create_report() -> DocumentUrl:
    """Return document URL."""
    return DocumentUrl(url="https://example.com/report.pdf")
```

## Error Handling

### ModelRetry for Recoverable Errors
```python
@agent.tool_plain
def external_api_call(endpoint: str) -> dict:
    """Call external API with retry logic."""
    try:
        response = requests.get(endpoint, timeout=10)
        return response.json()
    except requests.Timeout:
        raise ModelRetry("API timeout. Please retry with different parameters.")
    except requests.ConnectionError:
        raise ModelRetry("Connection failed. Please check network and retry.")
```

### Validation Errors
```python
@agent.tool_plain
def process_data(data: List[int]) -> dict:
    """Process numeric data with validation."""
    if not data:
        raise ValueError("Data list cannot be empty")
    if any(x < 0 for x in data):
        raise ValueError("All values must be non-negative")
    return {"sum": sum(data), "avg": sum(data) / len(data)}
```

## Integration Features

### Streaming Support 🌊
- Real-time response streaming
- Progressive tool execution
- Live result updates

### Testing with TestModel 🧪
```python
from pydantic_ai.models.test import TestModel

test_model = TestModel()
with agent.override(model=test_model):
    result = agent.run_sync("test query")
    tools = test_model.last_model_request_parameters.function_tools
    # Analyze tool schemas without API costs
```

### Usage Monitoring 📊
- Token usage tracking
- Request counting
- Cost monitoring
- Performance metrics

## Production Features

### Security 🛡️
- Input validation and sanitization
- Permission-based tool access
- Safe execution environments
- Audit logging capabilities

### Performance ⚡
- Tool result caching
- Batch operations
- Parallel execution
- Resource optimization

### CI/CD Integration 🔄
- Automated tool validation
- Schema verification
- TestModel-based testing
- Zero-cost tool inspection

## External Integrations

### Common Tool Libraries
```python
# DuckDuckGo Search
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
agent = Agent(tools=[duckduckgo_search_tool()])

# Tavily Search (Premium)
from pydantic_ai.common_tools.tavily import tavily_search_tool
agent = Agent(tools=[tavily_search_tool(api_key)])
```

### MCP Server Support
```python
from pydantic_ai.mcp import MCPServerHTTP, MCPServerStdio

# HTTP MCP Server
weather_server = MCPServerHTTP(url='http://localhost:3001/sse')

# Stdio MCP Server
python_server = MCPServerStdio('python', args=['-m', 'mcp_server'])

agent = Agent('claude-4-sonnet', mcp_servers=[weather_server, python_server])
```

## Tool Development Best Practices

### 1. Clear Documentation
```python
@agent.tool_plain
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compound_frequency: int = 12
) -> Dict[str, float]:
    """
    Calculate compound interest for investment.
    
    Args:
        principal: Initial investment amount in dollars
        rate: Annual interest rate as decimal (e.g., 0.05 for 5%)
        time: Investment period in years
        compound_frequency: Compounding frequency per year (default: monthly)
    
    Returns:
        Dictionary with final amount, interest earned, and effective rate
    """
    amount = principal * (1 + rate/compound_frequency) ** (compound_frequency * time)
    interest = amount - principal
    effective_rate = (amount / principal) ** (1/time) - 1
    
    return {
        "final_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
        "effective_annual_rate": round(effective_rate, 4)
    }
```

### 2. Type Safety
```python
from typing import Literal
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
) -> Dict[str, Any]:
    """Update order with type-safe status enum."""
    return {
        "order_id": order_id,
        "status": status.value,
        "tracking_number": tracking_number,
        "updated_at": datetime.now().isoformat()
    }
```

### 3. Error Handling
```python
@agent.tool_plain
def safe_division(dividend: float, divisor: float) -> Dict[str, Any]:
    """Perform division with comprehensive error handling."""
    try:
        if divisor == 0:
            raise ModelRetry("Division by zero. Please provide a non-zero divisor.")
        
        result = dividend / divisor
        
        return {
            "dividend": dividend,
            "divisor": divisor,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "dividend": dividend,
            "divisor": divisor,
            "error": str(e),
            "success": False
        }
```

### 4. Performance Optimization
```python
from functools import lru_cache
import asyncio

@lru_cache(maxsize=1000)
def expensive_calculation(value: int) -> int:
    """Cached expensive operation."""
    return value ** 3

@agent.tool_plain
def batch_process(values: List[int]) -> List[int]:
    """Process multiple values efficiently."""
    return [expensive_calculation(v) for v in values]
```

## Complete Feature Matrix

| Feature | Built-in | Custom | Status |
|---------|----------|--------|--------|
| Code Execution | ✅ | ✅ | Available |
| Web Search | ✅ | ✅ | Available |
| File Access | ✅ | ✅ | Available |
| Database Operations | ❌ | ✅ | Custom Only |
| API Integrations | ❌ | ✅ | Custom Only |
| Extended Thinking | ✅ | N/A | Built-in |
| Parallel Execution | ✅ | ✅ | Available |
| Memory/Persistence | ✅ | ✅ | Available |
| Structured Outputs | ✅ | ✅ | Available |
| Error Recovery | ✅ | ✅ | Available |
| Streaming | ✅ | ✅ | Available |
| Testing (TestModel) | N/A | ✅ | Development |

## Key Insights

1. **Built-in Tools**: 4 core tools (code execution, web search, file access, MCP)
2. **Custom Tools**: Unlimited via Pydantic AI registration
3. **Advanced Features**: Extended thinking, parallel execution, memory
4. **Zero-Cost Testing**: TestModel for development and CI/CD
5. **Production Ready**: Comprehensive error handling, monitoring, security
6. **Type Safety**: Full Pydantic validation for inputs and outputs
7. **Flexible Architecture**: Simple tools to complex context-aware operations

This represents the complete Claude 4 Sonnet tool ecosystem as of May 2025, providing both Anthropic's built-in capabilities and unlimited custom tool extensibility through Pydantic AI.