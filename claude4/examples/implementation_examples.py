#!/usr/bin/env python3
"""
Claude 4 Sonnet Implementation Examples
Complete technical examples for all Claude 4 Sonnet capabilities.
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic import BaseModel, Field


# =============================================================================
# DEPENDENCY INJECTION SETUP
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration for context injection."""
    connection_string: str
    timeout: int
    pool_size: int
    ssl_enabled: bool

@dataclass
class APICredentials:
    """API credentials for external services."""
    openai_key: str
    anthropic_key: str
    google_key: str
    
@dataclass
class UserSession:
    """User session context."""
    user_id: str
    session_id: str
    permissions: List[str]
    tenant_id: str

@dataclass
class ApplicationContext:
    """Complete application context for dependency injection."""
    user: UserSession
    database: DatabaseConfig
    credentials: APICredentials
    environment: Literal["dev", "staging", "prod"]


# =============================================================================
# STRUCTURED DATA MODELS
# =============================================================================

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    """Structured task representation."""
    id: str
    title: str
    description: Optional[str] = None
    priority: Priority
    status: TaskStatus = TaskStatus.PENDING
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    estimated_hours: Optional[float] = None

class SearchResult(BaseModel):
    """Web search result structure."""
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: Literal["web", "academic", "news"] = "web"

class CodeExecutionResult(BaseModel):
    """Code execution result structure."""
    success: bool
    output: str
    error_message: Optional[str] = None
    execution_time: float
    memory_used: int  # bytes


# =============================================================================
# SIMPLE TOOLS (NO CONTEXT REQUIRED)
# =============================================================================

def create_simple_tools_agent() -> Agent:
    """Create agent with simple tools that don't require context."""
    
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        system_prompt="You are Claude 4 Sonnet with advanced tool capabilities."
    )
    
    @agent.tool_plain
    def get_current_time(timezone: str = "UTC") -> str:
        """Get current time in specified timezone."""
        return datetime.now().isoformat() + f" ({timezone})"
    
    @agent.tool_plain
    def calculate_math(expression: str, precision: int = 2) -> Dict[str, Any]:
        """Calculate mathematical expressions with specified precision."""
        try:
            result = eval(expression.replace("^", "**"))  # Safe for demo
            return {
                "expression": expression,
                "result": round(float(result), precision),
                "precision": precision,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
    
    @agent.tool_plain
    def format_text(
        text: str,
        operation: Literal["uppercase", "lowercase", "title", "reverse"] = "lowercase",
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Format text with various operations."""
        operations = {
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "title": text.title(),
            "reverse": text[::-1]
        }
        
        result = operations[operation]
        if max_length:
            result = result[:max_length]
        
        return {
            "original": text,
            "formatted": result,
            "operation": operation,
            "length": len(result),
            "truncated": max_length and len(text) > max_length
        }
    
    @agent.tool_plain
    def generate_uuid(version: int = 4, count: int = 1) -> List[str]:
        """Generate UUID strings."""
        import uuid
        
        generators = {
            1: uuid.uuid1,
            4: uuid.uuid4
        }
        
        generator = generators.get(version, uuid.uuid4)
        return [str(generator()) for _ in range(min(count, 10))]
    
    @agent.tool_plain
    def validate_email(email: str) -> Dict[str, Any]:
        """Validate email address format."""
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = re.match(pattern, email) is not None
        
        return {
            "email": email,
            "valid": is_valid,
            "domain": email.split("@")[1] if "@" in email else None,
            "local_part": email.split("@")[0] if "@" in email else None
        }
    
    return agent


# =============================================================================
# CONTEXT-AWARE TOOLS (WITH DEPENDENCY INJECTION)
# =============================================================================

def create_context_aware_agent() -> Agent:
    """Create agent with context-aware tools using dependency injection."""
    
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        deps_type=ApplicationContext,
        system_prompt="You are Claude 4 Sonnet with full access to application context and services."
    )
    
    @agent.tool
    def get_user_profile(
        ctx: RunContext[ApplicationContext],
        target_user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user profile with permission checking."""
        user_id = target_user or ctx.deps.user.user_id
        
        # Check permissions
        if target_user and "read_users" not in ctx.deps.user.permissions:
            raise ModelRetry("Insufficient permissions to read other user profiles")
        
        return {
            "user_id": user_id,
            "session_id": ctx.deps.user.session_id,
            "permissions": ctx.deps.user.permissions,
            "tenant_id": ctx.deps.user.tenant_id,
            "environment": ctx.deps.environment,
            "database_pool_size": ctx.deps.database.pool_size
        }
    
    @agent.tool
    def execute_database_query(
        ctx: RunContext[ApplicationContext],
        query: str,
        table: str,
        limit: int = 100,
        safe_mode: bool = True
    ) -> Dict[str, Any]:
        """Execute database query with context and safety checks."""
        
        # Permission check
        if "database_access" not in ctx.deps.user.permissions:
            raise ModelRetry("Database access denied. Required permission: database_access")
        
        # Safety checks
        dangerous_ops = ["drop", "delete", "truncate", "alter"]
        if safe_mode and any(op in query.lower() for op in dangerous_ops):
            raise ModelRetry(f"Dangerous operation detected in safe mode: {query}")
        
        # Environment-specific connection
        db_config = ctx.deps.database
        connection_info = {
            "connection": db_config.connection_string,
            "timeout": db_config.timeout,
            "ssl": db_config.ssl_enabled
        }
        
        return {
            "query": query,
            "table": table,
            "limit": limit,
            "user_id": ctx.deps.user.user_id,
            "tenant_id": ctx.deps.user.tenant_id,
            "environment": ctx.deps.environment,
            "connection_info": connection_info,
            "safe_mode": safe_mode,
            "simulated_result_count": min(limit, 42),
            "execution_time_ms": 150
        }
    
    @agent.tool
    def call_external_api(
        ctx: RunContext[ApplicationContext],
        service: Literal["openai", "google", "custom"],
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE"] = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call external API using stored credentials."""
        
        # Get appropriate API key
        credentials = ctx.deps.credentials
        api_keys = {
            "openai": credentials.openai_key,
            "google": credentials.google_key,
            "custom": "custom_api_key"
        }
        
        api_key = api_keys.get(service)
        if not api_key:
            raise ModelRetry(f"No API key configured for service: {service}")
        
        return {
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "data": data,
            "user_id": ctx.deps.user.user_id,
            "environment": ctx.deps.environment,
            "api_key_present": bool(api_key),
            "simulated_response": {
                "status": 200,
                "data": f"Response from {service} API",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    @agent.tool
    def manage_user_session(
        ctx: RunContext[ApplicationContext],
        action: Literal["extend", "terminate", "refresh", "get_info"] = "get_info",
        extension_minutes: int = 30
    ) -> Dict[str, Any]:
        """Manage user session with context awareness."""
        
        session = ctx.deps.user
        
        actions = {
            "extend": f"Session extended by {extension_minutes} minutes",
            "terminate": "Session terminated",
            "refresh": "Session refreshed with new token",
            "get_info": "Session information retrieved"
        }
        
        return {
            "action": action,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "tenant_id": session.tenant_id,
            "permissions": session.permissions,
            "environment": ctx.deps.environment,
            "message": actions[action],
            "timestamp": datetime.now().isoformat(),
            "extension_minutes": extension_minutes if action == "extend" else None
        }
    
    return agent


# =============================================================================
# STRUCTURED OUTPUT TOOLS
# =============================================================================

def create_structured_output_agent() -> Agent:
    """Create agent with tools that return structured Pydantic models."""
    
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        deps_type=ApplicationContext,
        output_type=Dict[str, Any],
        system_prompt="You return structured data using Pydantic models for validation."
    )
    
    @agent.tool_plain
    def create_task(
        title: str,
        description: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        assignee: Optional[str] = None,
        estimated_hours: Optional[float] = None,
        tags: List[str] = None
    ) -> Task:
        """Create a new task with structured validation."""
        
        return Task(
            id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            description=description,
            priority=priority,
            assignee=assignee,
            estimated_hours=estimated_hours,
            tags=tags or []
        )
    
    @agent.tool_plain
    def simulate_web_search(
        query: str,
        max_results: int = 5,
        source_types: List[Literal["web", "academic", "news"]] = None
    ) -> List[SearchResult]:
        """Simulate web search returning structured results."""
        
        if source_types is None:
            source_types = ["web"]
        
        results = []
        for i in range(min(max_results, 5)):
            result = SearchResult(
                title=f"Search result {i+1} for '{query}'",
                url=f"https://example.com/result/{i+1}",
                snippet=f"This is a relevant snippet for {query} from result {i+1}",
                relevance_score=0.9 - (i * 0.1),
                source_type=source_types[i % len(source_types)]
            )
            results.append(result)
        
        return results
    
    @agent.tool_plain
    def execute_code_simulation(
        code: str,
        language: Literal["python", "javascript", "sql"] = "python",
        timeout: int = 30
    ) -> CodeExecutionResult:
        """Simulate code execution with structured results."""
        
        # Simulate different outcomes
        if "error" in code.lower():
            return CodeExecutionResult(
                success=False,
                output="",
                error_message="Simulated error in code execution",
                execution_time=0.5,
                memory_used=1024
            )
        
        return CodeExecutionResult(
            success=True,
            output=f"Simulated output from {language} code:\n{code[:100]}...",
            execution_time=2.3,
            memory_used=2048
        )
    
    return agent


# =============================================================================
# ADVANCED TOOL FEATURES
# =============================================================================

def create_advanced_features_agent() -> Agent:
    """Create agent demonstrating advanced tool features."""
    
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        deps_type=ApplicationContext,
        system_prompt="You demonstrate advanced tool features including retries, conditional availability, and error handling."
    )
    
    @agent.tool(retries=3)
    def unreliable_service(
        ctx: RunContext[ApplicationContext],
        operation: str,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        """Demonstrate retry functionality with unreliable service."""
        
        if simulate_failure and ctx.retry < 2:
            raise ModelRetry(f"Service temporarily unavailable (attempt {ctx.retry + 1}). Please retry.")
        
        return {
            "operation": operation,
            "attempt": ctx.retry + 1,
            "success": True,
            "user_id": ctx.deps.user.user_id,
            "message": f"Operation '{operation}' completed successfully on attempt {ctx.retry + 1}"
        }
    
    @agent.tool(strict=True)
    def strict_validation_tool(
        param1: str,
        param2: int,
        param3: bool = False
    ) -> Dict[str, Any]:
        """Demonstrate strict mode validation."""
        
        return {
            "param1": param1,
            "param1_type": type(param1).__name__,
            "param2": param2,
            "param2_type": type(param2).__name__,
            "param3": param3,
            "param3_type": type(param3).__name__,
            "strict_mode": True,
            "validation_passed": True
        }
    
    # Conditional tool availability
    async def conditional_tool_filter(
        ctx: RunContext[ApplicationContext],
        tool_defs: List[ToolDefinition]
    ) -> List[ToolDefinition]:
        """Filter tools based on user permissions and environment."""
        
        user_permissions = ctx.deps.user.permissions
        environment = ctx.deps.environment
        
        # Filter tools based on permissions
        allowed_tools = []
        for tool_def in tool_defs:
            tool_name = tool_def.name
            
            # Admin-only tools
            if tool_name in ["admin_operation", "system_reset"] and "admin" not in user_permissions:
                continue
                
            # Production-only tools
            if tool_name == "production_deploy" and environment != "prod":
                continue
            
            allowed_tools.append(tool_def)
        
        return allowed_tools
    
    # Set the conditional filter
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        deps_type=ApplicationContext,
        prepare_tools=conditional_tool_filter,
        system_prompt="You have conditional tool access based on permissions and environment."
    )
    
    @agent.tool
    def admin_operation(ctx: RunContext[ApplicationContext], action: str) -> Dict[str, Any]:
        """Admin-only operation requiring special permissions."""
        return {
            "action": action,
            "user_id": ctx.deps.user.user_id,
            "admin_access": True,
            "environment": ctx.deps.environment,
            "message": f"Admin operation '{action}' executed"
        }
    
    @agent.tool
    def production_deploy(ctx: RunContext[ApplicationContext], version: str) -> Dict[str, Any]:
        """Production deployment tool (prod environment only)."""
        return {
            "version": version,
            "environment": ctx.deps.environment,
            "user_id": ctx.deps.user.user_id,
            "deployment_id": f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "deployed"
        }
    
    return agent


# =============================================================================
# TOOL INSPECTION AND TESTING
# =============================================================================

def demonstrate_testmodel_inspection():
    """Demonstrate comprehensive tool inspection using TestModel."""
    
    print("🔍 Demonstrating TestModel Tool Inspection")
    print("=" * 60)
    
    # Create agent with comprehensive tools
    test_model = TestModel()
    agent = Agent(
        test_model,
        deps_type=ApplicationContext,
        system_prompt="You are Claude 4 Sonnet for tool inspection."
    )
    
    # Add sample tools
    @agent.tool_plain
    def sample_simple_tool(param: str) -> str:
        """Sample simple tool for inspection."""
        return f"Processed: {param}"
    
    @agent.tool
    def sample_context_tool(ctx: RunContext[ApplicationContext], data: str) -> Dict[str, Any]:
        """Sample context-aware tool for inspection."""
        return {"data": data, "user": ctx.deps.user.user_id}
    
    # Run agent to populate TestModel
    sample_context = ApplicationContext(
        user=UserSession(
            user_id="test_user",
            session_id="test_session",
            permissions=["read", "write"],
            tenant_id="test_tenant"
        ),
        database=DatabaseConfig(
            connection_string="postgresql://test",
            timeout=30,
            pool_size=10,
            ssl_enabled=True
        ),
        credentials=APICredentials(
            openai_key="test_openai",
            anthropic_key="test_anthropic",
            google_key="test_google"
        ),
        environment="dev"
    )
    
    result = agent.run_sync("Inspect tools", deps=sample_context)
    
    # Extract and analyze tools
    function_tools = test_model.last_model_request_parameters.function_tools
    
    print(f"📊 Extracted {len(function_tools)} tools from TestModel")
    
    for i, tool_def in enumerate(function_tools, 1):
        schema = tool_def.parameters_json_schema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        print(f"\n{i}. Tool: {tool_def.name}")
        print(f"   Description: {tool_def.description}")
        print(f"   Parameters: {len(properties)}")
        print(f"   Required: {required}")
        print(f"   Optional: {[p for p in properties.keys() if p not in required]}")
        print(f"   Strict: {getattr(tool_def, 'strict', None)}")
    
    return function_tools


# =============================================================================
# PRODUCTION EXAMPLE
# =============================================================================

def create_production_agent() -> Agent:
    """Create production-ready agent with comprehensive error handling."""
    
    agent = Agent(
        'anthropic:claude-4-sonnet-20250522',
        deps_type=ApplicationContext,
        system_prompt="You are a production Claude 4 Sonnet agent with comprehensive error handling and monitoring.",
        retries=2
    )
    
    @agent.tool
    def production_database_operation(
        ctx: RunContext[ApplicationContext],
        operation: Literal["read", "write", "update", "delete"],
        table: str,
        data: Optional[Dict[str, Any]] = None,
        transaction: bool = False,
        audit_log: bool = True
    ) -> Dict[str, Any]:
        """Production database operation with comprehensive error handling."""
        
        try:
            # Validate permissions
            required_permission = f"database_{operation}"
            if required_permission not in ctx.deps.user.permissions:
                raise ModelRetry(f"Permission denied. Required: {required_permission}")
            
            # Validate environment for write operations
            if operation in ["write", "update", "delete"] and ctx.deps.environment == "prod":
                if "prod_write" not in ctx.deps.user.permissions:
                    raise ModelRetry("Production write operations require prod_write permission")
            
            # Simulate operation
            operation_id = f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = {
                "operation_id": operation_id,
                "operation": operation,
                "table": table,
                "user_id": ctx.deps.user.user_id,
                "tenant_id": ctx.deps.user.tenant_id,
                "environment": ctx.deps.environment,
                "transaction": transaction,
                "audit_logged": audit_log,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            if data:
                result["data_size"] = len(str(data))
            
            return result
            
        except Exception as e:
            # Log error and re-raise as ModelRetry for LLM to handle
            error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            raise ModelRetry(f"Database operation failed (ID: {error_id}): {str(e)}")
    
    @agent.tool
    def production_monitoring(
        ctx: RunContext[ApplicationContext],
        metric_type: Literal["performance", "errors", "usage", "security"],
        time_range: Literal["1h", "24h", "7d", "30d"] = "1h"
    ) -> Dict[str, Any]:
        """Production monitoring and metrics collection."""
        
        if "monitoring" not in ctx.deps.user.permissions:
            raise ModelRetry("Monitoring access denied. Required permission: monitoring")
        
        # Simulate metrics based on type
        metrics = {
            "performance": {
                "avg_response_time": 150,
                "throughput": 1200,
                "error_rate": 0.05
            },
            "errors": {
                "total_errors": 15,
                "critical_errors": 2,
                "warning_count": 45
            },
            "usage": {
                "total_requests": 50000,
                "unique_users": 1250,
                "api_calls": 75000
            },
            "security": {
                "failed_logins": 23,
                "suspicious_activity": 5,
                "blocked_ips": 12
            }
        }
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "environment": ctx.deps.environment,
            "tenant_id": ctx.deps.user.tenant_id,
            "metrics": metrics[metric_type],
            "collected_at": datetime.now().isoformat(),
            "collector": ctx.deps.user.user_id
        }
    
    return agent


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

async def run_comprehensive_demo():
    """Run comprehensive demonstration of all Claude 4 Sonnet capabilities."""
    
    print("🚀 Claude 4 Sonnet Comprehensive Tool Demonstration")
    print("=" * 70)
    
    # Sample application context
    app_context = ApplicationContext(
        user=UserSession(
            user_id="demo_user_123",
            session_id="session_abc456",
            permissions=["read", "write", "database_access", "monitoring", "admin"],
            tenant_id="tenant_xyz789"
        ),
        database=DatabaseConfig(
            connection_string="postgresql://prod-db:5432/main",
            timeout=30,
            pool_size=20,
            ssl_enabled=True
        ),
        credentials=APICredentials(
            openai_key="sk-test-openai-key",
            anthropic_key="sk-test-anthropic-key",
            google_key="test-google-key"
        ),
        environment="prod"
    )
    
    # Test different agent types
    agents = {
        "Simple Tools": create_simple_tools_agent(),
        "Context-Aware": create_context_aware_agent(),
        "Structured Output": create_structured_output_agent(),
        "Advanced Features": create_advanced_features_agent(),
        "Production Ready": create_production_agent()
    }
    
    for agent_name, agent in agents.items():
        print(f"\n🔧 Testing {agent_name} Agent")
        print("-" * 50)
        
        try:
            if agent_name == "Simple Tools":
                result = agent.run_sync("Calculate 15 + 25 and format the text 'Hello World'")
            else:
                result = await agent.run("Get user profile and execute a database query", deps=app_context)
            
            print(f"✅ {agent_name}: Success")
            print(f"   Response: {str(result.data)[:100]}...")
            
        except Exception as e:
            print(f"❌ {agent_name}: Error - {e}")
    
    # Demonstrate TestModel inspection
    print(f"\n🔍 TestModel Inspection Results")
    print("-" * 50)
    tools = demonstrate_testmodel_inspection()
    print(f"✅ Successfully inspected {len(tools)} tools")


def main():
    """Main demonstration function."""
    print("Claude 4 Sonnet Implementation Examples")
    print("Comprehensive demonstration of all tool capabilities")
    
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo())


if __name__ == "__main__":
    main()