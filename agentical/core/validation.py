"""
Enhanced Request Validation Module

This module provides comprehensive request validation, input sanitization,
and performance-optimized validation patterns for the Agentical framework.

Features:
- Enhanced Pydantic models with custom validators
- Input sanitization for security
- Request size and rate limiting integration
- Field-level validation with descriptive errors
- Performance-optimized validation patterns
- Custom domain-specific validators
"""

import re
import html
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Type
from decimal import Decimal
from email_validator import validate_email, EmailNotValidError

import logfire
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.error_wrappers import ErrorWrapper, ValidationError
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from agentical.core.exceptions import ValidationError as AgenticalValidationError


# Custom validation utilities
class ValidationUtils:
    """Utility functions for enhanced validation"""

    # Security patterns
    XSS_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',
        r'<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>',
        r'<embed\b[^<]*(?:(?!<\/embed>)<[^<]*)*<\/embed>',
    ]

    SQL_INJECTION_PATTERNS = [
        r"('\s*(or|and)\s*'.*')",
        r"(union\s+select)",
        r"(drop\s+table)",
        r"(insert\s+into)",
        r"(delete\s+from)",
        r"(update\s+.*\s+set)",
        r"(-{2,})",  # SQL comments
    ]

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input for security"""
        if not isinstance(value, str):
            raise ValueError("Value must be a string")

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]

        # HTML escape
        value = html.escape(value)

        # Remove potential XSS patterns
        for pattern in ValidationUtils.XSS_PATTERNS:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        # Check for SQL injection patterns
        for pattern in ValidationUtils.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, flags=re.IGNORECASE):
                raise ValueError("Potentially malicious SQL pattern detected")

        return value.strip()

    @staticmethod
    def validate_agent_id(agent_id: str) -> str:
        """Validate agent ID format"""
        if not agent_id:
            raise ValueError("Agent ID cannot be empty")

        # Agent ID should be alphanumeric with hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            raise ValueError("Agent ID can only contain letters, numbers, hyphens, and underscores")

        if len(agent_id) > 50:
            raise ValueError("Agent ID must be 50 characters or less")

        return agent_id.lower()

    @staticmethod
    def validate_operation(operation: str) -> str:
        """Validate operation name"""
        if not operation:
            raise ValueError("Operation cannot be empty")

        # Operation should be descriptive but safe
        operation = ValidationUtils.sanitize_string(operation, max_length=100)

        if len(operation) < 3:
            raise ValueError("Operation must be at least 3 characters long")

        return operation

    @staticmethod
    def validate_json_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize JSON parameters"""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")

        # Limit parameter count
        if len(parameters) > 50:
            raise ValueError("Too many parameters (maximum 50)")

        # Validate and sanitize each parameter
        sanitized = {}
        for key, value in parameters.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(f"Parameter key must be string, got {type(key)}")

            if len(key) > 100:
                raise ValueError(f"Parameter key '{key}' too long (maximum 100 characters)")

            sanitized_key = ValidationUtils.sanitize_string(key, max_length=100)

            # Validate and sanitize value
            if isinstance(value, str):
                sanitized_value = ValidationUtils.sanitize_string(value, max_length=10000)
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized_value = value
            elif isinstance(value, dict):
                # Recursive validation for nested dictionaries (limited depth)
                if len(json.dumps(value)) > 50000:  # 50KB limit for nested objects
                    raise ValueError(f"Parameter '{key}' value too large")
                sanitized_value = ValidationUtils.validate_json_parameters(value)
            elif isinstance(value, list):
                if len(value) > 100:
                    raise ValueError(f"Parameter '{key}' list too long (maximum 100 items)")
                sanitized_value = [
                    ValidationUtils.sanitize_string(item, max_length=1000) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                raise ValueError(f"Parameter '{key}' has unsupported type {type(value)}")

            sanitized[sanitized_key] = sanitized_value

        return sanitized

    @staticmethod
    def validate_tool_names(tools: List[str]) -> List[str]:
        """Validate tool names list"""
        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")

        if len(tools) > 20:
            raise ValueError("Too many tools specified (maximum 20)")

        validated_tools = []
        for tool in tools:
            if not isinstance(tool, str):
                raise ValueError(f"Tool name must be string, got {type(tool)}")

            tool = ValidationUtils.sanitize_string(tool, max_length=50)

            if not re.match(r'^[a-zA-Z0-9_-]+$', tool):
                raise ValueError(f"Tool name '{tool}' contains invalid characters")

            if tool not in validated_tools:  # Remove duplicates
                validated_tools.append(tool)

        return validated_tools


# Enhanced Pydantic models
class EnhancedAgentRequest(BaseModel):
    """Enhanced agent execution request with comprehensive validation"""

    agent_id: str = Field(
        ...,
        description="Agent ID to execute",
        min_length=1,
        max_length=50
    )
    operation: str = Field(
        ...,
        description="Operation to perform",
        min_length=3,
        max_length=100
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Operation parameters"
    )
    use_knowledge: bool = Field(
        default=True,
        description="Whether to use knowledge base"
    )
    tools: Optional[List[str]] = Field(
        default_factory=list,
        description="Tools to use",
        max_items=20
    )
    priority: str = Field(
        default="normal",
        description="Request priority level",
        regex=r'^(low|normal|high|urgent)$'
    )
    timeout: Optional[int] = Field(
        default=300,
        description="Request timeout in seconds",
        ge=1,
        le=3600
    )
    metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Additional metadata",
        max_items=10
    )

    @validator('agent_id')
    def validate_agent_id(cls, v):
        return ValidationUtils.validate_agent_id(v)

    @validator('operation')
    def validate_operation(cls, v):
        return ValidationUtils.validate_operation(v)

    @validator('parameters')
    def validate_parameters(cls, v):
        if v is None:
            return {}
        return ValidationUtils.validate_json_parameters(v)

    @validator('tools')
    def validate_tools(cls, v):
        if v is None:
            return []
        return ValidationUtils.validate_tool_names(v)

    @validator('metadata')
    def validate_metadata(cls, v):
        if v is None:
            return {}

        if len(v) > 10:
            raise ValueError("Too many metadata items (maximum 10)")

        validated = {}
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Metadata keys and values must be strings")

            key = ValidationUtils.sanitize_string(key, max_length=50)
            value = ValidationUtils.sanitize_string(value, max_length=200)
            validated[key] = value

        return validated

    @root_validator
    def validate_request_consistency(cls, values):
        """Validate request consistency and relationships"""
        agent_id = values.get('agent_id')
        operation = values.get('operation')
        parameters = values.get('parameters', {})
        tools = values.get('tools', [])

        # Check for operation-specific requirements
        if operation and 'search' in operation.lower():
            if not parameters.get('query'):
                raise ValueError("Search operations require a 'query' parameter")

        if operation and 'create' in operation.lower():
            if not parameters.get('name'):
                raise ValueError("Create operations require a 'name' parameter")

        # Validate tool compatibility
        if tools and 'database' in tools and not parameters.get('database_config'):
            # Could add warnings or auto-configuration here
            pass

        return values

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True

        extra = "forbid"  # Reject extra fields


class EnhancedWorkflowRequest(BaseModel):
    """Enhanced workflow execution request with validation"""
    
    workflow_type: str = Field(
        ..., 
        description="Type of workflow pattern",
        regex=r'^[a-zA-Z0-9_-]+$',
        min_length=3,
        max_length=50
    )
    agents: List[str] = Field(
        ..., 
        description="Agents to coordinate",
        min_items=1,
        max_items=10
    )
    steps: List[Dict[str, Any]] = Field(
        ..., 
        description="Workflow steps",
        min_items=1,
        max_items=20
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Workflow parameters"
    )
    parallel_execution: bool = Field(
        default=False,
        description="Whether steps can be executed in parallel"
    )
    retry_policy: Optional[Dict[str, int]] = Field(
        default=None,
        description="Retry policy configuration"
    )
    
    @validator('agents')
    def validate_agents(cls, v):
        return ValidationUtils.validate_tool_names(v)  # Reuse tool validation
    
    @validator('steps')
    def validate_steps(cls, v):
        if len(v) > 20:
            raise ValueError("Too many workflow steps (maximum 20)")
        
        for i, step in enumerate(v):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary")
            
            if 'action' not in step:
                raise ValueError(f"Step {i} must have an 'action' field")
            
            step['action'] = ValidationUtils.sanitize_string(str(step['action']), max_length=100)
            
            # Validate step parameters
            if 'parameters' in step:
                step['parameters'] = ValidationUtils.validate_json_parameters(step['parameters'])
        
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        return ValidationUtils.validate_json_parameters(v)
    
    @validator('retry_policy')
    def validate_retry_policy(cls, v):
        if v is None:
            return None
        
        required_fields = ['max_retries', 'retry_delay']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Retry policy must include '{field}'")
        
        if not isinstance(v['max_retries'], int) or v['max_retries'] < 0 or v['max_retries'] > 5:
            raise ValueError("max_retries must be an integer between 0 and 5")
        
        if not isinstance(v['retry_delay'], int) or v['retry_delay'] < 1 or v['retry_delay'] > 300:
            raise ValueError("retry_delay must be an integer between 1 and 300 seconds")
        
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"


class EnhancedPlaybookRequest(BaseModel):
    """Enhanced playbook execution request with validation"""
    
    playbook_name: str = Field(
        ..., 
        description="Name of playbook to execute",
        regex=r'^[a-zA-Z0-9_.-]+$',
        min_length=3,
        max_length=100
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Playbook parameters"
    )
    agents: Optional[List[str]] = Field(
        default=None, 
        description="Override agent selection",
        max_items=10
    )
    tools: Optional[List[str]] = Field(
        default=None, 
        description="Override tool selection",
        max_items=20
    )
    environment: str = Field(
        default="development",
        description="Execution environment",
        regex=r'^(development|staging|production)$'
    )
    dry_run: bool = Field(
        default=False,
        description="Whether to perform a dry run"
    )
    
    @validator('parameters')
    def validate_parameters(cls, v):
        return ValidationUtils.validate_json_parameters(v)
    
    @validator('agents')
    def validate_agents(cls, v):
        if v is None:
            return None
        return ValidationUtils.validate_tool_names(v)
    
    @validator('tools')
    def validate_tools(cls, v):
        if v is None:
            return None
        return ValidationUtils.validate_tool_names(v)
    
    class Config:
        validate_assignment = True
        extra = "forbid"


# Request size and performance validation
class RequestSizeLimiter:
    """Request size and performance validation"""
    
    def __init__(self):
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_json_size = 5 * 1024 * 1024      # 5MB
        self.max_parameters = 100
        self.max_string_length = 50000
    
    def validate_request_size(self, content_length: Optional[int], content_type: str) -> None:
        """Validate request size limits"""
        if content_length is None:
            return
        
        if content_length > self.max_request_size:
            raise AgenticalValidationError(
                f"Request too large: {content_length} bytes (max: {self.max_request_size})",
                details={"content_length": content_length, "max_size": self.max_request_size}
            )
        
        if 'application/json' in content_type and content_length > self.max_json_size:
            raise AgenticalValidationError(
                f"JSON payload too large: {content_length} bytes (max: {self.max_json_size})",
                details={"content_length": content_length, "max_json_size": self.max_json_size}
            )
    
    async def validate_json_complexity(self, data: Dict[str, Any]) -> None:
        """Validate JSON complexity and structure"""
        json_str = json.dumps(data)
        
        if len(json_str) > self.max_json_size:
            raise AgenticalValidationError(
                f"JSON too complex: {len(json_str)} characters",
                details={"json_length": len(json_str), "max_length": self.max_json_size}
            )
        
        # Count nested levels
        max_depth = self._calculate_json_depth(data)
        if max_depth > 10:
            raise AgenticalValidationError(
                f"JSON too deeply nested: {max_depth} levels (max: 10)",
                details={"depth": max_depth, "max_depth": 10}
            )
    
    def _calculate_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested JSON structure"""
        if current_depth > 15:  # Safety limit
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                self._calculate_json_depth(value, current_depth + 1)
                for value in obj.values()
            )
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(
                self._calculate_json_depth(item, current_depth + 1)
                for item in obj
            )
        else:
            return current_depth


# Performance-optimized validation
class PerformanceValidator:
    """Performance-optimized validation patterns"""
    
    def __init__(self):
        self.validation_cache = {}
        self.cache_max_size = 1000
        self.cache_ttl = 300  # 5 minutes
    
    async def cached_validate(self, validation_key: str, validation_func: Callable, *args) -> Any:
        """Cache validation results for repeated requests"""
        cache_key = f"{validation_key}:{hash(str(args))}"
        
        # Check cache
        if cache_key in self.validation_cache:
            cached_result, timestamp = self.validation_cache[cache_key]
            if datetime.utcnow().timestamp() - timestamp < self.cache_ttl:
                return cached_result
        
        # Perform validation
        with logfire.span("Validation", validation_type=validation_key):
            result = await validation_func(*args) if asyncio.iscoroutinefunction(validation_func) else validation_func(*args)
        
        # Cache result
        if len(self.validation_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_key = min(self.validation_cache.keys(), 
                           key=lambda k: self.validation_cache[k][1])
            del self.validation_cache[oldest_key]
        
        self.validation_cache[cache_key] = (result, datetime.utcnow().timestamp())
        return result
    
    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()


# Global validation instances
request_limiter = RequestSizeLimiter()
performance_validator = PerformanceValidator()


# Validation middleware integration
async def validate_request_performance(request: Request) -> None:
    """Validate request for performance and security"""
    # Check content length
    content_length = request.headers.get('content-length')
    if content_length:
        content_length = int(content_length)
    
    content_type = request.headers.get('content-type', '')
    
    with logfire.span("Request validation", 
                      content_length=content_length,
                      content_type=content_type):
        request_limiter.validate_request_size(content_length, content_type)


# Custom exception handling for validation
def handle_validation_error(exc: ValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to Agentical format"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input", "")
        })
    
    agentical_error = AgenticalValidationError(
        message="Request validation failed",
        details={"validation_errors": errors}
    )
    
    return JSONResponse(
        status_code=agentical_error.status_code,
        content=agentical_error.to_dict()
    )