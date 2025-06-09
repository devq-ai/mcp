"""
Exceptions Module for Agentical

This module defines a hierarchy of custom exception classes and handlers
for use throughout the Agentical framework, ensuring consistent error
responses and proper logging.

Features:
- Structured exception hierarchy
- HTTP status code associations
- Standardized error response format
- Integration with Logfire observability
- Contextual error metadata
"""

import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple, Type, TypeVar, ClassVar
from uuid import uuid4

import logfire
from fastapi import status, Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Type variable for error class inheritance
T = TypeVar('T', bound='AgenticalError')


# Base exception classes
class AgenticalError(Exception):
    """Base exception for all Agentical-specific errors."""
    
    status_code: ClassVar[int] = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: ClassVar[str] = "internal_error"
    error_message: ClassVar[str] = "An internal error occurred"
    
    def __init__(
        self,
        message: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None,
    ):
        """Initialize exception with detailed information.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code to override the default
            error_code: Machine-readable error code to override the default
            details: Additional structured error details
            context: Contextual information about where/why the error occurred
            error_id: Unique identifier for this error instance (generated if None)
        """
        self.message = message or self.error_message
        self.status_code = status_code or self.status_code
        self.error_code = error_code or self.error_code
        self.details = details or {}
        self.context = context or {}
        self.error_id = error_id or str(uuid4())
        
        # Add traceback information in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            self.details['traceback'] = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for JSON response."""
        error_dict = {
            "error": self.error_code,
            "message": self.message,
            "error_id": self.error_id,
            "status_code": self.status_code
        }
        
        if self.details:
            error_dict["details"] = self.details
            
        return error_dict
    
    def log_error(self, request: Optional[Request] = None) -> None:
        """Log the error with context information."""
        log_data = {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "status_code": self.status_code,
        }
        
        if self.details:
            log_data["details"] = self.details
            
        if self.context:
            log_data["context"] = self.context
            
        if request:
            log_data["request_path"] = str(request.url.path)
            log_data["request_method"] = request.method
            log_data["client_ip"] = request.client.host if request.client else "unknown"
            
        logfire.error(self.message, **log_data)


# HTTP 4xx Errors (Client Errors)
class ClientError(AgenticalError):
    """Base class for all client-related errors (HTTP 4xx)."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "client_error"
    error_message = "Client error occurred"


class ValidationError(ClientError):
    """Validation error for input data."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "validation_error"
    error_message = "Invalid input data"
    
    @classmethod
    def from_pydantic(cls: Type[T], exc: ValidationError) -> T:
        """Create from a Pydantic ValidationError."""
        errors = []
        for error in exc.errors():
            errors.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", "")
            })
        
        return cls(
            message="Validation error",
            details={"errors": errors}
        )
    
    @classmethod
    def from_request_validation(cls: Type[T], exc: RequestValidationError) -> T:
        """Create from a FastAPI RequestValidationError."""
        errors = []
        for error in exc.errors():
            errors.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", "")
            })
        
        return cls(
            message="Request validation error",
            details={"errors": errors}
        )


class NotFoundError(ClientError):
    """Resource not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"
    error_message = "Resource not found"


class AuthenticationError(ClientError):
    """Authentication error."""
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "authentication_error"
    error_message = "Authentication failed"


class AuthorizationError(ClientError):
    """Authorization error."""
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "authorization_error"
    error_message = "Not authorized to access this resource"


class RateLimitError(ClientError):
    """Rate limit exceeded error."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"
    error_message = "Rate limit exceeded"
    
    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize with retry information.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds after which the client can retry
            **kwargs: Additional arguments passed to the parent constructor
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class BadRequestError(ClientError):
    """Bad request error."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "bad_request"
    error_message = "Bad request"


class ConflictError(ClientError):
    """Conflict error."""
    status_code = status.HTTP_409_CONFLICT
    error_code = "conflict"
    error_message = "Resource conflict"


# HTTP 5xx Errors (Server Errors)
class ServerError(AgenticalError):
    """Base class for all server-related errors (HTTP 5xx)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "server_error"
    error_message = "Server error occurred"


class DatabaseError(ServerError):
    """Database operation error."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "database_error"
    error_message = "Database operation failed"


class ExternalServiceError(ServerError):
    """External service communication error."""
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "external_service_error"
    error_message = "External service error"
    
    def __init__(
        self,
        message: Optional[str] = None,
        service_name: Optional[str] = None,
        response_status: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        """Initialize with external service details.
        
        Args:
            message: Human-readable error message
            service_name: Name of the external service
            response_status: HTTP status code from external service
            response_body: Response body from external service (may be truncated)
            **kwargs: Additional arguments passed to the parent constructor
        """
        super().__init__(message, **kwargs)
        
        details = {}
        if service_name:
            details["service_name"] = service_name
        if response_status:
            details["response_status"] = response_status
        if response_body:
            # Truncate very long response bodies
            details["response_body"] = (
                response_body[:1000] + "..." if len(response_body) > 1000 else response_body
            )
            
        self.details.update(details)


class ConfigurationError(ServerError):
    """Server configuration error."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "configuration_error"
    error_message = "Server configuration error"


class ServiceUnavailableError(ServerError):
    """Service temporarily unavailable."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "service_unavailable"
    error_message = "Service temporarily unavailable"


class TimeoutError(ServerError):
    """Operation timeout."""
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "timeout"
    error_message = "Operation timed out"


# Agent-specific errors
class AgentError(AgenticalError):
    """Base class for agent-related errors."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "agent_error"
    error_message = "Agent error occurred"


class AgentInitializationError(AgentError):
    """Error during agent initialization."""
    error_code = "agent_initialization_error"
    error_message = "Agent initialization failed"


class AgentExecutionError(AgentError):
    """Error during agent execution."""
    error_code = "agent_execution_error"
    error_message = "Agent execution failed"


class AgentNotFoundError(AgentError, NotFoundError):
    """Agent not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "agent_not_found"
    error_message = "Agent not found"


# Workflow-specific errors
class WorkflowError(AgenticalError):
    """Base class for workflow-related errors."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "workflow_error"
    error_message = "Workflow error occurred"


class WorkflowNotFoundError(WorkflowError, NotFoundError):
    """Workflow not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "workflow_not_found"
    error_message = "Workflow not found"


class WorkflowExecutionError(WorkflowError):
    """Error during workflow execution."""
    error_code = "workflow_execution_error"
    error_message = "Workflow execution failed"


class WorkflowValidationError(WorkflowError, ValidationError):
    """Workflow validation error."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "workflow_validation_error"
    error_message = "Workflow validation failed"


# Playbook-specific errors
class PlaybookError(AgenticalError):
    """Base class for playbook-related errors."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "playbook_error"
    error_message = "Playbook error occurred"


class PlaybookNotFoundError(PlaybookError, NotFoundError):
    """Playbook not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "playbook_not_found"
    error_message = "Playbook not found"


class PlaybookExecutionError(PlaybookError):
    """Error during playbook execution."""
    error_code = "playbook_execution_error"
    error_message = "Playbook execution failed"


# Knowledge-specific errors
class KnowledgeError(AgenticalError):
    """Base class for knowledge-related errors."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "knowledge_error"
    error_message = "Knowledge error occurred"


class KnowledgeNotFoundError(KnowledgeError, NotFoundError):
    """Knowledge item not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "knowledge_not_found"
    error_message = "Knowledge item not found"


class KnowledgeQueryError(KnowledgeError):
    """Error during knowledge query."""
    error_code = "knowledge_query_error"
    error_message = "Knowledge query failed"


# Exception handlers for FastAPI
def setup_exception_handlers(app: FastAPI) -> None:
    """Set up exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Handle Agentical-specific exceptions
    @app.exception_handler(AgenticalError)
    async def agentical_exception_handler(request: Request, exc: AgenticalError):
        """Handle all Agentical-specific exceptions."""
        exc.log_error(request)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    # Handle FastAPI RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI request validation errors."""
        error = ValidationError.from_request_validation(exc)
        error.log_error(request)
        return JSONResponse(
            status_code=error.status_code,
            content=error.to_dict()
        )
    
    # Handle Pydantic ValidationError
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        error = ValidationError.from_pydantic(exc)
        error.log_error(request)
        return JSONResponse(
            status_code=error.status_code,
            content=error.to_dict()
        )
    
    # Handle general exceptions as fallback
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions."""
        error_id = str(uuid4())
        
        # Detailed logging with traceback
        logfire.error(
            "Unhandled exception",
            error_id=error_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            request_path=str(request.url.path),
            request_method=request.method,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        # In production, hide detailed error information
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An internal server error occurred",
                "error_id": error_id,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )