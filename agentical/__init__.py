"""
Agentical Framework

A framework for orchestrating agents that leverages existing DevQ.ai infrastructure:
- Ptolemies Knowledge Base
- MCP Server Ecosystem
- DevQ.ai Standard Stack

This package provides:
- Agent orchestration
- Workflow management
- Tool integration
- Knowledge base connectivity
- Logging and observability
- Global exception handling
"""

__version__ = "0.1.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

from agentical.core import (
    configure_logging,
    log_operation,
    mask_sensitive_data,
    get_request_logger
)

# Import exceptions
from agentical.core.exceptions import (
    AgenticalError,
    ClientError,
    ServerError,
    NotFoundError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    BadRequestError,
    ConflictError,
    DatabaseError,
    ExternalServiceError,
    ConfigurationError,
    ServiceUnavailableError,
    TimeoutError,
    AgentError,
    AgentInitializationError,
    AgentExecutionError,
    AgentNotFoundError,
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowExecutionError,
    WorkflowValidationError,
    PlaybookError,
    PlaybookNotFoundError,
    PlaybookExecutionError,
    KnowledgeError,
    KnowledgeNotFoundError,
    KnowledgeQueryError,
    setup_exception_handlers
)

__all__ = [
    # Logging utilities
    "configure_logging",
    "log_operation",
    "mask_sensitive_data", 
    "get_request_logger",
    
    # Exception classes
    "AgenticalError",
    "ClientError",
    "ServerError",
    "NotFoundError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "BadRequestError",
    "ConflictError",
    "DatabaseError",
    "ExternalServiceError",
    "ConfigurationError",
    "ServiceUnavailableError",
    "TimeoutError",
    "AgentError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentNotFoundError",
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowExecutionError",
    "WorkflowValidationError",
    "PlaybookError",
    "PlaybookNotFoundError",
    "PlaybookExecutionError",
    "KnowledgeError",
    "KnowledgeNotFoundError",
    "KnowledgeQueryError",
    
    # Utility functions
    "setup_exception_handlers"
]