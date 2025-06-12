"""
Core Package for Agentical

This package contains core functionality and utilities used throughout
the Agentical framework, including logging, configuration, security,
and exception handling.
"""

from .logging import (
    configure_logging,
    log_operation,
    mask_sensitive_data,
    get_request_logger
)

from .exceptions import (
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