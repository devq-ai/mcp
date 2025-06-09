"""
Structured Logging System for Agentical

This module provides a comprehensive logging system for the Agentical framework.
It configures Logfire for structured logging with proper context and middleware
integration for FastAPI applications.

Features:
- Configures Logfire with environment-specific settings
- Provides request context middleware for request IDs and correlation
- Implements log masking for sensitive data
- Sets up logging levels based on environment
- Supports contextual logging with span tracking
"""

import os
import uuid
import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

import logfire
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# Configure standard Python logging to work with Logfire
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Sensitive keys that should be masked in logs
SENSITIVE_KEYS = [
    "password", "token", "api_key", "secret", "authorization", 
    "access_token", "refresh_token", "private_key", "client_secret"
]

def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data in logs to prevent leaking credentials."""
    if not isinstance(data, dict):
        return data
    
    masked_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value)
        elif isinstance(key, str) and any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
            masked_data[key] = "********"
        else:
            masked_data[key] = value
    
    return masked_data

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context to logs including request IDs."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID if not already present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Set request context in Logfire
        with logfire.context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("User-Agent", "unknown")
        ):
            # Create a span for the entire request
            with logfire.span("HTTP Request"):
                # Log the request
                logfire.info(
                    "Request started",
                    method=request.method,
                    path=request.url.path,
                    headers=mask_sensitive_data(dict(request.headers))
                )
                
                try:
                    # Process the request
                    response = await call_next(request)
                    
                    # Log the response
                    logfire.info(
                        "Request completed",
                        status_code=response.status_code,
                        processing_time=response.headers.get("X-Process-Time", "unknown")
                    )
                    
                    # Add request ID to response headers
                    response.headers["X-Request-ID"] = request_id
                    return response
                    
                except Exception as e:
                    # Log any unhandled exceptions
                    logfire.error(
                        "Unhandled exception",
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Middleware to track processing time for requests."""
    
    async def dispatch(self, request: Request, call_next):
        import time
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

def configure_logging(
    app: FastAPI, 
    token: Optional[str] = None,
    project_name: Optional[str] = None,
    service_name: Optional[str] = None,
    environment: Optional[str] = None
) -> None:
    """
    Configure Logfire and structured logging for the application.
    
    Args:
        app: FastAPI application instance
        token: Logfire token (defaults to LOGFIRE_TOKEN env var)
        project_name: Project name for Logfire (defaults to LOGFIRE_PROJECT_NAME env var)
        service_name: Service name for Logfire (defaults to LOGFIRE_SERVICE_NAME env var)
        environment: Environment name (defaults to LOGFIRE_ENVIRONMENT env var)
    """
    # Configure Logfire
    logfire.configure(
        token=token or os.getenv("LOGFIRE_TOKEN"),
        project_name=project_name or os.getenv("LOGFIRE_PROJECT_NAME", "agentical"),
        service_name=service_name or os.getenv("LOGFIRE_SERVICE_NAME", "agentical-api"),
        environment=environment or os.getenv("LOGFIRE_ENVIRONMENT", "development")
    )
    
    # Configure log level based on environment
    log_level = logging.DEBUG if environment == "development" else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Instrument FastAPI with Logfire
    logfire.instrument_fastapi(app, capture_headers=True)
    
    # Add request ID middleware
    app.add_middleware(RequestContextMiddleware)
    
    # Add processing time middleware
    app.add_middleware(ProcessTimeMiddleware)
    
    logger.info(f"Logging configured for environment: {environment}")
    logfire.info(
        "Logging system initialized",
        environment=environment,
        log_level=logging.getLevelName(log_level),
        service=service_name
    )

@contextmanager
def log_operation(operation_name: str, **context_data):
    """
    Context manager for logging operations with context data.
    
    Args:
        operation_name: Name of the operation being performed
        context_data: Additional context data to include in logs
    """
    # Mask any sensitive data in context
    masked_context = mask_sensitive_data(context_data)
    
    # Create a span for the operation
    with logfire.span(operation_name, **masked_context):
        logfire.info(f"Starting {operation_name}", **masked_context)
        try:
            yield
            logfire.info(f"Completed {operation_name}", **masked_context)
        except Exception as e:
            logfire.error(
                f"Error in {operation_name}",
                error=str(e),
                error_type=type(e).__name__,
                **masked_context
            )
            raise

def get_request_logger(request: Request):
    """
    Get a logger with request context for dependency injection.
    
    Args:
        request: FastAPI request object
        
    Returns:
        A LoggerAdapter with request context
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    def log_with_context(level: str, message: str, **kwargs):
        """Log with request context."""
        context = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            **kwargs
        }
        
        if level == "info":
            logfire.info(message, **context)
        elif level == "warning":
            logfire.warning(message, **context)
        elif level == "error":
            logfire.error(message, **context)
        elif level == "debug":
            logfire.debug(message, **context)
    
    return log_with_context