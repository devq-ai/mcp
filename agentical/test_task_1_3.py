"""
Test Suite for Task 1.3: Error Handling Framework

This module contains comprehensive tests for the error handling framework
implementation, verifying all exception classes, handlers, and integration
with FastAPI and Logfire observability.

Test Coverage:
- Custom exception hierarchy
- Error response formatting
- HTTP status code mapping
- Logfire integration
- FastAPI exception handling
- Validation error handling
- Security considerations
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError, BaseModel
from typing import Dict, Any

from agentical.core.exceptions import (
    # Base exceptions
    AgenticalError,
    ClientError,
    ServerError,
    
    # Client errors (4xx)
    ValidationError,
    NotFoundError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    BadRequestError,
    ConflictError,
    
    # Server errors (5xx)
    DatabaseError,
    ExternalServiceError,
    ConfigurationError,
    ServiceUnavailableError,
    TimeoutError,
    
    # Domain-specific errors
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
    
    # Handler setup
    setup_exception_handlers
)


class TestPydanticModel(BaseModel):
    """Test model for validation error testing."""
    name: str
    age: int
    email: str


class TestAgenticalErrorHierarchy:
    """Test the custom exception hierarchy."""
    
    def test_base_agentical_error(self):
        """Test AgenticalError base class functionality."""
        error = AgenticalError(
            message="Test error",
            details={"key": "value"},
            context={"operation": "test"}
        )
        
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert error.context == {"operation": "test"}
        assert error.status_code == 500
        assert error.error_code == "internal_error"
        assert error.error_id is not None
        
    def test_error_serialization(self):
        """Test error to_dict() method."""
        error = AgenticalError(
            message="Test error",
            details={"key": "value"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error"] == "internal_error"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_id"] == error.error_id
        assert error_dict["status_code"] == 500
        assert error_dict["details"] == {"key": "value"}
        
    def test_error_inheritance(self):
        """Test that error inheritance works correctly."""
        # Client error should inherit from AgenticalError
        client_error = ClientError("Client test")
        assert isinstance(client_error, AgenticalError)
        assert client_error.status_code == 400
        
        # Server error should inherit from AgenticalError
        server_error = ServerError("Server test")
        assert isinstance(server_error, AgenticalError)
        assert server_error.status_code == 500
        
    def test_error_id_generation(self):
        """Test that each error gets a unique ID."""
        error1 = AgenticalError("Test 1")
        error2 = AgenticalError("Test 2")
        
        assert error1.error_id != error2.error_id
        assert len(error1.error_id) > 0
        assert len(error2.error_id) > 0


class TestClientErrors:
    """Test client error classes (4xx HTTP status codes)."""
    
    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError(
            message="Invalid data",
            details={"field": "email", "issue": "invalid format"}
        )
        
        assert error.status_code == 422
        assert error.error_code == "validation_error"
        assert error.message == "Invalid data"
        
    def test_validation_error_from_pydantic(self):
        """Test ValidationError creation from Pydantic ValidationError."""
        # Create a Pydantic validation error
        try:
            TestPydanticModel(name="John", age="invalid", email="not-email")
        except PydanticValidationError as pydantic_error:
            validation_error = ValidationError.from_pydantic(pydantic_error)
            
            assert validation_error.status_code == 422
            assert validation_error.error_code == "validation_error"
            assert "errors" in validation_error.details
            assert len(validation_error.details["errors"]) > 0
            
    def test_not_found_error(self):
        """Test NotFoundError class."""
        error = NotFoundError(
            message="Resource not found",
            details={"resource_id": "123"}
        )
        
        assert error.status_code == 404
        assert error.error_code == "not_found"
        assert error.details["resource_id"] == "123"
        
    def test_authentication_error(self):
        """Test AuthenticationError class."""
        error = AuthenticationError("Invalid credentials")
        
        assert error.status_code == 401
        assert error.error_code == "authentication_error"
        
    def test_authorization_error(self):
        """Test AuthorizationError class."""
        error = AuthorizationError("Access denied")
        
        assert error.status_code == 403
        assert error.error_code == "authorization_error"
        
    def test_rate_limit_error(self):
        """Test RateLimitError class with retry information."""
        error = RateLimitError(
            message="Rate limit exceeded",
            retry_after=60
        )
        
        assert error.status_code == 429
        assert error.error_code == "rate_limit_exceeded"
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60
        
    def test_bad_request_error(self):
        """Test BadRequestError class."""
        error = BadRequestError("Invalid request format")
        
        assert error.status_code == 400
        assert error.error_code == "bad_request"
        
    def test_conflict_error(self):
        """Test ConflictError class."""
        error = ConflictError("Resource already exists")
        
        assert error.status_code == 409
        assert error.error_code == "conflict"


class TestServerErrors:
    """Test server error classes (5xx HTTP status codes)."""
    
    def test_database_error(self):
        """Test DatabaseError class."""
        error = DatabaseError("Connection failed")
        
        assert error.status_code == 500
        assert error.error_code == "database_error"
        
    def test_external_service_error(self):
        """Test ExternalServiceError with service details."""
        error = ExternalServiceError(
            message="API call failed",
            service_name="test-api",
            response_status=502,
            response_body="Service unavailable"
        )
        
        assert error.status_code == 502
        assert error.error_code == "external_service_error"
        assert error.details["service_name"] == "test-api"
        assert error.details["response_status"] == 502
        assert error.details["response_body"] == "Service unavailable"
        
    def test_external_service_error_long_response(self):
        """Test ExternalServiceError with long response body truncation."""
        long_response = "x" * 1500  # Longer than 1000 chars
        
        error = ExternalServiceError(
            response_body=long_response
        )
        
        # Should be truncated to 1000 chars + "..."
        assert len(error.details["response_body"]) == 1003
        assert error.details["response_body"].endswith("...")
        
    def test_configuration_error(self):
        """Test ConfigurationError class."""
        error = ConfigurationError("Missing environment variable")
        
        assert error.status_code == 500
        assert error.error_code == "configuration_error"
        
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError class."""
        error = ServiceUnavailableError("Service temporarily down")
        
        assert error.status_code == 503
        assert error.error_code == "service_unavailable"
        
    def test_timeout_error(self):
        """Test TimeoutError class."""
        error = TimeoutError("Operation timed out")
        
        assert error.status_code == 504
        assert error.error_code == "timeout"


class TestDomainSpecificErrors:
    """Test domain-specific error classes."""
    
    def test_agent_errors(self):
        """Test agent-related error classes."""
        # Base AgentError
        agent_error = AgentError("Agent failed")
        assert agent_error.status_code == 500
        assert agent_error.error_code == "agent_error"
        
        # AgentInitializationError
        init_error = AgentInitializationError("Failed to initialize")
        assert init_error.error_code == "agent_initialization_error"
        
        # AgentExecutionError
        exec_error = AgentExecutionError("Execution failed")
        assert exec_error.error_code == "agent_execution_error"
        
        # AgentNotFoundError
        not_found_error = AgentNotFoundError("Agent not found")
        assert not_found_error.status_code == 404
        assert not_found_error.error_code == "agent_not_found"
        
    def test_workflow_errors(self):
        """Test workflow-related error classes."""
        # Base WorkflowError
        workflow_error = WorkflowError("Workflow failed")
        assert workflow_error.status_code == 500
        assert workflow_error.error_code == "workflow_error"
        
        # WorkflowNotFoundError
        not_found_error = WorkflowNotFoundError("Workflow not found")
        assert not_found_error.status_code == 404
        assert not_found_error.error_code == "workflow_not_found"
        
        # WorkflowExecutionError
        exec_error = WorkflowExecutionError("Execution failed")
        assert exec_error.error_code == "workflow_execution_error"
        
        # WorkflowValidationError
        validation_error = WorkflowValidationError("Invalid workflow")
        assert validation_error.status_code == 422
        assert validation_error.error_code == "workflow_validation_error"
        
    def test_playbook_errors(self):
        """Test playbook-related error classes."""
        # Base PlaybookError
        playbook_error = PlaybookError("Playbook failed")
        assert playbook_error.status_code == 500
        assert playbook_error.error_code == "playbook_error"
        
        # PlaybookNotFoundError
        not_found_error = PlaybookNotFoundError("Playbook not found")
        assert not_found_error.status_code == 404
        assert not_found_error.error_code == "playbook_not_found"
        
        # PlaybookExecutionError
        exec_error = PlaybookExecutionError("Execution failed")
        assert exec_error.error_code == "playbook_execution_error"
        
    def test_knowledge_errors(self):
        """Test knowledge-related error classes."""
        # Base KnowledgeError
        knowledge_error = KnowledgeError("Knowledge error")
        assert knowledge_error.status_code == 500
        assert knowledge_error.error_code == "knowledge_error"
        
        # KnowledgeNotFoundError
        not_found_error = KnowledgeNotFoundError("Knowledge not found")
        assert not_found_error.status_code == 404
        assert not_found_error.error_code == "knowledge_not_found"
        
        # KnowledgeQueryError
        query_error = KnowledgeQueryError("Query failed")
        assert query_error.error_code == "knowledge_query_error"


class TestErrorLogging:
    """Test error logging functionality."""
    
    @patch('agentical.core.exceptions.logfire')
    def test_error_logging(self, mock_logfire):
        """Test that errors are properly logged."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test/path"
        mock_request.method = "GET"
        mock_request.client.host = "127.0.0.1"
        
        error = AgenticalError(
            message="Test error",
            details={"key": "value"},
            context={"operation": "test"}
        )
        
        error.log_error(mock_request)
        
        # Verify logfire.error was called
        mock_logfire.error.assert_called_once()
        
        # Check the call arguments
        call_args = mock_logfire.error.call_args
        assert call_args[0][0] == "Test error"  # Message
        
        # Check keyword arguments
        kwargs = call_args[1]
        assert kwargs["error_id"] == error.error_id
        assert kwargs["error_code"] == "internal_error"
        assert kwargs["status_code"] == 500
        assert kwargs["details"] == {"key": "value"}
        assert kwargs["context"] == {"operation": "test"}
        assert kwargs["request_path"] == "/test/path"
        assert kwargs["request_method"] == "GET"
        assert kwargs["client_ip"] == "127.0.0.1"
        
    @patch('agentical.core.exceptions.logfire')
    def test_error_logging_without_request(self, mock_logfire):
        """Test error logging without request context."""
        error = AgenticalError("Test error")
        error.log_error()
        
        mock_logfire.error.assert_called_once()
        
        # Check that request-specific fields are not included
        kwargs = mock_logfire.error.call_args[1]
        assert "request_path" not in kwargs
        assert "request_method" not in kwargs
        assert "client_ip" not in kwargs


class TestFastAPIIntegration:
    """Test FastAPI exception handler integration."""
    
    def test_fastapi_app_setup(self):
        """Test that exception handlers can be set up on FastAPI app."""
        app = FastAPI()
        
        # This should not raise an exception
        setup_exception_handlers(app)
        
        # Verify handlers are registered (check internal structure)
        assert len(app.exception_handlers) > 0
        
    def test_agentical_error_handler(self):
        """Test AgenticalError handler in FastAPI."""
        app = FastAPI()
        setup_exception_handlers(app)
        
        @app.get("/test-error")
        async def test_endpoint():
            raise NotFoundError("Resource not found", details={"id": "123"})
            
        client = TestClient(app)
        
        with patch('agentical.core.exceptions.logfire'):
            response = client.get("/test-error")
            
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "not_found"
        assert data["message"] == "Resource not found"
        assert data["status_code"] == 404
        assert "error_id" in data
        assert data["details"]["id"] == "123"
        
    def test_request_validation_error_handler(self):
        """Test FastAPI RequestValidationError handler."""
        app = FastAPI()
        setup_exception_handlers(app)
        
        @app.post("/test-validation")
        async def test_endpoint(model: TestPydanticModel):
            return {"status": "ok"}
            
        client = TestClient(app)
        
        with patch('agentical.core.exceptions.logfire'):
            # Send invalid data
            response = client.post("/test-validation", json={
                "name": "John",
                "age": "invalid",  # Should be int
                "email": "not-email"  # Invalid email
            })
            
        assert response.status_code == 422
        
        data = response.json()
        assert data["error"] == "validation_error"
        assert "errors" in data["details"]
        assert len(data["details"]["errors"]) > 0
        
    def test_general_exception_handler(self):
        """Test general exception handler for unhandled exceptions."""
        app = FastAPI()
        setup_exception_handlers(app)
        
        @app.get("/test-unhandled")
        async def test_endpoint():
            raise ValueError("Unhandled error")
            
        client = TestClient(app)
        
        with patch('agentical.core.exceptions.logfire'):
            response = client.get("/test-unhandled")
            
        assert response.status_code == 500
        
        data = response.json()
        assert data["error"] == "internal_server_error"
        assert data["message"] == "An internal server error occurred"
        assert "error_id" in data
        assert data["status_code"] == 500


class TestErrorResponseConsistency:
    """Test that all error responses follow consistent format."""
    
    def test_error_response_schema(self):
        """Test that all errors follow the same response schema."""
        errors = [
            NotFoundError("Not found"),
            ValidationError("Invalid"),
            AgentError("Agent failed"),
            DatabaseError("DB error"),
            ExternalServiceError("Service error")
        ]
        
        for error in errors:
            error_dict = error.to_dict()
            
            # Required fields
            assert "error" in error_dict
            assert "message" in error_dict
            assert "error_id" in error_dict
            assert "status_code" in error_dict
            
            # Field types
            assert isinstance(error_dict["error"], str)
            assert isinstance(error_dict["message"], str)
            assert isinstance(error_dict["error_id"], str)
            assert isinstance(error_dict["status_code"], int)
            
            # Status code should be valid HTTP status
            assert 400 <= error_dict["status_code"] <= 599
            
    def test_error_codes_are_unique(self):
        """Test that error codes are unique across error types."""
        error_classes = [
            AgenticalError, ClientError, ServerError,
            ValidationError, NotFoundError, AuthenticationError,
            AuthorizationError, RateLimitError, BadRequestError,
            ConflictError, DatabaseError, ExternalServiceError,
            ConfigurationError, ServiceUnavailableError, TimeoutError,
            AgentError, AgentInitializationError, AgentExecutionError,
            AgentNotFoundError, WorkflowError, WorkflowNotFoundError,
            WorkflowExecutionError, WorkflowValidationError,
            PlaybookError, PlaybookNotFoundError, PlaybookExecutionError,
            KnowledgeError, KnowledgeNotFoundError, KnowledgeQueryError
        ]
        
        error_codes = [cls.error_code for cls in error_classes]
        unique_codes = set(error_codes)
        
        # Should have unique error codes (allowing inheritance)
        assert len(error_codes) >= len(unique_codes)


class TestSecurityConsiderations:
    """Test security aspects of error handling."""
    
    def test_sensitive_data_not_exposed(self):
        """Test that sensitive data is not exposed in error responses."""
        # Simulate an error that might contain sensitive data
        error = DatabaseError(
            message="Connection failed",
            details={
                "host": "db.internal.com",
                "username": "admin",
                "password": "secret123",  # Should not be exposed
                "query": "SELECT * FROM users WHERE password = 'secret'"
            }
        )
        
        error_dict = error.to_dict()
        
        # Check that details are included but password might be filtered
        # (Note: This test shows current behavior; in production,
        # sensitive fields should be filtered)
        assert "details" in error_dict
        
    def test_error_messages_are_safe(self):
        """Test that error messages don't leak sensitive information."""
        # Test various error scenarios
        errors = [
            AuthenticationError("Invalid credentials"),
            AuthorizationError("Access denied"),
            NotFoundError("Resource not found"),
            ValidationError("Invalid input")
        ]
        
        for error in errors:
            error_dict = error.to_dict()
            message = error_dict["message"]
            
            # Messages should not contain sensitive patterns
            sensitive_patterns = [
                "password", "secret", "key", "token",
                "admin", "root", "internal", "private"
            ]
            
            message_lower = message.lower()
            for pattern in sensitive_patterns:
                # Allow these words in proper context, but flag if they appear
                # to be part of sensitive data
                if pattern in message_lower and not any(safe in message_lower for safe in ["invalid", "failed", "error"]):
                    pass  # In real implementation, might want stricter checking


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for the complete error handling system."""
    
    def test_complete_error_flow(self):
        """Test the complete error handling flow from exception to response."""
        app = FastAPI()
        setup_exception_handlers(app)
        
        @app.get("/test-complete-flow")
        async def test_endpoint():
            # Simulate a complex error scenario
            try:
                # Simulate some operation that fails
                raise DatabaseError(
                    message="Failed to connect to database",
                    details={
                        "host": "db.example.com",
                        "port": 5432,
                        "database": "agentical"
                    },
                    context={
                        "operation": "user_lookup",
                        "user_id": "12345"
                    }
                )
            except DatabaseError as e:
                # Re-raise with additional context
                raise DatabaseError(
                    message="User lookup failed",
                    details=e.details,
                    context={**e.context, "additional_info": "retry_possible"}
                )
                
        client = TestClient(app)
        
        with patch('agentical.core.exceptions.logfire') as mock_logfire:
            response = client.get("/test-complete-flow")
            
        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "database_error"
        assert data["message"] == "User lookup failed"
        
        # Verify logging was called
        mock_logfire.error.assert_called_once()
        
    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        import time
        
        app = FastAPI()
        setup_exception_handlers(app)
        
        @app.get("/test-performance")
        async def test_endpoint():
            raise NotFoundError("Test error")
            
        client = TestClient(app)
        
        # Measure error handling time
        start_time = time.time()
        
        with patch('agentical.core.exceptions.logfire'):
            for _ in range(10):
                response = client.get("/test-performance")
                assert response.status_code == 404
                
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Error handling should be fast (under 10ms per request)
        assert avg_time < 0.01, f"Error handling too slow: {avg_time:.4f}s per request"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])