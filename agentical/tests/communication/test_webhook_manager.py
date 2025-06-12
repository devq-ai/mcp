"""
Tests for Webhook Manager module.

This module provides comprehensive test coverage for webhook management
functionality including endpoint registration, event triggering,
delivery processing, and signature verification.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Import the modules to test
from agentical.tools.communication.webhook_manager import (
    WebhookManager,
    WebhookEndpoint,
    WebhookEvent,
    WebhookDelivery,
    AvailabilitySlot,
    WebhookMethod,
    SignatureMethod,
    WebhookStatus,
    EventStatus,
    RetryStrategy,
    HMACSignatureVerifier,
    JWTSignatureVerifier,
    EventFilter,
    RateLimiter,
    MemoryStorage,
    RedisStorage
)


class TestWebhookEndpoint:
    """Test webhook endpoint functionality."""

    def test_endpoint_creation(self):
        """Test creating a webhook endpoint."""
        endpoint = WebhookEndpoint(
            id="test_endpoint_123",
            url="https://api.example.com/webhook",
            method=WebhookMethod.POST,
            secret="test_secret",
            event_types=["user.created", "user.updated"]
        )

        assert endpoint.id == "test_endpoint_123"
        assert endpoint.url == "https://api.example.com/webhook"
        assert endpoint.method == WebhookMethod.POST
        assert endpoint.secret == "test_secret"
        assert "user.created" in endpoint.event_types
        assert endpoint.status == WebhookStatus.ACTIVE
        assert endpoint.timeout == 30.0

    def test_endpoint_to_dict(self):
        """Test converting endpoint to dictionary."""
        endpoint = WebhookEndpoint(
            id="test_endpoint_123",
            url="https://api.example.com/webhook",
            method=WebhookMethod.POST,
            secret="test_secret",
            event_types=["user.created"],
            headers={"Authorization": "Bearer token"}
        )

        result = endpoint.to_dict()

        assert result['id'] == "test_endpoint_123"
        assert result['url'] == "https://api.example.com/webhook"
        assert result['method'] == "POST"
        assert result['headers']['Authorization'] == "Bearer token"

    def test_endpoint_from_dict(self):
        """Test creating endpoint from dictionary."""
        data = {
            'id': 'test_endpoint_123',
            'url': 'https://api.example.com/webhook',
            'method': 'POST',
            'secret': 'test_secret',
            'event_types': ['user.created'],
            'status': 'active',
            'signature_method': 'hmac_sha256',
            'timeout': 45.0
        }

        endpoint = WebhookEndpoint.from_dict(data)

        assert endpoint.id == "test_endpoint_123"
        assert endpoint.url == "https://api.example.com/webhook"
        assert endpoint.method == WebhookMethod.POST
        assert endpoint.timeout == 45.0


class TestWebhookEvent:
    """Test webhook event functionality."""

    def test_event_creation(self):
        """Test creating a webhook event."""
        event = WebhookEvent(
            id="event_123",
            event_type="user.created",
            data={"user_id": 123, "email": "test@example.com"},
            source="user_service"
        )

        assert event.id == "event_123"
        assert event.event_type == "user.created"
        assert event.data["user_id"] == 123
        assert event.source == "user_service"
        assert isinstance(event.timestamp, datetime)

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event_data = {"user_id": 123, "name": "Test User"}
        event = WebhookEvent(
            id="event_123",
            event_type="user.created",
            data=event_data,
            source="user_service"
        )

        result = event.to_dict()

        assert result['id'] == "event_123"
        assert result['event_type'] == "user.created"
        assert result['data'] == event_data
        assert result['source'] == "user_service"
        assert 'timestamp' in result

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        timestamp = datetime.utcnow()
        data = {
            'id': 'event_123',
            'event_type': 'user.created',
            'data': {'user_id': 123},
            'source': 'user_service',
            'timestamp': timestamp.isoformat()
        }

        event = WebhookEvent.from_dict(data)

        assert event.id == "event_123"
        assert event.event_type == "user.created"
        assert event.data["user_id"] == 123
        assert event.source == "user_service"


class TestWebhookDelivery:
    """Test webhook delivery functionality."""

    def test_delivery_creation(self):
        """Test creating a webhook delivery."""
        delivery = WebhookDelivery(
            id="delivery_123",
            endpoint_id="endpoint_456",
            event_id="event_789",
            attempt_number=1
        )

        assert delivery.id == "delivery_123"
        assert delivery.endpoint_id == "endpoint_456"
        assert delivery.event_id == "event_789"
        assert delivery.attempt_number == 1
        assert delivery.status == EventStatus.PENDING

    def test_delivery_is_success(self):
        """Test checking if delivery was successful."""
        delivery = WebhookDelivery(
            id="delivery_123",
            endpoint_id="endpoint_456",
            event_id="event_789",
            status=EventStatus.DELIVERED,
            response_code=200
        )

        assert delivery.is_success()

        delivery.status = EventStatus.FAILED
        assert not delivery.is_success()

    def test_delivery_to_dict(self):
        """Test converting delivery to dictionary."""
        delivery = WebhookDelivery(
            id="delivery_123",
            endpoint_id="endpoint_456",
            event_id="event_789",
            status=EventStatus.DELIVERED,
            response_code=200,
            latency=0.234
        )

        result = delivery.to_dict()

        assert result['id'] == "delivery_123"
        assert result['endpoint_id'] == "endpoint_456"
        assert result['event_id'] == "event_789"
        assert result['status'] == "delivered"
        assert result['response_code'] == 200
        assert result['latency'] == 0.234


class TestHMACSignatureVerifier:
    """Test HMAC signature verification."""

    def test_hmac_signature_generation(self):
        """Test generating HMAC signature."""
        verifier = HMACSignatureVerifier('sha256')
        payload = b'{"test": "data"}'
        secret = "test_secret"

        signature = verifier.generate_signature(payload, secret)

        assert signature.startswith('sha256=')
        assert len(signature) > 10

    def test_hmac_signature_verification(self):
        """Test verifying HMAC signature."""
        verifier = HMACSignatureVerifier('sha256')
        payload = b'{"test": "data"}'
        secret = "test_secret"

        signature = verifier.generate_signature(payload, secret)
        assert verifier.verify_signature(payload, signature, secret)

        # Test with wrong signature
        wrong_signature = "sha256=wrong_signature"
        assert not verifier.verify_signature(payload, wrong_signature, secret)

    def test_hmac_different_algorithms(self):
        """Test HMAC with different algorithms."""
        payload = b'{"test": "data"}'
        secret = "test_secret"

        verifier_256 = HMACSignatureVerifier('sha256')
        verifier_512 = HMACSignatureVerifier('sha512')

        sig_256 = verifier_256.generate_signature(payload, secret)
        sig_512 = verifier_512.generate_signature(payload, secret)

        assert sig_256.startswith('sha256=')
        assert sig_512.startswith('sha512=')
        assert sig_256 != sig_512


class TestEventFilter:
    """Test event filtering functionality."""

    def test_event_filter_creation(self):
        """Test creating an event filter."""
        filter_config = {
            'event_types': ['user.created', 'user.updated'],
            'conditions': {
                'data.user_id': {'operator': 'eq', 'value': 123}
            }
        }

        event_filter = EventFilter(filter_config)
        assert event_filter.config == filter_config

    def test_event_filter_matching(self):
        """Test event filter matching."""
        filter_config = {
            'event_types': ['user.created'],
            'conditions': {
                'data.user_id': {'operator': 'eq', 'value': 123}
            }
        }

        event_filter = EventFilter(filter_config)

        # Matching event
        matching_event = WebhookEvent(
            id="event_1",
            event_type="user.created",
            data={"user_id": 123, "email": "test@example.com"}
        )

        assert event_filter.matches_filter(matching_event)

        # Non-matching event type
        non_matching_event = WebhookEvent(
            id="event_2",
            event_type="user.deleted",
            data={"user_id": 123}
        )

        assert not event_filter.matches_filter(non_matching_event)

        # Non-matching condition
        non_matching_condition = WebhookEvent(
            id="event_3",
            event_type="user.created",
            data={"user_id": 456}
        )

        assert not event_filter.matches_filter(non_matching_condition)

    def test_event_filter_complex_conditions(self):
        """Test complex event filter conditions."""
        filter_config = {
            'event_types': ['order.created'],
            'conditions': {
                'data.amount': {'operator': 'gt', 'value': 100},
                'data.status': {'operator': 'eq', 'value': 'pending'}
            }
        }

        event_filter = EventFilter(filter_config)

        # Matching event
        matching_event = WebhookEvent(
            id="event_1",
            event_type="order.created",
            data={"amount": 150, "status": "pending"}
        )

        assert event_filter.matches_filter(matching_event)

        # Non-matching amount
        non_matching_event = WebhookEvent(
            id="event_2",
            event_type="order.created",
            data={"amount": 50, "status": "pending"}
        )

        assert not event_filter.matches_filter(non_matching_event)


class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_creation(self):
        """Test creating a rate limiter."""
        limiter = RateLimiter()
        assert limiter.requests == {}

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limits."""
        limiter = RateLimiter()

        # First few requests should be allowed
        for i in range(5):
            assert limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=60)

    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests."""
        limiter = RateLimiter()

        # Make requests up to the limit
        for i in range(10):
            assert limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=60)

        # 11th request should be blocked
        assert not limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=60)

    def test_rate_limiter_window_reset(self):
        """Test rate limiter window reset."""
        limiter = RateLimiter()

        # Make requests up to the limit
        for i in range(10):
            assert limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=1)

        # Request should be blocked
        assert not limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=1)

        # Wait for window to reset (simulate time passage)
        time.sleep(1.1)

        # Request should now be allowed
        assert limiter.is_allowed("endpoint_1", max_requests=10, window_seconds=1)


class TestMemoryStorage:
    """Test memory storage functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()

    @pytest.mark.asyncio
    async def test_save_and_get_endpoint(self):
        """Test saving and retrieving endpoint."""
        endpoint = WebhookEndpoint(
            id="test_endpoint",
            url="https://api.example.com/webhook",
            method=WebhookMethod.POST,
            secret="test_secret"
        )

        await self.storage.save_endpoint(endpoint)
        retrieved = await self.storage.get_endpoint("test_endpoint")

        assert retrieved is not None
        assert retrieved.id == endpoint.id
        assert retrieved.url == endpoint.url

    @pytest.mark.asyncio
    async def test_list_endpoints(self):
        """Test listing endpoints."""
        endpoint1 = WebhookEndpoint(
            id="endpoint_1",
            url="https://api1.example.com/webhook",
            method=WebhookMethod.POST
        )

        endpoint2 = WebhookEndpoint(
            id="endpoint_2",
            url="https://api2.example.com/webhook",
            method=WebhookMethod.POST
        )

        await self.storage.save_endpoint(endpoint1)
        await self.storage.save_endpoint(endpoint2)

        endpoints = await self.storage.list_endpoints()

        assert len(endpoints) == 2
        endpoint_ids = [ep.id for ep in endpoints]
        assert "endpoint_1" in endpoint_ids
        assert "endpoint_2" in endpoint_ids

    @pytest.mark.asyncio
    async def test_delete_endpoint(self):
        """Test deleting endpoint."""
        endpoint = WebhookEndpoint(
            id="test_endpoint",
            url="https://api.example.com/webhook",
            method=WebhookMethod.POST
        )

        await self.storage.save_endpoint(endpoint)
        assert await self.storage.get_endpoint("test_endpoint") is not None

        await self.storage.delete_endpoint("test_endpoint")
        assert await self.storage.get_endpoint("test_endpoint") is None

    @pytest.mark.asyncio
    async def test_save_event(self):
        """Test saving event."""
        event = WebhookEvent(
            id="test_event",
            event_type="user.created",
            data={"user_id": 123}
        )

        # Should not raise exception
        await self.storage.save_event(event)

    @pytest.mark.asyncio
    async def test_save_delivery(self):
        """Test saving delivery."""
        delivery = WebhookDelivery(
            id="test_delivery",
            endpoint_id="endpoint_1",
            event_id="event_1"
        )

        # Should not raise exception
        await self.storage.save_delivery(delivery)


class TestWebhookManager:
    """Test webhook manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_endpoints': 100,
            'default_timeout': 30.0,
            'max_retries': 3,
            'storage': 'memory',
            'rate_limiting': True,
            'signature_verification': True
        }
        self.manager = WebhookManager(self.config)

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test webhook manager initialization."""
        assert self.manager.max_endpoints == 100
        assert self.manager.default_timeout == 30.0
        assert self.manager.max_retries == 3
        assert self.manager.enable_rate_limiting is True
        assert self.manager.enable_signature_verification is True

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping webhook manager."""
        await self.manager.start()
        assert len(self.manager.processing_tasks) > 0

        await self.manager.stop()
        assert all(task.done() for task in self.manager.processing_tasks)

    @pytest.mark.asyncio
    async def test_register_endpoint(self):
        """Test registering webhook endpoint."""
        await self.manager.start()

        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created', 'user.updated'],
            'secret': 'test_secret'
        }

        endpoint = await self.manager.register_endpoint(endpoint_data)

        assert endpoint is not None
        assert endpoint.url == endpoint_data['url']
        assert 'user.created' in endpoint.event_types
        assert self.manager.metrics['endpoints_registered'] == 1

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_unregister_endpoint(self):
        """Test unregistering webhook endpoint."""
        await self.manager.start()

        # First register an endpoint
        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created']
        }

        endpoint = await self.manager.register_endpoint(endpoint_data)
        endpoint_id = endpoint.id

        # Then unregister it
        result = await self.manager.unregister_endpoint(endpoint_id)

        assert result is True
        assert self.manager.metrics['endpoints_unregistered'] == 1

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_trigger_event(self):
        """Test triggering webhook event."""
        await self.manager.start()

        # Register an endpoint first
        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created']
        }

        await self.manager.register_endpoint(endpoint_data)

        # Trigger an event
        event_data = {
            'event_type': 'user.created',
            'data': {'user_id': 123, 'email': 'test@example.com'},
            'source': 'test_service'
        }

        await self.manager.trigger_event(event_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        assert self.manager.metrics['events_triggered'] == 1

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_event_filtering(self):
        """Test event filtering for endpoints."""
        await self.manager.start()

        # Register endpoint with specific event types
        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created'],  # Only user.created events
            'filters': {
                'conditions': {
                    'data.user_id': {'operator': 'gt', 'value': 100}
                }
            }
        }

        await self.manager.register_endpoint(endpoint_data)

        # Trigger matching event
        matching_event = {
            'event_type': 'user.created',
            'data': {'user_id': 123},
            'source': 'test_service'
        }

        await self.manager.trigger_event(matching_event)

        # Trigger non-matching event (wrong type)
        non_matching_type = {
            'event_type': 'user.deleted',
            'data': {'user_id': 123},
            'source': 'test_service'
        }

        await self.manager.trigger_event(non_matching_type)

        # Trigger non-matching event (wrong condition)
        non_matching_condition = {
            'event_type': 'user.created',
            'data': {'user_id': 50},  # Less than 100
            'source': 'test_service'
        }

        await self.manager.trigger_event(non_matching_condition)

        await asyncio.sleep(0.1)

        # Should have triggered 3 events but only 1 should match
        assert self.manager.metrics['events_triggered'] == 3

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_signature_verification(self):
        """Test webhook signature verification."""
        payload = b'{"test": "data"}'
        signature = "sha256=test_signature"
        secret = "test_secret"

        # Mock the signature verifier
        mock_verifier = Mock()
        mock_verifier.verify_signature.return_value = True
        self.manager.signature_verifiers[SignatureMethod.HMAC_SHA256] = mock_verifier

        result = self.manager.verify_webhook_signature(
            payload, signature, secret, SignatureMethod.HMAC_SHA256
        )

        assert result is True
        mock_verifier.verify_signature.assert_called_once_with(payload, signature, secret)

    @pytest.mark.asyncio
    async def test_get_endpoint_statistics(self):
        """Test getting endpoint statistics."""
        await self.manager.start()

        # Register an endpoint
        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created']
        }

        endpoint = await self.manager.register_endpoint(endpoint_data)

        # Get statistics
        stats = await self.manager.get_endpoint_statistics(endpoint.id)

        assert stats is not None
        assert 'endpoint_id' in stats
        assert 'total_deliveries' in stats
        assert 'successful_deliveries' in stats
        assert 'failed_deliveries' in stats

        await self.manager.stop()

    def test_get_metrics(self):
        """Test getting manager metrics."""
        # Add some test metrics
        self.manager.metrics['events_triggered'] = 10
        self.manager.metrics['deliveries_successful'] = 8
        self.manager.metrics['deliveries_failed'] = 2

        metrics = self.manager.get_metrics()

        assert metrics['events_triggered'] == 10
        assert metrics['deliveries_successful'] == 8
        assert metrics['deliveries_failed'] == 2

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        await self.manager.start()

        result = await self.manager.health_check()

        assert result is True

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        await self.manager.start()

        # Add some test data
        self.manager.metrics['test'] = 1

        await self.manager.cleanup()

        assert len(self.manager.metrics) == 0


class TestWebhookManagerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = WebhookManager()

    @pytest.mark.asyncio
    async def test_register_endpoint_validation(self):
        """Test endpoint registration validation."""
        await self.manager.start()

        # Test with invalid URL
        invalid_endpoint = {
            'url': 'not-a-url',
            'event_types': ['user.created']
        }

        with pytest.raises(ValueError):
            await self.manager.register_endpoint(invalid_endpoint)

        # Test with empty event types
        empty_events = {
            'url': 'https://api.example.com/webhook',
            'event_types': []
        }

        with pytest.raises(ValueError):
            await self.manager.register_endpoint(empty_events)

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_endpoint(self):
        """Test unregistering non-existent endpoint."""
        await self.manager.start()

        result = await self.manager.unregister_endpoint("nonexistent_id")

        assert result is False

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_trigger_event_no_endpoints(self):
        """Test triggering event with no registered endpoints."""
        await self.manager.start()

        event_data = {
            'event_type': 'user.created',
            'data': {'user_id': 123},
            'source': 'test_service'
        }

        # Should not raise exception
        await self.manager.trigger_event(event_data)

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_delivery_failure_retry(self):
        """Test delivery failure and retry mechanism."""
        await self.manager.start()

        # Mock HTTP client to simulate failure
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        self.manager.http_client = mock_client

        # Register endpoint
        endpoint_data = {
            'url': 'https://api.example.com/webhook',
            'event_types': ['user.created'],
            'max_retries': 2
        }

        await self.manager.register_endpoint(endpoint_data)

        # Trigger event
        event_data = {
            'event_type': 'user.created',
            'data': {'user_id': 123},
            'source': 'test_service'
        }

        await self.manager.trigger_event(event_data)

        # Give time for processing and retries
        await asyncio.sleep(0.5)

        # Should have attempted delivery (original + retries)
        assert self.manager.metrics['deliveries_failed'] > 0

        await self.manager.stop()

    def test_signature_verification_unsupported_method(self):
        """Test signature verification with unsupported method."""
        payload = b'{"test": "data"}'
        signature = "custom=test_signature"
        secret = "test_secret"

        result = self.manager.verify_webhook_signature(
            payload, signature, secret, SignatureMethod.NONE
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_max_endpoints_limit(self):
        """Test maximum endpoints limit."""
        manager = WebhookManager({'max_endpoints': 1})
        await manager.start()

        # Register first endpoint (should succeed)
        endpoint1 = {
            'url': 'https://api1.example.com/webhook',
            'event_types': ['user.created']
        }

        result1 = await manager.register_endpoint(endpoint1)
        assert result1 is not None

        # Register second endpoint (should fail due to limit)
        endpoint2 = {
            'url': 'https://api2.example.com/webhook',
            'event_types': ['user.created']
        }

        with pytest.raises(ValueError, match="Maximum number of endpoints"):
            await manager.register_endpoint(endpoint2)

        await manager.stop()


if __name__ == '__main__':
    pytest.main([__file__])
