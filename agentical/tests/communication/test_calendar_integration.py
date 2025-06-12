"""
Tests for Calendar Integration module.

This module provides comprehensive test coverage for calendar integration
functionality including Google Calendar provider, event management,
and scheduling automation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Import the modules to test
from agentical.tools.communication.calendar_integration import (
    CalendarIntegration,
    CalendarEvent,
    CalendarAttendee,
    CalendarRecurrence,
    AvailabilitySlot,
    GoogleCalendarProvider,
    CalendarProvider,
    EventStatus,
    AttendeeStatus,
    RecurrenceFrequency,
    Visibility
)


class TestCalendarAttendee:
    """Test calendar attendee functionality."""

    def test_attendee_creation(self):
        """Test creating a calendar attendee."""
        attendee = CalendarAttendee(
            email="test@example.com",
            name="Test User",
            status=AttendeeStatus.ACCEPTED
        )

        assert attendee.email == "test@example.com"
        assert attendee.name == "Test User"
        assert attendee.status == AttendeeStatus.ACCEPTED
        assert not attendee.is_organizer
        assert not attendee.optional

    def test_attendee_to_dict(self):
        """Test converting attendee to dictionary."""
        attendee = CalendarAttendee(
            email="test@example.com",
            name="Test User",
            status=AttendeeStatus.ACCEPTED,
            is_organizer=True
        )

        result = attendee.to_dict()

        assert result['email'] == "test@example.com"
        assert result['name'] == "Test User"
        assert result['status'] == "accepted"
        assert result['is_organizer'] is True

    def test_attendee_from_google_data(self):
        """Test creating attendee from Google Calendar data."""
        google_data = {
            'email': 'test@example.com',
            'displayName': 'Test User',
            'responseStatus': 'accepted',
            'organizer': True,
            'optional': False
        }

        attendee = CalendarAttendee.from_google_data(google_data)

        assert attendee.email == "test@example.com"
        assert attendee.name == "Test User"
        assert attendee.status == AttendeeStatus.ACCEPTED
        assert attendee.is_organizer is True
        assert not attendee.optional


class TestCalendarRecurrence:
    """Test calendar recurrence functionality."""

    def test_recurrence_creation(self):
        """Test creating a recurrence pattern."""
        recurrence = CalendarRecurrence(
            frequency=RecurrenceFrequency.WEEKLY,
            interval=1,
            count=10
        )

        assert recurrence.frequency == RecurrenceFrequency.WEEKLY
        assert recurrence.interval == 1
        assert recurrence.count == 10
        assert recurrence.until is None

    def test_recurrence_to_dict(self):
        """Test converting recurrence to dictionary."""
        end_date = datetime(2024, 12, 31, 23, 59, 59)
        recurrence = CalendarRecurrence(
            frequency=RecurrenceFrequency.DAILY,
            interval=2,
            until=end_date
        )

        result = recurrence.to_dict()

        assert result['frequency'] == "daily"
        assert result['interval'] == 2
        assert result['until'] == end_date.isoformat()

    def test_recurrence_to_rrule_string(self):
        """Test converting recurrence to RRULE string."""
        recurrence = CalendarRecurrence(
            frequency=RecurrenceFrequency.WEEKLY,
            interval=1,
            count=5,
            by_weekday=[0, 2, 4]  # Monday, Wednesday, Friday
        )

        rrule = recurrence.to_rrule_string()

        assert "FREQ=WEEKLY" in rrule
        assert "INTERVAL=1" in rrule
        assert "COUNT=5" in rrule
        assert "BYDAY=" in rrule


class TestCalendarEvent:
    """Test calendar event functionality."""

    def test_event_creation(self):
        """Test creating a calendar event."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)

        event = CalendarEvent(
            title="Test Meeting",
            start_time=start_time,
            end_time=end_time,
            description="Test description"
        )

        assert event.title == "Test Meeting"
        assert event.start_time == start_time
        assert event.end_time == end_time
        assert event.description == "Test description"
        assert event.status == EventStatus.CONFIRMED

    def test_event_duration(self):
        """Test calculating event duration."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 30, 0)

        event = CalendarEvent(
            title="Test Meeting",
            start_time=start_time,
            end_time=end_time
        )

        duration = event.duration
        assert duration == timedelta(hours=1, minutes=30)

    def test_event_is_recurring(self):
        """Test checking if event is recurring."""
        event = CalendarEvent(
            title="Test Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

        assert not event.is_recurring

        event.recurrence = CalendarRecurrence(
            frequency=RecurrenceFrequency.WEEKLY,
            count=5
        )

        assert event.is_recurring

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)
        attendee = CalendarAttendee(email="test@example.com")

        event = CalendarEvent(
            title="Test Meeting",
            start_time=start_time,
            end_time=end_time,
            attendees=[attendee],
            location="Conference Room A"
        )

        result = event.to_dict()

        assert result['title'] == "Test Meeting"
        assert result['start_time'] == start_time.isoformat()
        assert result['end_time'] == end_time.isoformat()
        assert result['location'] == "Conference Room A"
        assert len(result['attendees']) == 1

    def test_event_from_google_data(self):
        """Test creating event from Google Calendar data."""
        google_data = {
            'id': 'test_event_123',
            'summary': 'Test Meeting',
            'description': 'Test description',
            'location': 'Conference Room A',
            'start': {'dateTime': '2024-06-15T10:00:00Z'},
            'end': {'dateTime': '2024-06-15T11:00:00Z'},
            'status': 'confirmed',
            'visibility': 'default',
            'attendees': [
                {
                    'email': 'test@example.com',
                    'displayName': 'Test User',
                    'responseStatus': 'accepted'
                }
            ]
        }

        event = CalendarEvent.from_google_data(google_data)

        assert event.id == "test_event_123"
        assert event.title == "Test Meeting"
        assert event.description == "Test description"
        assert event.location == "Conference Room A"
        assert len(event.attendees) == 1
        assert event.attendees[0].email == "test@example.com"


class TestAvailabilitySlot:
    """Test availability slot functionality."""

    def test_availability_slot_creation(self):
        """Test creating an availability slot."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)

        slot = AvailabilitySlot(
            start_time=start_time,
            end_time=end_time,
            busy=False
        )

        assert slot.start_time == start_time
        assert slot.end_time == end_time
        assert not slot.busy

    def test_availability_slot_to_dict(self):
        """Test converting availability slot to dictionary."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)

        slot = AvailabilitySlot(
            start_time=start_time,
            end_time=end_time,
            busy=True,
            event_id="event_123"
        )

        result = slot.to_dict()

        assert result['start_time'] == start_time.isoformat()
        assert result['end_time'] == end_time.isoformat()
        assert result['busy'] is True
        assert result['event_id'] == "event_123"


class TestGoogleCalendarProvider:
    """Test Google Calendar provider functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'credentials_file': '/path/to/credentials.json',
            'token_file': '/path/to/token.json',
            'scopes': ['https://www.googleapis.com/auth/calendar']
        }
        self.provider = GoogleCalendarProvider(self.config)

    def test_provider_initialization(self):
        """Test Google Calendar provider initialization."""
        assert self.provider.credentials_file == '/path/to/credentials.json'
        assert self.provider.token_file == '/path/to/token.json'
        assert len(self.provider.scopes) == 1

    @pytest.mark.asyncio
    async def test_provider_authentication_mock(self):
        """Test provider authentication with mocked service."""
        with patch('agentical.tools.communication.calendar_integration.build') as mock_build:
            mock_service = Mock()
            mock_build.return_value = mock_service

            with patch.object(self.provider, '_load_credentials') as mock_load_creds:
                mock_creds = Mock()
                mock_creds.valid = True
                mock_load_creds.return_value = mock_creds

                result = await self.provider.authenticate()

                assert result is True
                assert self.provider.service == mock_service

    def test_to_google_event_conversion(self):
        """Test converting CalendarEvent to Google Calendar format."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)
        attendee = CalendarAttendee(email="test@example.com", name="Test User")

        event = CalendarEvent(
            title="Test Meeting",
            start_time=start_time,
            end_time=end_time,
            description="Test description",
            location="Conference Room A",
            attendees=[attendee]
        )

        google_event = self.provider._to_google_event(event)

        assert google_event['summary'] == "Test Meeting"
        assert google_event['description'] == "Test description"
        assert google_event['location'] == "Conference Room A"
        assert len(google_event['attendees']) == 1
        assert google_event['attendees'][0]['email'] == "test@example.com"

    def test_is_available(self):
        """Test checking if provider is available."""
        # Without service, should not be available
        assert not self.provider.is_available()

        # With service, should be available
        self.provider.service = Mock()
        assert self.provider.is_available()


class TestCalendarIntegration:
    """Test calendar integration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'provider': 'google',
            'timezone': 'UTC',
            'default_calendar_id': 'primary',
            'cache_duration_minutes': 15
        }

        with patch('agentical.tools.communication.calendar_integration.GoogleCalendarProvider'):
            self.integration = CalendarIntegration(self.config)

    def test_integration_initialization(self):
        """Test calendar integration initialization."""
        assert self.integration.provider_type == CalendarProvider.GOOGLE
        assert self.integration.default_timezone == 'UTC'
        assert self.integration.default_calendar_id == 'primary'
        assert self.integration.cache_duration_minutes == 15

    @pytest.mark.asyncio
    async def test_create_event(self):
        """Test creating a calendar event."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        end_time = datetime(2024, 6, 15, 11, 0, 0)

        event = CalendarEvent(
            title="Test Meeting",
            start_time=start_time,
            end_time=end_time
        )

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.create_event.return_value = event
        self.integration.provider = mock_provider

        result = await self.integration.create_event(event)

        assert result == event
        mock_provider.create_event.assert_called_once_with(event, 'primary')
        assert self.integration.metrics['events_created'] == 1

    @pytest.mark.asyncio
    async def test_update_event(self):
        """Test updating a calendar event."""
        event = CalendarEvent(
            id="test_event_123",
            title="Updated Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.update_event.return_value = event
        self.integration.provider = mock_provider

        result = await self.integration.update_event(event)

        assert result == event
        mock_provider.update_event.assert_called_once_with(event, 'primary')
        assert self.integration.metrics['events_updated'] == 1

    @pytest.mark.asyncio
    async def test_delete_event(self):
        """Test deleting a calendar event."""
        event_id = "test_event_123"

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.delete_event.return_value = True
        self.integration.provider = mock_provider

        result = await self.integration.delete_event(event_id)

        assert result is True
        mock_provider.delete_event.assert_called_once_with(event_id, 'primary')
        assert self.integration.metrics['events_deleted'] == 1

    @pytest.mark.asyncio
    async def test_get_event(self):
        """Test getting a calendar event."""
        event_id = "test_event_123"
        expected_event = CalendarEvent(
            id=event_id,
            title="Test Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.get_event.return_value = expected_event
        self.integration.provider = mock_provider

        result = await self.integration.get_event(event_id)

        assert result == expected_event
        mock_provider.get_event.assert_called_once_with(event_id, 'primary')

    @pytest.mark.asyncio
    async def test_list_events(self):
        """Test listing calendar events."""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=7)

        expected_events = [
            CalendarEvent(
                id="event_1",
                title="Meeting 1",
                start_time=start_time + timedelta(hours=1),
                end_time=start_time + timedelta(hours=2)
            ),
            CalendarEvent(
                id="event_2",
                title="Meeting 2",
                start_time=start_time + timedelta(days=1),
                end_time=start_time + timedelta(days=1, hours=1)
            )
        ]

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.list_events.return_value = expected_events
        self.integration.provider = mock_provider

        result = await self.integration.list_events(start_time, end_time)

        assert len(result) == 2
        assert result == expected_events
        mock_provider.list_events.assert_called_once_with(start_time, end_time, 'primary')

    @pytest.mark.asyncio
    async def test_check_availability(self):
        """Test checking availability."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=8)

        expected_slots = [
            AvailabilitySlot(
                start_time=start_time,
                end_time=start_time + timedelta(hours=2),
                busy=False
            ),
            AvailabilitySlot(
                start_time=start_time + timedelta(hours=2),
                end_time=start_time + timedelta(hours=3),
                busy=True,
                event_id="busy_event"
            ),
            AvailabilitySlot(
                start_time=start_time + timedelta(hours=3),
                end_time=end_time,
                busy=False
            )
        ]

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.check_availability.return_value = expected_slots
        self.integration.provider = mock_provider

        result = await self.integration.check_availability(start_time, end_time)

        assert len(result) == 3
        assert result == expected_slots
        mock_provider.check_availability.assert_called_once_with(start_time, end_time, 'primary')

    @pytest.mark.asyncio
    async def test_find_free_time(self):
        """Test finding free time slots."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=4)

        # Mock availability slots with some free time
        availability_slots = [
            AvailabilitySlot(
                start_time=start_time,
                end_time=start_time + timedelta(hours=2),
                busy=False
            ),
            AvailabilitySlot(
                start_time=start_time + timedelta(hours=2),
                end_time=start_time + timedelta(hours=3),
                busy=True
            ),
            AvailabilitySlot(
                start_time=start_time + timedelta(hours=3),
                end_time=end_time,
                busy=False
            )
        ]

        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.check_availability.return_value = availability_slots
        self.integration.provider = mock_provider

        # Find 60-minute slots
        free_slots = await self.integration.find_free_time(60, start_time, end_time)

        # Should find slots in both free periods
        assert len(free_slots) > 0

        # Each slot should be 60 minutes long
        for slot_start, slot_end in free_slots:
            assert slot_end - slot_start == timedelta(minutes=60)

    @pytest.mark.asyncio
    async def test_schedule_meeting(self):
        """Test automatically scheduling a meeting."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=4)

        # Mock free time finding
        with patch.object(self.integration, 'find_free_time') as mock_find_free:
            free_slot = (start_time + timedelta(hours=1), start_time + timedelta(hours=2))
            mock_find_free.return_value = [free_slot]

            with patch.object(self.integration, 'create_event') as mock_create:
                expected_event = CalendarEvent(
                    title="Team Meeting",
                    start_time=free_slot[0],
                    end_time=free_slot[1]
                )
                mock_create.return_value = expected_event

                result = await self.integration.schedule_meeting(
                    title="Team Meeting",
                    attendees=["user1@example.com", "user2@example.com"],
                    duration_minutes=60,
                    preferred_start=start_time,
                    preferred_end=end_time
                )

                assert result == expected_event
                assert self.integration.metrics['meetings_scheduled'] == 1

    @pytest.mark.asyncio
    async def test_get_upcoming_events(self):
        """Test getting upcoming events."""
        now = datetime.utcnow()

        # Create mock events (some past, some future)
        past_event = CalendarEvent(
            id="past_event",
            title="Past Event",
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1)
        )

        future_event1 = CalendarEvent(
            id="future_event1",
            title="Future Event 1",
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2)
        )

        future_event2 = CalendarEvent(
            id="future_event2",
            title="Future Event 2",
            start_time=now + timedelta(days=2),
            end_time=now + timedelta(days=2, hours=1)
        )

        all_events = [past_event, future_event1, future_event2]

        with patch.object(self.integration, 'list_events') as mock_list:
            mock_list.return_value = all_events

            upcoming = await self.integration.get_upcoming_events(days_ahead=7)

            # Should only return future events, sorted by start time
            assert len(upcoming) == 2
            assert upcoming[0] == future_event1
            assert upcoming[1] == future_event2

    @pytest.mark.asyncio
    async def test_create_recurring_event(self):
        """Test creating a recurring event."""
        event = CalendarEvent(
            title="Weekly Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

        recurrence = CalendarRecurrence(
            frequency=RecurrenceFrequency.WEEKLY,
            count=10
        )

        with patch.object(self.integration, 'create_event') as mock_create:
            expected_event = CalendarEvent(
                title="Weekly Meeting",
                start_time=event.start_time,
                end_time=event.end_time,
                recurrence=recurrence
            )
            mock_create.return_value = expected_event

            result = await self.integration.create_recurring_event(event, recurrence)

            assert result.recurrence == recurrence
            assert self.integration.metrics['recurring_events_created'] == 1

    def test_get_metrics(self):
        """Test getting integration metrics."""
        # Add some test metrics
        self.integration.metrics['events_created'] = 5
        self.integration.metrics['events_updated'] = 3
        self.integration.metrics['errors'] = 1

        metrics = self.integration.get_metrics()

        assert metrics['events_created'] == 5
        assert metrics['events_updated'] == 3
        assert metrics['errors'] == 1

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        # Mock provider availability and list_events
        mock_provider = AsyncMock()
        mock_provider.is_available.return_value = True
        mock_provider.list_events.return_value = []
        self.integration.provider = mock_provider

        with patch.object(self.integration, 'list_events') as mock_list:
            mock_list.return_value = []

            result = await self.integration.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        # Mock provider as unavailable
        mock_provider = Mock()
        mock_provider.is_available.return_value = False
        self.integration.provider = mock_provider

        result = await self.integration.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        # Add some test data to caches
        self.integration.event_cache['test'] = ([], datetime.now())
        self.integration.availability_cache['test'] = ([], datetime.now())
        self.integration.metrics['test'] = 1

        await self.integration.cleanup()

        assert len(self.integration.event_cache) == 0
        assert len(self.integration.availability_cache) == 0
        assert len(self.integration.metrics) == 0


class TestCalendarIntegrationEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('agentical.tools.communication.calendar_integration.GoogleCalendarProvider'):
            self.integration = CalendarIntegration()

    @pytest.mark.asyncio
    async def test_create_event_error_handling(self):
        """Test error handling in create_event."""
        event = CalendarEvent(
            title="Test Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

        # Mock provider to raise exception
        mock_provider = AsyncMock()
        mock_provider.create_event.side_effect = Exception("API Error")
        self.integration.provider = mock_provider

        with pytest.raises(Exception, match="API Error"):
            await self.integration.create_event(event)

        # Should increment error count
        assert self.integration.metrics['errors'] == 1

    @pytest.mark.asyncio
    async def test_schedule_meeting_no_free_time(self):
        """Test scheduling meeting when no free time is available."""
        with patch.object(self.integration, 'find_free_time') as mock_find_free:
            mock_find_free.return_value = []  # No free slots

            result = await self.integration.schedule_meeting(
                title="Team Meeting",
                attendees=["user1@example.com"],
                duration_minutes=60,
                preferred_start=datetime.now(),
                preferred_end=datetime.now() + timedelta(hours=4)
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_find_free_time_no_availability(self):
        """Test finding free time when no availability data exists."""
        mock_provider = AsyncMock()
        mock_provider.check_availability.return_value = []
        self.integration.provider = mock_provider

        free_slots = await self.integration.find_free_time(
            60, datetime.now(), datetime.now() + timedelta(hours=4)
        )

        assert len(free_slots) == 0

    def test_event_validation(self):
        """Test event validation with invalid data."""
        # Test with end time before start time
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # Invalid: end before start

        with pytest.raises(ValueError):
            CalendarEvent(
                title="Invalid Event",
                start_time=start_time,
                end_time=end_time
            )


if __name__ == '__main__':
    pytest.main([__file__])
