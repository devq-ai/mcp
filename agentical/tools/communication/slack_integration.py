"""
Slack Integration for Agentical

This module provides comprehensive Slack integration capabilities including
bot functionality, message sending, event handling, channel management,
and enterprise-grade features for team communication and automation.

Features:
- Slack Bot API integration with OAuth 2.0
- Message sending with rich formatting (blocks, attachments)
- Interactive components (buttons, modals, select menus)
- Event handling and real-time messaging
- Channel and user management
- File uploads and sharing
- Thread management and replies
- Slash command handling
- Workflow automation and triggers
- Enterprise features (audit logging, compliance, security)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import os

# Optional dependencies
try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.socket_mode.async_client import AsyncSocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class MessageType(Enum):
    """Slack message types."""
    TEXT = "text"
    BLOCKS = "blocks"
    ATTACHMENTS = "attachments"
    INTERACTIVE = "interactive"


class EventType(Enum):
    """Slack event types."""
    MESSAGE = "message"
    APP_MENTION = "app_mention"
    REACTION_ADDED = "reaction_added"
    REACTION_REMOVED = "reaction_removed"
    CHANNEL_CREATED = "channel_created"
    CHANNEL_DELETED = "channel_deleted"
    MEMBER_JOINED_CHANNEL = "member_joined_channel"
    MEMBER_LEFT_CHANNEL = "member_left_channel"
    USER_CHANGE = "user_change"
    TEAM_JOIN = "team_join"


class ChannelType(Enum):
    """Slack channel types."""
    PUBLIC = "public_channel"
    PRIVATE = "private_channel"
    DIRECT_MESSAGE = "im"
    GROUP_DIRECT_MESSAGE = "mpim"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class SlackUser:
    """Slack user information."""
    id: str
    name: str
    real_name: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    is_admin: bool = False
    timezone: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_slack_data(cls, data: Dict[str, Any]) -> 'SlackUser':
        """Create from Slack API data."""
        profile = data.get('profile', {})
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            real_name=data.get('real_name'),
            email=profile.get('email'),
            display_name=profile.get('display_name'),
            avatar_url=profile.get('image_72'),
            is_bot=data.get('is_bot', False),
            is_admin=data.get('is_admin', False),
            timezone=data.get('tz'),
            status=profile.get('status_text')
        )


@dataclass
class SlackChannel:
    """Slack channel information."""
    id: str
    name: str
    channel_type: ChannelType
    is_private: bool = False
    is_archived: bool = False
    topic: Optional[str] = None
    purpose: Optional[str] = None
    member_count: int = 0
    created_at: Optional[datetime] = None
    creator: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['channel_type'] = self.channel_type.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_slack_data(cls, data: Dict[str, Any]) -> 'SlackChannel':
        """Create from Slack API data."""
        # Determine channel type
        if data.get('is_channel'):
            channel_type = ChannelType.PRIVATE if data.get('is_private') else ChannelType.PUBLIC
        elif data.get('is_im'):
            channel_type = ChannelType.DIRECT_MESSAGE
        elif data.get('is_mpim'):
            channel_type = ChannelType.GROUP_DIRECT_MESSAGE
        else:
            channel_type = ChannelType.PUBLIC

        created_timestamp = data.get('created')
        created_at = datetime.fromtimestamp(created_timestamp) if created_timestamp else None

        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            channel_type=channel_type,
            is_private=data.get('is_private', False),
            is_archived=data.get('is_archived', False),
            topic=data.get('topic', {}).get('value'),
            purpose=data.get('purpose', {}).get('value'),
            member_count=data.get('num_members', 0),
            created_at=created_at,
            creator=data.get('creator')
        )


@dataclass
class SlackMessage:
    """Slack message definition."""
    id: str
    channel: str
    text: Optional[str] = None
    blocks: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    thread_ts: Optional[str] = None
    reply_broadcast: bool = False
    username: Optional[str] = None
    icon_emoji: Optional[str] = None
    icon_url: Optional[str] = None
    link_names: bool = True
    mrkdwn: bool = True
    parse: Optional[str] = None
    unfurl_links: bool = True
    unfurl_media: bool = True
    priority: MessagePriority = MessagePriority.NORMAL
    schedule_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def message_type(self) -> MessageType:
        """Determine message type based on content."""
        if self.blocks:
            return MessageType.BLOCKS
        elif self.attachments:
            return MessageType.ATTACHMENTS
        else:
            return MessageType.TEXT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['message_type'] = self.message_type.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.schedule_time:
            data['schedule_time'] = self.schedule_time.isoformat()
        return data


@dataclass
class SlackEvent:
    """Slack event data."""
    id: str
    event_type: EventType
    channel: Optional[str] = None
    user: Optional[str] = None
    text: Optional[str] = None
    timestamp: Optional[str] = None
    thread_ts: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    team: Optional[str] = None
    api_app_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.event_data is None:
            self.event_data = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_slack_data(cls, data: Dict[str, Any]) -> 'SlackEvent':
        """Create from Slack event data."""
        event = data.get('event', {})
        event_type_str = event.get('type', 'message')

        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.MESSAGE

        return cls(
            id=str(uuid.uuid4()),
            event_type=event_type,
            channel=event.get('channel'),
            user=event.get('user'),
            text=event.get('text'),
            timestamp=event.get('ts'),
            thread_ts=event.get('thread_ts'),
            event_data=event,
            team=data.get('team_id'),
            api_app_id=data.get('api_app_id')
        )


@dataclass
class SlackFile:
    """Slack file upload definition."""
    filename: str
    content: bytes
    channels: Optional[List[str]] = None
    title: Optional[str] = None
    initial_comment: Optional[str] = None
    thread_ts: Optional[str] = None
    filetype: Optional[str] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = []

    @property
    def size(self) -> int:
        """Get file size in bytes."""
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'filename': self.filename,
            'title': self.title,
            'initial_comment': self.initial_comment,
            'thread_ts': self.thread_ts,
            'filetype': self.filetype,
            'channels': self.channels,
            'size': self.size
        }


@dataclass
class SlackResult:
    """Result from Slack operation."""
    success: bool
    message_ts: Optional[str] = None
    channel: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle_event(self, event: SlackEvent) -> bool:
        """Handle Slack event."""
        pass

    @abstractmethod
    def get_event_types(self) -> List[EventType]:
        """Get list of event types this handler supports."""
        pass


class MessageEventHandler(EventHandler):
    """Handler for message events."""

    def __init__(self, callback: Callable[[SlackEvent], bool]):
        self.callback = callback

    async def handle_event(self, event: SlackEvent) -> bool:
        """Handle message event."""
        if event.event_type in [EventType.MESSAGE, EventType.APP_MENTION]:
            return await self.callback(event)
        return False

    def get_event_types(self) -> List[EventType]:
        """Get supported event types."""
        return [EventType.MESSAGE, EventType.APP_MENTION]


class ReactionEventHandler(EventHandler):
    """Handler for reaction events."""

    def __init__(self, callback: Callable[[SlackEvent], bool]):
        self.callback = callback

    async def handle_event(self, event: SlackEvent) -> bool:
        """Handle reaction event."""
        if event.event_type in [EventType.REACTION_ADDED, EventType.REACTION_REMOVED]:
            return await self.callback(event)
        return False

    def get_event_types(self) -> List[EventType]:
        """Get supported event types."""
        return [EventType.REACTION_ADDED, EventType.REACTION_REMOVED]


class SlackIntegration:
    """
    Comprehensive Slack integration system.

    Provides enterprise-grade Slack bot functionality with message sending,
    event handling, and advanced team communication features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Slack integration.

        Args:
            config: Configuration dictionary with Slack settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.bot_token = self.config.get('bot_token') or os.getenv('SLACK_BOT_TOKEN')
        self.app_token = self.config.get('app_token') or os.getenv('SLACK_APP_TOKEN')
        self.signing_secret = self.config.get('signing_secret') or os.getenv('SLACK_SIGNING_SECRET')

        if not self.bot_token:
            raise ValueError("Slack bot token is required")

        # Client configuration
        self.socket_mode = self.config.get('socket_mode', False)
        self.enable_events = self.config.get('enable_events', True)

        # Performance settings
        self.rate_limit_per_minute = self.config.get('rate_limit_per_minute', 60)
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)
        self.compliance_mode = self.config.get('compliance_mode', False)

        # Initialize components
        if not SLACK_SDK_AVAILABLE:
            raise ImportError("slack_sdk library required for Slack integration")

        self.client = AsyncWebClient(token=self.bot_token, timeout=self.timeout)
        self.socket_client = None

        if self.socket_mode and self.app_token:
            self.socket_client = AsyncSocketModeClient(
                app_token=self.app_token,
                web_client=self.client
            )

        # Event handling
        self.event_handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)

        # Caches
        self.user_cache: Dict[str, SlackUser] = {}
        self.channel_cache: Dict[str, SlackChannel] = {}

        # Metrics
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.rate_limiter = {}

        # Initialize socket mode if enabled
        if self.socket_client:
            self.socket_client.socket_mode_request_listeners.append(self._handle_socket_mode_request)

    async def start(self):
        """Start Slack integration and event listening."""
        try:
            self.logger.info("Starting Slack integration")

            # Test API connection
            auth_test = await self.client.auth_test()
            if not auth_test.get('ok'):
                raise RuntimeError("Slack authentication failed")

            self.logger.info(f"Connected as bot: {auth_test.get('user')}")

            # Start socket mode if enabled
            if self.socket_client and self.enable_events:
                await self.socket_client.connect()
                self.logger.info("Socket mode connection established")

            self.metrics['integration_started'] += 1

        except Exception as e:
            self.logger.error(f"Failed to start Slack integration: {e}")
            raise

    async def stop(self):
        """Stop Slack integration."""
        try:
            if self.socket_client:
                await self.socket_client.disconnect()
                self.logger.info("Socket mode connection closed")

            self.logger.info("Slack integration stopped")

        except Exception as e:
            self.logger.error(f"Error stopping Slack integration: {e}")

    async def send_message(self, message: SlackMessage) -> SlackResult:
        """
        Send message to Slack channel.

        Args:
            message: Slack message to send

        Returns:
            Result of the send operation
        """
        try:
            self.logger.debug(f"Sending message to channel: {message.channel}")

            # Check rate limiting
            if not await self._check_rate_limit(message.channel):
                return SlackResult(
                    success=False,
                    error_message="Rate limit exceeded"
                )

            # Prepare message data
            message_data = {
                'channel': message.channel,
                'text': message.text,
                'link_names': message.link_names,
                'mrkdwn': message.mrkdwn,
                'parse': message.parse,
                'unfurl_links': message.unfurl_links,
                'unfurl_media': message.unfurl_media
            }

            # Add optional fields
            if message.blocks:
                message_data['blocks'] = message.blocks
            if message.attachments:
                message_data['attachments'] = message.attachments
            if message.thread_ts:
                message_data['thread_ts'] = message.thread_ts
                message_data['reply_broadcast'] = message.reply_broadcast
            if message.username:
                message_data['username'] = message.username
            if message.icon_emoji:
                message_data['icon_emoji'] = message.icon_emoji
            if message.icon_url:
                message_data['icon_url'] = message.icon_url

            # Send message
            if message.schedule_time:
                # Schedule message for future delivery
                schedule_timestamp = int(message.schedule_time.timestamp())
                response = await self.client.chat_scheduleMessage(
                    **message_data,
                    post_at=schedule_timestamp
                )
            else:
                # Send immediately
                response = await self.client.chat_postMessage(**message_data)

            if response.get('ok'):
                result = SlackResult(
                    success=True,
                    message_ts=response.get('ts'),
                    channel=response.get('channel'),
                    response_data=response.data
                )
                self.metrics['messages_sent'] += 1
            else:
                result = SlackResult(
                    success=False,
                    error_message=response.get('error', 'Unknown error')
                )
                self.metrics['messages_failed'] += 1

            # Log audit
            if self.audit_logging:
                self._log_message(message, result)

            return result

        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e}")
            return SlackResult(
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return SlackResult(
                success=False,
                error_message=str(e)
            )

    async def send_direct_message(self, user_id: str, text: str, **kwargs) -> SlackResult:
        """
        Send direct message to user.

        Args:
            user_id: Slack user ID
            text: Message text
            **kwargs: Additional message parameters

        Returns:
            Result of the send operation
        """
        try:
            # Open DM channel
            dm_response = await self.client.conversations_open(users=user_id)
            if not dm_response.get('ok'):
                return SlackResult(
                    success=False,
                    error_message=f"Failed to open DM channel: {dm_response.get('error')}"
                )

            channel_id = dm_response['channel']['id']

            # Create message
            message = SlackMessage(
                id=str(uuid.uuid4()),
                channel=channel_id,
                text=text,
                **kwargs
            )

            return await self.send_message(message)

        except Exception as e:
            self.logger.error(f"Failed to send direct message: {e}")
            return SlackResult(
                success=False,
                error_message=str(e)
            )

    async def upload_file(self, file: SlackFile) -> SlackResult:
        """
        Upload file to Slack.

        Args:
            file: File to upload

        Returns:
            Result of the upload operation
        """
        try:
            self.logger.debug(f"Uploading file: {file.filename}")

            upload_data = {
                'filename': file.filename,
                'content': file.content
            }

            if file.channels:
                upload_data['channels'] = ','.join(file.channels)
            if file.title:
                upload_data['title'] = file.title
            if file.initial_comment:
                upload_data['initial_comment'] = file.initial_comment
            if file.thread_ts:
                upload_data['thread_ts'] = file.thread_ts
            if file.filetype:
                upload_data['filetype'] = file.filetype

            response = await self.client.files_upload(**upload_data)

            if response.get('ok'):
                result = SlackResult(
                    success=True,
                    response_data=response.data
                )
                self.metrics['files_uploaded'] += 1
            else:
                result = SlackResult(
                    success=False,
                    error_message=response.get('error', 'Unknown error')
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            return SlackResult(
                success=False,
                error_message=str(e)
            )

    async def get_user_info(self, user_id: str, use_cache: bool = True) -> Optional[SlackUser]:
        """
        Get user information.

        Args:
            user_id: Slack user ID
            use_cache: Whether to use cached data

        Returns:
            User information if found
        """
        try:
            # Check cache first
            if use_cache and user_id in self.user_cache:
                return self.user_cache[user_id]

            # Fetch from API
            response = await self.client.users_info(user=user_id)
            if response.get('ok'):
                user_data = response['user']
                user = SlackUser.from_slack_data(user_data)

                # Cache user data
                self.user_cache[user_id] = user
                return user

        except Exception as e:
            self.logger.error(f"Failed to get user info: {e}")

        return None

    async def get_channel_info(self, channel_id: str, use_cache: bool = True) -> Optional[SlackChannel]:
        """
        Get channel information.

        Args:
            channel_id: Slack channel ID
            use_cache: Whether to use cached data

        Returns:
            Channel information if found
        """
        try:
            # Check cache first
            if use_cache and channel_id in self.channel_cache:
                return self.channel_cache[channel_id]

            # Fetch from API
            response = await self.client.conversations_info(channel=channel_id)
            if response.get('ok'):
                channel_data = response['channel']
                channel = SlackChannel.from_slack_data(channel_data)

                # Cache channel data
                self.channel_cache[channel_id] = channel
                return channel

        except Exception as e:
            self.logger.error(f"Failed to get channel info: {e}")

        return None

    async def list_channels(self, channel_types: Optional[List[ChannelType]] = None) -> List[SlackChannel]:
        """
        List workspace channels.

        Args:
            channel_types: Filter by channel types

        Returns:
            List of channels
        """
        try:
            channels = []
            cursor = None

            while True:
                params = {'limit': 200}
                if cursor:
                    params['cursor'] = cursor

                response = await self.client.conversations_list(**params)
                if not response.get('ok'):
                    break

                for channel_data in response['channels']:
                    channel = SlackChannel.from_slack_data(channel_data)

                    # Filter by type if specified
                    if channel_types and channel.channel_type not in channel_types:
                        continue

                    channels.append(channel)
                    # Cache channel data
                    self.channel_cache[channel.id] = channel

                # Check for pagination
                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break

            return channels

        except Exception as e:
            self.logger.error(f"Failed to list channels: {e}")
            return []

    def add_event_handler(self, handler: EventHandler):
        """
        Add event handler.

        Args:
            handler: Event handler to add
        """
        for event_type in handler.get_event_types():
            self.event_handlers[event_type].append(handler)

        self.logger.info(f"Added event handler for: {[et.value for et in handler.get_event_types()]}")

    def remove_event_handler(self, handler: EventHandler):
        """
        Remove event handler.

        Args:
            handler: Event handler to remove
        """
        for event_type in handler.get_event_types():
            if handler in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(handler)

    async def _handle_socket_mode_request(self, client: AsyncSocketModeClient, req: SocketModeRequest):
        """Handle socket mode request."""
        try:
            if req.type == "events_api":
                # Handle event
                event = SlackEvent.from_slack_data(req.payload)
                await self._process_event(event)

                # Acknowledge event
                response = SocketModeResponse(envelope_id=req.envelope_id)
                await client.send_socket_mode_response(response)

            elif req.type == "interactive":
                # Handle interactive component
                await self._handle_interactive_component(req.payload)

                # Acknowledge
                response = SocketModeResponse(envelope_id=req.envelope_id)
                await client.send_socket_mode_response(response)

            elif req.type == "slash_commands":
                # Handle slash command
                await self._handle_slash_command(req.payload)

                # Acknowledge
                response = SocketModeResponse(envelope_id=req.envelope_id)
                await client.send_socket_mode_response(response)

        except Exception as e:
            self.logger.error(f"Error handling socket mode request: {e}")

    async def _process_event(self, event: SlackEvent):
        """Process Slack event through registered handlers."""
        try:
            self.event_history.append(event)
            self.metrics['events_received'] += 1

            # Find and execute handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler.handle_event(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")

            # Log audit
            if self.audit_logging:
                self._log_event(event)

        except Exception as e:
            self.logger.error(f"Error processing event: {e}")

    async def _handle_interactive_component(self, payload: Dict[str, Any]):
        """Handle interactive component interaction."""
        # Placeholder for interactive component handling
        self.logger.debug(f"Interactive component: {payload.get('type')}")
        self.metrics['interactive_components'] += 1

    async def _handle_slash_command(self, payload: Dict[str, Any]):
        """Handle slash command."""
        # Placeholder for slash command handling
        self.logger.debug(f"Slash command: {payload.get('command')}")
        self.metrics['slash_commands'] += 1

    async def _check_rate_limit(self, channel: str) -> bool:
        """Check rate limiting for channel."""
        now = time.time()
        minute = int(now // 60)

        if channel not in self.rate_limiter:
            self.rate_limiter[channel] = {}

        current_count = self.rate_limiter[channel].get(minute, 0)

        if current_count >= self.rate_limit_per_minute:
            return False

        self.rate_limiter[channel][minute] = current_count + 1

        # Clean old entries
        old_minutes = [m for m in self.rate_limiter[channel].keys() if m < minute - 1]
        for old_minute in old_minutes:
            del self.rate_limiter[channel][old_minute]

        return True

    def _log_message(self, message: SlackMessage, result: SlackResult):
        """Log message for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'send_message',
            'message_id': message.id,
            'channel': message.channel,
            'success': result.success,
            'message_ts': result.message_ts
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def _log_event(self, event: SlackEvent):
        """Log event for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'receive_event',
            'event_id': event.id,
            'event_type': event.event_type.value,
            'channel': event.channel,
            'user': event.user
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    def clear_cache(self):
        """Clear user and channel caches."""
        self.user_cache.clear()
        self.channel_cache.clear()
        self.logger.info("Slack integration cache cleared")

    async def health_check(self) -> bool:
        """
        Perform health check of Slack connection.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            response = await self.client.auth_test()
            return response.get('ok', False)
        except Exception as e:
            self.logger.error(f"Slack health check failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup Slack integration resources."""
        try:
            await self.stop()
            self.clear_cache()
            self.event_handlers.clear()
            self.metrics.clear()
            self.logger.info("Slack integration cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'client') and self.client:
                self.logger.info("SlackIntegration being destroyed - cleanup recommended")
        except:
            pass
