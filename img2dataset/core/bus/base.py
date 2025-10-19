"""
EventBus interface for the producer/consumer architecture.

This module defines the abstract interface for event buses used in img2dataset's
producer/consumer pipeline. The event bus carries lightweight commands and facts
(never large payloads like image bytes).

Topics:
    - ingest.items: Commands for items to ingest (URLs + metadata)
    - segments.events: Facts about segment operations (APPEND, SEGMENT_CLOSED, TOMBSTONE)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Dict, Any, Optional
import time


@dataclass
class Event:
    """
    Represents a single event from the bus.

    Attributes:
        topic: The topic this event was published to
        key: The event key (used for partitioning/ordering)
        value: The event payload as a dictionary
        timestamp: Unix timestamp when event was published
        offset: Sequential offset within the topic (for resumption)
    """
    topic: str
    key: str
    value: Dict[str, Any]
    timestamp: int
    offset: int


class EventBus(ABC):
    """
    Abstract interface for event buses.

    Implementations must provide:
    - At-least-once delivery semantics
    - Ordered delivery within a key
    - Topic creation and management
    - Consumer group offsets for resumption
    """

    @abstractmethod
    def publish(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        """
        Publish an event to a topic.

        Args:
            topic: The topic to publish to
            key: The event key (for ordering/partitioning)
            value: The event payload (must be JSON-serializable)

        Note:
            This should be idempotent where possible. The value should not
            contain large payloads (>1MB) - only metadata and pointers.
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        topic: str,
        group: str,
        auto_commit: bool = True,
        start_offset: Optional[int] = None
    ) -> Iterator[Event]:
        """
        Subscribe to a topic as part of a consumer group.

        Args:
            topic: The topic to subscribe to
            group: Consumer group ID (for offset tracking)
            auto_commit: Whether to automatically commit offsets after yield
            start_offset: Optional specific offset to start from (overrides group offset)

        Yields:
            Events from the topic in order

        Note:
            - Should resume from last committed offset for the group
            - Must track offsets per (topic, group) pair
            - If auto_commit=False, caller must call commit() manually
        """
        pass

    @abstractmethod
    def commit(self, topic: str, group: str, offset: int) -> None:
        """
        Manually commit an offset for a consumer group.

        Args:
            topic: The topic
            group: Consumer group ID
            offset: The offset to commit
        """
        pass

    @abstractmethod
    def get_offset(self, topic: str, group: str) -> Optional[int]:
        """
        Get the last committed offset for a consumer group.

        Args:
            topic: The topic
            group: Consumer group ID

        Returns:
            Last committed offset, or None if no offset committed yet
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the event bus and release resources.
        """
        pass


def create_event_envelope(
    event_id: str,
    entity_type: str,
    entity_id: str,
    kind: str,
    payload: Dict[str, Any],
    payload_version: int = 1,
    producer_id: Optional[str] = None,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Create a standardized event envelope for forwards/backwards compatibility.

    Args:
        event_id: Unique event ID (e.g., ULID)
        entity_type: Type of entity ("item" or "segment")
        entity_id: ID of the entity
        kind: Event kind (APPEND, SEGMENT_CLOSED, TOMBSTONE, etc.)
        payload: Event-specific payload
        payload_version: Schema version of the payload
        producer_id: Optional identifier of the producer
        attempt: Attempt number for this operation

    Returns:
        Standardized event envelope dictionary
    """
    return {
        "event_id": event_id,
        "occurred_at": int(time.time()),
        "entity": {
            "type": entity_type,
            "id": entity_id
        },
        "kind": kind,
        "payload_version": payload_version,
        "payload": payload,
        "trace": {
            "producer": producer_id or "unknown",
            "attempt": attempt
        }
    }
