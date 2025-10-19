"""
Segment Appender - The single source of writes in the producer/consumer architecture.

The Segment Appender is responsible for:
1. Consuming items from the ingest.items topic
2. Fetching bytes from source URLs
3. Computing item_id (sha256) and deduplicating via index
4. Appending bytes sequentially to segments
5. Updating the index
6. Publishing APPEND and SEGMENT_CLOSED events

This is the ONLY component that writes to segments and the index.
"""

import time
import os
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass
import mimetypes

from .bus import EventBus, create_event_envelope
from .index_store import IndexStore, compute_item_id
from .io import SegmentWriter, download_image_with_retry


def generate_ulid() -> str:
    """
    Generate a ULID-like unique ID.

    For simplicity, we use timestamp + random suffix.
    In production, consider using the `ulid-py` library.
    """
    import random
    import string
    timestamp = int(time.time() * 1000)
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    return f"{timestamp:013d}{random_suffix}"


def guess_mime_type(data: bytes, url: str) -> str:
    """
    Guess MIME type from data and URL.

    Args:
        data: Image bytes
        url: Source URL

    Returns:
        MIME type string
    """
    # Try to guess from URL extension
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type:
        return mime_type

    # Try to detect from magic bytes
    if data[:2] == b'\xff\xd8':
        return 'image/jpeg'
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    elif data[:2] == b'GIF':
        return 'image/gif'

    return 'application/octet-stream'


@dataclass
class AppenderStats:
    """Statistics for segment appender."""
    items_processed: int = 0
    items_appended: int = 0
    items_deduplicated: int = 0
    items_failed: int = 0
    bytes_appended: int = 0
    segments_created: int = 0
    segments_closed: int = 0


class SegmentAppender:
    """
    Single source of writes for the producer/consumer architecture.

    Consumes from ingest.items, fetches bytes, deduplicates, appends to segments,
    updates index, and publishes events.
    """

    def __init__(
        self,
        bus: EventBus,
        index: IndexStore,
        segment_writer: SegmentWriter,
        consumer_group: str = "segment_appender",
        fetch_retries: int = 3,
        fetch_timeout: int = 10,
        user_agent: str = "img2dataset/2.0",
        disallowed_header_directives: Optional[list] = None,
        producer_id: Optional[str] = None
    ):
        """
        Initialize segment appender.

        Args:
            bus: Event bus for consuming and publishing
            index: Index store for deduplication and metadata
            segment_writer: Segment writer for appending bytes
            consumer_group: Consumer group ID
            fetch_retries: Number of HTTP retry attempts
            fetch_timeout: HTTP timeout in seconds
            user_agent: User-Agent string for HTTP requests
            disallowed_header_directives: X-Robots-Tag directives to respect
            producer_id: Optional producer identifier
        """
        self.bus = bus
        self.index = index
        self.segment_writer = segment_writer
        self.consumer_group = consumer_group
        self.fetch_retries = fetch_retries
        self.fetch_timeout = fetch_timeout
        self.user_agent = user_agent
        self.disallowed_header_directives = disallowed_header_directives
        self.producer_id = producer_id or f"appender@{os.uname().nodename}"

        self.stats = AppenderStats()
        self._running = False

    def _process_item(self, event_data: Dict[str, Any]) -> bool:
        """
        Process a single item from ingest.items.

        Args:
            event_data: Event payload from ingest.items

        Returns:
            True if successful, False otherwise
        """
        source_url = event_data.get("source_url")
        if not source_url:
            return False

        meta = event_data.get("meta", {})

        # Fetch bytes from source
        data, error = download_image_with_retry(
            url=source_url,
            retries=self.fetch_retries,
            timeout=self.fetch_timeout,
            user_agent=self.user_agent,
            disallowed_header_directives=self.disallowed_header_directives
        )

        if error or data is None:
            self.stats.items_failed += 1
            return False

        # Compute item_id (sha256)
        item_id = compute_item_id(data)
        sha256 = item_id  # Same as item_id

        # Check for deduplication
        if self.index.exists(item_id):
            self.stats.items_deduplicated += 1
            return True  # Already exists, skip (idempotent)

        # Guess MIME type
        mime = guess_mime_type(data, source_url)

        # Append to segment
        try:
            segment_id, offset, length = self.segment_writer.append(
                item_id=item_id,
                data=data,
                mime=mime
            )
        except Exception as e:
            # Failed to append
            self.stats.items_failed += 1
            return False

        # Insert into index
        ts_ingest = int(time.time())
        inserted = self.index.insert(
            item_id=item_id,
            segment_id=segment_id,
            offset=offset,
            length=length,
            mime=mime,
            sha256=sha256,
            ts_ingest=ts_ingest
        )

        if not inserted:
            # Race condition: another process inserted first
            # This is OK, we skip (idempotent)
            self.stats.items_deduplicated += 1
            return True

        # Update stats
        self.stats.items_appended += 1
        self.stats.bytes_appended += length

        # Publish APPEND event
        event_payload = {
            "type": "APPEND",
            "item_id": item_id,
            "segment_id": segment_id,
            "offset": offset,
            "length": length,
            "mime": mime,
            "ts_ingest": ts_ingest,
            "source_url": source_url
        }

        envelope = create_event_envelope(
            event_id=generate_ulid(),
            entity_type="item",
            entity_id=item_id,
            kind="APPEND",
            payload=event_payload,
            producer_id=self.producer_id
        )

        self.bus.publish(
            topic="segments.events",
            key=item_id,
            value=envelope
        )

        # Check if segment should be sealed
        current_segment = self.segment_writer.get_current_segment()
        if current_segment and self.segment_writer._should_roll():
            self._seal_current_segment()

        return True

    def _seal_current_segment(self):
        """Seal the current segment and publish SEGMENT_CLOSED event."""
        sealed = self.segment_writer.seal()
        if sealed is None:
            return

        self.stats.segments_closed += 1

        # Publish SEGMENT_CLOSED event
        event_payload = {
            "type": "SEGMENT_CLOSED",
            "segment_id": sealed.segment_id,
            "items": sealed.items,
            "bytes": sealed.bytes,
            "uri": f"file://{sealed.path}",
            "ts_close": sealed.ts_closed
        }

        envelope = create_event_envelope(
            event_id=generate_ulid(),
            entity_type="segment",
            entity_id=sealed.segment_id,
            kind="SEGMENT_CLOSED",
            payload=event_payload,
            producer_id=self.producer_id
        )

        self.bus.publish(
            topic="segments.events",
            key=sealed.segment_id,
            value=envelope
        )

    def run(self, max_items: Optional[int] = None):
        """
        Run the segment appender (consume and process items).

        Args:
            max_items: Maximum number of items to process (None = unlimited)
        """
        self._running = True
        items_processed = 0

        print(f"Segment Appender starting (consumer_group={self.consumer_group})")

        try:
            # Subscribe to ingest.items
            for event in self.bus.subscribe(
                topic="ingest.items",
                group=self.consumer_group,
                auto_commit=True
            ):
                if not self._running:
                    break

                self.stats.items_processed += 1
                items_processed += 1

                # Process the item
                self._process_item(event.value)

                # Progress logging
                if items_processed % 100 == 0:
                    print(f"Processed {items_processed} items "
                          f"(appended={self.stats.items_appended}, "
                          f"dedup={self.stats.items_deduplicated}, "
                          f"failed={self.stats.items_failed})")

                # Check max items
                if max_items is not None and items_processed >= max_items:
                    break

        finally:
            # Seal any open segment
            if self.segment_writer.get_current_segment():
                self._seal_current_segment()

            print(f"Segment Appender finished: {self.stats}")

    def stop(self):
        """Stop the appender gracefully."""
        self._running = False

    def get_stats(self) -> AppenderStats:
        """Get current statistics."""
        return self.stats
