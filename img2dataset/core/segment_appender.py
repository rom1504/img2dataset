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
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import mimetypes
from threading import Semaphore, Lock
from multiprocessing.pool import ThreadPool

from .bus import EventBus, create_event_envelope
from .index_store import IndexStore, compute_item_id
from .io import SegmentWriter, download_image_with_retry


def generate_ulid() -> str:
    """
    Generate a ULID-like unique ID.

    For simplicity, we use timestamp + random suffix.
    In production, consider using the `ulid-py` library.
    """
    # pylint: disable=import-outside-toplevel
    import random
    import string

    timestamp = int(time.time() * 1000)
    random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
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
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    elif data[:2] == b"GIF":
        return "image/gif"

    return "application/octet-stream"


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
        producer_id: Optional[str] = None,
        thread_count: int = 32,
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
            thread_count: Number of download threads (default: 32)
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
        self.thread_count = thread_count

        self.stats = AppenderStats()
        self._running = False
        self._stats_lock = Lock()  # For thread-safe stats updates

    def _download_item(self, event_data: Dict[str, Any]) -> Tuple[str, Optional[bytes], Optional[str]]:
        """
        Download a single item (for use in thread pool).

        Args:
            event_data: Event payload from ingest.items

        Returns:
            Tuple of (source_url, data, error)
        """
        source_url = event_data.get("source_url")
        if not source_url:
            return (source_url or "", None, "No source URL")

        # Fetch bytes from source
        data, error = download_image_with_retry(
            url=source_url,
            retries=self.fetch_retries,
            timeout=self.fetch_timeout,
            user_agent=self.user_agent,
            disallowed_header_directives=self.disallowed_header_directives,
        )

        return (source_url, data, error)

    def _write_downloaded_item(self, source_url: str, data: bytes) -> bool:
        """
        Write a downloaded item to segments (called after parallel download).

        Args:
            source_url: Source URL of the item
            data: Downloaded image bytes

        Returns:
            True if successful, False otherwise
        """
        # Compute item_id (sha256)
        item_id = compute_item_id(data)
        sha256 = item_id  # Same as item_id

        # Check for deduplication
        if self.index.exists(item_id):
            with self._stats_lock:
                self.stats.items_deduplicated += 1
            return True  # Already exists, skip (idempotent)

        # Guess MIME type
        mime = guess_mime_type(data, source_url)

        # Append to segment
        try:
            segment_id, offset, length = self.segment_writer.append(item_id=item_id, data=data, mime=mime)
        except Exception:  # pylint: disable=broad-exception-caught
            # Failed to append - catch all exceptions for robustness
            with self._stats_lock:
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
            ts_ingest=ts_ingest,
        )

        if not inserted:
            # Race condition: another process inserted first
            # This is OK, we skip (idempotent)
            with self._stats_lock:
                self.stats.items_deduplicated += 1
            return True

        # Update stats
        with self._stats_lock:
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
            "source_url": source_url,
        }

        envelope = create_event_envelope(
            event_id=generate_ulid(),
            entity_type="item",
            entity_id=item_id,
            kind="APPEND",
            payload=event_payload,
            producer_id=self.producer_id,
        )

        self.bus.publish(topic="segments.events", key=item_id, value=envelope)

        # Check if segment should be sealed
        current_segment = self.segment_writer.get_current_segment()
        # pylint: disable=protected-access
        if current_segment and self.segment_writer._should_roll():
            self._seal_current_segment()

        return True

    def _seal_current_segment(self):
        """Seal the current segment and publish SEGMENT_CLOSED event."""
        sealed = self.segment_writer.seal()
        if sealed is None:
            return

        with self._stats_lock:
            self.stats.segments_closed += 1

        # Publish SEGMENT_CLOSED event
        event_payload = {
            "type": "SEGMENT_CLOSED",
            "segment_id": sealed.segment_id,
            "items": sealed.items,
            "bytes": sealed.bytes,
            "uri": f"file://{sealed.path}",
            "ts_close": sealed.ts_closed,
        }

        envelope = create_event_envelope(
            event_id=generate_ulid(),
            entity_type="segment",
            entity_id=sealed.segment_id,
            kind="SEGMENT_CLOSED",
            payload=event_payload,
            producer_id=self.producer_id,
        )

        self.bus.publish(topic="segments.events", key=sealed.segment_id, value=envelope)

    def run(self, max_items: Optional[int] = None):
        """
        Run the segment appender with parallel downloads.

        Downloads happen in parallel using a thread pool, but writes to segments
        are serialized to maintain consistency.

        Args:
            max_items: Maximum number of items to process (None = unlimited)
        """
        self._running = True
        items_processed = 0

        print(f"Segment Appender starting (consumer_group={self.consumer_group}, threads={self.thread_count})")

        # Semaphore to control memory usage (like old implementation)
        semaphore = Semaphore(self.thread_count * 2)

        try:
            # Collect events in batches for parallel processing
            batch = []
            batch_size = self.thread_count * 2  # Process 2x thread_count at a time

            # Subscribe to ingest.items
            for event in self.bus.subscribe(topic="ingest.items", group=self.consumer_group, auto_commit=True):
                if not self._running:
                    break

                batch.append(event.value)

                # Process batch when full or reached max_items
                if len(batch) >= batch_size or (max_items and items_processed + len(batch) >= max_items):
                    self._process_batch(batch, semaphore)
                    items_processed += len(batch)

                    # Progress logging
                    if items_processed % 100 == 0:
                        print(
                            f"Processed {items_processed} items "
                            f"(appended={self.stats.items_appended}, "
                            f"dedup={self.stats.items_deduplicated}, "
                            f"failed={self.stats.items_failed})"
                        )

                    batch = []

                    # Check max items
                    if max_items is not None and items_processed >= max_items:
                        break

            # Process remaining batch
            if batch and self._running:
                self._process_batch(batch, semaphore)
                items_processed += len(batch)

        finally:
            # Seal any open segment
            if self.segment_writer.get_current_segment():
                self._seal_current_segment()

            print(f"Segment Appender finished: {self.stats}")

    def _process_batch(self, batch: list, semaphore: Semaphore):
        """
        Process a batch of items with parallel downloads.

        Args:
            batch: List of event payloads
            semaphore: Semaphore for memory control
        """
        # Create thread pool and download in parallel
        with ThreadPool(self.thread_count) as pool:
            # Generator that yields items and acquires semaphore
            def item_generator():
                for item in batch:
                    semaphore.acquire()  # pylint: disable=consider-using-with
                    yield item

            # Download in parallel using imap_unordered (unordered for speed)
            for source_url, data, error in pool.imap_unordered(self._download_item, item_generator()):
                try:
                    with self._stats_lock:
                        self.stats.items_processed += 1

                    if error or data is None:
                        with self._stats_lock:
                            self.stats.items_failed += 1
                    else:
                        # Write to segments (serialized, thread-safe)
                        self._write_downloaded_item(source_url, data)

                finally:
                    semaphore.release()

    def stop(self):
        """Stop the appender gracefully."""
        self._running = False

    def get_stats(self) -> AppenderStats:
        """Get current statistics."""
        return self.stats
