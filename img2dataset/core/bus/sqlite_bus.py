"""
SQLite-based EventBus implementation for local/single-node operation.

This provides a simple, zero-dependency event bus suitable for:
- Single-node img2dataset operation
- Development and testing
- Small to medium datasets

For large-scale production deployments, consider using kafka_bus.py instead.
"""

import sqlite3
import json
import threading
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from contextlib import contextmanager

from .base import EventBus, Event


class SQLiteBus(EventBus):
    """
    SQLite-based event bus with file-based persistence.

    Schema:
        events table:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT (offset)
            - topic: TEXT
            - key: TEXT
            - value: TEXT (JSON)
            - timestamp: INTEGER
            - INDEX on (topic, id)

        consumer_offsets table:
            - topic: TEXT
            - consumer_group: TEXT
            - offset: INTEGER
            - PRIMARY KEY (topic, consumer_group)

    Thread-safety: Uses connection per thread and table-level locking
    """

    def __init__(self, db_path: str = "eventbus.sqlite3"):
        """
        Initialize the SQLite event bus.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path,
                isolation_level="IMMEDIATE",  # Use IMMEDIATE for better concurrency
                check_same_thread=False,
            )
            # Enable WAL mode for better concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema if not exists."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_topic_id
                ON events(topic, id)
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consumer_offsets (
                    topic TEXT NOT NULL,
                    consumer_group TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (topic, consumer_group)
                )
            """
            )

            conn.commit()

    def publish(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        """
        Publish an event to a topic.

        Args:
            topic: The topic to publish to
            key: The event key
            value: The event payload (will be JSON-serialized)
        """
        timestamp = int(time.time())
        value_json = json.dumps(value)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (topic, key, value, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (topic, key, value_json, timestamp),
            )
            conn.commit()

    def subscribe(
        self, topic: str, group: str, auto_commit: bool = True, start_offset: Optional[int] = None
    ) -> Iterator[Event]:
        """
        Subscribe to a topic as part of a consumer group.

        Args:
            topic: The topic to subscribe to
            group: Consumer group ID
            auto_commit: Whether to automatically commit offsets
            start_offset: Optional specific offset to start from

        Yields:
            Events from the topic in order
        """
        # Determine starting offset
        if start_offset is not None:
            current_offset: int = start_offset
        else:
            offset_result = self.get_offset(topic, group)
            current_offset = offset_result if offset_result is not None else 0

        with self._get_connection() as conn:
            while True:
                # Fetch next batch of events
                cursor = conn.execute(
                    """
                    SELECT id, topic, key, value, timestamp
                    FROM events
                    WHERE topic = ? AND id > ?
                    ORDER BY id
                    LIMIT 100
                    """,
                    (topic, current_offset),
                )

                rows = cursor.fetchall()
                if not rows:
                    # No more events available
                    break

                for row in rows:
                    event_id, topic, key, value_json, timestamp = row

                    event = Event(
                        topic=topic, key=key, value=json.loads(value_json), timestamp=timestamp, offset=event_id
                    )

                    yield event

                    if auto_commit:
                        self.commit(topic, group, event_id)

                    current_offset = event_id

    def commit(self, topic: str, group: str, offset: int) -> None:
        """
        Manually commit an offset for a consumer group.

        Args:
            topic: The topic
            group: Consumer group ID
            offset: The offset to commit
        """
        timestamp = int(time.time())

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO consumer_offsets (topic, consumer_group, offset, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(topic, consumer_group)
                DO UPDATE SET
                    offset = excluded.offset,
                    updated_at = excluded.updated_at
                """,
                (topic, group, offset, timestamp),
            )
            conn.commit()

    def get_offset(self, topic: str, group: str) -> Optional[int]:
        """
        Get the last committed offset for a consumer group.

        Args:
            topic: The topic
            group: Consumer group ID

        Returns:
            Last committed offset, or None if no offset committed yet
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT offset FROM consumer_offsets
                WHERE topic = ? AND consumer_group = ?
                """,
                (topic, group),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_topic_count(self, topic: str) -> int:
        """
        Get the total number of events in a topic.

        Args:
            topic: The topic

        Returns:
            Number of events
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events WHERE topic = ?", (topic,))
            return cursor.fetchone()[0]

    def get_latest_offset(self, topic: str) -> Optional[int]:
        """
        Get the latest offset in a topic.

        Args:
            topic: The topic

        Returns:
            Latest offset, or None if topic is empty
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT MAX(id) FROM events WHERE topic = ?", (topic,))
            result = cursor.fetchone()[0]
            return result

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")
