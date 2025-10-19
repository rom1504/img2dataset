"""
Global index store for segment-based storage.

The index is the single source of truth for mapping item_id to (segment_id, offset, length).
Supports SQLite for fast queries and optional Parquet export for analytics.

Schema:
    item_id TEXT PRIMARY KEY    -- sha256(bytes) of the item
    segment_id TEXT NOT NULL    -- which segment contains this item
    offset INTEGER NOT NULL     -- byte offset within segment
    length INTEGER NOT NULL     -- byte length of item
    mime TEXT                   -- MIME type (e.g., image/jpeg)
    ts_ingest INTEGER NOT NULL  -- Unix timestamp of ingestion
    sha256 TEXT NOT NULL        -- duplicate of item_id for auditing

Indexes:
    PRIMARY KEY (item_id)
    INDEX (segment_id, offset) -- for sequential scans
"""

import sqlite3
import threading
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class IndexEntry:
    """
    Represents a single item in the index.
    """
    item_id: str
    segment_id: str
    offset: int
    length: int
    mime: str
    ts_ingest: int
    sha256: str


class IndexStore:
    """
    Thread-safe index store using SQLite with optional Parquet export.

    The index tracks all items and their locations within segments.
    """

    def __init__(self, db_path: str = "index.sqlite3"):
        """
        Initialize the index store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                isolation_level='IMMEDIATE',
                check_same_thread=False
            )
            # Enable WAL mode for better concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            # Row factory for easier access
            self._local.conn.row_factory = sqlite3.Row

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    segment_id TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    mime TEXT,
                    ts_ingest INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                )
            """)

            # Index for sequential reading by segment
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_seg_off
                ON items(segment_id, offset)
            """)

            # Index for timestamp queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_ts
                ON items(ts_ingest)
            """)

            conn.commit()

    def insert(
        self,
        item_id: str,
        segment_id: str,
        offset: int,
        length: int,
        mime: str,
        sha256: str,
        ts_ingest: Optional[int] = None
    ) -> bool:
        """
        Insert a new item into the index (idempotent).

        Args:
            item_id: Unique item identifier (typically sha256)
            segment_id: Segment containing this item
            offset: Byte offset within segment
            length: Byte length of item
            mime: MIME type
            sha256: SHA256 hash of item bytes
            ts_ingest: Ingestion timestamp (defaults to now)

        Returns:
            True if inserted, False if already exists (idempotent)
        """
        if ts_ingest is None:
            ts_ingest = int(time.time())

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO items (item_id, segment_id, offset, length, mime, ts_ingest, sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (item_id, segment_id, offset, length, mime, ts_ingest, sha256)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Item already exists (duplicate)
                return False

    def get(self, item_id: str) -> Optional[IndexEntry]:
        """
        Get an item from the index by ID.

        Args:
            item_id: Item identifier

        Returns:
            IndexEntry if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                FROM items
                WHERE item_id = ?
                """,
                (item_id,)
            )
            row = cursor.fetchone()
            if row:
                return IndexEntry(**dict(row))
            return None

    def exists(self, item_id: str) -> bool:
        """
        Check if an item exists in the index.

        Args:
            item_id: Item identifier

        Returns:
            True if exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM items WHERE item_id = ? LIMIT 1",
                (item_id,)
            )
            return cursor.fetchone() is not None

    def get_by_segment(self, segment_id: str) -> List[IndexEntry]:
        """
        Get all items in a segment, ordered by offset.

        Args:
            segment_id: Segment identifier

        Returns:
            List of IndexEntry objects
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                FROM items
                WHERE segment_id = ?
                ORDER BY offset
                """,
                (segment_id,)
            )
            return [IndexEntry(**dict(row)) for row in cursor.fetchall()]

    def sample_sequential(
        self,
        limit: int,
        start_segment: Optional[str] = None,
        start_offset: int = 0
    ) -> List[IndexEntry]:
        """
        Sample items sequentially for optimal IO performance.

        This is the key API for trainers that want to read segments sequentially.
        Items are returned in (segment_id, offset) order for cache-friendly access.

        Args:
            limit: Maximum number of items to return
            start_segment: Optional segment to start from (for resumption)
            start_offset: Offset within start_segment to begin

        Returns:
            List of IndexEntry objects in sequential order
        """
        with self._get_connection() as conn:
            if start_segment is not None:
                cursor = conn.execute(
                    """
                    SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                    FROM items
                    WHERE (segment_id > ? OR (segment_id = ? AND offset >= ?))
                    ORDER BY segment_id, offset
                    LIMIT ?
                    """,
                    (start_segment, start_segment, start_offset, limit)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                    FROM items
                    ORDER BY segment_id, offset
                    LIMIT ?
                    """,
                    (limit,)
                )

            return [IndexEntry(**dict(row)) for row in cursor.fetchall()]

    def count(self) -> int:
        """
        Get total number of items in index.

        Returns:
            Item count
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM items")
            return cursor.fetchone()[0]

    def count_by_segment(self, segment_id: str) -> int:
        """
        Get number of items in a specific segment.

        Args:
            segment_id: Segment identifier

        Returns:
            Item count
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM items WHERE segment_id = ?",
                (segment_id,)
            )
            return cursor.fetchone()[0]

    def get_segments(self) -> List[str]:
        """
        Get list of all segment IDs.

        Returns:
            List of segment IDs
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT segment_id FROM items ORDER BY segment_id"
            )
            return [row[0] for row in cursor.fetchall()]

    def iter_all(self, batch_size: int = 1000) -> Iterator[List[IndexEntry]]:
        """
        Iterate over all items in batches.

        Args:
            batch_size: Number of items per batch

        Yields:
            Batches of IndexEntry objects
        """
        offset = 0
        with self._get_connection() as conn:
            while True:
                cursor = conn.execute(
                    """
                    SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                    FROM items
                    ORDER BY segment_id, offset
                    LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset)
                )
                rows = cursor.fetchall()
                if not rows:
                    break

                yield [IndexEntry(**dict(row)) for row in rows]
                offset += len(rows)

    def export_to_parquet(self, output_path: str) -> None:
        """
        Export the entire index to a Parquet file.

        Requires: pyarrow

        Args:
            output_path: Path to output Parquet file
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")

        # Read all data from SQLite
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT item_id, segment_id, offset, length, mime, ts_ingest, sha256
                FROM items
                ORDER BY segment_id, offset
                """
            )
            rows = cursor.fetchall()

        if not rows:
            # Create empty parquet with schema
            schema = pa.schema([
                ('item_id', pa.string()),
                ('segment_id', pa.string()),
                ('offset', pa.int64()),
                ('length', pa.int64()),
                ('mime', pa.string()),
                ('ts_ingest', pa.int64()),
                ('sha256', pa.string()),
            ])
            table = pa.Table.from_pydict({}, schema=schema)
        else:
            # Convert to Arrow table
            data = {
                'item_id': [row['item_id'] for row in rows],
                'segment_id': [row['segment_id'] for row in rows],
                'offset': [row['offset'] for row in rows],
                'length': [row['length'] for row in rows],
                'mime': [row['mime'] for row in rows],
                'ts_ingest': [row['ts_ingest'] for row in rows],
                'sha256': [row['sha256'] for row in rows],
            }
            table = pa.Table.from_pydict(data)

        # Write to Parquet
        pq.write_table(table, output_path)

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')


def compute_item_id(data: bytes) -> str:
    """
    Compute item_id from bytes.

    Args:
        data: Item bytes

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(data).hexdigest()
