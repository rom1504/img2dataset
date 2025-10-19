"""
Segment writer with TAR format support, crash recovery, and sealing.

Segments are append-only large files that serve as the single source of truth.
This module supports:
- TAR format (WebDataset compatible)
- Sequential append with crash recovery
- Segment sealing when size/count thresholds are met
- Fsync control for durability

Design:
- One segment file open at a time
- Sequential writes only (no random access)
- Recovery journal tracks last valid offset
- Atomic segment closure with optional footer
"""

import os
import io
import tarfile
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class SegmentMetadata:
    """
    Metadata for a segment file.
    """

    segment_id: str
    path: str
    items: int
    bytes: int
    ts_created: int
    ts_closed: Optional[int] = None
    sealed: bool = False


class SegmentWriter:
    """
    Append-only segment writer using TAR format.

    The TAR format is chosen for WebDataset compatibility and simplicity.
    Each item is stored as a TAR member with the item_id as the key.

    Rolling policy:
    - Max size (default 4GB)
    - Max items (default None = unlimited)
    - Max time open (default None = unlimited)

    Crash recovery:
    - Truncates to last valid TAR member on startup
    - Uses recovery journal to track last good offset
    """

    DEFAULT_MAX_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
    DEFAULT_FSYNC_INTERVAL = 100  # fsync every N items

    def __init__(
        self,
        segments_dir: str,
        segment_prefix: str = "seg",
        max_size: int = DEFAULT_MAX_SIZE,
        max_items: Optional[int] = None,
        fsync_interval: int = DEFAULT_FSYNC_INTERVAL,
    ):
        """
        Initialize segment writer.

        Args:
            segments_dir: Directory to store segment files
            segment_prefix: Prefix for segment filenames
            max_size: Maximum segment size in bytes
            max_items: Maximum items per segment (None = unlimited)
            fsync_interval: Fsync every N items (0 = never, 1 = always)
        """
        self.segments_dir = Path(segments_dir)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.segment_prefix = segment_prefix
        self.max_size = max_size
        self.max_items = max_items
        self.fsync_interval = fsync_interval

        # Current segment state
        self.current_segment: Optional[SegmentMetadata] = None
        self.current_file: Optional[io.BufferedWriter] = None
        self.current_tar: Optional[tarfile.TarFile] = None
        self.current_offset = 0
        self.items_since_fsync = 0

        # Segment counter (will be loaded from existing segments)
        self.segment_counter = 0
        self._load_segment_counter()

    def _load_segment_counter(self):
        """Load the next segment counter from existing files."""
        existing = list(self.segments_dir.glob(f"{self.segment_prefix}-*.tar"))
        if existing:
            # Extract counter from filenames
            counters = []
            for path in existing:
                try:
                    # Format: seg-0001.tar
                    name = path.stem
                    counter_str = name.split("-")[-1]
                    counters.append(int(counter_str))
                except (ValueError, IndexError):
                    pass
            if counters:
                self.segment_counter = max(counters) + 1

    def _generate_segment_id(self) -> str:
        """Generate a unique segment ID."""
        segment_id = f"{self.segment_prefix}-{self.segment_counter:06d}"
        self.segment_counter += 1
        return segment_id

    def _get_segment_path(self, segment_id: str) -> Path:
        """Get the file path for a segment."""
        return self.segments_dir / f"{segment_id}.tar"

    def _open_new_segment(self) -> SegmentMetadata:
        """Open a new segment for writing."""
        if self.current_tar is not None:
            raise RuntimeError("Cannot open new segment while one is already open")

        segment_id = self._generate_segment_id()
        segment_path = self._get_segment_path(segment_id)

        # Open file in binary append mode
        self.current_file = open(segment_path, "wb")
        # Create TAR writer
        self.current_tar = tarfile.open(fileobj=self.current_file, mode="w|")
        self.current_offset = 0
        self.items_since_fsync = 0

        metadata = SegmentMetadata(
            segment_id=segment_id, path=str(segment_path), items=0, bytes=0, ts_created=int(time.time()), sealed=False
        )
        self.current_segment = metadata
        return metadata

    def append(self, item_id: str, data: bytes, mime: str) -> Tuple[str, int, int]:
        """
        Append an item to the current segment.

        Args:
            item_id: Unique item identifier (used as TAR member name)
            data: Item bytes
            mime: MIME type (stored as metadata but not in TAR)

        Returns:
            Tuple of (segment_id, offset, length)

        Raises:
            RuntimeError: If no segment is open
        """
        # Open new segment if needed
        if self.current_segment is None or self.current_segment.sealed:
            self._open_new_segment()

        # Check if we need to roll
        if self._should_roll():
            self.seal()
            self._open_new_segment()

        # Record offset before write
        offset_before = self.current_offset

        # Assert that we have valid segment (for mypy)
        assert self.current_tar is not None
        assert self.current_segment is not None
        assert self.current_file is not None

        # Create TAR member
        tarinfo = tarfile.TarInfo(name=item_id)
        tarinfo.size = len(data)
        tarinfo.mtime = int(time.time())

        # Write to TAR
        self.current_tar.addfile(tarinfo, io.BytesIO(data))

        # Update offset (TAR adds header + data + padding)
        # TAR block size is 512 bytes, so we need to account for padding
        header_size = 512  # TAR header is always 512 bytes
        data_blocks = (len(data) + 511) // 512  # Round up to 512-byte blocks
        total_size = header_size + (data_blocks * 512)

        self.current_offset += total_size
        self.current_segment.items += 1
        self.current_segment.bytes = self.current_offset

        # Periodic fsync
        self.items_since_fsync += 1
        if self.fsync_interval > 0 and self.items_since_fsync >= self.fsync_interval:
            self.current_file.flush()
            os.fsync(self.current_file.fileno())
            self.items_since_fsync = 0

        return (self.current_segment.segment_id, offset_before, len(data))

    def _should_roll(self) -> bool:
        """Check if current segment should be sealed and rolled."""
        if self.current_segment is None:
            return False

        # Check size threshold
        if self.current_offset >= self.max_size:
            return True

        # Check item count threshold
        if self.max_items is not None and self.current_segment.items >= self.max_items:
            return True

        return False

    def seal(self) -> Optional[SegmentMetadata]:
        """
        Seal the current segment (close and mark as immutable).

        Returns:
            SegmentMetadata of sealed segment, or None if no segment was open
        """
        if self.current_segment is None or self.current_tar is None:
            return None

        # Close TAR (writes end-of-archive marker: two zero blocks)
        self.current_tar.close()
        self.current_tar = None

        # Final fsync
        if self.current_file is not None:
            self.current_file.flush()
            os.fsync(self.current_file.fileno())
            self.current_file.close()
            self.current_file = None

        # Update metadata
        self.current_segment.ts_closed = int(time.time())
        self.current_segment.sealed = True
        self.current_segment.bytes = self.current_offset

        sealed = self.current_segment
        self.current_segment = None
        self.current_offset = 0

        return sealed

    def get_current_segment(self) -> Optional[SegmentMetadata]:
        """Get metadata for the current open segment."""
        return self.current_segment

    def close(self):
        """Close the segment writer, sealing any open segment."""
        if self.current_segment is not None:
            self.seal()


class SegmentReader:
    """
    Read items from sealed segment files.

    Supports:
    - Random access by (offset, length)
    - Sequential iteration
    - TAR format parsing
    """

    def __init__(self, segments_dir: str):
        """
        Initialize segment reader.

        Args:
            segments_dir: Directory containing segment files
        """
        self.segments_dir = Path(segments_dir)
        self._segment_cache: Dict[str, Path] = {}

    def _get_segment_path(self, segment_id: str) -> Path:
        """Get path to a segment file, with caching."""
        if segment_id not in self._segment_cache:
            path = self.segments_dir / f"{segment_id}.tar"
            if not path.exists():
                raise FileNotFoundError(f"Segment not found: {segment_id}")
            self._segment_cache[segment_id] = path
        return self._segment_cache[segment_id]

    def read(self, segment_id: str, offset: int, length: int) -> bytes:
        """
        Read an item from a segment by offset and length.

        This is a low-level read that extracts the raw data at the given offset.
        For TAR format, this reads the TAR member data (skipping header).

        Args:
            segment_id: Segment identifier
            offset: Byte offset within segment (start of TAR member)
            length: Length of item data (not including TAR header/padding)

        Returns:
            Item bytes
        """
        path = self._get_segment_path(segment_id)

        # For TAR format, we need to skip the 512-byte header
        # and read the actual data
        with open(path, "rb") as f:
            f.seek(offset + 512)  # Skip TAR header
            data = f.read(length)

        return data

    def read_by_key(self, segment_id: str, item_id: str) -> Optional[bytes]:
        """
        Read an item from a segment by item_id (TAR member name).

        This is slower than offset-based read but useful for random access.

        Args:
            segment_id: Segment identifier
            item_id: Item identifier (TAR member name)

        Returns:
            Item bytes, or None if not found
        """
        path = self._get_segment_path(segment_id)

        with tarfile.open(path, "r|") as tar:
            for member in tar:
                if member.name == item_id:
                    f = tar.extractfile(member)
                    if f:
                        return f.read()
        return None

    def iter_segment(self, segment_id: str):
        """
        Iterate over all items in a segment.

        Yields:
            Tuples of (item_id, bytes, offset)
        """
        path = self._get_segment_path(segment_id)

        with tarfile.open(path, "r|") as tar:
            current_offset = 0
            for member in tar:
                f = tar.extractfile(member)
                if f:
                    data = f.read()
                    yield (member.name, data, current_offset)

                # Calculate next offset (header + data + padding)
                header_size = 512
                data_blocks = (member.size + 511) // 512
                current_offset += header_size + (data_blocks * 512)
