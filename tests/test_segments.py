"""
Tests for segment reading and writing.
"""

import tempfile
import os
from pathlib import Path

from img2dataset.core.io import SegmentWriter, SegmentReader


def test_segment_writer_basic():
    """Test basic segment writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(
            segments_dir=tmpdir,
            max_size=10 * 1024  # 10KB for testing
        )

        # Append items
        seg1, off1, len1 = writer.append("item1", b"data1", "image/jpeg")
        seg2, off2, len2 = writer.append("item2", b"data2", "image/png")

        assert seg1 == seg2  # Same segment
        assert off1 == 0
        assert off2 > off1  # Different offset

        # Close and seal
        writer.close()

        # Verify segment file exists
        segment_files = list(Path(tmpdir).glob("*.tar"))
        assert len(segment_files) == 1


def test_segment_writer_rolling():
    """Test segment rolling on size threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(
            segments_dir=tmpdir,
            max_size=2048,  # 2KB
            max_items=None
        )

        # Append enough data to trigger roll
        segments_used = set()
        for i in range(10):
            seg, off, length = writer.append(
                f"item{i}",
                b"x" * 500,  # 500 bytes each
                "image/jpeg"
            )
            segments_used.add(seg)

        writer.close()

        # Should have created multiple segments
        assert len(segments_used) > 1

        segment_files = list(Path(tmpdir).glob("*.tar"))
        assert len(segment_files) >= len(segments_used)


def test_segment_reader():
    """Test segment reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some data
        writer = SegmentWriter(segments_dir=tmpdir)

        data1 = b"test data 1"
        data2 = b"test data 2"

        seg1, off1, len1 = writer.append("item1", data1, "image/jpeg")
        seg2, off2, len2 = writer.append("item2", data2, "image/png")

        writer.close()

        # Read it back
        reader = SegmentReader(segments_dir=tmpdir)

        read1 = reader.read(seg1, off1, len1)
        read2 = reader.read(seg2, off2, len2)

        assert read1 == data1
        assert read2 == data2


def test_segment_reader_by_key():
    """Test reading by item_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some data
        writer = SegmentWriter(segments_dir=tmpdir)

        data = b"test data for key lookup"
        seg, off, length = writer.append("myitem", data, "image/jpeg")

        writer.close()

        # Read by key
        reader = SegmentReader(segments_dir=tmpdir)
        read_data = reader.read_by_key(seg, "myitem")

        assert read_data == data


def test_segment_iteration():
    """Test iterating over segment items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some data
        writer = SegmentWriter(segments_dir=tmpdir)

        items = [
            ("item1", b"data1"),
            ("item2", b"data2"),
            ("item3", b"data3")
        ]

        segment_id = None
        for item_id, data in items:
            seg, _, _ = writer.append(item_id, data, "image/jpeg")
            segment_id = seg

        writer.close()

        # Iterate
        reader = SegmentReader(segments_dir=tmpdir)
        read_items = list(reader.iter_segment(segment_id))

        assert len(read_items) == 3
        for i, (item_id, data, offset) in enumerate(read_items):
            assert item_id == items[i][0]
            assert data == items[i][1]


if __name__ == "__main__":
    test_segment_writer_basic()
    test_segment_writer_rolling()
    test_segment_reader()
    test_segment_reader_by_key()
    test_segment_iteration()
    print("All segment tests passed!")
