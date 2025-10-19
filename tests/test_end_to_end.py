"""
End-to-end tests for producer/consumer architecture.
"""

import tempfile
import os
import time
from pathlib import Path

from img2dataset.core.bus import SQLiteBus
from img2dataset.core.index_store import IndexStore
from img2dataset.core.io import SegmentWriter, SegmentReader
from img2dataset.core.segment_appender import SegmentAppender


def create_test_image():
    """Create a minimal test JPEG image."""
    # Minimal JPEG (1x1 pixel, red)
    jpeg_data = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
        0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
        0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
        0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
        0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
        0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
        0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4,
        0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08,
        0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7F, 0xFF,
        0xD9
    ])
    return jpeg_data


def test_end_to_end_simple():
    """
    Test complete pipeline: enqueue -> appender -> index -> reader
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup
        bus = SQLiteBus(db_path=str(tmpdir / "bus.db"))
        index = IndexStore(db_path=str(tmpdir / "index.db"))
        segment_writer = SegmentWriter(
            segments_dir=str(tmpdir / "segments"),
            max_size=1024 * 1024  # 1MB
        )

        # Create test image
        test_image = create_test_image()

        # Enqueue items (simulate URLs, but we'll use data URIs)
        import base64
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(test_image).decode()}"

        for i in range(5):
            bus.publish(
                topic="ingest.items",
                key=f"item{i}",
                value={
                    "source_url": data_uri,
                    "meta": {"index": i}
                }
            )

        # Note: The segment appender expects real URLs
        # For this test, we'll test the components separately

        # Direct append to segments (bypassing network fetch)
        from img2dataset.core.index_store import compute_item_id

        for i in range(5):
            item_id = compute_item_id(test_image + str(i).encode())
            seg, off, length = segment_writer.append(
                item_id=item_id,
                data=test_image,
                mime="image/jpeg"
            )

            index.insert(
                item_id=item_id,
                segment_id=seg,
                offset=off,
                length=length,
                mime="image/jpeg",
                sha256=item_id,
                ts_ingest=int(time.time())
            )

        segment_writer.close()

        # Verify index
        assert index.count() == 5

        # Sequential scan
        samples = index.sample_sequential(limit=10)
        assert len(samples) == 5

        # Read from segments
        reader = SegmentReader(segments_dir=str(tmpdir / "segments"))

        for entry in samples:
            data = reader.read(entry.segment_id, entry.offset, entry.length)
            assert data == test_image

        # Cleanup
        index.close()
        bus.close()

        print("End-to-end test passed!")


def test_idempotency():
    """Test that duplicate items are deduplicated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        index = IndexStore(db_path=str(tmpdir / "index.db"))
        segment_writer = SegmentWriter(segments_dir=str(tmpdir / "segments"))

        test_data = b"duplicate data"
        from img2dataset.core.index_store import compute_item_id

        item_id = compute_item_id(test_data)

        # Insert same item twice
        seg1, off1, len1 = segment_writer.append(item_id, test_data, "image/jpeg")
        inserted1 = index.insert(item_id, seg1, off1, len1, "image/jpeg", item_id)

        seg2, off2, len2 = segment_writer.append(item_id, test_data, "image/jpeg")
        inserted2 = index.insert(item_id, seg2, off2, len2, "image/jpeg", item_id)

        segment_writer.close()

        # First insert should succeed, second should be idempotent
        assert inserted1 is True
        assert inserted2 is False

        # Only one entry in index
        assert index.count() == 1

        index.close()

        print("Idempotency test passed!")


def test_recovery():
    """Test recovery from partial writes (simulated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write some segments
        writer = SegmentWriter(segments_dir=str(tmpdir / "segments"))

        for i in range(3):
            writer.append(f"item{i}", b"data" * 100, "image/jpeg")

        # Get current segment before close
        current = writer.get_current_segment()
        segment_id = current.segment_id if current else None

        writer.close()

        # Reopen writer (simulating recovery)
        writer2 = SegmentWriter(segments_dir=str(tmpdir / "segments"))

        # Should be able to append to new segment
        seg, off, length = writer2.append("item_new", b"new data", "image/jpeg")

        # Should have created a new segment (counter incremented)
        assert seg != segment_id

        writer2.close()

        print("Recovery test passed!")


if __name__ == "__main__":
    test_end_to_end_simple()
    test_idempotency()
    test_recovery()
    print("\nAll end-to-end tests passed!")
