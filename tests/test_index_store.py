"""
Tests for IndexStore.
"""

import tempfile
import os

from img2dataset.core.index_store import IndexStore, compute_item_id


def test_index_store_basic():
    """Test basic index operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "index.db")
        index = IndexStore(db_path=db_path)

        # Insert items
        inserted = index.insert(
            item_id="item1",
            segment_id="seg-001",
            offset=0,
            length=1024,
            mime="image/jpeg",
            sha256="abc123"
        )
        assert inserted is True

        # Duplicate insert should be idempotent
        inserted = index.insert(
            item_id="item1",
            segment_id="seg-001",
            offset=0,
            length=1024,
            mime="image/jpeg",
            sha256="abc123"
        )
        assert inserted is False

        # Get item
        entry = index.get("item1")
        assert entry is not None
        assert entry.item_id == "item1"
        assert entry.segment_id == "seg-001"
        assert entry.offset == 0
        assert entry.length == 1024

        # Check exists
        assert index.exists("item1") is True
        assert index.exists("nonexistent") is False

        index.close()


def test_index_store_sequential_scan():
    """Test sequential scanning for trainer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "index.db")
        index = IndexStore(db_path=db_path)

        # Insert items across multiple segments
        for seg_id in range(3):
            for item_idx in range(10):
                index.insert(
                    item_id=f"item-{seg_id}-{item_idx}",
                    segment_id=f"seg-{seg_id:03d}",
                    offset=item_idx * 1000,
                    length=1000,
                    mime="image/jpeg",
                    sha256=f"hash-{seg_id}-{item_idx}"
                )

        # Sequential scan should return items in segment order
        samples = index.sample_sequential(limit=100)
        assert len(samples) == 30

        # Verify ordering
        prev_segment = None
        prev_offset = -1
        for entry in samples:
            if prev_segment == entry.segment_id:
                assert entry.offset > prev_offset
            prev_segment = entry.segment_id
            prev_offset = entry.offset

        # Resume from middle
        samples = index.sample_sequential(
            limit=10,
            start_segment="seg-001",
            start_offset=5000
        )
        assert len(samples) == 10
        assert samples[0].segment_id == "seg-001"
        assert samples[0].offset >= 5000

        index.close()


def test_index_store_count():
    """Test counting functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "index.db")
        index = IndexStore(db_path=db_path)

        # Insert items
        for i in range(5):
            index.insert(
                item_id=f"item-{i}",
                segment_id="seg-001",
                offset=i * 1000,
                length=1000,
                mime="image/jpeg",
                sha256=f"hash-{i}"
            )

        for i in range(3):
            index.insert(
                item_id=f"item-seg2-{i}",
                segment_id="seg-002",
                offset=i * 1000,
                length=1000,
                mime="image/jpeg",
                sha256=f"hash-seg2-{i}"
            )

        assert index.count() == 8
        assert index.count_by_segment("seg-001") == 5
        assert index.count_by_segment("seg-002") == 3

        segments = index.get_segments()
        assert len(segments) == 2
        assert "seg-001" in segments
        assert "seg-002" in segments

        index.close()


def test_compute_item_id():
    """Test item_id computation."""
    data = b"test data"
    item_id = compute_item_id(data)

    # Should be hex sha256
    assert len(item_id) == 64
    assert all(c in "0123456789abcdef" for c in item_id)

    # Should be deterministic
    assert compute_item_id(data) == item_id


if __name__ == "__main__":
    test_index_store_basic()
    test_index_store_sequential_scan()
    test_index_store_count()
    test_compute_item_id()
    print("All IndexStore tests passed!")
