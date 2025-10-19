"""
Tests for EventBus implementations.
"""

import tempfile
import os
from pathlib import Path

from img2dataset.core.bus import SQLiteBus, create_event_envelope


def test_sqlite_bus_basic():
    """Test basic publish/subscribe with SQLite bus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        bus = SQLiteBus(db_path=db_path)

        # Publish events
        bus.publish("test.topic", "key1", {"data": "value1"})
        bus.publish("test.topic", "key2", {"data": "value2"})
        bus.publish("test.topic", "key3", {"data": "value3"})

        # Subscribe and consume
        events = list(bus.subscribe("test.topic", "group1"))

        assert len(events) == 3
        assert events[0].key == "key1"
        assert events[0].value["data"] == "value1"
        assert events[1].key == "key2"
        assert events[2].key == "key3"

        bus.close()


def test_sqlite_bus_offset_tracking():
    """Test consumer offset tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        bus = SQLiteBus(db_path=db_path)

        # Publish events
        for i in range(10):
            bus.publish("test.topic", f"key{i}", {"index": i})

        # Consume first 5 events
        events = []
        for i, event in enumerate(bus.subscribe("test.topic", "group1")):
            events.append(event)
            if i >= 4:
                break

        assert len(events) == 5

        # Check offset was committed
        offset = bus.get_offset("test.topic", "group1")
        assert offset == events[-1].offset

        # Resume consumption should get remaining events
        remaining = list(bus.subscribe("test.topic", "group1"))
        assert len(remaining) == 5
        assert remaining[0].value["index"] == 5

        bus.close()


def test_sqlite_bus_multiple_groups():
    """Test multiple consumer groups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        bus = SQLiteBus(db_path=db_path)

        # Publish events
        for i in range(5):
            bus.publish("test.topic", f"key{i}", {"index": i})

        # Two groups should each get all events
        group1_events = list(bus.subscribe("test.topic", "group1"))
        group2_events = list(bus.subscribe("test.topic", "group2"))

        assert len(group1_events) == 5
        assert len(group2_events) == 5

        bus.close()


def test_event_envelope():
    """Test event envelope creation."""
    envelope = create_event_envelope(
        event_id="test-123",
        entity_type="item",
        entity_id="item-456",
        kind="APPEND",
        payload={"data": "test"},
        producer_id="test-producer"
    )

    assert envelope["event_id"] == "test-123"
    assert envelope["entity"]["type"] == "item"
    assert envelope["entity"]["id"] == "item-456"
    assert envelope["kind"] == "APPEND"
    assert envelope["payload"]["data"] == "test"
    assert envelope["trace"]["producer"] == "test-producer"
    assert "occurred_at" in envelope


if __name__ == "__main__":
    test_sqlite_bus_basic()
    test_sqlite_bus_offset_tracking()
    test_sqlite_bus_multiple_groups()
    test_event_envelope()
    print("All EventBus tests passed!")
