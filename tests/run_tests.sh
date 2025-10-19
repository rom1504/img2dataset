#!/bin/bash
# Test runner for producer/consumer architecture

set -e

echo "Running img2dataset producer/consumer architecture tests..."
echo

echo "=== Test 1: EventBus ==="
python tests/test_eventbus.py
echo

echo "=== Test 2: IndexStore ==="
python tests/test_index_store.py
echo

echo "=== Test 3: Segments ==="
python tests/test_segments.py
echo

echo "=== Test 4: End-to-End ==="
python tests/test_end_to_end.py
echo

echo "✓ All tests passed!"
