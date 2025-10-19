# Producer/Consumer Mode (Segment-as-Truth Architecture)

## Overview

img2dataset now supports a **producer/consumer architecture** with **segments as the single source of truth**. This architecture enables:

- **Scalable ingestion**: Separate producers from storage writers
- **No data duplication**: Segments are the only copy of image bytes
- **Idempotent operations**: Deduplicated by content hash (SHA256)
- **Sequential IO**: Optimized for training with cache-friendly access
- **Event-driven**: Lightweight event stream for coordination
- **Extensible**: Easy to add consumers (enrichers, trainers, etc.)

## Architecture

### Core Components

1. **Event Bus** (`ingest.items`, `segments.events`)
   - Carries lightweight commands and facts
   - Never carries image bytes (only pointers)
   - SQLite implementation for local use (Kafka adapter available)

2. **Segments** (TAR files)
   - Append-only large files (default: 4GB)
   - WebDataset-compatible TAR format
   - The single source of truth for image bytes

3. **Index** (SQLite + optional Parquet)
   - Maps `item_id` → `(segment_id, offset, length)`
   - Enables deduplication and sequential access
   - Indexed by `(segment_id, offset)` for trainer performance

4. **Segment Appender**
   - The ONLY process that writes to segments/index
   - Consumes from `ingest.items`, fetches bytes, appends
   - Publishes `APPEND` and `SEGMENT_CLOSED` events

5. **Consumers** (optional)
   - Shard materializer: Build WebDataset shards or manifests
   - Enrichers: Extract metadata (dimensions, safety, embeddings)
   - Trainers: Read directly from segments

### Data Flow

```
URLs → Producer (enqueue) → ingest.items → Segment Appender
                                                ↓
                                          Segments + Index
                                                ↓
                                          segments.events
                                                ↓
                                    Consumers (materializer, trainer)
```

## Usage

### Mode 1: Service Mode (Decoupled)

Run producer and consumer as separate processes:

```bash
# Terminal 1: Start segment appender service
img2dataset service --output_folder output/

# Terminal 2: Enqueue URLs
img2dataset enqueue \
    --url_list urls.txt \
    --output_folder output/ \
    --input_format txt

# Terminal 3: Materialize dataset (optional)
img2dataset materialize \
    --output_folder output/ \
    --manifest_path manifest.json \
    --output_format manifest
```

### Mode 2: Traditional (Backward Compatible)

The traditional CLI still works, but internally uses the new architecture:

```bash
img2dataset download \
    --url_list urls.txt \
    --output_folder output/ \
    --output_format webdataset \
    --processes_count 8 \
    --thread_count 32
```

## API Reference

### CLI Commands

#### `img2dataset service`

Start the segment appender service.

**Arguments:**
- `--output_folder`: Output directory for segments, index, and bus (default: `output`)
- `--max_segment_size`: Maximum segment size in bytes (default: 4GB)
- `--fetch_retries`: HTTP retry attempts (default: 3)
- `--fetch_timeout`: HTTP timeout in seconds (default: 10)
- `--user_agent_token`: Optional User-Agent token
- `--max_items`: Maximum items to process (default: unlimited)

**Example:**
```bash
img2dataset service \
    --output_folder /data/output \
    --max_segment_size 8589934592 \
    --fetch_retries 5
```

#### `img2dataset enqueue`

Enqueue URLs to the ingest queue.

**Arguments:**
- `--url_list`: Path to URL list file (required)
- `--output_folder`: Output directory (default: `output`)
- `--input_format`: Input format: txt, csv, json, parquet (default: `txt`)
- `--url_col`: Column name for URLs in structured formats (default: `url`)

**Example:**
```bash
img2dataset enqueue \
    --url_list urls.parquet \
    --output_folder /data/output \
    --input_format parquet \
    --url_col image_url
```

#### `img2dataset materialize`

Materialize dataset from segments.

**Arguments:**
- `--output_folder`: Output directory (default: `output`)
- `--manifest_path`: Path to output manifest (default: `manifest.json`)
- `--output_format`: Output format: manifest, parquet (default: `manifest`)

**Example:**
```bash
img2dataset materialize \
    --output_folder /data/output \
    --manifest_path dataset.json \
    --output_format manifest
```

### Python API

#### SegmentAppender

```python
from img2dataset.core.bus import SQLiteBus
from img2dataset.core.index_store import IndexStore
from img2dataset.core.io import SegmentWriter
from img2dataset.core.segment_appender import SegmentAppender

# Initialize components
bus = SQLiteBus(db_path="eventbus.sqlite3")
index = IndexStore(db_path="index.sqlite3")
segment_writer = SegmentWriter(segments_dir="segments/")

# Create appender
appender = SegmentAppender(
    bus=bus,
    index=index,
    segment_writer=segment_writer,
    fetch_retries=3
)

# Run
appender.run(max_items=10000)
```

#### SegmentDataLoader (Training)

```python
from img2dataset.core.index_store import IndexStore
from img2dataset.core.io import SegmentReader
from img2dataset.consumers.trainer_example import SegmentDataLoader

# Initialize
index = IndexStore(db_path="index.sqlite3")
reader = SegmentReader(segments_dir="segments/")

# Create data loader
loader = SegmentDataLoader(
    index=index,
    segment_reader=reader,
    batch_size=32
)

# Iterate over batches
for batch_ids, batch_images in loader.iter_batches():
    # Your training code here
    pass
```

#### ShardMaterializer

```python
from img2dataset.consumers.shard_materializer import materialize_shards

# Materialize manifest-only shards (no data duplication)
output_dir = materialize_shards(
    index_path="index.sqlite3",
    segments_dir="segments/",
    output_dir="shards/",
    dataset_name="my_dataset",
    shard_size=10000,
    mode="manifest"
)
```

## Schema

### ingest.items (Commands)

Events published to this topic represent work to be done.

```json
{
  "source_url": "https://example.com/image.jpg",
  "meta": {
    "license": "CC-BY",
    "extra": {}
  },
  "ts_enq": 1739990000
}
```

### segments.events (Facts)

#### APPEND Event

Published when an item is appended to a segment.

```json
{
  "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "occurred_at": 1739990000,
  "entity": {
    "type": "item",
    "id": "abc123..."
  },
  "kind": "APPEND",
  "payload_version": 1,
  "payload": {
    "type": "APPEND",
    "item_id": "abc123...",
    "segment_id": "seg-000042",
    "offset": 12345678,
    "length": 183742,
    "mime": "image/jpeg",
    "ts_ingest": 1739990000,
    "source_url": "https://..."
  },
  "trace": {
    "producer": "appender@host",
    "attempt": 1
  }
}
```

#### SEGMENT_CLOSED Event

Published when a segment is sealed.

```json
{
  "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "occurred_at": 1739992222,
  "entity": {
    "type": "segment",
    "id": "seg-000042"
  },
  "kind": "SEGMENT_CLOSED",
  "payload_version": 1,
  "payload": {
    "type": "SEGMENT_CLOSED",
    "segment_id": "seg-000042",
    "items": 48231,
    "bytes": 4096000000,
    "uri": "file:///data/segments/seg-000042.tar",
    "ts_close": 1739992222
  },
  "trace": {
    "producer": "appender@host",
    "attempt": 1
  }
}
```

### Index Schema

SQLite table with the following schema:

```sql
CREATE TABLE items (
    item_id TEXT PRIMARY KEY,         -- sha256(bytes)
    segment_id TEXT NOT NULL,         -- which segment contains this item
    offset INTEGER NOT NULL,          -- byte offset within segment
    length INTEGER NOT NULL,          -- byte length of item
    mime TEXT,                        -- MIME type
    ts_ingest INTEGER NOT NULL,       -- Unix timestamp
    sha256 TEXT NOT NULL              -- duplicate of item_id for auditing
);

CREATE INDEX idx_items_seg_off ON items(segment_id, offset);
CREATE INDEX idx_items_ts ON items(ts_ingest);
```

## Performance Tuning

### Segment Size

- **Default**: 4GB (good for most use cases)
- **Smaller** (1-2GB): Faster recovery, more granular control
- **Larger** (8-16GB): Fewer files, less overhead

```bash
img2dataset service --max_segment_size 8589934592  # 8GB
```

### Sequential Reading

For optimal training performance, the index is ordered by `(segment_id, offset)`. This ensures:

- Sequential disk IO (cache-friendly)
- Minimal seeks
- High throughput

```python
# Good: Sequential access
samples = index.sample_sequential(limit=1000)

# Bad: Random access
samples = [index.get(random_id) for _ in range(1000)]
```

### Filesystem

- **Recommended**: XFS or ext4 with `noatime`
- **Large readahead**: `blockdev --setra 8192 /dev/sdX`
- **Sequential writes dominate**: Use fast storage for segments

### HTTP Fetching

- Connection pooling (built-in)
- DNS caching (built-in)
- Tune retries and timeout for your network:

```bash
img2dataset service \
    --fetch_retries 5 \
    --fetch_timeout 30
```

## Monitoring

### Metrics to Track

1. **Append rate**: items/sec being written
2. **Queue lag**: backlog in `ingest.items`
3. **Fsync latency**: time to sync segment writes
4. **Decode failures**: failed image downloads
5. **Deduplication rate**: % of items skipped

### Logging

The segment appender logs progress every 100 items:

```
Processed 100 items (appended=95, dedup=3, failed=2)
Processed 200 items (appended=192, dedup=5, failed=3)
...
```

## Migration Guide

### From Traditional img2dataset

Your existing commands will continue to work:

```bash
# Old (still works)
img2dataset --url_list urls.txt --output_folder out --output_format webdataset

# New (equivalent, more flexible)
img2dataset download --url_list urls.txt --output_folder out --output_format webdataset
```

### To Service Mode

1. Start the service:
   ```bash
   img2dataset service --output_folder /data/output
   ```

2. Enqueue your URLs:
   ```bash
   img2dataset enqueue --url_list urls.txt --output_folder /data/output
   ```

3. (Optional) Materialize shards:
   ```bash
   img2dataset materialize --output_folder /data/output
   ```

## FAQ

### Q: What if I just want WebDataset shards like before?

A: Use the traditional CLI or materialize after ingestion:

```bash
img2dataset materialize \
    --output_folder output/ \
    --output_format physical  # Creates physical TAR shards
```

### Q: How do I resume after a crash?

A: Just restart the service. The segment appender will:
- Resume from last committed offset in the event bus
- Skip already-ingested items (deduplication by item_id)
- Truncate any incomplete segment writes

### Q: Can I use this with Kafka/Redis instead of SQLite?

A: Yes! The EventBus is pluggable. Implement the `EventBus` interface for your broker:

```python
from img2dataset.core.bus import EventBus

class KafkaBus(EventBus):
    # Implement publish(), subscribe(), etc.
    pass
```

### Q: How do I scale horizontally?

A: Run multiple segment appenders with different consumer groups:

```bash
# Worker 1
img2dataset service --output_folder /shared/output --consumer_group worker1

# Worker 2
img2dataset service --output_folder /shared/output --consumer_group worker2
```

Note: Use a shared filesystem or object store for segments.

### Q: What about deduplication across runs?

A: Deduplication is automatic via content hash (SHA256). If you re-enqueue the same URL and it produces the same bytes, it will be skipped.

## Troubleshooting

### Segment appender stuck

Check queue lag:

```python
from img2dataset.core.bus import SQLiteBus

bus = SQLiteBus("eventbus.sqlite3")
count = bus.get_topic_count("ingest.items")
offset = bus.get_offset("ingest.items", "segment_appender")
print(f"Total events: {count}, consumed: {offset}")
```

### Index getting large

Export to Parquet periodically:

```python
from img2dataset.core.index_store import IndexStore

index = IndexStore("index.sqlite3")
index.export_to_parquet("index.parquet")
```

### Segments not readable

Verify integrity:

```python
from img2dataset.core.io import SegmentReader

reader = SegmentReader("segments/")
for item_id, data, offset in reader.iter_segment("seg-000001"):
    print(f"{item_id}: {len(data)} bytes")
```

## Examples

### Example 1: Simple Ingestion

```bash
# Create URL list
echo "https://example.com/image1.jpg" > urls.txt
echo "https://example.com/image2.jpg" >> urls.txt

# Start service
img2dataset service --output_folder data/

# In another terminal: enqueue
img2dataset enqueue --url_list urls.txt --output_folder data/
```

### Example 2: Training from Segments

```python
from img2dataset.consumers.trainer_example import train_example

train_example(
    index_path="data/index.sqlite3",
    segments_dir="data/segments/",
    batch_size=32,
    max_batches=100
)
```

### Example 3: Building Manifests

```python
from img2dataset.consumers.shard_materializer import materialize_shards

materialize_shards(
    index_path="data/index.sqlite3",
    segments_dir="data/segments/",
    output_dir="data/manifests/",
    dataset_name="my_dataset",
    shard_size=10000,
    mode="manifest"  # No data duplication
)
```

## Contributing

Contributions are welcome! Key areas:

- Additional event bus adapters (Kafka, Redis, NATS)
- Cloud storage backends (S3, GCS, Azure)
- Recovery improvements
- Performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 (same as img2dataset)
