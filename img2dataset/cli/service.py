"""
Service subcommands for producer/consumer mode.

This module provides service-oriented commands:
- service start: Start local broker + segment appender
- enqueue: Push URLs to ingest.items
- materialize: Build manifests/shards from index
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from ..core.bus import SQLiteBus
from ..core.index_store import IndexStore
from ..core.io import SegmentWriter
from ..core.segment_appender import SegmentAppender


def start_service(
    output_folder: str = "output",
    max_segment_size: int = 4 * 1024 * 1024 * 1024,  # 4GB
    fetch_retries: int = 3,
    fetch_timeout: int = 10,
    user_agent_token: Optional[str] = None,
    max_items: Optional[int] = None
):
    """
    Start a local service (bus + segment appender).

    This runs a segment appender that consumes from ingest.items and writes
    to segments and index.

    Args:
        output_folder: Output directory for segments, index, and bus
        max_segment_size: Maximum segment size in bytes
        fetch_retries: Number of HTTP retry attempts
        fetch_timeout: HTTP timeout in seconds
        user_agent_token: Optional token for User-Agent string
        max_items: Maximum items to process (None = unlimited)
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    bus_path = output_path / "eventbus.sqlite3"
    index_path = output_path / "index.sqlite3"
    segments_dir = output_path / "segments"

    print(f"Starting img2dataset service in {output_folder}")
    print(f"  Event bus: {bus_path}")
    print(f"  Index: {index_path}")
    print(f"  Segments: {segments_dir}")

    bus = SQLiteBus(db_path=str(bus_path))
    index = IndexStore(db_path=str(index_path))
    segment_writer = SegmentWriter(
        segments_dir=str(segments_dir),
        max_size=max_segment_size
    )

    # Build user agent
    user_agent = f"img2dataset/2.0"
    if user_agent_token:
        user_agent += f" ({user_agent_token})"

    # Create and run appender
    appender = SegmentAppender(
        bus=bus,
        index=index,
        segment_writer=segment_writer,
        fetch_retries=fetch_retries,
        fetch_timeout=fetch_timeout,
        user_agent=user_agent
    )

    try:
        appender.run(max_items=max_items)
    except KeyboardInterrupt:
        print("\nShutting down...")
        appender.stop()
    finally:
        segment_writer.close()
        index.close()
        bus.close()


def enqueue(
    url_list: str,
    output_folder: str = "output",
    input_format: str = "txt",
    url_col: str = "url"
):
    """
    Enqueue URLs to ingest.items topic.

    Args:
        url_list: Path to URL list file
        output_folder: Output directory (for bus)
        input_format: Input format (txt, csv, json, parquet)
        url_col: Column name for URLs (for structured formats)
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    bus_path = output_path / "eventbus.sqlite3"
    bus = SQLiteBus(db_path=str(bus_path))

    print(f"Enqueuing URLs from {url_list}")

    try:
        # Read URLs based on format
        urls = []

        if input_format == "txt":
            with open(url_list, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]

        elif input_format == "csv":
            import pandas as pd
            df = pd.read_csv(url_list)
            if url_col not in df.columns:
                print(f"Error: Column '{url_col}' not found in CSV")
                return
            urls = df[url_col].tolist()

        elif input_format == "json":
            import json
            with open(url_list, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    urls = [item.get(url_col) if isinstance(item, dict) else item for item in data]
                else:
                    print("Error: JSON must be a list")
                    return

        elif input_format == "parquet":
            import pandas as pd
            df = pd.read_parquet(url_list)
            if url_col not in df.columns:
                print(f"Error: Column '{url_col}' not found in Parquet")
                return
            urls = df[url_col].tolist()

        else:
            print(f"Error: Unsupported input format '{input_format}'")
            return

        # Publish to ingest.items
        for i, url in enumerate(urls):
            if not url:
                continue

            event = {
                "source_url": url,
                "meta": {},
                "ts_enq": int(time.time())
            }

            bus.publish(
                topic="ingest.items",
                key=url,
                value=event
            )

            if (i + 1) % 1000 == 0:
                print(f"Enqueued {i + 1} URLs...")

        print(f"Enqueued {len(urls)} URLs total")

    finally:
        bus.close()


def materialize(
    output_folder: str = "output",
    manifest_path: str = "manifest.json",
    output_format: str = "manifest"
):
    """
    Materialize a dataset from segments.

    Args:
        output_folder: Output directory (for index)
        manifest_path: Path to output manifest
        output_format: Output format (manifest, webdataset)
    """
    output_path = Path(output_folder)
    index_path = output_path / "index.sqlite3"

    if not index_path.exists():
        print(f"Error: Index not found at {index_path}")
        return

    index = IndexStore(db_path=str(index_path))

    try:
        if output_format == "manifest":
            # Export manifest as JSON
            import json

            manifest = []
            for batch in index.iter_all(batch_size=1000):
                for entry in batch:
                    manifest.append({
                        "item_id": entry.item_id,
                        "segment_id": entry.segment_id,
                        "offset": entry.offset,
                        "length": entry.length,
                        "mime": entry.mime
                    })

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            print(f"Materialized {len(manifest)} items to {manifest_path}")

        elif output_format == "parquet":
            # Export as Parquet
            index.export_to_parquet(manifest_path)
            print(f"Exported index to {manifest_path}")

        else:
            print(f"Error: Unsupported output format '{output_format}'")

    finally:
        index.close()
