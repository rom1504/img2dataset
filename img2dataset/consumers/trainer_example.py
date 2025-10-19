"""
Example trainer that reads directly from segments.

This demonstrates how to build a training pipeline that reads from
segment files using the index for efficient sequential IO.

Features:
- Sequential reading for optimal disk IO
- Batching by segment for locality
- Minimal memory footprint
- No data duplication
"""

import io
import time
from typing import Iterator, Tuple, Optional, List, Dict, Any
from PIL import Image
import numpy as np

from ..core.index_store import IndexStore, IndexEntry
from ..core.io import SegmentReader


class SegmentDataLoader:
    """
    DataLoader that reads directly from segments.

    Provides sequential access to images with optimal IO performance.
    """

    def __init__(
        self,
        index: IndexStore,
        segment_reader: SegmentReader,
        batch_size: int = 32,
        start_segment: Optional[str] = None,
        start_offset: int = 0,
    ):
        """
        Initialize segment data loader.

        Args:
            index: Index store
            segment_reader: Segment reader
            batch_size: Batch size
            start_segment: Optional segment to resume from
            start_offset: Offset to resume from
        """
        self.index = index
        self.segment_reader = segment_reader
        self.batch_size = batch_size
        self.start_segment = start_segment
        self.start_offset = start_offset

    def iter_items(self) -> Iterator[Tuple[str, bytes, str]]:
        """
        Iterate over items (item_id, bytes, mime).

        Reads sequentially for optimal IO performance.

        Yields:
            Tuples of (item_id, image_bytes, mime)
        """
        # Sequential scan from index
        cursor_segment = self.start_segment
        cursor_offset = self.start_offset

        while True:
            # Fetch next batch from index
            batch = self.index.sample_sequential(limit=1000, start_segment=cursor_segment, start_offset=cursor_offset)

            if not batch:
                break

            # Group by segment for locality
            by_segment: Dict[str, List[Any]] = {}
            for entry in batch:
                if entry.segment_id not in by_segment:
                    by_segment[entry.segment_id] = []
                by_segment[entry.segment_id].append(entry)

            # Read each segment's items
            for segment_id in sorted(by_segment.keys()):
                entries = sorted(by_segment[segment_id], key=lambda e: e.offset)

                for entry in entries:
                    # Read bytes from segment
                    data = self.segment_reader.read(entry.segment_id, entry.offset, entry.length)

                    yield (entry.item_id, data, entry.mime)

            # Update cursor to last item
            last = batch[-1]
            cursor_segment = last.segment_id
            cursor_offset = last.offset + 1

    def iter_images(self) -> Iterator[Tuple[str, Image.Image]]:
        """
        Iterate over images (item_id, PIL.Image).

        Yields:
            Tuples of (item_id, image)
        """
        for item_id, data, mime in self.iter_items():
            try:
                image = Image.open(io.BytesIO(data))
                yield (item_id, image)
            except Exception as e:
                # Skip corrupted images
                continue

    def iter_batches(self) -> Iterator[Tuple[list, list]]:
        """
        Iterate over batches of images.

        Yields:
            Tuples of (item_ids, images)
        """
        batch_ids = []
        batch_images = []

        for item_id, image in self.iter_images():
            batch_ids.append(item_id)
            batch_images.append(image)

            if len(batch_ids) >= self.batch_size:
                yield (batch_ids, batch_images)
                batch_ids = []
                batch_images = []

        # Yield remaining
        if batch_ids:
            yield (batch_ids, batch_images)


def train_example(index_path: str, segments_dir: str, batch_size: int = 32, max_batches: Optional[int] = None):
    """
    Example training loop reading from segments.

    This demonstrates:
    - Sequential reading for optimal IO
    - Batching
    - Simple preprocessing

    Args:
        index_path: Path to index database
        segments_dir: Directory containing segments
        batch_size: Batch size
        max_batches: Maximum batches to process (for demo)
    """
    print("Initializing segment data loader...")

    index = IndexStore(db_path=index_path)
    reader = SegmentReader(segments_dir=segments_dir)

    loader = SegmentDataLoader(index=index, segment_reader=reader, batch_size=batch_size)

    try:
        print(f"Starting training (batch_size={batch_size})")
        start_time = time.time()
        batches_processed = 0
        images_processed = 0

        for batch_ids, batch_images in loader.iter_batches():
            batches_processed += 1
            images_processed += len(batch_images)

            # Example preprocessing: convert to arrays and normalize
            batch_arrays_list: List[np.ndarray] = []
            for image in batch_images:
                # Resize to fixed size
                image = image.resize((224, 224))
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                # To numpy array
                arr = np.array(image, dtype=np.float32) / 255.0
                batch_arrays_list.append(arr)

            batch_arrays = np.stack(batch_arrays_list)

            # Your training code here
            # model.train_step(batch_arrays)

            # Progress logging
            if batches_processed % 10 == 0:
                elapsed = time.time() - start_time
                throughput = images_processed / elapsed if elapsed > 0 else 0
                print(f"Batch {batches_processed}: {images_processed} images " f"({throughput:.1f} img/sec)")

            # Stop if max reached
            if max_batches and batches_processed >= max_batches:
                break

        elapsed = time.time() - start_time
        throughput = images_processed / elapsed if elapsed > 0 else 0
        print(f"\nTraining complete:")
        print(f"  Batches: {batches_processed}")
        print(f"  Images: {images_processed}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.1f} img/sec")

    finally:
        index.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m img2dataset.consumers.trainer_example <index_path> <segments_dir>")
        sys.exit(1)

    index_path = sys.argv[1]
    segments_dir = sys.argv[2]

    train_example(index_path, segments_dir, max_batches=100)
