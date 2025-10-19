"""
Shard Materializer Consumer

Builds WebDataset or other shard formats from segments using the index.
Can work in two modes:
1. Manifest-only: Creates pointers to (segment_id, offset, length)
2. Physical shards: Copies data to new TAR files (optional)
"""

import os
import tarfile
import io
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..core.index_store import IndexStore, IndexEntry
from ..core.io import SegmentReader


class ShardMaterializer:
    """
    Materializes shards from segments.

    Supports manifest-only mode (preferred) or physical TAR creation.
    """

    def __init__(
        self,
        index: IndexStore,
        segment_reader: SegmentReader,
        output_dir: str,
        shard_size: int = 10000,
        mode: str = "manifest"
    ):
        """
        Initialize shard materializer.

        Args:
            index: Index store
            segment_reader: Segment reader
            output_dir: Output directory for shards/manifests
            shard_size: Items per shard
            mode: "manifest" or "physical"
        """
        self.index = index
        self.segment_reader = segment_reader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.mode = mode

    def materialize_manifest(self, dataset_name: str = "dataset") -> str:
        """
        Create manifest-only shards (preferred mode).

        Manifests are JSONL files containing pointers to segments.
        This avoids data duplication.

        Args:
            dataset_name: Name prefix for manifest files

        Returns:
            Path to manifest directory
        """
        manifest_dir = self.output_dir / f"{dataset_name}_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        shard_id = 0
        items_in_shard = 0
        current_manifest = []

        print(f"Materializing manifests to {manifest_dir}")

        for batch in self.index.iter_all(batch_size=1000):
            for entry in batch:
                current_manifest.append({
                    "item_id": entry.item_id,
                    "segment_id": entry.segment_id,
                    "offset": entry.offset,
                    "length": entry.length,
                    "mime": entry.mime
                })
                items_in_shard += 1

                # Write shard when full
                if items_in_shard >= self.shard_size:
                    self._write_manifest_shard(
                        manifest_dir,
                        dataset_name,
                        shard_id,
                        current_manifest
                    )
                    shard_id += 1
                    items_in_shard = 0
                    current_manifest = []

        # Write remaining items
        if current_manifest:
            self._write_manifest_shard(
                manifest_dir,
                dataset_name,
                shard_id,
                current_manifest
            )

        print(f"Created {shard_id + 1} manifest shards")
        return str(manifest_dir)

    def _write_manifest_shard(
        self,
        manifest_dir: Path,
        dataset_name: str,
        shard_id: int,
        items: List[Dict[str, Any]]
    ):
        """Write a single manifest shard."""
        manifest_path = manifest_dir / f"{dataset_name}-{shard_id:06d}.jsonl"

        with open(manifest_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')

    def materialize_physical(self, dataset_name: str = "dataset") -> str:
        """
        Create physical WebDataset TAR shards.

        This duplicates data from segments into new TAR files.
        Only use if manifest mode is not suitable.

        Args:
            dataset_name: Name prefix for shard files

        Returns:
            Path to shard directory
        """
        shard_dir = self.output_dir / f"{dataset_name}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_id = 0
        items_in_shard = 0
        current_tar = None
        current_tar_path = None

        print(f"Materializing physical shards to {shard_dir}")

        try:
            for batch in self.index.iter_all(batch_size=1000):
                for entry in batch:
                    # Open new shard if needed
                    if current_tar is None:
                        current_tar_path = shard_dir / f"{dataset_name}-{shard_id:06d}.tar"
                        current_tar = tarfile.open(current_tar_path, 'w')

                    # Read item from segment
                    data = self.segment_reader.read(
                        entry.segment_id,
                        entry.offset,
                        entry.length
                    )

                    # Write to TAR
                    tarinfo = tarfile.TarInfo(name=entry.item_id)
                    tarinfo.size = len(data)
                    current_tar.addfile(tarinfo, io.BytesIO(data))

                    items_in_shard += 1

                    # Close shard when full
                    if items_in_shard >= self.shard_size:
                        current_tar.close()
                        current_tar = None
                        shard_id += 1
                        items_in_shard = 0

                        if (shard_id) % 10 == 0:
                            print(f"Created {shard_id} shards...")

        finally:
            # Close any open TAR
            if current_tar is not None:
                current_tar.close()

        print(f"Created {shard_id + 1} physical shards")
        return str(shard_dir)

    def run(self, dataset_name: str = "dataset") -> str:
        """
        Run materialization based on configured mode.

        Args:
            dataset_name: Dataset name prefix

        Returns:
            Path to output directory
        """
        if self.mode == "manifest":
            return self.materialize_manifest(dataset_name)
        elif self.mode == "physical":
            return self.materialize_physical(dataset_name)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def materialize_shards(
    index_path: str,
    segments_dir: str,
    output_dir: str,
    dataset_name: str = "dataset",
    shard_size: int = 10000,
    mode: str = "manifest"
) -> str:
    """
    Convenience function to materialize shards.

    Args:
        index_path: Path to index database
        segments_dir: Directory containing segments
        output_dir: Output directory
        dataset_name: Dataset name prefix
        shard_size: Items per shard
        mode: "manifest" or "physical"

    Returns:
        Path to output directory
    """
    index = IndexStore(db_path=index_path)
    reader = SegmentReader(segments_dir=segments_dir)

    try:
        materializer = ShardMaterializer(
            index=index,
            segment_reader=reader,
            output_dir=output_dir,
            shard_size=shard_size,
            mode=mode
        )
        return materializer.run(dataset_name=dataset_name)
    finally:
        index.close()
