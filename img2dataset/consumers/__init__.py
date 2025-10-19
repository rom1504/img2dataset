"""
Consumer modules for producer/consumer architecture.
"""

from .shard_materializer import ShardMaterializer, materialize_shards
from .trainer_example import SegmentDataLoader, train_example

__all__ = ["ShardMaterializer", "materialize_shards", "SegmentDataLoader", "train_example"]
