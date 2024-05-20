"""distributor defines the distribution strategies for img2dataset"""

from contextlib import contextmanager
from multiprocessing import get_context
from itertools import islice, chain

from tqdm import tqdm


def retrier(runf, failed_shards, max_shard_retry):
    # retry failed shards max_shard_retry times
    for i in range(max_shard_retry):
        if len(failed_shards) == 0:
            break
        print(f"Retrying {len(failed_shards)} shards, try {i+1}")
        failed_shards = runf(failed_shards)
    if len(failed_shards) != 0:
        print(
            f"Retried {max_shard_retry} times, but {len(failed_shards)} shards "
            "still failed. You may restart the same command to retry again."
        )


def multiprocessing_distributor(processes_count, downloader, reader, _, max_shard_retry):
    """Distribute the work to the processes using multiprocessing"""
    ctx = get_context("spawn")
    with ctx.Pool(processes_count, maxtasksperchild=5) as process_pool:

        def run(gen):
            failed_shards = []
            for status, row in tqdm(process_pool.imap_unordered(downloader, gen)):
                if status is False:
                    failed_shards.append(row)
            return failed_shards

        failed_shards = run(reader)

        retrier(run, failed_shards, max_shard_retry)

        process_pool.terminate()
        process_pool.join()
        del process_pool


def pyspark_distributor(processes_count, downloader, reader, subjob_size, max_shard_retry):
    """Distribute the work to the processes using pyspark"""

    with _spark_session(processes_count) as spark:

        def batcher(iterable, batch_size):
            iterator = iter(iterable)
            for first in iterator:
                yield list(chain([first], islice(iterator, batch_size - 1)))

        def run(gen):
            failed_shards = []
            for batch in batcher(gen, subjob_size):
                rdd = spark.sparkContext.parallelize(batch, len(batch))
                for status, row in rdd.map(downloader).collect():
                    if status is False:
                        failed_shards.append(row)
            return failed_shards

        failed_shards = run(reader)

        retrier(run, failed_shards, max_shard_retry)


try:
    import ray  # pylint: disable=import-outside-toplevel

    @ray.remote
    def ray_download(downloader, shards):
        status, row = downloader(shards)
        return status, row

    def ray_distributor(processes_count, downloader, reader, _, max_shard_retry):  # type: ignore
        # pylint: disable=unused-argument
        ret = []
        count = 0
        for task in reader:
            count += 1
            ret.append(ray_download.remote(downloader, task))
        ray.get(ret)

except ModuleNotFoundError as e:

    def ray_distributor(processes_count, downloader, reader, subjob_size, max_shard_retry):  # type: ignore  # pylint: disable=unused-argument
        return None


@contextmanager
def _spark_session(processes_count: int):
    """Create and close a spark session if none exist"""

    from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel
    import pyspark  # pylint: disable=import-outside-toplevel

    spark_major_version = int(pyspark.version.__version__[0])
    if spark_major_version >= 3:
        spark = SparkSession.getActiveSession()
    else:
        spark = pyspark.sql.SparkSession._instantiatedSession  # type: ignore  # pylint: disable=protected-access

    if spark is None:
        print("No pyspark session found, creating a new one!")
        owned = True
        spark = (
            SparkSession.builder.config("spark.driver.memory", "16G")
            .master("local[" + str(processes_count) + "]")
            .appName("spark-stats")
            .getOrCreate()
        )
    else:
        owned = False

    try:
        yield spark
    finally:
        if owned:
            spark.stop()
