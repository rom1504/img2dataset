"""distributor defines the distribution strategies for img2dataset"""

from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice, chain


def multiprocessing_distributor(
    processes_count, downloader, reader, _,
):
    """Distribute the work to the processes using multiprocessing"""
    with Pool(processes_count, maxtasksperchild=5) as process_pool:
        for _ in tqdm(process_pool.imap_unordered(downloader, reader),):
            pass

        process_pool.terminate()
        process_pool.join()
        del process_pool


def pyspark_distributor(
    processes_count, downloader, reader, subjob_size,
):
    """Distribute the work to the processes using pyspark"""

    from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

    spark = SparkSession.getActiveSession()

    if spark is None:
        print("No pyspark session found, creating a new one!")
        spark = (
            SparkSession.builder.config("spark.driver.memory", "16G")
            .master("local[" + str(processes_count) + "]")
            .appName("spark-stats")
            .getOrCreate()
        )

    def batcher(iterable, batch_size):
        iterator = iter(iterable)
        for first in iterator:
            yield list(chain([first], islice(iterator, batch_size - 1)))

    for batch in batcher(reader, subjob_size):
        rdd = spark.sparkContext.parallelize(batch, len(batch))
        rdd.foreach(downloader)
