from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

output_dir = os.path.abspath("bench")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

spark = (
    SparkSession.builder.config("spark.driver.memory", "16G").master("local[16]").appName("spark-stats").getOrCreate()
)

download(
    processes_count=16,
    thread_count=32,
    url_list="../tests/test_files/test_10000.parquet",
    image_size=256,
    output_folder=output_dir,
    output_format="webdataset",
    input_format="parquet",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=True,
    number_sample_per_shard=1000,
    distributor="pyspark",
)

# rm -rf bench
