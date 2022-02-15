## Laion5B

Laion5B has 5.86B samples

### Download the metadata

Download from ...


### Download the images

This one is big so I advise doing it in distributed mode. I followed distributed_img2dataset_tutorial.md.
Note some aws specifics in that guide (in particular regarding VPC and security group configs to allow worker and master to talk together)
Below is some specifics.

#### What infra

In practice I advise to rent 1 master node and 10 worker nodes with the instance type c6i.4xlarge (16 intel cores).
That makes it possible to download laion5B in a week.

Each instance downloads at around 1000 sample/s.
The below config produces a dataset of size 220TB. You can choose to resize to 256 instead to get a 50TB dataset.

#### Script

Example of master config:
```
./spark-3.2.0-bin-hadoop3.2/sbin/start-master.sh -p 7077
```

Example of worker config:
```
parallel-ssh -l $USER -i -h  ips.txt  './spark-3.2.0-bin-hadoop3.2/sbin/start-worker.sh -c 16 -m 24G "spark://172.31.46.59:7077"'
```

bash
```
aws s3 rm --recursive s3://laion-us-east-1/test_output/
./img2dataset.pex download.py
```


```python
from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext

def create_spark_session():
    # this must be a path that is available on all worker nodes
    pex_file = "/home/ubuntu/img2dataset.pex"
    
    os.environ['PYSPARK_PYTHON'] = pex_file
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client") \
        #.config("spark.files", pex_file) \ # you may choose to uncomment this option if you want spark to automatically download the pex file, but it may be slow
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        #.config("spark.executor.cores", "16")
        #.config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "172.31.44.42")
        .config("spark.driver.bindAddress", "172.31.44.42")
        .config("spark.executor.memory", "16G") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .master("spark://172.31.44.42:7077") # this should point to your master node, if using the tunnelling version, keep this to localhost
        .appName("spark-stats")
        .getOrCreate()
    )
    return spark

spark = create_spark_session()

url_list = "s3://laion-us-east-1/laion-metadata/laion2B-en/"
output_dir = "s3://laion-us-east-1/laion-data/laion2B-data"

download(
    processes_count=1,
    thread_count=64,
    url_list = url_list,
    image_size=384,
    resize_only_if_bigger=True,
    resize_mode="keep_ratio",
    skip_reencode=True,
    output_folder=output_dir,
    output_format="webdataset",
    input_format="parquet",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=True,
    number_sample_per_shard=10000,
    distributor="pyspark",
    save_additional_columns=["NSFW","similarity","LICENSE"],
    oom_shard_count=6,
)
```

Will result in :
```
Total Objects: 694047
   Total Size: 84.8 TiB
```

Same config for laion2B-multi and laion1B-nolang
