## Laion5B

Laion5B has 5.86B samples
See https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/ and https://rom1504.medium.com/semantic-search-at-billions-scale-95f21695689a for details. 

### Download the metadata

Download from https://huggingface.co/datasets/laion/laion2B-en https://huggingface.co/datasets/laion/laion2B-multi https://huggingface.co/datasets/laion/laion1B-nolang

```
mkdir laion2B-en && cd laion2B-en
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$i-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet; done
cd ..
```

```
mkdir laion2B-multi && cd laion2B-multi
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-multi/resolve/main/part-$i-fc82da14-99c9-4ff6-ab6a-ac853ac82819-c000.snappy.parquet; done
cd ..
```

```
mkdir laion1B-nolang && cd laion1B-nolang
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion1B-nolang/resolve/main/part-$i-d6a94da9-d368-4d5b-9ab7-3f6d3c7abdb3-c000.snappy.parquet; done
```


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
