# Distributed img2dataset tutorial

Img2dataset can be used on a single machine to download and resize at around 100 sample/s/core.
For large node, that has been measure to go up to 4000 samples/s (with 40 cores).

However, what if you have billion of samples and you don't want to wait weeks ?

To support that use case, img2dataset proposes to use multiple machines by setting up a pyspark cluster.
This document will help you setup such a cluster and run img2dataset on it.

## Setting up a pyspark cluster

### You already got a cluster

That option is of course the best. If you have an existing on-premise cluster, or you're using a cloud cluster like amazon emr, then you're all set, go directly to the use img2dataset section.
You may want to put https://github.com/rom1504/img2dataset/releases/download/1.22.3/img2dataset.pex in a place that is available to all your nodes.

### You don't have a cluster, but you have access to N machines over ssh

That's a common case, you have access to N machines, and you have a place to store the data.
This is actually fairly easy to use this to setup a pyspark cluster. Let's see how to do it.

Tools:
* spark and pyspark
* parallel ssh
* ssh tunnels
* pex

An additional assumption we will be making is all the nodes are not in the same network, hence we will need to make tunnels.
If the master and worker nodes are all in the same network, no tunnel will be needed.


#### Setup the master node
On the master node:

First download spark:
```bash
wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar xf spark-3.2.0-bin-hadoop3.2.tgz
```

Then download img2dataset:
```bash
wget https://github.com/rom1504/img2dataset/releases/download/1.22.3/img2dataset.pex -O img2dataset_new.pex
```

Pick an output folder and link it to a fixed place (should be the same location as in worker nodes):
```bash
OUTPUT_FOLDER=some/output/folder
ln -s $OUTPUT_FOLDER /tmp/bench
```

You can do a tunnel between your local machine and the master node to be able to see the spark ui (at http://localhost:8080)
```bash
ssh -L 8080:localhost:8080 -L 4040:localhost:4040 master_node
```


#### Setup the worker nodes

##### ssh basic setup

We will start many tunnels from the worker nodes, so first increase your sshd parameter in the master node:
```bash
sudo vim /etc/ssh/sshd_config 
MaxSessions 200
MaxStartups 200:30:200
```

Still in the master node, create a ips.txt with the ips of all the nodes

```bash
ssh-keyscan `cat ips.txt` >> ~/.ssh/known_hosts
```

Install pssh with `sudo apt install pssh`

Pick the right username (MASTER_USER) for the master node, and (USER) for the worker nodes, then run this to check your parallel ssh setup:
```bash
MASTER_USER=laion
USER=rom1504
parallel-ssh -l $USER -i -h  ips.txt uname -a
```

##### Install some packages

```bash
parallel-ssh -l $USER "sudo apt update"
parallel-ssh -l $USER "sudo apt install openjdk-11-jre-headless libgl1 htop tmux bwm-ng sshfs"
```

##### Optional swap disk

Optionally you may want to create a swap disk if you don't have a lot of ram in the worker nodes:
```bash
parallel-ssh -l $USER -i -h   ips.txt  "dd if=/dev/zero of=/home/$USER/swapfile.img bs=1024 count=5M"
parallel-ssh -l $USER -i -h   ips.txt  "mkswap /home/$USER/swapfile.img"
parallel-ssh -l $USER -i -h  ips.txt  "sudo swapon /home/$USER/swapfile.img"
```


##### Download img2dataset on all nodes

Download img2dataset on all node by retrying this N times until parallel ssh says success for all:
```bash
parallel-ssh -i -h ips.txt  "wget -c https://github.com/rom1504/img2dataset/releases/download/1.22.3/img2dataset.pex -O img2dataset_new.pex"
```
Then:
```bash
parallel-ssh -i -h ips.txt  "mv img2dataset_new.pex img2dataset.pex"
parallel-ssh -i -h ips.txt  "chmod +x img2dataset.pex"
```

##### Creating the tunnels and mounting the shared output folder

For this step, we will need to let the workers connect to the master node in order to establish sshfs, so do it by allowing them to ssh to your master node:
first create a key in a worker with ssh-keygen, and put it on all workers (with scp and parallel-scp)
then add it to the authorized keys on the master node::
```bash
cat ~/the_id_rsa_pub >> ~/.ssh/authorized_keys
```

Then create the location of the mounted folder on the workers:
```bash
parallel-ssh -l $USER -i -h  ips.txt  "mkdir -p /tmp/bench"
```

Start the tunnels (7077 is the main spark master port, 5678 is the driver port, 6678 is the block manager port):
```bash
OUTPUT_PATH=/some/output/path
for IP in `cat ips.txt`
do
    ssh -R 7077:localhost:7077 -R 5678:localhost:5678 -R 6678:localhost:6678 -R 2300:localhost:22  $USER@$IP "sshfs  -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no  -p 2300 $MASTER_USER@localhost:$OUTPUT_PATH /tmp/bench && sleep infinity" &
done
```

#### Start the master node

When you're ready, you can start the master node with:

```bash
./spark-3.2.0-bin-hadoop3.2/sbin/start-master.sh -h 127.0.0.1 -p 7077
```


#### Start the worker nodes

When you're ready, you can start the worker nodes with:

```bash
parallel-ssh -l rom1504 -i -h ips.txt  "./spark-3.2.0-bin-hadoop3.2/sbin/start-worker.sh -c 2 -m 1G -h 127.0.0.1 spark://127.0.0.1:7077"
```


#### Stop the worker nodes

When you're done, you can stop the worker nodes with:

```bash
parallel-ssh -l rom1504 -i -h ips.txt "rm -rf ~/spark-3.2.0-bin-hadoop3.2/work/*"
pkill -f "ssh -R"
parallel-ssh -l rom1504 -i -h  ips.txt  "pkill java"
```


#### Stop the master node

When you're done, you can stop the master node with:

```bash
pkill java
```


### Running img2dataset on it

Once your spark cluster is setup, you're ready to start img2dataset in distributed mode.
Make sure to open your spark UI, at http://localhost:8080 (or the ip where the master node is running)

Save this script to download.py.

Then run ./img2dataset.pex download.py

```python
from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext

def create_spark_session():
    # this must be a path that is available on all worker nodes
    pex_file = "/home/rom1504/img2dataset.pex"
    
    os.environ['PYSPARK_PYTHON'] = pex_file
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client") \
        #.config("spark.files", pex_file) \ # you may choose to uncomment this option if you want spark to automatically download the pex file, but it may be slow
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "200") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executor.memory", "800M") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "300M")
        .config("spark.task.maxFailures", "100")
        .master("spark://127.0.0.1:7077") # this should point to your master node, if using the tunnelling version, keep this to localhost
        .appName("spark-stats")
        .getOrCreate()
    )
    return spark

output_dir = "/tmp/bench"


spark = create_spark_session()

url_list = "some_file.parquet"

download(
    processes_count=1,
    thread_count=32,
    retries=0,
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
    enable_wandb=False,
    number_sample_per_shard=10000,
    distributor="pyspark",
    save_additional_columns=["NSFW","similarity","LICENSE"]
)
```