# Distributed img2dataset tutorial

Img2dataset can be used on a single machine to download and resize at around 100 sample/s/core.
For large node, that has been measure to go up to 4000 samples/s (with 40 cores).

However, what if you have billion of samples and you don't want to wait weeks ?

To support that use case, img2dataset proposes to use multiple machines by setting up a pyspark cluster.
This document will help you setup such a cluster and run img2dataset on it.

## Where to get a cluster, what machines to use?

These providers have been tested to work well with img2dataset:
* aliyun small 2 cores nodes ($4.5/month for 40 sample/s)
* aws c6i.4xlarge nodes ($0.68/h for 1000 sample/s)

Ubuntu 20.04 works well with img2dataset. Centos7 also works.
Other providers probably work too but haven't been tested.

## Setting up a pyspark cluster

### You already got a cluster

That option is of course the best. If you have an existing on-premise cluster, or you're using a cloud cluster like amazon emr, then you're all set, go directly to the use img2dataset section.
You may want to put https://github.com/rom1504/img2dataset/releases/latest/download/img2dataset.pex in a place that is available to all your nodes.

### You don't have a cluster, but you have access to N machines over ssh

That's a common case, you have access to N machines, and you have a place to store the data.
This is actually fairly easy to use this to setup a pyspark cluster. Let's see how to do it.

Tools:
* spark and pyspark
* parallel ssh
* pex

We will be assuming ubuntu 20.04.


#### Setup the master node

On the master node:

First download spark:
```bash
wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar xf spark-3.2.0-bin-hadoop3.2.tgz
```

Then download img2dataset:
```bash
wget https://github.com/rom1504/img2dataset/releases/latest/download/img2dataset.pex -O img2dataset.pex
```

Pick an output folder and link it to a fixed place (should be the same location as in worker nodes):
```bash
OUTPUT_FOLDER=some/output/folder
ln -s $OUTPUT_FOLDER /tmp/bench
```

If the master node cannot open ports that are visible from your local machine, you can do a tunnel between your local machine and the master node to be able to see the spark ui (at http://localhost:8080)
```bash
ssh -L 8080:localhost:8080 -L 4040:localhost:4040 master_node
```


#### Setup the worker nodes

##### ssh basic setup

Still in the master node, create a ips.txt with the ips of all the nodes

```bash
ssh-keyscan `cat ips.txt` >> ~/.ssh/known_hosts
```

You may use a script like this to fill your .ssh/config file
```
def generate(ip):
    print(
        f"Host {ip}\n"
        f"        HostName {ip}\n"
        "        User ubuntu\n"
        "        IdentityFile ~/yourkey.pem"
        )

with open("ips.txt") as f:
    lines = f.readlines()
    for line in lines:
        generate(line.strip())
```
python3 generate.py >> ~/.ssh/config

Install pssh with `sudo apt install pssh`

Pick the right username (MASTER_USER) for the master node, and (USER) for the worker nodes, then run this to check your parallel ssh setup:
```bash
MASTER_USER=rom1504
USER=rom1504
```

Optionally, if another node than the current one has access to the worker nodes, you may need to add a ssh key to all the nodes with:
```
for IP in `cat ips.txt`
do
        ssh-copy-id -i the_new_id_rsa $USER@$IP
done
```

Check you can connect to all the nodes with:
```
parallel-ssh -l $USER -i -h  ips.txt uname -a
```

##### Install some packages

```bash
parallel-ssh -l $USER -i -h  ips.txt "sudo apt update"
parallel-ssh -l $USER -i -h  ips.txt "sudo apt install openjdk-11-jre-headless libgl1 htop tmux bwm-ng sshfs -y"
```


#### Network setting

on master:
```bash
sudo sh -c 'echo `hostname -I` `hostname` >> /etc/hosts'
```

on workers
```bash
parallel-ssh -l $USER -i -h  ips.txt  "sudo sh -c 'echo \`hostname -I\` \`hostname\` >> /etc/hosts'"
```


### Install knot resolver

```bash
parallel-ssh -l $USER -i -h  ips.txt "sudo apt update && sudo apt install libgl1 htop tmux bwm-ng python3.8-venv awscli -y"
parallel-ssh -l $USER -i -h  ips.txt "wget https://secure.nic.cz/files/knot-resolver/knot-resolver-release.deb && sudo dpkg -i knot-resolver-release.deb && sudo apt update && sudo apt install -y knot-resolver"
```

```bash
parallel-ssh -l $USER -i -h  ips.txt "sudo systemctl stop systemd-resolved"
parallel-ssh -l $USER -i -h  ips.txt "sudo systemctl start kresd@{1..4}.service"
parallel-ssh -l $USER -i -h  ips.txt 'sudo sh -c "echo nameserver 127.0.0.1 > /etc/resolv.conf"'
parallel-ssh -l $USER -i -h  ips.txt 'dig @localhost google.com'
```


##### Download img2dataset on all nodes

Download img2dataset on all node by retrying this N times until parallel ssh says success for all:
```bash
parallel-ssh -i -h ips.txt  "wget -c https://github.com/rom1504/img2dataset/releases/latest/download/img2dataset.pex -O img2dataset_new.pex"
```
Then:
```bash
parallel-ssh -l $USER -i -h  ips.txt  "mv img2dataset_new.pex img2dataset.pex"
parallel-ssh -l $USER -i -h  ips.txt  "chmod +x img2dataset.pex"
```

##### Download spark on workers

```bash
parallel-ssh -l $USER -i -h  ips.txt  "wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz"
parallel-ssh -l $USER -i -h  ips.txt  "tar xf spark-3.2.0-bin-hadoop3.2.tgz"
```

#### Start the master node

When you're ready, you can start the master node with:

```bash
./spark-3.2.0-bin-hadoop3.2/sbin/start-master.sh -h master_node -p 7077
```

Replace master_node by the master node ip.


#### Start the worker nodes

When you're ready, you can start the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt  "./spark-3.2.0-bin-hadoop3.2/sbin/start-worker.sh -c 16 -m 16G spark://master_node:7077"
```

Replace master_node by the master node ip.
Replace -c 16 -m 16g but the number of cores and ram you want to use on each worker.


#### Stop the worker nodes

When you're done, you can stop the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt "rm -rf ~/spark-3.2.0-bin-hadoop3.2/work/*"
pkill -f "ssh -R"
parallel-ssh -l $USER -i -h  ips.txt  "pkill java"
```


#### Stop the master node

When you're done, you can stop the master node with:

```bash
pkill java
```


### Running img2dataset on it

Once your spark cluster is setup, you're ready to start img2dataset in distributed mode.
Make sure to open your spark UI, at http://master_node:8080

Save this script to download.py.

Then run ./img2dataset.pex download.py

Replace master_node by the master node ip.

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
        #.config("spark.executor.cores", "2")
        #.config("spark.cores.max", "200") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "master_node")
        .config("spark.driver.bindAddress", "master_node")
        .config("spark.executor.memory", "16GB") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8GB")
        .config("spark.task.maxFailures", "100")
        .master("spark://master_node:7077") # this should point to your master node, if using the tunnelling version, keep this to localhost
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
