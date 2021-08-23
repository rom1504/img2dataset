# img2dataset
[![pypi](https://img.shields.io/pypi/v/img2dataset.svg)](https://pypi.python.org/pypi/img2dataset)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/img2dataset/blob/master/notebook/img2dataset_getting_started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/img2dataset)

Easily turn large sets of image urls to an image dataset.
Can download, resize and package 100M urls in 20h on one machine.

Also supports saving captions for url+caption datasets.

## Install

pip install img2dataset

## Usage

First get some image url list. For example:
```
echo 'https://placekitten.com/200/305' >> myimglist.txt
echo 'https://placekitten.com/200/304' >> myimglist.txt
echo 'https://placekitten.com/200/303' >> myimglist.txt
```

Then, run the tool:

```
img2dataset --url_list=myimglist.txt --output_folder=output_folder --thread_count=64 --image_size=256
```

The tool will then automatically download the urls, resize them, and store them with that format:
* output_folder
    * 0
        * 0.jpg
        * 1.jpg
        * 2.jpg

or as this format if choosing webdataset:
* output_folder
    * 0.tar containing:
        * 0.jpg
        * 1.jpg
        * 2.jpg

with each number being the position in the list. The subfolders avoids having too many files in a single folder.

If **captions** are provided, they will be saved as 0.txt, 1.txt, ...

This can then easily be fed into machine learning training or any other use case.

If **save_metadata** option is turned on (that's the default), then .json files named 0.json, 1.json,... are saved with these keys:
* url
* caption
* key
* shard_id
* status : whether the download succeeded
* error_message
* width
* height
* original_width
* original_height
* exif
Also a .parquet file will be saved with the same name as the subfolder/tar files containing these same metadata.
It can be used to analyze the results efficiently.

## API

This module exposes a single function `download` which takes the same arguments as the command line tool:

* **url_list** A file with the list of url of images to download. It can be a folder of such files. (*required*)
* **image_size** The side to resize image to (default *256*)
* **output_folder** The path to the output folder. If existing subfolder are present, the tool will continue to the next number. (default *"images"*)
* **processes_count** The number of processes used for downloading the pictures. This is important to be high for performance. (default *1*)
* **thread_count** The number of threads used for downloading the pictures. This is important to be high for performance. (default *256*)
* **resize_mode** The way to resize pictures, can be no, border or keep_ratio (default *border*)
  * **no** doesn't resize at all
  * **border** will make the image image_size x image_size and add a border
  * **keep_ratio** will keep the ratio and make the smallest side of the picture image_size
* **resize_only_if_bigger** resize pictures only if bigger that the image_size (default *False*)
* **output_format** decides how to save pictures (default *files*)
  * **files** saves as a set of subfolder containing pictures
  * **webdataset** saves as tars containing pictures
* **input_format** decides how to load the urls (default *txt*)
  * **txt** loads the urls as a text file of url, one per line
  * **csv** loads the urls and optional caption as a csv
  * **parquet** loads the urls and optional caption as a parquet
* **url_col** the name of the url column for parquet and csv (default *url*)
* **caption_col** the name of the caption column for parquet and csv (default *None*)
* **number_sample_per_shard** the number of sample that will be downloaded in one shard (default *10000*)
* **save_metadata** if true, saves one parquet file per folder/tar and json files with metadata (default *True*)
* **save_additional_columns** list of additional columns to take from the csv/parquet files and save in metadata files (default *None*)

## How to tweak the options

The default values should be good enough for small sized dataset. For larger ones, these tips may help you get the best performance:

* set the processes_count as the number of cores your machine has
* increase thread_count as long as your bandwidth and cpu are below the limits
* I advise to set output_format to webdataset if your dataset has more than 1M elements, it will be easier to manipulate few tars rather than million of files
* keeping metdata to True can be useful to check what items were already saved and avoid redownloading them

## Road map

This tool works very well in the current state for up to 100M elements. Future goals include:

* a benchmark for 1B pictures which may require
  * further optimization on the resizing part
  * better multi node support
  * integrated support for incremental support (only download new elements)

## Architecture notes

This tool is designed to download pictures as fast as possible.
This put a stress on various kind of resources. Some numbers assuming 1350 image/s:
* Bandwidth: downloading a thousand average image per second requires about 130MB/s
* CPU: resizing one image may take several milliseconds, several thousand per second can use up to 16 cores
* DNS querying: million of urls mean million of domains, default OS setting usually are not enough. Setting up a local bind9 resolver may be required
* Disk: if using resizing, up to 30MB/s write speed is necessary. If not using resizing, up to 130MB/s. Writing in few tar files make it possible to use rotational drives instead of a SSD.

With these information in mind, the design choice was done in this way:
* the list of urls is split in N shards. N is usually chosen as the number of cores
* N processes are started (using multiprocessing process pool)
  * each process starts M threads. M should be maximized in order to use as much network as possible while keeping cpu usage below 100%.
  * each of this thread download 1 image and returns it
  * the parent thread handle resizing (which means there is at most N resizing running at once, using up the cores but not more)
  * the parent thread saves to a tar file that is different from other process

This design make it possible to use the CPU resource efficiently by doing only 1 resize per core, reduce disk overhead by opening 1 file per core, while using the bandwidth resource as much as possible by using M thread per process.

## Setting up a bind9 resolver

In order to keep the success rate high, it is necessary to use an efficient DNS resolver.
I tried several options: systemd-resolved, dnsmaskq and bind9 and reached the conclusion that bind9 reaches the best performance for this use case.
Here is how to set this up on ubuntu:
```
sudo apt install bind9
sudo vim /etc/bind/named.conf.options

Add this in options:
        recursive-clients 10000;
        resolver-query-timeout 30000;
        max-clients-per-query 10000;

sudo systemctl restart bind9

sudo vim /etc/resolv.conf

Put this content:
nameserver 127.0.0.1
```
This will make it possible to keep an high success rate while doing thousands of dns queries.
You may also want to [setup bind9 logging](https://nsrc.org/activities/agendas/en/dnssec-3-days/dns/materials/labs/en/dns-bind-logging.html) in order to check that few dns errors happen.

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/img2dataset) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
python -m pytest -v tests -s
```

## Benchmarks

### 10000 image benchmark

```
cd tests
bash benchmark.sh
```


### 18M image benchmark

Download crawling at home first part, then:

```
cd tests
bash large_bench.sh
```
It takes 3.7h to download 18M pictures

1350 images/s is the currently observed performance. 4.8M images per hour, 116M images per 24h.


### 36M image benchmark

downloading 2 parquet files of 18M items (result 936GB) took 7h24
average of 1345 image/s
