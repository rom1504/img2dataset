# img2dataset
[![pypi](https://img.shields.io/pypi/v/img2dataset.svg)](https://pypi.python.org/pypi/img2dataset)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/img2dataset/blob/master/notebook/img2dataset_getting_started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/img2dataset)

Easily turn a set of image urls to an image dataset

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

This can then easily be fed into machine learning training or any other use case.

## API

This module exposes a single function `download` which takes the same arguments as the command line tool:

* **url_list** A file with the list of url of images to download, one by line (required)
* **image_size** The side to resize image to (default 256)
* **output_folder** The path to the output folder (default "images")
* **thread_count** The number of threads used for downloading the pictures. This is important to be high for performance. (default 256)
* **resize_mode** The way to resize pictures, can be no, border or keep_ratio (default border)
  * **no** doesn't resize at all
  * **border** will make the image image_size x image_size and add a border
  * **keep_ratio** will keep the ratio and make the smallest side of the picture image_size
* **resize_only_if_bigger** resize pictures only if bigger that the image_size (default False)
* **output_format** decides how to save pictures
  * **files** saves as a set of subfolder containing pictures
  * **webdataset** saves as tars containing pictures

## Road map

This tool work as it. However in the future goals will include:

* WebDataset format option
* support for multiple input files
* support for csv or parquet files as input
* benchmarks for 1M, 10M, 100M pictures

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

```
cd tests
bash benchmark.js
```

500 images/s is the currently observed performance. 1.8M images/s
