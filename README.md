# img2dataset
[![pypi](https://img.shields.io/pypi/v/img2dataset.svg)](https://pypi.python.org/pypi/img2dataset)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/img2dataset/blob/master/notebook/example.ipynb)
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

with each number being the position in the list. The subfolders avoids having too many files in a single folder.

This can then easily be fed into machine learning training or any other use case.

## Road map

This tool work as it. However in the future goals will include:

* WebDataset format option
* support for multiple input files
* support for csv or parquet files as input
* more resizing options (currently resizing with borders, other options are often useful)

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/img2dataset) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

