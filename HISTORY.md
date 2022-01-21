## 1.21.1

* decrease default log interval to 5s

## 1.21.0

* add option to retry http download

## 1.20.2

* add original_width by default for a consistent schema

## 1.20.1

* fix relative path handling

## 1.20.0

* Add multi distributor support : multiprocessing and pyspark

## 1.19.0

* make the reader emits file paths instead of samples

## 1.18.0

* use a logger process to make logging distribution friendly, also save json stat files next to folder/tar files

## 1.17.0

* Use fsspec to support all filesystems

## 1.16.0

* implement md5 of images feature

## 1.15.1

* fix null convert in writer

## 1.15.0

* add parquet writer

## 1.14.0

* make reader memory efficient by using feather files

## 1.13.0

* large refactoring of the whole code in submodules
* Enhance image resize processing (esp re downscale) (@rwightman)

## 1.12.0

* handle transparency (thanks @borisdayma)
* add json input file support


## 1.11.0

* Add support for .tsv.gz files (thanks @robvanvolt)

## 1.10.1

* raise clean exception on image decoding error
* remove the \n in urls for txt inputs
* save the error message when resizing fails in metadata
* add type hints to download function

## 1.10.0

* use semaphores to decrease memory usage

## 1.9.9

* fix an issue with resize_mode "no"

## 1.9.8

* optimize listing files is back, sorted is eager so the iterator returned by iglob is ok

## 1.9.7

* revert last commit, it could cause double iteration on an iterator which can cause surprising behaviors

## 1.9.6

* optimize listing files (thanks @Skylion)

## 1.9.5

* fix a bug affecting downloading multiple files

## 1.9.4

* ensure sharded_images_to_dl is removed from memory at the end of downloading a file

## 1.9.3

* solve the stemming issue: make keys uniques

## 1.9.2

* Save empty caption if caption are none instead of not having the caption file

## 1.9.1

* fix for the new logging feature when cleaning the status dict

## 1.9.0

* wandb support is back

## 1.8.5

* support for python 3.6

## 1.8.4

* convert caption to str before writing

## 1.8.3

* add back timeout properly

## 1.8.2

* fixes

## 1.8.1

* revert wandb for now, code is too complex and there are issues

## 1.8.0

* feat: custom timeout (thanks @borisdayma)
* feat: support wandb (thanks @borisdayma)

## 1.7.0

* use albumentations for resizing (thanks @borisdayma)

## 1.6.1

* depend on pyyaml to be able to use the last webdataset

## 1.6.0

* feat: handle tsv + center crop (thanks @borisdayma)

## 1.5.3

* increase stability by closing the pool and tarwriter explictely

## 1.5.2

* improve memory usage

## 1.5.1

* glob only input files of the right ext

## 1.5.0

* add a save_additional_columns option

## 1.4.0

* Multiple file support
* Status dataframe

## 1.3.0

* Uses a resizing method less prone to aliasing (thanks @skylion)
* multi processing + multi threading

## 1.2.0

* add webdataset support and benchmarks
* supports reading as parquet and csv

## 1.1.1

* fix cli

## 1.1.0

* add image resizing mode

## 1.0.1

* fixes

## 1.0.0

* it works