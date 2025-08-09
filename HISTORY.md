## 1.47.0

* Increase range of pyarrow

## 1.46.0

* Fix albumentations deprecation and NumPy 2.0+ compatibility issues
* Fix glob pattern when gcs url path has a trailing slash (thanks @kafonek)
* Fix usage example in README (thanks @johnbradley2008)

## 1.45.0

* update pyarrow
* add incremental model extend (thanks @edwardguil)

## 1.44.1

* extend fire dep range

## 1.44.0

* Deps update

## 1.43.0

* Remove version restriction for fsspec

## 1.42.0

* ray distibutor (thanks @Vaishaal)
* Remove tmp_dir only if the output dir is not in s3 (thanks @ezzarum)
* support more input formats (thanks @ldfandian)

## 1.41.0

* Verify hashes during download. (thanks @GeorgiosSmyrnis and @carlini)
* opencv-python => opencv-python-headless (thanks @shionhonda)

## 1.40.0

* Add SBU captions benchmark
* Bump ffspec version
* Fix face blurring when padding/cropping
* Add support for other hash functions

## 1.39.0

* Make opt out the default, add warning about ethical issues with slowing down democratization of skills and art.

## 1.38.0

* Incorporate face blurring with bounding boxes. (thanks @GeorgiosSmyrnis)

## 1.37.0

* Add support for resizing with fixed aspect ratio while fixing the largest image dimension (thanks @gabrielilharco)

## 1.36.0

* bumping webdataset version to 0.2.5+

## 1.35.0

* added max_image_area flag (thanks @sagadre)

## 1.34.0

* Add argument validator in main.
* Respect noai and noimageai directives when downloading image files (thanks @raincoastchris)
* add list of int, float feature in TFRecordSampleWriter (thanks @justHungryMan)

## 1.33.0

* feat: support pyspark < 3 when distributing image-to-dataset job (thanks @nateagr)

## 1.32.0

* feat: support min image size + max aspect ratio (@borisdayma)

## 1.31.0

* feat: allow encoding in different formats (thanks @borisdayma)

## 1.30.2

* Fix error message for incorrect input format 

## 1.30.1

* Bug fix: shard id was incorrect when resuming (thanks @lxj616)

## 1.30.0

* Implement shard retrying

## 1.29.0

* Validate input and output format
* Implement incremental mode

## 1.28.0

* use pyarrow in the reader to make it much faster

## 1.27.4

* use 2022.1.0 of fsspec for python3.6

## 1.27.3

* fix fsspec version

## 1.27.2

* fix fsspec version

## 1.27.1

* add gcsfs to pex

## 1.27.0

* buffered writer fix: release ram more often
* feat: accept numpy arrays (thanks @borisdayma)

## 1.26.0

* add tfrecord output format (thanks @borisdayma)

## 1.25.6

* fix an interaction between md5 and exif option

## 1.25.5

* fix dependency ranges

## 1.25.4

* use exifread-nocycle to avoid cycle in exifread

## 1.25.3

* retry whole sharding if it fails

## 1.25.2

* retry writing the shard in reader in case of error

## 1.25.1

* small fix for logger and continuing
* use time instead of perf_counter to measure shard duration

## 1.25.0

* make metadata writer much faster by building the schema in the downloader instead of guessing it
* add new option allowing to disable reencoding

## 1.24.1

* hide opencv warning

## 1.24.0

* force one thread for opencv
* make total logger start time the minimum of workers start time
* add s3fs into the released pex for convenience
* make sharding faster on high latency fs by using a thread pool

## 1.23.1

* fix logger on s3: do not use listing caching in logger

## 1.23.0

* add tutorial on how to setup a spark cluster and use it for distributed img2dataset
better aws s3 support:
* initialize logger fs in subprocess to avoid moving fs over a fork()
* use spawn instead of fork method

* make total logging more intuitive and convenient by logging every worker return

## 1.22.3

* fix release regex

## 1.22.2

* fix fsspec support by using tmp_dir in main.py

## 1.22.1

* fix pex creation

## 1.22.0

* add option not to write

## 1.21.2

* try catch in the logger for json.load
* prevent error if logger sync is called when no call has been done
* Add a build-pex target in Makefile and CI

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

* increase stability by closing the pool and tarwriter explicitly

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
