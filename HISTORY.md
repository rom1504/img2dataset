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