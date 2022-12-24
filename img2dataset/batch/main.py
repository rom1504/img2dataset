"""Img2dataset"""

from typing import List, Optional
import fire
import logging
from ..core.logger import LoggerProcess
from ..core.reader import Reader
from .distributor import multiprocessing_distributor, pyspark_distributor
import fsspec
import sys
import signal
import os
import sys

import pydantic

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


def arguments_validator(params):
    """Validate the arguments"""
    if params["save_additional_columns"] is not None:
        save_additional_columns_set = set(params["save_additional_columns"])

        forbidden_columns = set(
            [
                "key",
                "caption",
                "url",
                "width",
                "height",
                "original_width",
                "original_height",
                "status",
                "error_message",
                "exif",
                "md5",
            ]
        )
        intersection = save_additional_columns_set.intersection(forbidden_columns)
        if intersection:
            raise ValueError(
                f"You cannot use in save_additional_columns the following columns: {intersection}."
                + "img2dataset reserves these columns for its own use. Please remove them from save_additional_columns."
            )

    
from ..core.writer import WriterOptions
from ..core.downloader import DownloaderOptions
from ..core.resizer import ResizingOptions
from ..batch.distributor import DistributorOptions
from ..core.logger import LoggerOptions
from ..core.reader import ReaderOptions

MainOptions = pydantic.create_model("MainOptions", __base__=(
    ResizingOptions,
    WriterOptions,
    DownloaderOptions,
    DistributorOptions,
    LoggerOptions,
    ReaderOptions,))

DownloaderWorkerInMainOptions = pydantic.create_model("DownloaderWorkerOptions", __base__=(
    WriterOptions,
    DownloaderOptions,
    ResizingOptions,))

# do kwargs + have a parsing thing for validating
def download(**kwargs):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    opts = MainOptions(**kwargs)

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    opts.output_folder = make_path_absolute(opts.output_folder)
    opts.url_list = make_path_absolute(opts.url_list)
    config_parameters = opts.dict()
    arguments_validator(config_parameters)

    logger_process = LoggerProcess(opts.output_folder, opts.enable_wandb, opts.wandb_project, config_parameters)

    tmp_path = opts.output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    fs, output_path = fsspec.core.url_to_fs(opts.output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if opts.incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif opts.incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {opts.incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    reader = Reader(
        opts.url_list,
        opts.input_format,
        opts.url_col,
        opts.caption_col,
        opts.save_additional_columns,
        opts.number_sample_per_shard,
        done_shards,
        tmp_path,
    )

    def worker_config_generator():
        for (shard_id, input_file) in reader:
            shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=opts.oom_shard_count
            )
            output_file_prefix = f"{opts.output_folder}/{shard_name}"
            param_keys = set(DownloaderWorkerInMainOptions.__fields__.keys())
            param_keys.add("caption_col")
            param_keys.add("input_format")
            param_keys.add("save_additional_columns")
            conf = {k: config_parameters[k] for k in param_keys}
            conf["input_file"] = input_file
            conf["output_file_prefix"] = output_file_prefix
            yield conf

    print("Starting the downloading of this file")
    if opts.distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
    elif opts.distributor == "pyspark":
        distributor_fn = pyspark_distributor
    else:
        raise ValueError(f"Distributor {opts.distributor} not supported")

    distributor_fn(
        opts.processes_count,
        worker_config_generator(),
        opts.subjob_size,
        opts.max_shard_retry,
    )
    logger_process.join()
    fs.rm(tmp_dir, recursive=True)


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
