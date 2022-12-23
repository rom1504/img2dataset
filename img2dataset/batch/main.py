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


# do kwargs + have a parsing thing for validating
def download(
    url_list: str,
    image_size: int = 256,
    output_folder: str = "images",
    processes_count: int = 1,
    resize_mode: str = "border",
    resize_only_if_bigger: bool = False,
    upscale_interpolation: str = "lanczos",
    downscale_interpolation: str = "area",
    encode_quality: int = 95,
    encode_format: str = "jpg",
    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    thread_count: int = 256,
    number_sample_per_shard: int = 10000,
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 7,
    compute_md5: bool = True,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    retries: int = 0,
    disable_all_reencoding: bool = False,
    min_image_size: int = 0,
    max_image_area: float = float("inf"),
    max_aspect_ratio: float = float("inf"),
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    user_agent_token: Optional[str] = None,
    disallowed_header_directives: Optional[List[str]] = None,
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())
    arguments_validator(config_parameters)

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters)

    tmp_path = output_folder + "/_tmp"
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

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    reader = Reader(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
    )

    def worker_config_generator():
        for (shard_id, input_file) in reader:
            shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=oom_shard_count
            )
            output_file_prefix = f"{output_folder}/{shard_name}"
            param_keys = [
                "image_size",
                "resize_mode",
                "resize_only_if_bigger",
                "upscale_interpolation",
                "downscale_interpolation",
                "encode_quality",
                "encode_format",
                "skip_reencode",
                "output_format",
                "input_format",
                "caption_col",
                "thread_count",
                "number_sample_per_shard",
                "extract_exif",
                "save_additional_columns",
                "timeout",
                "compute_md5",
                "retries",
                "disable_all_reencoding",
                "min_image_size",
                "max_image_area",
                "max_aspect_ratio",
                "user_agent_token",
                "disallowed_header_directives",
            ]
            conf = {k: config_parameters[k] for k in param_keys}
            conf["input_file"] = input_file
            conf["output_file_prefix"] = output_file_prefix
            yield conf

    print("Starting the downloading of this file")
    if distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
    elif distributor == "pyspark":
        distributor_fn = pyspark_distributor
    else:
        raise ValueError(f"Distributor {distributor} not supported")

    distributor_fn(
        processes_count,
        worker_config_generator(),
        subjob_size,
        max_shard_retry,
    )
    logger_process.join()
    fs.rm(tmp_dir, recursive=True)


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
