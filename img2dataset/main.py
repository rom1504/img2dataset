"""Img2dataset"""

from multiprocessing import Pool
from typing import List, Optional
from tqdm import tqdm
import os
import fire
import glob
import logging
import time
import wandb
from .logger import CappedCounter, SpeedLogger, StatusTableLogger
from .resizer import Resizer
from .writer import WebDatasetSampleWriter, FilesSampleWriter
from .reader import Reader
from .downloader import Downloader

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


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
    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    thread_count: int = 256,
    number_sample_per_shard: int = 10000,
    save_metadata: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 5,
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())
    if enable_wandb:
        wandb.init(project=wandb_project, config=config_parameters, anonymous="allow")
    total_speed_logger = SpeedLogger("total", processes_count=processes_count, enable_wandb=enable_wandb)
    status_table_logger = StatusTableLogger(processes_count=processes_count, enable_wandb=enable_wandb)
    start_time = time.perf_counter()
    total_status_dict = CappedCounter()
    save_caption = caption_col is not None

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        start_shard_id = 0
    else:
        existing_top_level_files = glob.glob(output_folder + "/*")
        if len(existing_top_level_files) == 0:
            start_shard_id = 0
        else:
            start_shard_id = max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files]) + 1

    reader = Reader(
        url_list, input_format, url_col, caption_col, save_additional_columns, number_sample_per_shard, start_shard_id
    )

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore

    resizer = Resizer(
        image_size=image_size,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
        upscale_interpolation=upscale_interpolation,
        downscale_interpolation=downscale_interpolation,
        encode_quality=encode_quality,
        skip_reencode=skip_reencode,
    )

    downloader = Downloader(
        sample_writer_class=sample_writer_class,
        resizer=resizer,
        thread_count=thread_count,
        save_caption=save_caption,
        save_metadata=save_metadata,
        output_folder=output_folder,
        column_list=reader.column_list,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        oom_shard_count=oom_shard_count,
    )

    print("Starting the downloading of this file")
    with Pool(processes_count, maxtasksperchild=5) as process_pool:
        for count, successes, failed_to_download, failed_to_resize, duration, status_dict in tqdm(
            process_pool.imap_unordered(downloader, reader),
        ):
            SpeedLogger("worker", enable_wandb=enable_wandb)(
                duration=duration,
                count=count,
                success=successes,
                failed_to_download=failed_to_download,
                failed_to_resize=failed_to_resize,
            )
            total_speed_logger(
                duration=time.perf_counter() - start_time,
                count=count,
                success=successes,
                failed_to_download=failed_to_download,
                failed_to_resize=failed_to_resize,
            )
            total_status_dict.update(status_dict)
            status_table_logger(total_status_dict, total_speed_logger.count)
            pass

        # ensure final sync
        total_speed_logger.sync()
        status_table_logger.sync()
        process_pool.terminate()
        process_pool.join()
        del process_pool


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
