"""Img2dataset"""

from multiprocessing import Pool
from typing import List, Optional
from tqdm import tqdm
import fire
import logging
from .logger import LoggerProcess
from .resizer import Resizer
from .writer import WebDatasetSampleWriter, FilesSampleWriter, ParquetSampleWriter
from .reader import Reader
from .downloader import Downloader
import fsspec
import sys
import signal

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
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 5,
    compute_md5: bool = True,
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())
    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters, processes_count)
    logger_process.start()

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        start_shard_id = 0
    else:
        existing_top_level_files = fs.glob(output_path + "/*")
        if len(existing_top_level_files) == 0:
            start_shard_id = 0
        else:
            start_shard_id = max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files]) + 1

    reader = Reader(
        url_list, input_format, url_col, caption_col, save_additional_columns, number_sample_per_shard, start_shard_id
    )

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
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
        extract_exif=extract_exif,
        output_folder=output_folder,
        column_list=reader.column_list,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        oom_shard_count=oom_shard_count,
        compute_md5=compute_md5,
    )

    print("Starting the downloading of this file")
    with Pool(processes_count, maxtasksperchild=5) as process_pool:
        for _ in tqdm(process_pool.imap_unordered(downloader, reader),):
            pass

        process_pool.terminate()
        process_pool.join()
        del process_pool
    logger_process.join()


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
