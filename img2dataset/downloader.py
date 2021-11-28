"""Img2dataset"""

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Optional
from tqdm import tqdm
import os
import urllib.request
import fire
import functools
import io
import pandas as pd
import math
import exifread
import json
import glob
import logging
import time
import wandb
from .logging_utils import CappedCounter, SpeedLogger, StatusTableLogger
from .resizer import Resizer
from .writer import WebDatasetSampleWriter, FilesSampleWriter
from .reader import Reader

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


def download_image(row, timeout):
    """Download an image with urllib"""
    key, url = row
    img_stream = None
    try:
        request = urllib.request.Request(
            url,
            data=None,
            headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            img_stream = io.BytesIO(r.read())
        return key, img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return key, None, str(err)


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10 ** oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(key_format=key_format, true_key=true_key)
    return str_key


def one_process_downloader(
    row,
    sample_writer_class,
    resizer,
    thread_count,
    save_caption,
    save_metadata,
    output_folder,
    column_list,
    timeout,
    number_sample_per_shard,
    oom_shard_count,
):
    """Function to start an image downloading in one process"""
    shard_id, shard_to_dl = row

    start_time = time.perf_counter()
    status_dict = CappedCounter()

    if save_metadata:
        metadatas = []

    count = len(shard_to_dl)
    successes = 0
    failed_to_download = 0
    failed_to_resize = 0
    url_indice = column_list.index("url")
    caption_indice = column_list.index("caption") if "caption" in column_list else None
    key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

    # this prevents an accumulation of more than twice the number of threads in sample ready to resize
    # limit the memory usage
    semaphore = Semaphore(thread_count * 2)

    def data_generator():
        for e in key_url_list:
            semaphore.acquire()
            yield e

    loader = data_generator()

    sample_writer = sample_writer_class(shard_id, output_folder, save_caption, save_metadata, oom_shard_count)
    oom_sample_per_shard = math.ceil(math.log10(number_sample_per_shard))
    with ThreadPool(thread_count) as thread_pool:
        for key, img_stream, error_message in thread_pool.imap_unordered(
            lambda x: download_image(x, timeout=timeout), loader
        ):
            _, sample_data = shard_to_dl[key]
            str_key = compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count)
            meta = {
                **{column_list[i]: sample_data[i] for i in range(len(column_list))},
                "key": str_key,
                "status": None,
                "error_message": error_message,
                "width": None,
                "height": None,
                "exif": None,
            }
            if error_message is not None:
                failed_to_download += 1
                status = "failed_to_download"
                status_dict.increment(error_message)
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
                semaphore.release()
                continue
            (img, width, height, original_width, original_height, error_message,) = resizer(img_stream)
            if error_message is not None:
                failed_to_resize += 1
                status = "failed_to_resize"
                status_dict.increment(error_message)
                if save_metadata:
                    meta["status"] = status
                    meta["error_message"] = error_message
                    metadatas.append(meta)
                img_stream.close()
                del img_stream
                semaphore.release()
                continue
            successes += 1
            status = "success"
            status_dict.increment(status)

            if save_metadata:
                try:
                    exif = json.dumps(
                        {
                            k: str(v).strip()
                            for k, v in exifread.process_file(img_stream, details=False).items()
                            if v is not None
                        }
                    )
                except Exception as _:  # pylint: disable=broad-except
                    exif = None
                meta["status"] = status
                meta["width"] = width
                meta["height"] = height
                meta["original_width"] = original_width
                meta["original_height"] = original_height
                meta["exif"] = exif
                metadatas.append(meta)
            else:
                meta = None
            img_stream.close()
            del img_stream

            sample_writer.write(
                img, str_key, sample_data[caption_indice] if caption_indice is not None else None, meta,
            )
            semaphore.release()

        sample_writer.close()
        thread_pool.terminate()
        thread_pool.join()
        del thread_pool

    if save_metadata:
        df = pd.DataFrame(metadatas)
        shard_name = "%05d" % shard_id
        df.to_parquet(output_folder + "/" + shard_name + ".parquet")

    end_time = time.perf_counter()
    return (count, successes, failed_to_download, failed_to_resize, end_time - start_time, status_dict)


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

    downloader = functools.partial(
        one_process_downloader,
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
