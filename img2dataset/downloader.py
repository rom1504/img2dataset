"""Img2dataset"""

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Optional
from tqdm import tqdm
import albumentations as A
import cv2
import os
import urllib.request
import fire
import functools
import webdataset as wds
import io
import numpy as np
import pandas as pd
import math
import exifread
import json
import glob
import logging
import time
import wandb
from .logging_utils import CappedCounter, SpeedLogger, StatusTableLogger

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


class Resizer:
    """Resize images"""

    def __init__(self, image_size, resize_mode, resize_only_if_bigger):
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.resize_only_if_bigger = resize_only_if_bigger

        if resize_mode not in ["no", "keep_ratio", "center_crop", "border"]:
            raise Exception(f"Invalid option for resize_mode: {resize_mode}")

        if resize_mode == "keep_ratio":
            self.resize_tfm = A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4)
        elif resize_mode == "center_crop":
            self.resize_tfm = A.Compose(
                [A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4), A.CenterCrop(image_size, image_size),]
            )
        elif resize_mode == "border":
            self.resize_tfm = A.Compose(
                [
                    A.LongestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4),
                    A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255],),
                ]
            )
        elif resize_mode == "no":
            pass

    def __call__(self, img_stream):
        try:
            img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
            if img is None:
                raise Exception("Image decoding error")
            original_height, original_width = img.shape[:2]

            # resizing in following conditions
            if self.resize_mode != "no" and (
                not self.resize_only_if_bigger
                or (
                    self.resize_mode in ["keep_ratio", "center_crop"]
                    # smallest side contained to image_size (largest cropped)
                    and min(img.shape[:2]) > self.image_size
                )
                or (
                    self.resize_mode == "border"
                    # largest side contained to image_size
                    and max(img.shape[:2]) > self.image_size
                )
            ):
                img = self.resize_tfm(image=img)["image"]

            height, width = img.shape[:2]
            img_str = cv2.imencode(".jpg", img)[1].tobytes()
            del img
            return img_str, width, height, original_width, original_height, None

        except Exception as err:  # pylint: disable=broad-except
            return None, None, None, None, None, str(err)


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(self, shard_id, output_folder, save_caption, save_metadata, oom_shard_count):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
        self.shard_id = shard_id
        self.tarwriter = wds.TarWriter(f"{output_folder}/{shard_name}.tar")
        self.save_caption = save_caption
        self.save_metadata = save_metadata

    def write(self, img_str, key, caption, meta):
        sample = {"__key__": key, "jpg": img_str}
        if self.save_caption:
            sample["txt"] = str(caption) if caption is not None else ""
        if self.save_metadata:
            sample["json"] = json.dumps(meta, indent=4)
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(self, shard_id, output_folder, save_caption, save_metadata, oom_shard_count):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
        self.shard_id = shard_id
        self.subfolder = f"{output_folder}/{shard_name}"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.save_metadata = save_metadata

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        filename = f"{self.subfolder}/{key}.jpg"
        with open(filename, "wb") as f:
            f.write(img_str)
        if self.save_caption:
            caption = str(caption) if caption is not None else ""
            caption_filename = f"{self.subfolder}/{key}.txt"
            with open(caption_filename, "w") as f:
                f.write(str(caption))
        if self.save_metadata:
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with open(meta_filename, "w") as f:
                f.write(j)

    def close(self):
        pass


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

    if os.path.isdir(url_list):
        input_files = sorted(glob.iglob(url_list + "/*." + input_format))
    else:
        input_files = [url_list]

    for i, input_file in enumerate(input_files):
        print("Downloading file number " + str(i + 1) + " of " + str(len(input_files)) + " called " + input_file)
        print("Loading the input file")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            start_shard_id = 0
        else:
            existing_top_level_files = glob.glob(output_folder + "/*")
            if len(existing_top_level_files) == 0:
                start_shard_id = 0
            else:
                start_shard_id = max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files]) + 1

        if input_format == "txt":
            images_to_dl = []
            with open(input_file, encoding="utf-8") as file:
                images_to_dl = [(url.rstrip(),) for url in file.readlines()]
            column_list = ["url"]
        elif input_format in ["csv", "tsv", "parquet"]:
            if input_format == "csv":
                df = pd.read_csv(input_file)
            elif input_format == "tsv":
                df = pd.read_table(input_file)
            elif input_format == "parquet":
                columns_to_read = [url_col]
                if caption_col is not None:
                    columns_to_read += [caption_col]
                if save_additional_columns is not None:
                    columns_to_read += save_additional_columns
                df = pd.read_parquet(input_file, columns=columns_to_read)
            column_list = save_additional_columns if save_additional_columns is not None else []
            df = df.rename(columns={caption_col: "caption", url_col: "url"})
            if caption_col is not None:
                column_list = column_list + ["caption", "url"]
            else:
                column_list = column_list + ["url"]
            images_to_dl = df[column_list].to_records(index=False).tolist()
            del df

        sharded_images_to_dl = []
        number_samples = len(images_to_dl)
        number_shards = math.ceil(number_samples / number_sample_per_shard)
        print(f"Splitting the {number_samples} samples in {number_shards} shards of size {number_sample_per_shard}")
        for shard_id in range(number_shards):
            begin_shard = shard_id * number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * number_sample_per_shard)
            sharded_images_to_dl.append(
                (shard_id + start_shard_id, list(enumerate(images_to_dl[begin_shard:end_shard])),)
            )
        del images_to_dl
        print("Done sharding the input file")

        if output_format == "webdataset":
            sample_writer_class = WebDatasetSampleWriter
        elif output_format == "files":
            sample_writer_class = FilesSampleWriter  # type: ignore

        resizer = Resizer(image_size=image_size, resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger,)

        downloader = functools.partial(
            one_process_downloader,
            sample_writer_class=sample_writer_class,
            resizer=resizer,
            thread_count=thread_count,
            save_caption=save_caption,
            save_metadata=save_metadata,
            output_folder=output_folder,
            column_list=column_list,
            timeout=timeout,
            number_sample_per_shard=number_sample_per_shard,
            oom_shard_count=oom_shard_count,
        )

        print("Starting the downloading of this file")
        with Pool(processes_count, maxtasksperchild=5) as process_pool:
            for count, successes, failed_to_download, failed_to_resize, duration, status_dict in tqdm(
                process_pool.imap_unordered(downloader, sharded_images_to_dl), total=len(sharded_images_to_dl),
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
            del sharded_images_to_dl


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
