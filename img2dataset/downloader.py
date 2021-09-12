"""Img2dataset"""

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
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
            self.resize_mode = None

    def __call__(self, img_stream):
        try:
            img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
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
            return img_str, width, height, original_width, original_height, None

        except Exception as err:  # pylint: disable=broad-except
            return None, None, None, None, None, str(err)


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(self, shard_id, output_folder):
        shard_name = "%05d" % shard_id
        self.tarwriter = wds.TarWriter(f"{output_folder}/{shard_name}.tar")

    def write(self, img_str, key, caption, meta):
        key = "%09d" % key
        sample = {"__key__": key, "jpg": img_str}
        if caption is not None:
            sample["txt"] = caption
        if meta is not None:
            sample["json"] = json.dumps(meta, indent=4)
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(self, shard_id, output_folder):
        shard_name = "%05d" % shard_id
        self.subfolder = f"{output_folder}/{shard_name}"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        key = "%04d" % key
        filename = f"{self.subfolder}/{key}.jpg"
        with open(filename, "wb") as f:
            f.write(img_str)
        if caption is not None:
            caption_filename = f"{self.subfolder}/{key}.txt"
            with open(caption_filename, "w") as f:
                f.write(str(caption))
        if meta is not None:
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with open(meta_filename, "w") as f:
                f.write(j)

    def close(self):
        pass


def one_process_downloader(
    row, sample_writer_class, resizer, thread_count, save_metadata, output_folder, column_list, timeout,
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

    sample_writer = sample_writer_class(shard_id, output_folder)
    with ThreadPool(thread_count) as thread_pool:
        for key, img_stream, error_message in thread_pool.imap_unordered(
            lambda x: download_image(x, timeout=timeout), key_url_list
        ):
            _, sample_data = shard_to_dl[key]
            meta = {
                **{column_list[i]: sample_data[i] for i in range(len(column_list))},
                "key": key,
                "shard_id": shard_id,
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
                continue
            (img, width, height, original_width, original_height, error_message,) = resizer(img_stream)
            if error_message is not None:
                failed_to_resize += 1
                status = "failed_to_resize"
                status_dict.increment(error_message)
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
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

            sample_writer.write(
                img, key, sample_data[caption_indice] if caption_indice is not None else None, meta,
            )

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
    url_list,
    image_size=256,
    output_folder="images",
    processes_count=1,
    resize_mode="border",
    resize_only_if_bigger=False,
    output_format="files",
    input_format="txt",
    url_col="url",
    caption_col=None,
    thread_count=256,
    number_sample_per_shard=10000,
    save_metadata=True,
    save_additional_columns=None,
    timeout=10,
    enable_wandb=False,
    wandb_project="img2dataset",
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())
    if enable_wandb:
        wandb.init(project=wandb_project, config=config_parameters, anonymous="allow")
    total_speed_logger = SpeedLogger("total", processes_count=processes_count, enable_wandb=enable_wandb)
    status_table_logger = StatusTableLogger(processes_count=processes_count, enable_wandb=enable_wandb)
    start_time = time.perf_counter()
    total_status_dict = CappedCounter()

    if os.path.isdir(url_list):
        input_files = glob.glob(url_list + "/*." + input_format)
    else:
        input_files = [url_list]

    for i, input_file in enumerate(input_files):
        print("Downloading file number " + str(i + 1) + " of " + str(len(input_files)) + " called " + input_file)
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
            with open(url_list, encoding="utf-8") as file:
                images_to_dl = [(url,) for url in file.readlines()]
            column_list = ["url"]
        elif input_format in ["csv", "tsv", "parquet"]:
            if input_format == "csv":
                df = pd.read_csv(url_list)
            elif input_format == "tsv":
                df = pd.read_table(url_list)
            elif input_format == "parquet":
                df = pd.read_parquet(url_list)
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
        for shard_id in range(number_shards):
            begin_shard = shard_id * number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * number_sample_per_shard)
            sharded_images_to_dl.append(
                (shard_id + start_shard_id, list(enumerate(images_to_dl[begin_shard:end_shard])),)
            )
        del images_to_dl

        if output_format == "webdataset":
            sample_writer_class = WebDatasetSampleWriter
        elif output_format == "files":
            sample_writer_class = FilesSampleWriter

        resizer = Resizer(image_size=image_size, resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger,)

        downloader = functools.partial(
            one_process_downloader,
            sample_writer_class=sample_writer_class,
            resizer=resizer,
            thread_count=thread_count,
            save_metadata=save_metadata,
            output_folder=output_folder,
            column_list=column_list,
            timeout=timeout,
        )

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


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
