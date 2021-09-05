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
import sys
import time
import wandb

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)

# define global counters
total_count = 0
total_success = 0
total_failed_to_download = 0
total_failed_to_resize = 0
total_status_dict = {}


class Logger:
    def __init__(self, processes_count=1, min_interval=0):
        """Log only every processes_count and if min_interval (seconds) have elapsed since last log"""
        # wait for all processes to return
        self.processes_count = processes_count
        self.processes_returned = 0
        # min time (in seconds) before logging a new table (avoids too many logs)
        self.min_interval = min_interval
        self.last = time.perf_counter()
        # keep track of whether we logged the last call
        self.last_call_logged = False
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.processes_returned += 1
        if (
            self.processes_returned % self.processes_count == 0
            and time.perf_counter() - self.last > self.min_interval
        ):
            self.do_log(*args, **kwargs)
            self.last = time.perf_counter()
            self.last_call_logged = True
        else:
            self.last_call_logged = False
            self.last_args = args
            self.last_kwargs = kwargs

    def do_log(self, *args, **kwargs):
        raise NotImplemented

    def sync(self):
        """Ensure last call is logged"""
        if not self.last_call_logged:
            self.do_log(*self.last_args, **self.last_kwargs)
            # reset for next file
            self.processes_returned = 0


class SpeedLogger(Logger):
    """Log performance metrics"""

    def __init__(self, prefix, **logger_args):
        super().__init__(**logger_args)
        self.prefix = prefix
        self.start = time.perf_counter()

    def do_log(self, count, success, failed_to_download, failed_to_resize):
        duration = time.perf_counter() - self.start
        img_per_sec = count / duration
        success_ratio = 1.0 * success / count
        failed_to_download_ratio = 1.0 * failed_to_download / count
        failed_to_resize_ratio = 1.0 * failed_to_resize / count

        print(
            " - ".join(
                [
                    f"{self.prefix:<7}",
                    f"success: {success_ratio:.3f}",
                    f"failed to download: {failed_to_download_ratio:.3f}",
                    f"failed to resize: {failed_to_resize_ratio:.3f}",
                    f"images per sec: {img_per_sec:.0f}",
                    f"count: {count}",
                ]
            )
        )

        wandb.log(
            {
                f"{self.prefix}/img_per_sec": img_per_sec,
                f"{self.prefix}/success": success_ratio,
                f"{self.prefix}/failed_to_download": failed_to_download_ratio,
                f"{self.prefix}/failed_to_resize": failed_to_resize_ratio,
                f"{self.prefix}/count": count,
            }
        )


class StatusTableLogger(Logger):
    """Log status table to W&B, up to `max_status` most frequent items"""

    def __init__(self, max_status=100, min_interval=60, **logger_args):
        super().__init__(min_interval=min_interval, **logger_args)
        # avoids too many errors unique to a specific website (SSL certificates, etc)
        self.max_status = max_status

    def do_log(self, status_dict, count):
        status_table = wandb.Table(
            columns=["status", "frequency", "count"],
            data=[
                [k, 1.0 * v / count, v]
                for k, v in sorted(
                    status_dict.items(), key=lambda x: x[1], reverse=True
                )[: self.max_status]
            ],
        )
        wandb.run.log({"status": status_table})


def download_image(row, timeout):
    key, url = row
    try:
        request = urllib.request.Request(
            url,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            img_stream = io.BytesIO(r.read())
        return key, img_stream, None
    except Exception as err:
        return key, None, str(err)


class Resizer:
    """Resize images"""

    def __init__(self, image_size, resize_mode, resize_only_if_bigger):
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.resize_only_if_bigger = resize_only_if_bigger

        # define transform
        if resize_mode not in ["no", "keep_ratio", "center_crop", "border"]:
            raise Exception(f"Invalid option for resize_mode: {resize_mode}")
        self.resize_tfm = (
            A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4)
            if resize_mode == "keep_ratio"
            else A.Compose(
                [
                    A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4),
                    A.CenterCrop(image_size, image_size),
                ]
            )
            if resize_mode == "center_crop"
            else A.Compose(
                [
                    A.LongestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4),
                    A.PadIfNeeded(
                        image_size,
                        image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    ),
                ]
            )
            if resize_mode == "border"
            else None
        )

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

        except Exception as err:
            return None, None, None, None, None, str(err)


class WebDatasetSampleWriter:
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
    def __init__(self, shard_id, output_folder):
        shard_name = "%05d" % shard_id
        self.subfolder = f"{output_folder}/{shard_name}"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)

    def write(self, img_str, key, caption, meta):
        key = "%04d" % key
        filename = f"{self.subfolder}/{key}.jpg"
        with open(filename, "wb") as f:
            f.write(img_str)
        if caption is not None:
            caption_filename = f"{self.subfolder}/{key}.txt"
            with open(caption_filename, "w") as f:
                f.write(caption)
        if meta is not None:
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with open(meta_filename, "w") as f:
                f.write(j)

    def close(self):
        pass


def one_process_downloader(
    row,
    sample_writer_class,
    resizer,
    thread_count,
    save_metadata,
    output_folder,
    column_list,
    timeout,
):
    speed_logger = SpeedLogger("process")
    shard_id, shard_to_dl = row

    if save_metadata:
        metadatas = []

    count = len(shard_to_dl)
    successes = 0
    failed_to_download = 0
    failed_to_resize = 0
    url_indice = column_list.index("url")
    caption_indice = column_list.index("caption") if "caption" in column_list else None
    key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]
    # capture error/success
    status_dict = dict()

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
                status_dict[error_message] = status_dict.get(error_message, 0) + 1
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
                continue
            (
                img,
                width,
                height,
                original_width,
                original_height,
                error_message,
            ) = resizer(img_stream)
            if error_message is not None:
                failed_to_resize += 1
                status = "failed_to_resize"
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
                continue
            successes += 1
            status = "success"
            status_dict["success"] = status_dict.get("success", 0) + 1

            if save_metadata:
                try:
                    exif = json.dumps(
                        {
                            k: str(v).strip()
                            for k, v in exifread.process_file(
                                img_stream, details=False
                            ).items()
                            if v is not None
                        }
                    )
                except Exception as _:
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
                img,
                key,
                sample_data[caption_indice] if caption_indice is not None else None,
                meta,
            )

        sample_writer.close()
        thread_pool.terminate()
        thread_pool.join()
        del thread_pool

    if save_metadata:
        df = pd.DataFrame(metadatas)
        shard_name = "%05d" % shard_id
        df.to_parquet(output_folder + "/" + shard_name + ".parquet")

    return (
        count,
        successes,
        failed_to_download,
        failed_to_resize,
        speed_logger,
        status_dict,
    )


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
    wandb_project="img2dataset",
):
    # capture all config parameters
    config_parameters = dict(locals())

    def download_one_file(url_list, total_speed_logger, status_table_logger):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            start_shard_id = 0
        else:
            existing_top_level_files = glob.glob(output_folder + "/*")
            if len(existing_top_level_files) == 0:
                start_shard_id = 0
            else:
                start_shard_id = (
                    max(
                        [
                            int(x.split("/")[-1].split(".")[0])
                            for x in existing_top_level_files
                        ]
                    )
                    + 1
                )

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
            column_list = (
                save_additional_columns if save_additional_columns is not None else []
            )
            df = df.rename(columns={caption_col: "caption", url_col: "url"})
            if caption_col is not None:
                column_list = column_list + ["caption", "url"]
            else:
                column_list = column_list + ["url"]
            images_to_dl = df[column_list].to_records(index=False)
            del df

        sharded_images_to_dl = []
        number_samples = len(images_to_dl)
        number_shards = math.ceil(number_samples / number_sample_per_shard)
        for shard_id in range(number_shards):
            begin_shard = shard_id * number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * number_sample_per_shard)
            sharded_images_to_dl.append(
                (
                    shard_id + start_shard_id,
                    list(enumerate(images_to_dl[begin_shard:end_shard])),
                )
            )
        del images_to_dl

        if output_format == "webdataset":
            sample_writer_class = WebDatasetSampleWriter
        elif output_format == "files":
            sample_writer_class = FilesSampleWriter

        resizer = Resizer(
            image_size=image_size,
            resize_mode=resize_mode,
            resize_only_if_bigger=resize_only_if_bigger,
        )

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

        # Start a W&B run
        wandb.init(project=wandb_project, config=config_parameters, anonymous="allow")

        # retrieve global variables
        global total_count, total_success, total_failed_to_download, total_failed_to_resize, total_status_dict

        with Pool(processes_count, maxtasksperchild=5) as process_pool:
            for (
                count,
                successes,
                failed_to_download,
                failed_to_resize,
                speed_logger,
                status_dict,
            ) in tqdm(
                process_pool.imap_unordered(downloader, sharded_images_to_dl),
                total=len(sharded_images_to_dl),
                file=sys.stdout,
            ):
                total_count += count
                total_success += successes
                total_failed_to_download += failed_to_download
                total_failed_to_resize += failed_to_resize

                speed_logger(
                    count=count,
                    success=successes,
                    failed_to_download=failed_to_download,
                    failed_to_resize=failed_to_resize,
                )
                total_speed_logger(
                    count=total_count,
                    success=total_success,
                    failed_to_download=total_failed_to_download,
                    failed_to_resize=total_failed_to_resize,
                )

                # update status table
                for k, v in status_dict.items():
                    total_status_dict[k] = total_status_dict.get(k, 0) + v
                status_table_logger(total_status_dict, total_count)

            # ensure final sync
            total_speed_logger.sync()
            status_table_logger.sync()

            process_pool.terminate()
            process_pool.join()
            del process_pool

    if os.path.isdir(url_list):
        input_files = glob.glob(url_list + "/*." + input_format)
    else:
        input_files = [url_list]

    total_speed_logger = SpeedLogger("total", processes_count=processes_count)
    status_table_logger = StatusTableLogger(processes_count=processes_count)

    for i, input_file in enumerate(input_files):
        print(
            "Downloading file number "
            + str(i + 1)
            + " of "
            + str(len(input_files))
            + " called "
            + input_file
        )
        download_one_file(input_file, total_speed_logger, status_table_logger)


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
