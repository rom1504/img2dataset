"""the downloader module handles the downloading"""

from multiprocessing.pool import ThreadPool
from threading import Semaphore
import urllib.request
import io
import math
import exifread
import json
import time
import hashlib
import pandas as pd

import fsspec
from .logger import CappedCounter
from .logger import write_stats


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


def download_image_with_retry(row, timeout, retries):
    for _ in range(retries + 1):
        key, img_stream, err = download_image(row, timeout)
        if img_stream is not None:
            return key, img_stream, err
    return key, None, err


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10 ** oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(key_format=key_format, true_key=true_key)
    return str_key


class Downloader:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        resizer,
        thread_count,
        save_caption,
        extract_exif,
        output_folder,
        column_list,
        timeout,
        number_sample_per_shard,
        oom_shard_count,
        compute_md5,
        retries,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.resizer = resizer
        self.thread_count = thread_count
        self.save_caption = save_caption
        self.extract_exif = extract_exif
        self.output_folder = output_folder
        self.column_list = column_list
        self.timeout = timeout
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.compute_md5 = compute_md5
        self.retries = retries

    def __call__(
        self, row,
    ):
        try:
            return self.download_shard(row)
        except Exception as err:  # pylint: disable=broad-except
            print(err)
            return (False, 0, 0, 0, 0, 0, None)

    def download_shard(
        self, row,
    ):
        """Function to start an image downloading in one process"""

        shard_id, shard_file = row

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        with fs.open(shard_path, "rb") as f:
            df = pd.read_feather(f)
        shard_to_dl = list(enumerate(df[self.column_list].to_records(index=False).tolist()))
        del df

        start_time = time.perf_counter()
        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed_to_download = 0
        failed_to_resize = 0
        url_indice = self.column_list.index("url")
        caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        # this prevents an accumulation of more than twice the number of threads in sample ready to resize
        # limit the memory usage
        semaphore = Semaphore(self.thread_count * 2)

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()
                yield e

        loader = data_generator()

        sample_writer = self.sample_writer_class(shard_id, self.output_folder, self.save_caption, self.oom_shard_count)
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))
        with ThreadPool(self.thread_count) as thread_pool:
            for key, img_stream, error_message in thread_pool.imap_unordered(
                lambda x: download_image_with_retry(x, timeout=self.timeout, retries=self.retries), loader
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                        "width": None,
                        "height": None,
                        "original_width": None,
                        "original_height": None,
                    }
                    if self.extract_exif:
                        meta["exif"] = None
                    if self.compute_md5:
                        meta["md5"] = None
                    if error_message is not None:
                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        sample_writer.write(
                            None, str_key, sample_data[caption_indice] if caption_indice is not None else None, meta
                        )
                        semaphore.release()
                        continue
                    (img, width, height, original_width, original_height, error_message,) = self.resizer(img_stream)
                    if error_message is not None:
                        failed_to_resize += 1
                        status = "failed_to_resize"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        meta["error_message"] = error_message
                        sample_writer.write(
                            None, str_key, sample_data[caption_indice] if caption_indice is not None else None, meta
                        )
                        img_stream.close()
                        del img_stream
                        semaphore.release()
                        continue
                    successes += 1
                    status = "success"
                    status_dict.increment(status)

                    if self.extract_exif:
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
                        meta["exif"] = exif

                    if self.compute_md5:
                        meta["md5"] = hashlib.md5(img_stream.read()).hexdigest()

                    meta["status"] = status
                    meta["width"] = width
                    meta["height"] = height
                    meta["original_width"] = original_width
                    meta["original_height"] = original_height
                    img_stream.close()
                    del img_stream

                    sample_writer.write(
                        img, str_key, sample_data[caption_indice] if caption_indice is not None else None, meta,
                    )
                except Exception as err:  # pylint: disable=broad-except
                    print(err)
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.perf_counter()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed_to_download,
            failed_to_resize,
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
        fs.rm(shard_path)
