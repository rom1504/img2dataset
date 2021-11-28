"""the downloader module handles the downloading"""

from multiprocessing.pool import ThreadPool
from threading import Semaphore
import urllib.request
import io
import pandas as pd
import math
import exifread
import json
import time
from .logger import CappedCounter


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


class Downloader:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
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
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.resizer = resizer
        self.thread_count = thread_count
        self.save_caption = save_caption
        self.save_metadata = save_metadata
        self.output_folder = output_folder
        self.column_list = column_list
        self.timeout = timeout
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count

    def __call__(
        self, row,
    ):
        """Function to start an image downloading in one process"""
        shard_id, shard_to_dl = row

        start_time = time.perf_counter()
        status_dict = CappedCounter()

        if self.save_metadata:
            metadatas = []

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

        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.save_metadata, self.oom_shard_count
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))
        with ThreadPool(self.thread_count) as thread_pool:
            for key, img_stream, error_message in thread_pool.imap_unordered(
                lambda x: download_image(x, timeout=self.timeout), loader
            ):
                _, sample_data = shard_to_dl[key]
                str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                meta = {
                    **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
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
                    if self.save_metadata:
                        meta["status"] = status
                        metadatas.append(meta)
                    semaphore.release()
                    continue
                (img, width, height, original_width, original_height, error_message,) = self.resizer(img_stream)
                if error_message is not None:
                    failed_to_resize += 1
                    status = "failed_to_resize"
                    status_dict.increment(error_message)
                    if self.save_metadata:
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

                if self.save_metadata:
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

        if self.save_metadata:
            df = pd.DataFrame(metadatas)
            shard_name = "%05d" % shard_id
            df.to_parquet(self.output_folder + "/" + shard_name + ".parquet")

        end_time = time.perf_counter()
        return (count, successes, failed_to_download, failed_to_resize, end_time - start_time, status_dict)
