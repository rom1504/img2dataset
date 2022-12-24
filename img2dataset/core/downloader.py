"""Img2dataset downloader, downloads one shard"""

from multiprocessing.pool import ThreadPool
from threading import Semaphore
import urllib.request
import io
import exifread
import json
import time
import hashlib
import pyarrow as pa
import traceback
import uuid

import fsspec
from .logger import CappedCounter
from .logger import write_stats
from typing import List, Optional
import fire
import logging
from .resizer import Resizer
from .writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from pydantic import BaseModel
import pydantic


logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


class DownloaderOptions(BaseModel):
    thread_count: int = 256
    extract_exif: bool = True
    timeout: int = 10
    compute_md5: bool = True
    retries: int = 0
    user_agent_token: Optional[str] = None
    disallowed_header_directives: Optional[List[str]] = None
    delete_input_shard: bool = True

class DownloaderWorkerSpecificOptions(BaseModel):
    input_file: str
    output_file_prefix: str
    caption_col: Optional[str] = None
    input_format: str = "txt"
    save_additional_columns: Optional[List[str]] = None


from ..core.writer import WriterOptions
from ..core.downloader import DownloaderOptions
from ..core.resizer import ResizingOptions

DownloaderWorkerOptions = pydantic.create_model("DownloaderWorkerOptions", __base__=(
    WriterOptions,
    DownloaderOptions,
    ResizingOptions,
    DownloaderWorkerSpecificOptions,))


def is_disallowed(headers, user_agent_token, disallowed_header_directives):
    """Check if HTTP headers contain an X-Robots-Tag directive disallowing usage"""
    for values in headers.get_all("X-Robots-Tag", []):
        try:
            uatoken_directives = values.split(":", 1)
            directives = [x.strip().lower() for x in uatoken_directives[-1].split(",")]
            ua_token = uatoken_directives[0].lower() if len(uatoken_directives) == 2 else None
            if (ua_token is None or ua_token == user_agent_token) and any(
                x in disallowed_header_directives for x in directives
            ):
                return True
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"Failed to parse X-Robots-Tag: {values}: {err}")
    return False


def download_image(row, timeout, user_agent_token, disallowed_header_directives):
    """Download an image with urllib"""
    key, url = row
    img_stream = None
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/rom1504/img2dataset)"
    try:
        request = urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string})
        with urllib.request.urlopen(request, timeout=timeout) as r:
            if disallowed_header_directives and is_disallowed(
                r.headers,
                user_agent_token,
                disallowed_header_directives,
            ):
                return key, None, "Use of image disallowed by X-Robots-Tag directive"
            img_stream = io.BytesIO(r.read())
        return key, img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return key, None, str(err)


def download_image_with_retry(row, timeout, retries, user_agent_token, disallowed_header_directives):
    for _ in range(retries + 1):
        key, img_stream, err = download_image(row, timeout, user_agent_token, disallowed_header_directives)
        if img_stream is not None:
            return key, img_stream, err
    return key, None, err

def download_shard(**kwargs):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    opts = DownloaderWorkerOptions(**kwargs)

    save_caption = opts.caption_col is not None


    if opts.output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif opts.output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif opts.output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif opts.output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif opts.output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {opts.output_format}")

    resizer = Resizer(
        image_size=opts.image_size,
        resize_mode=opts.resize_mode,
        resize_only_if_bigger=opts.resize_only_if_bigger,
        upscale_interpolation=opts.upscale_interpolation,
        downscale_interpolation=opts.downscale_interpolation,
        encode_quality=opts.encode_quality,
        encode_format=opts.encode_format,
        skip_reencode=opts.skip_reencode,
        disable_all_reencoding=opts.disable_all_reencoding,
        min_image_size=opts.min_image_size,
        max_image_area=opts.max_image_area,
        max_aspect_ratio=opts.max_aspect_ratio,
    )

    if opts.input_format == "txt":
        column_list = ["url"]
    elif opts.input_format in ["json", "csv", "tsv", "tsv.gz", "parquet"]:
        column_list = opts.save_additional_columns if opts.save_additional_columns is not None else []
        if opts.caption_col is not None:
            column_list = column_list + ["caption", "url"]
        else:
            column_list = column_list + ["url"]
    else:
        raise ValueError(f"Invalid input format {opts.input_format}")

    opts.disallowed_header_directives = (
        None
        if opts.disallowed_header_directives is None
        else {directive.strip().lower() for directive in opts.disallowed_header_directives}
    )
    opts.user_agent_token = None if opts.user_agent_token is None else opts.user_agent_token.strip().lower()

    try:
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(opts.input_file)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
            .append(pa.field("width", pa.int32()))
            .append(pa.field("height", pa.int32()))
            .append(pa.field("original_width", pa.int32()))
            .append(pa.field("original_height", pa.int32()))
        )
        if opts.extract_exif:
            schema = schema.append(pa.field("exif", pa.string()))

        if opts.compute_md5:
            schema = schema.append(pa.field("md5", pa.string()))

        pydict = df.select(column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in column_list))))
        del pydict
        del df

        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed_to_download = 0
        failed_to_resize = 0
        url_indice = column_list.index("url")
        caption_indice = column_list.index("caption") if "caption" in column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        # this prevents an accumulation of more than twice the number of threads in sample ready to resize
        # limit the memory usage
        semaphore = Semaphore(opts.thread_count * 2)

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield e

        loader = data_generator()

        # give schema to writer
        sample_writer = sample_writer_class(
            opts.output_file_prefix,
            save_caption,
            schema,
            opts.encode_format,
        )
        with ThreadPool(opts.thread_count) as thread_pool:
            for key, img_stream, error_message in thread_pool.imap_unordered(
                lambda x: download_image_with_retry(
                    x,
                    timeout=opts.timeout,
                    retries=opts.retries,
                    user_agent_token=opts.user_agent_token,
                    disallowed_header_directives=opts.disallowed_header_directives,
                ),
                loader,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = str(uuid.uuid4())
                    meta = {
                        **{column_list[i]: sample_data[i] for i in range(len(column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                        "width": None,
                        "height": None,
                        "original_width": None,
                        "original_height": None,
                    }
                    if opts.extract_exif:
                        meta["exif"] = None
                    if opts.compute_md5:
                        meta["md5"] = None
                    if error_message is not None:
                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        sample_writer.write(
                            None,
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue
                    img_stream.seek(0)
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
                        status_dict.increment(error_message)
                        meta["status"] = status
                        meta["error_message"] = error_message
                        sample_writer.write(
                            None,
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        img_stream.close()
                        del img_stream
                        semaphore.release()
                        continue
                    successes += 1
                    status = "success"
                    status_dict.increment(status)

                    if opts.extract_exif:
                        try:
                            img_stream.seek(0)
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

                    if opts.compute_md5:
                        img_stream.seek(0)
                        meta["md5"] = hashlib.md5(img_stream.read()).hexdigest()

                    meta["status"] = status
                    meta["width"] = width
                    meta["height"] = height
                    meta["original_width"] = original_width
                    meta["original_height"] = original_height
                    img_stream.close()
                    del img_stream

                    sample_writer.write(
                        img,
                        str_key,
                        sample_data[caption_indice] if caption_indice is not None else None,
                        meta,
                    )
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            opts.output_file_prefix,
            count,
            successes,
            failed_to_download,
            failed_to_resize,
            start_time,
            end_time,
            status_dict,
        )
        if opts.delete_input_shard:
            fs.rm(shard_path)
        return (True, (opts.input_file, opts.output_file_prefix))
    except Exception as err:  # pylint: disable=broad-except
        traceback.print_exc()
        print(f"shard {opts.input_file} failed with error {err}")
        return (False, (opts.input_file, opts.output_file_prefix))


def main():
    fire.Fire(download_shard)


if __name__ == "__main__":
    main()
