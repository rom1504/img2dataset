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

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


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


def download_shard(
    input_file: str,
    output_file_prefix: str,
    image_size: int = 256,
    resize_mode: str = "border",
    resize_only_if_bigger: bool = False,
    upscale_interpolation: str = "lanczos",
    downscale_interpolation: str = "area",
    encode_quality: int = 95,
    encode_format: str = "jpg",
    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    caption_col: Optional[str] = None,
    thread_count: int = 256,
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    compute_md5: bool = True,
    retries: int = 0,
    disable_all_reencoding: bool = False,
    min_image_size: int = 0,
    max_image_area: float = float("inf"),
    max_aspect_ratio: float = float("inf"),
    user_agent_token: Optional[str] = None,
    disallowed_header_directives: Optional[List[str]] = None,
    delete_input_shard: bool = True,
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())
    arguments_validator(config_parameters)

    save_caption = caption_col is not None


    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {output_format}")

    resizer = Resizer(
        image_size=image_size,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
        upscale_interpolation=upscale_interpolation,
        downscale_interpolation=downscale_interpolation,
        encode_quality=encode_quality,
        encode_format=encode_format,
        skip_reencode=skip_reencode,
        disable_all_reencoding=disable_all_reencoding,
        min_image_size=min_image_size,
        max_image_area=max_image_area,
        max_aspect_ratio=max_aspect_ratio,
    )

    if input_format == "txt":
        column_list = ["url"]
    elif input_format in ["json", "csv", "tsv", "tsv.gz", "parquet"]:
        column_list = save_additional_columns if save_additional_columns is not None else []
        if caption_col is not None:
            column_list = column_list + ["caption", "url"]
        else:
            column_list = column_list + ["url"]
    else:
        raise ValueError(f"Invalid input format {input_format}")

    disallowed_header_directives = (
        None
        if disallowed_header_directives is None
        else {directive.strip().lower() for directive in disallowed_header_directives}
    )
    user_agent_token = None if user_agent_token is None else user_agent_token.strip().lower()

    try:
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(input_file)
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
        if extract_exif:
            schema = schema.append(pa.field("exif", pa.string()))

        if compute_md5:
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
        semaphore = Semaphore(thread_count * 2)

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield e

        loader = data_generator()

        # give schema to writer
        sample_writer = sample_writer_class(
            output_file_prefix,
            save_caption,
            schema,
            encode_format,
        )
        with ThreadPool(thread_count) as thread_pool:
            for key, img_stream, error_message in thread_pool.imap_unordered(
                lambda x: download_image_with_retry(
                    x,
                    timeout=timeout,
                    retries=retries,
                    user_agent_token=user_agent_token,
                    disallowed_header_directives=disallowed_header_directives,
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
                    if extract_exif:
                        meta["exif"] = None
                    if compute_md5:
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

                    if extract_exif:
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

                    if compute_md5:
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
            output_file_prefix,
            count,
            successes,
            failed_to_download,
            failed_to_resize,
            start_time,
            end_time,
            status_dict,
        )
        if delete_input_shard:
            fs.rm(shard_path)
        return (True, (input_file, output_file_prefix))
    except Exception as err:  # pylint: disable=broad-except
        traceback.print_exc()
        print(f"shard {input_file} failed with error {err}")
        return (False, (input_file, output_file_prefix))


def main():
    fire.Fire(download_shard)


if __name__ == "__main__":
    main()
