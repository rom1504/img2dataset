"""Img2dataset worker, downloads one shard"""

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
from .downloader import Downloader
import fsspec
import os

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


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


def download_worker(
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
    number_sample_per_shard: int = 10000,
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

    downloader = Downloader(
        sample_writer_class=sample_writer_class,
        resizer=resizer,
        thread_count=thread_count,
        save_caption=save_caption,
        extract_exif=extract_exif,
        column_list=column_list,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        compute_md5=compute_md5,
        encode_format=encode_format,
        retries=retries,
        user_agent_token=user_agent_token,
        disallowed_header_directives=disallowed_header_directives,
        delete_input_shard=delete_input_shard,
    )

    return downloader((input_file, output_file_prefix))


def main():
    fire.Fire(download_worker)


if __name__ == "__main__":
    main()
