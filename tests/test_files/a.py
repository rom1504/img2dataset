import sys

import pydantic
from pydantic import BaseModel
import fire

from typing import List, Optional

class ResizingOptions(BaseModel):
    disable_all_reencoding: bool = False,
    image_size: int = 256,
    resize_mode: str = "border"
    resize_only_if_bigger: bool = False
    upscale_interpolation: str = "lanczos"
    downscale_interpolation: str = "area"
    encode_quality: int = 95
    encode_format: str = "jpg"
    skip_reencode: bool = False
    min_image_size: int = 0
    max_image_area: float = float("inf")
    max_aspect_ratio: float = float("inf")

class WriterOptions(BaseModel):
    output_folder: str = "images"
    output_format: str = "files"
    oom_shard_count: int = 7

class ReaderOptions(BaseModel):
    url_list: str
    input_format: str = "txt"
    url_col: str = "url"
    caption_col: Optional[str] = None
    save_additional_columns: Optional[List[str]] = None
    incremental_mode: str = "incremental"

class DownloaderOptions(BaseModel):
    thread_count: int = 256
    extract_exif: bool = True
    timeout: int = 10
    compute_md5: bool = True
    retries: int = 0
    user_agent_token: Optional[str] = None
    disallowed_header_directives: Optional[List[str]] = None
    delete_input_shard: bool = True

class DistributorOptions(BaseModel):
    distributor: str = "multiprocessing"
    subjob_size: int = 1000
    max_shard_retry: int = 1

class LoggerOptions(BaseModel):
    enable_wandb: bool = False
    wandb_project: str = "img2dataset"


class DownloaderWorkerSpecificOptions(BaseModel):
    input_file: str
    output_file_prefix: str
    


DownloaderWorkerOptions = pydantic.create_model("DownloaderWorkerOptions", __base__=(
    WriterOptions,
    DownloaderOptions,
    ResizingOptions,
    DownloaderWorkerSpecificOptions,))

MainOptions = pydantic.create_model("MainOptions", __base__=(
    ResizingOptions,
    WriterOptions,
    DownloaderOptions,
    DistributorOptions,
    LoggerOptions,
    ReaderOptions,))


def example_runner(**opts) -> int:
    opts = MainOptions(**opts)
    print(f"Mock example running with options {opts}")
    return 0

if __name__ == '__main__':
    # to_runner will return a function that takes the args list to run and 
    # will return an integer exit code

    fire.Fire(example_runner)
