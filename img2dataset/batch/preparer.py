"""Img2dataset"""

from typing import List, Optional
import fire
import logging
from ..core.reader import Reader
import fsspec
import os

# is this needed? maybe only for slurm

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


def preparer(
    url_list: str,
    output_folder: str = "images",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    number_sample_per_shard: int = 10000,
    save_additional_columns: Optional[List[str]] = None,
    incremental_mode: str = "incremental",
):
    """Prepare the dataset for downloading"""
    config_parameters = dict(locals())
    arguments_validator(config_parameters)

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)


    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")


    reader = Reader(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
    )

    reader.prepare(1)



def main():
    fire.Fire(preparer)


if __name__ == "__main__":
    main()
