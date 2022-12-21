import shutil
import pytest
import json
from fixtures import setup_fixtures
from img2dataset.resizer import Resizer
from img2dataset.writer import FilesSampleWriter
from img2dataset.downloader import Downloader

import os
import pandas as pd


@pytest.mark.parametrize("compute_hash", ["md5", "sha256", "sha512"])
def test_valid_hash(compute_hash, tmp_path):
    test_folder = str(tmp_path)
    current_folder = os.path.dirname(__file__)
    input_file = os.path.join(current_folder, "test_files", "sample_image.txt")
    with open(input_file, "r") as file:
        test_list = pd.DataFrame([(url.rstrip(),) for url in file.readlines()], columns=["url"])

    image_folder_name = os.path.join(test_folder, "images")
    os.mkdir(image_folder_name)

    resizer = Resizer(256, "border", False)
    writer = FilesSampleWriter

    downloader = Downloader(
        writer,
        resizer,
        thread_count=32,
        save_caption=False,
        extract_exif=True,
        output_folder=image_folder_name,
        column_list=["url"],
        timeout=10,
        number_sample_per_shard=10,
        oom_shard_count=5,
        compute_hash=compute_hash,
        verify_hash_type=None,
        encode_format="jpg",
        retries=0,
        user_agent_token="img2dataset",
        disallowed_header_directives=["noai", "noindex"],
    )

    tmp_file = os.path.join(test_folder, "sample_image.feather")
    df = pd.DataFrame(test_list, columns=["url"])
    df.to_feather(tmp_file)

    downloader((0, tmp_file))

    df = pd.read_parquet(image_folder_name + "/00000.parquet")

    desired_output_file = os.path.join(current_folder, "test_files", "hashes.json")
    with open(desired_output_file, "r") as f:
        hashes_dict = json.load(f)

    assert df[compute_hash][0] == hashes_dict[compute_hash]


@pytest.mark.parametrize("compute_hash", ["md5", "sha256", "sha512"])
def test_unique_hash(compute_hash, tmp_path):
    current_folder = os.path.dirname(__file__)
    input_file = os.path.join(current_folder, "test_files", "unique_images.txt")
    with open(input_file, "r") as file:
        test_list = pd.DataFrame([(url.rstrip(),) for url in file.readlines()], columns=["url"])

    test_folder = str(tmp_path)

    image_folder_name = os.path.join(test_folder, "images")
    os.mkdir(image_folder_name)

    resizer = Resizer(256, "border", False)
    writer = FilesSampleWriter

    downloader = Downloader(
        writer,
        resizer,
        thread_count=32,
        save_caption=False,
        extract_exif=True,
        output_folder=image_folder_name,
        column_list=["url"],
        timeout=10,
        number_sample_per_shard=10,
        oom_shard_count=5,
        compute_hash=compute_hash,
        verify_hash_type=None,
        encode_format="jpg",
        retries=0,
        user_agent_token="img2dataset",
        disallowed_header_directives=["noai", "noindex"],
    )

    tmp_file = os.path.join(test_folder, "test_list.feather")
    df = pd.DataFrame(test_list, columns=["url"])
    df.to_feather(tmp_file)

    downloader((0, tmp_file))

    assert len(os.listdir(image_folder_name + "/00000")) >= 3 * 10

    df = pd.read_parquet(image_folder_name + "/00000.parquet")

    success = df[df[compute_hash].notnull()]

    assert len(success) > 10

    assert len(success) == len(success.drop_duplicates(compute_hash))


def test_downloader(tmp_path):
    test_folder = str(tmp_path)
    n_allowed = 5
    n_disallowed = 5
    test_list = setup_fixtures(count=n_allowed, disallowed=n_disallowed)

    assert len(test_list) == n_allowed + n_disallowed

    image_folder_name = os.path.join(test_folder, "images")

    os.mkdir(image_folder_name)

    resizer = Resizer(256, "border", False)
    writer = FilesSampleWriter

    downloader = Downloader(
        writer,
        resizer,
        thread_count=32,
        save_caption=True,
        extract_exif=True,
        output_folder=image_folder_name,
        column_list=["caption", "url"],
        timeout=10,
        number_sample_per_shard=10,
        oom_shard_count=5,
        compute_hash="md5",
        verify_hash_type="None",
        encode_format="jpg",
        retries=0,
        user_agent_token="img2dataset",
        disallowed_header_directives=["noai", "noindex"],
    )

    tmp_file = os.path.join(test_folder, "test_list.feather")
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_feather(tmp_file)

    downloader((0, tmp_file))

    assert len(os.listdir(image_folder_name + "/00000")) == 3 * n_allowed
