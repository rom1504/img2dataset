import pandas as pd
import cv2
import glob
import random
import os
import sys
import gzip


def setup_fixtures(count=5, disallowed=0):
    test_list = []
    current_folder = os.path.dirname(__file__)
    test_folder = current_folder + "/" + "resize_test_image"
    port = f"123{sys.version_info.minor}"
    image_paths = glob.glob(test_folder + "/*")
    for i in range(count):
        item = random.randint(0, len(image_paths) - 1)
        test_list.append(
            (
                f"caption {i}" if i != 0 else "",
                image_paths[item].replace(test_folder, f"http://localhost:{port}/allowed"),
            )
        )
    test_list = test_list[:count]

    for i in range(disallowed):
        item = random.randint(0, len(image_paths) - 1)
        test_list.append(
            (
                f"caption {i}" if i != 0 else "",
                image_paths[item].replace(test_folder, f"http://localhost:{port}/disallowed"),
            )
        )
    test_list = test_list[: count + disallowed]

    return test_list


def generate_url_list_txt(output_file, test_list, compression_on=False):
    if compression_on:
        f = gzip.open(output_file, "wt")
    else:
        f = open(output_file, "w")
    with f:
        for _, url in test_list:
            f.write(url + "\n")


def generate_csv(output_file, test_list, compression=None):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file, compression=compression)


def generate_tsv(output_file, test_list, compression=None):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file, sep="\t", compression=compression)


def generate_json(output_file, test_list, compression=None):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_json(output_file, compression=compression)


def generate_jsonl(output_file, test_list, compression=None):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_json(output_file, orient="records", lines=True, compression=compression)


def generate_parquet(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_parquet(output_file)


def generate_input_file(input_format, url_list_name, test_list):
    if input_format == "txt":
        url_list_name += ".txt"
        generate_url_list_txt(url_list_name, test_list)
    elif input_format == "txt.gz":
        url_list_name += ".txt.gz"
        generate_url_list_txt(url_list_name, test_list, True)
    elif input_format == "csv":
        url_list_name += ".csv"
        generate_csv(url_list_name, test_list)
    elif input_format == "csv.gz":
        url_list_name += ".csv.gz"
        generate_csv(url_list_name, test_list, "gzip")
    elif input_format == "tsv":
        url_list_name += ".tsv"
        generate_tsv(url_list_name, test_list)
    elif input_format == "tsv.gz":
        url_list_name += ".tsv.gz"
        generate_tsv(url_list_name, test_list, "gzip")
    elif input_format == "json":
        url_list_name += ".json"
        generate_json(url_list_name, test_list)
    elif input_format == "json.gz":
        url_list_name += ".json.gz"
        generate_json(url_list_name, test_list, "gzip")
    elif input_format == "jsonl":
        url_list_name += ".jsonl"
        generate_jsonl(url_list_name, test_list)
    elif input_format == "jsonl.gz":
        url_list_name += ".jsonl.gz"
        generate_jsonl(url_list_name, test_list, "gzip")
    elif input_format == "parquet":
        url_list_name += ".parquet"
        generate_parquet(url_list_name, test_list)

    return url_list_name


def get_all_files(folder, ext):
    return sorted(list(glob.glob(folder + "/**/*." + ext, recursive=True)))


def check_one_image_size(img, img_unresized, image_size, resize_mode, resize_only_if_bigger):
    width = img.shape[1]
    height = img.shape[0]
    width_unresized = img_unresized.shape[1]
    height_unresized = img_unresized.shape[0]
    resized = True
    if resize_only_if_bigger:
        if (
            max(width_unresized, height_unresized) <= image_size
            and resize_mode in ["border", "fixed"]
            or min(width_unresized, height_unresized) <= image_size
            and resize_mode in ["keep_ratio", "center_crop"]
        ):
            if width_unresized != width or height_unresized != height:
                raise Exception(
                    f"Image size is not the same as the original one in resize only if bigger mode,"
                    f"expected={width_unresized}, {height_unresized} found={width}, {height}"
                )
            else:
                resized = False

    if not resized:
        return

    if resize_mode in ["border", "fixed"]:
        if width != image_size or height != image_size:
            raise Exception(f"Image size is not 256x256 in border mode found={width}x{height}")
    elif resize_mode == "keep_ratio":
        ratio = float(image_size) / min(width_unresized, height_unresized)
        new_size = tuple([round(x * ratio) for x in [width_unresized, height_unresized]])
        if new_size != (width, height):
            raise Exception(
                f"Image size is not of the right size in keep ratio mode"
                f"expected = {new_size[0]},  {new_size[1]} found = {width},  {height} "
            )


def check_image_size(file_list, l_unresized, image_size, resize_mode, resize_only_if_bigger):
    for file, file_unresized in zip(file_list, l_unresized):
        img = cv2.imread(file)
        img_unresized = cv2.imread(file_unresized)
        check_one_image_size(img, img_unresized, image_size, resize_mode, resize_only_if_bigger)
