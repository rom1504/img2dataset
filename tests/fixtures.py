import pandas as pd
import cv2
import os
import glob
import random


def setup_fixtures(count=5):
    current_folder = os.path.dirname(__file__)
    test_folder = current_folder + "/" + "test_folder"
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    test_list = []
    while len(test_list) < count:
        for i in range(count):
            test_list.append(
                (
                    f"caption {i}" if i != 0 else None,
                    "https://picsum.photos/{}/{}".format(random.randint(200, 600), random.randint(200, 600)),
                )
            )
        test_list = list(set(test_list))
    test_list = test_list[:count]

    return test_folder, test_list, current_folder


def generate_url_list_txt(output_file, test_list):
    with open(output_file, "w") as f:
        for _, url in test_list:
            f.write(url + "\n")


def generate_csv(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file)


def generate_tsv(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file, sep="\t")


def generate_tsv_gz(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file, sep="\t", compression="gzip")


def generate_json(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_json(output_file)


def generate_parquet(output_file, test_list):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_parquet(output_file)


def generate_input_file(input_format, url_list_name, test_list):
    if input_format == "txt":
        url_list_name += ".txt"
        generate_url_list_txt(url_list_name, test_list)
    elif input_format == "csv":
        url_list_name += ".csv"
        generate_csv(url_list_name, test_list)
    elif input_format == "tsv":
        url_list_name += ".tsv"
        generate_tsv(url_list_name, test_list)
    elif input_format == "tsv.gz":
        url_list_name += ".tsv.gz"
        generate_tsv_gz(url_list_name, test_list)
    elif input_format == "json":
        url_list_name += ".json"
        generate_json(url_list_name, test_list)
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
            and resize_mode == "border"
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

    if resize_mode == "border":
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
