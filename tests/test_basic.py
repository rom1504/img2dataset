from img2dataset import download
import os
import shutil
import cv2
import pytest
import glob
import time
import pandas as pd
import tarfile


def test_basic():
    print("it works !")


current_folder = os.path.dirname(__file__)
test_folder = current_folder + "/" + "test_folder"
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

test_list = [
    ["first", "https://placekitten.com/400/600"],
    ["second", "https://placekitten.com/200/300"],
    ["third", "https://placekitten.com/300/200"],
    ["fourth", "https://placekitten.com/400/400"],
    ["fifth", "https://placekitten.com/200/200"],
    [None, "https://placekitten.com/200/200"],
]


def generate_url_list_txt(output_file):
    with open(output_file, "w") as f:
        for _, url in test_list:
            f.write(url + "\n")


def generate_csv(output_file):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file)


def generate_tsv(output_file):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_csv(output_file, sep="\t")


def generate_parquet(output_file):
    df = pd.DataFrame(test_list, columns=["caption", "url"])
    df.to_parquet(output_file)


def get_all_files(folder, ext):
    return sorted(list(glob.glob(folder + "/**/*." + ext, recursive=True)))


def check_image_size(file_list, l_unresized, image_size, resize_mode, resize_only_if_bigger):
    for file, file_unresized in zip(file_list, l_unresized):
        img = cv2.imread(file)
        img_unresized = cv2.imread(file_unresized)
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
            continue

        if resize_mode == "border":
            if width != image_size or height != image_size:
                raise Exception(f"Image size is not 256x256 in border mode found={width}x{height}")
        elif resize_mode == "keep_ratio":
            ratio = float(image_size) / min(width_unresized, height_unresized)
            new_size = tuple([int(x * ratio) for x in [width_unresized, height_unresized]])
            if new_size != (width, height):
                raise Exception(
                    f"Image size is not of the right size in keep ratio mode"
                    f"expected = {new_size[0]},  {new_size[0]} found = {width},  {height} "
                )


testdata = [
    ("border", False),
    ("border", True),
    ("keep_ratio", True),
    ("keep_ratio", False),
    ("center_crop", True),
    ("center_crop", False),
]


@pytest.mark.parametrize("resize_mode, resize_only_if_bigger", testdata)
def test_download_resize(resize_mode, resize_only_if_bigger):
    prefix = resize_mode + "_" + str(resize_only_if_bigger) + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list.txt")
    image_folder_name = os.path.join(test_folder, prefix + "images")
    unresized_folder = os.path.join(test_folder, prefix + "unresized_images")

    generate_url_list_txt(url_list_name)

    download(
        url_list_name,
        image_size=256,
        output_folder=unresized_folder,
        thread_count=32,
        resize_mode="no",
        resize_only_if_bigger=resize_only_if_bigger,
    )

    download(
        url_list_name,
        image_size=256,
        output_folder=image_folder_name,
        thread_count=32,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
    )

    l = get_all_files(image_folder_name, "jpg")
    j = get_all_files(image_folder_name, "json")
    assert len(j) == len(test_list)
    p = get_all_files(image_folder_name, "parquet")
    assert len(p) == 1
    l_unresized = get_all_files(unresized_folder, "jpg")
    assert len(l) == len(test_list)
    check_image_size(l, l_unresized, 256, resize_mode, resize_only_if_bigger)

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)
    shutil.rmtree(unresized_folder)


@pytest.mark.parametrize(
    "input_format, output_format",
    [
        ["txt", "files"],
        ["txt", "webdataset"],
        ["csv", "files"],
        ["csv", "webdataset"],
        ["tsv", "files"],
        ["tsv", "webdataset"],
        ["parquet", "files"],
        ["parquet", "webdataset"],
    ],
)
def test_download_input_format(input_format, output_format):
    prefix = input_format + "_" + output_format + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list.txt")
    image_folder_name = os.path.join(test_folder, prefix + "images")

    if input_format == "txt":
        generate_url_list_txt(url_list_name)
    elif input_format == "csv":
        generate_csv(url_list_name)
    elif input_format == "tsv":
        generate_tsv(url_list_name)
    elif input_format == "parquet":
        generate_parquet(url_list_name)

    download(
        url_list_name,
        image_size=256,
        output_folder=image_folder_name,
        thread_count=32,
        input_format=input_format,
        output_format=output_format,
        url_col="url",
        caption_col="caption",
    )

    expected_file_count = len(test_list)
    if output_format == "files":
        l = get_all_files(image_folder_name, "jpg")
        assert len(l) == expected_file_count
    elif output_format == "webdataset":
        l = glob.glob(image_folder_name + "/*.tar")
        assert len(l) == 1
        if l[0] != image_folder_name + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")

        assert (
            len([x for x in tarfile.open(image_folder_name + "/00000.tar").getnames() if x.endswith(".jpg")])
            == expected_file_count
        )

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)


@pytest.mark.parametrize(
    "save_caption, output_format", [[True, "files"], [False, "files"], [True, "webdataset"], [False, "webdataset"],],
)
def test_captions_saving(save_caption, output_format):
    input_format = "parquet"
    prefix = str(save_caption) + "_" + input_format + "_" + output_format + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list.txt")
    image_folder_name = os.path.join(test_folder, prefix + "images")
    generate_parquet(url_list_name)
    download(
        url_list_name,
        image_size=256,
        output_folder=image_folder_name,
        thread_count=32,
        input_format=input_format,
        output_format=output_format,
        url_col="url",
        caption_col="caption" if save_caption else None,
    )

    expected_file_count = len(test_list)
    if output_format == "files":
        l = get_all_files(image_folder_name, "jpg")
        assert len(l) == expected_file_count
        l = get_all_files(image_folder_name, "txt")
        if save_caption:
            assert len(l) == expected_file_count
            for expected, real in zip(test_list, l):
                true_real = open(real).read()
                true_expected = expected[0] if expected[0] is not None else ""
                assert true_expected == true_real
        else:
            assert len(l) == 0
    elif output_format == "webdataset":
        l = glob.glob(image_folder_name + "/*.tar")
        assert len(l) == 1
        if l[0] != image_folder_name + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")

        with tarfile.open(image_folder_name + "/00000.tar") as f:
            assert len([x for x in f.getnames() if x.endswith(".jpg")]) == expected_file_count
            txt_files = sorted([x for x in f.getnames() if x.endswith(".txt")])
            if save_caption:
                assert len(txt_files) == expected_file_count
                for expected, real in zip(test_list, txt_files):
                    true_expected = expected[0] if expected[0] is not None else ""
                    true_real = f.extractfile(real).read().decode("utf-8")
                    assert true_expected == true_real
            else:
                assert len(txt_files) == 0

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)


def test_webdataset():
    url_list_name = os.path.join(test_folder, "url_list.txt")
    image_folder_name = os.path.join(test_folder, "images")

    generate_url_list_txt(url_list_name)

    download(
        url_list_name, image_size=256, output_folder=image_folder_name, thread_count=32, output_format="webdataset"
    )

    l = glob.glob(image_folder_name + "/*.tar")
    assert len(l) == 1
    if l[0] != image_folder_name + "/00000.tar":
        raise Exception(l[0] + " is not 00000.tar")

    assert len(tarfile.open(image_folder_name + "/00000.tar").getnames()) == len(test_list) * 2

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)


@pytest.mark.skip(reason="slow")
@pytest.mark.parametrize("output_format", ["webdataset", "files"])
def test_benchmark(output_format):
    prefix = output_format + "_"
    url_list_name = os.path.join(current_folder, "test_1000.parquet")
    image_folder_name = os.path.join(test_folder, prefix + "images")

    t = time.time()

    download(
        url_list_name,
        image_size=256,
        output_folder=image_folder_name,
        thread_count=32,
        output_format=output_format,
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
    )

    took = time.time() - t

    print("Took " + str(took) + "s")

    if took > 100:
        raise Exception("Very slow, took " + str(took))

    shutil.rmtree(image_folder_name)
