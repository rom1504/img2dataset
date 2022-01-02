from img2dataset import download
import os
import shutil
import pytest
import glob
import time
import tarfile
from fixtures import (
    get_all_files,
    check_image_size,
    generate_input_file,
    setup_fixtures,
)

testdata = [
    ("border", False, False),
    ("border", False, True),
    ("border", True, False),
    ("keep_ratio", False, False),
    ("keep_ratio", True, False),
    ("keep_ratio", True, True),
    ("center_crop", False, False),
    ("center_crop", True, False),
    ("no", False, False),
    ("no", False, True),
]


@pytest.mark.parametrize("image_size", [256, 512])
@pytest.mark.parametrize("resize_mode, resize_only_if_bigger, skip_reencode", testdata)
def test_download_resize(image_size, resize_mode, resize_only_if_bigger, skip_reencode):
    test_folder, test_list, _ = setup_fixtures()
    prefix = resize_mode + "_" + str(resize_only_if_bigger) + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list")
    image_folder_name = os.path.join(test_folder, prefix + "images")
    unresized_folder = os.path.join(test_folder, prefix + "unresized_images")

    url_list_name = generate_input_file("txt", url_list_name, test_list)

    download(
        url_list_name,
        image_size=image_size,
        output_folder=unresized_folder,
        thread_count=32,
        resize_mode="no",
        resize_only_if_bigger=resize_only_if_bigger,
    )

    download(
        url_list_name,
        image_size=image_size,
        output_folder=image_folder_name,
        thread_count=32,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
        skip_reencode=skip_reencode,
    )

    l = get_all_files(image_folder_name, "jpg")
    j = get_all_files(image_folder_name, "json")
    assert len(j) == len(test_list)
    p = get_all_files(image_folder_name, "parquet")
    assert len(p) == 1
    l_unresized = get_all_files(unresized_folder, "jpg")
    assert len(l) == len(test_list)
    check_image_size(l, l_unresized, image_size, resize_mode, resize_only_if_bigger)

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
        ["tsv.gz", "files"],
        ["tsv.gz", "webdataset"],
        ["json", "files"],
        ["json", "webdataset"],
        ["parquet", "files"],
        ["parquet", "webdataset"],
    ],
)
def test_download_input_format(input_format, output_format):
    test_folder, test_list, _ = setup_fixtures()

    prefix = input_format + "_" + output_format + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list")
    image_folder_name = os.path.join(test_folder, prefix + "images")

    url_list_name = generate_input_file(input_format, url_list_name, test_list)

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
    "input_format, output_format",
    [
        ["txt", "files"],
        ["txt", "webdataset"],
        ["csv", "files"],
        ["csv", "webdataset"],
        ["tsv", "files"],
        ["tsv", "webdataset"],
        ["tsv.gz", "files"],
        ["tsv.gz", "webdataset"],
        ["json", "files"],
        ["json", "webdataset"],
        ["parquet", "files"],
        ["parquet", "webdataset"],
    ],
)
def test_download_multiple_input_files(input_format, output_format):
    test_folder, test_list, _ = setup_fixtures()
    prefix = input_format + "_" + output_format + "_"

    subfolder = test_folder + "/" + prefix + "input_folder"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)
    url_list_names = [os.path.join(subfolder, prefix + "url_list1"), os.path.join(subfolder, prefix + "url_list2")]
    image_folder_name = os.path.join(test_folder, prefix + "images")

    for url_list_name in url_list_names:
        url_list_name = generate_input_file(input_format, url_list_name, test_list)

    download(
        subfolder,
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
        assert len(l) == expected_file_count * 2
    elif output_format == "webdataset":
        l = sorted(glob.glob(image_folder_name + "/*.tar"))
        assert len(l) == 2
        if l[0] != image_folder_name + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")
        if l[1] != image_folder_name + "/00001.tar":
            raise Exception(l[1] + " is not 00001.tar")

        assert (
            len([x for x in tarfile.open(image_folder_name + "/00000.tar").getnames() if x.endswith(".jpg")])
            == expected_file_count
        )
        assert (
            len([x for x in tarfile.open(image_folder_name + "/00001.tar").getnames() if x.endswith(".jpg")])
            == expected_file_count
        )

    shutil.rmtree(subfolder)
    shutil.rmtree(image_folder_name)


@pytest.mark.parametrize(
    "save_caption, output_format", [[True, "files"], [False, "files"], [True, "webdataset"], [False, "webdataset"],],
)
def test_captions_saving(save_caption, output_format):
    test_folder, test_list, _ = setup_fixtures()
    input_format = "parquet"
    prefix = str(save_caption) + "_" + input_format + "_" + output_format + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list")
    image_folder_name = os.path.join(test_folder, prefix + "images")
    url_list_name = generate_input_file("parquet", url_list_name, test_list)
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
    test_folder, test_list, _ = setup_fixtures()
    url_list_name = os.path.join(test_folder, "url_list")
    image_folder_name = os.path.join(test_folder, "images")

    url_list_name = generate_input_file("txt", url_list_name, test_list)

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


# pytest.mark.skip(reason="slow")
@pytest.mark.parametrize("output_format", ["webdataset", "files"])
def test_benchmark(output_format):
    test_folder, _, current_folder = setup_fixtures()
    prefix = output_format + "_"
    url_list_name = os.path.join(current_folder, "test_files/test_1000.parquet")
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
