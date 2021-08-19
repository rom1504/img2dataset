from img2dataset import download
import os
import shutil
import cv2
import pytest
import glob
import time

def test_basic():
    print("it works !")

def generate_url_list_txt(output_file):
    with open(output_file, "w") as f:
        # in python
        f.write("https://placekitten.com/400/600\n")
        f.write("https://placekitten.com/200/300\n")
        f.write("https://placekitten.com/300/200\n")
        f.write("https://placekitten.com/400/400\n")
        f.write("https://placekitten.com/200/200\n")

def get_all_files(folder):
    l = []
    for root, _, files in os.walk(folder):
        for file in files:
            l.append(os.path.join(root, file))
    return l

def check_image_size(file_list,l_unresized, image_size, resize_mode, resize_only_if_bigger):
    for file, file_unresized in zip(file_list, l_unresized):
        img = cv2.imread(file)
        img_unresized = cv2.imread(file_unresized)
        width = img.shape[1]
        height = img.shape[0]
        width_unresized = img_unresized.shape[1]
        height_unresized = img_unresized.shape[0]
        resized = True
        if resize_only_if_bigger:
            if width_unresized <= image_size and height_unresized <= image_size:
                if width_unresized != width or height_unresized != height:
                    raise Exception(f"Image size is not the same as the original one in resize only if bigger mode,"
                    f"expected={width_unresized}, {height_unresized} found={width}, {height}")
                else:
                    resized = False

        if not resized:
            continue

        if resize_mode == "border":
            if width != image_size or height != image_size:
                raise Exception(f"Image size is not 256x256 in border mode found={width}x{height}")
        elif resize_mode == "keep_ratio":
            ratio = float(image_size)/min(width_unresized, height_unresized)
            new_size = tuple([int(x*ratio) for x in [width_unresized, height_unresized]])
            if new_size != (width, height):
                raise Exception(f"Image size is not of the right size in keep ratio mode"
                f"expected = {new_size[0]},  {new_size[0]} found = {width},  {height} ")


testdata = [
    ("border", False),
    ("border", True),
    ("keep_ratio", True),
    ("keep_ratio", False),
]


@pytest.mark.parametrize("resize_mode, resize_only_if_bigger", testdata)
def test_download(resize_mode, resize_only_if_bigger):
    current_folder = os.path.dirname(__file__)

    prefix = resize_mode +"_" + str(resize_only_if_bigger) + "_"
    url_list_name = os.path.join(current_folder, prefix + "url_list.txt")
    image_folder_name = os.path.join(current_folder, prefix + "images")
    unresized_folder = os.path.join(current_folder, prefix + "unresized_images")

    generate_url_list_txt(url_list_name)

    download(url_list_name, image_size=256, output_folder=unresized_folder,\
         thread_count=32, resize_mode="no", resize_only_if_bigger=resize_only_if_bigger)

    download(url_list_name, image_size=256, output_folder=image_folder_name,\
         thread_count=32, resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger)

    l = glob.glob("")
    l = get_all_files(image_folder_name)
    l_unresized = get_all_files(unresized_folder)
    assert(len(l) == 5)
    check_image_size(l, l_unresized, 256, resize_mode, resize_only_if_bigger)

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)
    shutil.rmtree(unresized_folder)


def test_webdataset():
    current_folder = os.path.dirname(__file__)

    url_list_name = os.path.join(current_folder, "url_list.txt")
    image_folder_name = os.path.join(current_folder, "images")

    generate_url_list_txt(url_list_name)

    download(url_list_name, image_size=256, output_folder=image_folder_name, thread_count=32, output_format="webdataset")

    l = glob.glob(image_folder_name+"/*")
    assert(len(l) == 1)
    if l[0] != image_folder_name+"/000000.tar":
        raise Exception(l[0] + "is not 000000.tar")

    os.remove(url_list_name)
    shutil.rmtree(image_folder_name)

@pytest.mark.parametrize("output_format", ["webdataset", "files"])
def test_benchmark(output_format):
    current_folder = os.path.dirname(__file__)

    prefix = output_format + "_"
    url_list_name = os.path.join(current_folder, "test1000.txt")
    image_folder_name = os.path.join(current_folder, prefix+"images")

    t = time.time()

    download(url_list_name, image_size=256, output_folder=image_folder_name, thread_count=32, output_format=output_format)

    took = time.time() - t

    print("Took "+ str(took)+"s")

    if took > 100:
        raise Exception("Very slow, took "+str(took))

    shutil.rmtree(image_folder_name)