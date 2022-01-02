from img2dataset.writer import FilesSampleWriter, WebDatasetSampleWriter

import os
import glob
import shutil
import pytest
import tarfile


@pytest.mark.parametrize("writer_type", ["files", "webdataset"])
def test_writer(writer_type):
    current_folder = os.path.dirname(__file__)
    input_folder = current_folder + "/" + "resize_test_image"
    output_folder = current_folder + "/" + "test_write"
    os.mkdir(output_folder)
    image_paths = glob.glob(input_folder + "/*")
    if writer_type == "files":
        writer = FilesSampleWriter(0, output_folder, True, True, 5)
    elif writer_type == "webdataset":
        writer = WebDatasetSampleWriter(0, output_folder, True, True, 5)
    for i, image_path in enumerate(image_paths):
        with open(image_path, "rb") as f:
            img_str = f.read()
            writer.write(img_str=img_str, key=str(i), caption=str(i), meta={"caption": str(i)})
    writer.close()

    if writer_type == "files":
        saved_files = list(glob.glob(output_folder + "/00000/*"))
        assert len(saved_files) == 3 * len(image_paths)
    elif writer_type == "webdataset":
        l = glob.glob(output_folder + "/*.tar")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")

        assert len(tarfile.open(output_folder + "/00000.tar").getnames()) == len(image_paths) * 3

    shutil.rmtree(output_folder)
