from img2dataset.writer import (
    FilesSampleWriter,
    WebDatasetSampleWriter,
    ParquetSampleWriter,
    DummySampleWriter,
    TFRecordSampleWriter,
)

import os
import glob
import pytest
import tarfile
import pandas as pd
import pyarrow as pa


@pytest.mark.parametrize("writer_type", ["files", "webdataset", "parquet", "dummy", "tfrecord"])
def test_writer(writer_type, tmp_path):
    current_folder = os.path.dirname(__file__)
    test_folder = str(tmp_path)
    input_folder = current_folder + "/" + "resize_test_image"
    output_folder = test_folder + "/" + "test_write"
    os.mkdir(output_folder)
    image_paths = glob.glob(input_folder + "/*")
    schema = pa.schema(
        [
            pa.field("key", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("status", pa.string()),
            pa.field("error_message", pa.string()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("original_width", pa.int32()),
            pa.field("original_height", pa.int32()),
            pa.field("labels", pa.list_(pa.int32())),
        ]
    )
    if writer_type == "files":
        writer_class = FilesSampleWriter
    elif writer_type == "webdataset":
        writer_class = WebDatasetSampleWriter
    elif writer_type == "parquet":
        writer_class = ParquetSampleWriter
    elif writer_type == "dummy":
        writer_class = DummySampleWriter
    elif writer_type == "tfrecord":
        writer_class = TFRecordSampleWriter

    writer = writer_class(0, output_folder, True, 5, schema, "jpg")

    for i, image_path in enumerate(image_paths):
        with open(image_path, "rb") as f:
            img_str = f.read()
            writer.write(
                img_str=img_str,
                key=str(i),
                caption=str(i),
                meta={
                    "key": str(i),
                    "caption": str(i),
                    "status": "ok",
                    "error_message": "",
                    "width": 100,
                    "height": 100,
                    "original_width": 100,
                    "original_height": 100,
                    "labels": [0, 100, 200],
                },
            )
    writer.close()

    if writer_type != "dummy":
        df = pd.read_parquet(output_folder + "/00000.parquet")

        expected_columns = [
            "key",
            "caption",
            "status",
            "error_message",
            "width",
            "height",
            "original_width",
            "original_height",
            "labels",
        ]

        if writer_type == "parquet":
            expected_columns.append("jpg")

        assert df.columns.tolist() == expected_columns

        assert df["key"].iloc[0] == "0"
        assert df["caption"].iloc[0] == "0"
        assert df["status"].iloc[0] == "ok"
        assert df["error_message"].iloc[0] == ""
        assert df["width"].iloc[0] == 100
        assert df["height"].iloc[0] == 100
        assert df["original_width"].iloc[0] == 100
        assert df["original_height"].iloc[0] == 100
        assert (df["labels"].iloc[0] == [0, 100, 200]).all()

    if writer_type == "files":
        saved_files = list(glob.glob(output_folder + "/00000/*"))
        assert len(saved_files) == 3 * len(image_paths)
    elif writer_type == "webdataset":
        l = glob.glob(output_folder + "/*.tar")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")

        assert len(tarfile.open(output_folder + "/00000.tar").getnames()) == len(image_paths) * 3
    elif writer_type == "parquet":
        l = glob.glob(output_folder + "/*.parquet")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.parquet":
            raise Exception(l[0] + " is not 00000.parquet")

        assert len(df.index) == len(image_paths)
    elif writer_type == "dummy":
        l = glob.glob(output_folder + "/*")
        assert len(l) == 0
    elif writer_type == "tfrecord":
        l = glob.glob(output_folder + "/*.tfrecord")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.tfrecord":
            raise Exception(l[0] + " is not 00000.tfrecord")
