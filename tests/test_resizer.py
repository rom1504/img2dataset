from img2dataset.resizer import Resizer
import os
import glob
import pytest
from fixtures import check_one_image_size
import io
import cv2
import numpy as np

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

testformat = [
    (95, "jpg"),
    (95, "webp"),
    (9, "png"),
]


@pytest.mark.parametrize("image_size", [256, 512])
@pytest.mark.parametrize("resize_mode, resize_only_if_bigger, skip_reencode", testdata)
@pytest.mark.parametrize("encode_quality, encode_format", testformat)
def test_resizer(image_size, resize_mode, resize_only_if_bigger, skip_reencode, encode_quality, encode_format):
    current_folder = os.path.dirname(__file__)
    test_folder = current_folder + "/" + "resize_test_image"
    image_paths = glob.glob(test_folder + "/*")
    resizer = Resizer(
        image_size,
        resize_mode,
        resize_only_if_bigger,
        encode_quality=encode_quality,
        encode_format=encode_format,
        skip_reencode=skip_reencode,
    )
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            img = f.read()
            image_original_stream = io.BytesIO(img)
        image_resized_str, width, height, original_width, original_height, err = resizer(image_original_stream)
        assert err is None
        image_original_stream = io.BytesIO(img)
        image_original = cv2.imdecode(np.frombuffer(image_original_stream.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image_resized = cv2.imdecode(np.frombuffer(image_resized_str, np.uint8), cv2.IMREAD_UNCHANGED)
        width_resized = image_resized.shape[1]
        height_resized = image_resized.shape[0]
        width_original = image_original.shape[1]
        height_original = image_original.shape[0]
        assert width_resized == width
        assert height_resized == height
        assert width_original == original_width
        assert height_original == original_height
        check_one_image_size(image_resized, image_original, image_size, resize_mode, resize_only_if_bigger)
