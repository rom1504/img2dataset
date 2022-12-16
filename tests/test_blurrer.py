"""Tests for the bounding box blurring module."""

from img2dataset.blurrer import BoundingBoxBlurrer
import os
import pytest
import cv2
import numpy as np


@pytest.mark.parametrize("bbox_format", ["albumentations", "coco"])
def test_blurrer(bbox_format):
    """Test whether blurrer works properly."""
    current_folder = os.path.dirname(__file__)
    test_folder = os.path.join(current_folder, "blur_test_files")
    orig_image_path = os.path.join(test_folder, "original.png")
    blur_image_path = os.path.join(test_folder, "blurred.png")
    bbox_path = os.path.join(test_folder, f"{bbox_format}_bbox.npy")

    blurrer = BoundingBoxBlurrer(bbox_format=bbox_format)
    orig_image = cv2.imread(orig_image_path)
    blur_image = cv2.imread(blur_image_path)
    with open(bbox_path, "rb") as f:
        bbox = np.load(f)

    blur_image_test = blurrer(orig_image, bbox)

    assert np.array_equal(blur_image, blur_image_test)  # Also checks for shape
