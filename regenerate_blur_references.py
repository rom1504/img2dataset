#!/usr/bin/env python3
"""Script to regenerate reference images for blur+resize tests"""

import os
import cv2
import tempfile
import shutil
from img2dataset.main import download


def regenerate_reference_images():
    """Regenerate all reference images for blur+resize tests"""
    current_folder = os.path.dirname(__file__)
    test_folder = os.path.join(current_folder, "tests", "blur_test_files")
    input_parquet = os.path.join(test_folder, "test_bbox.parquet")

    resize_modes = ["no", "border", "keep_ratio", "keep_ratio_largest", "center_crop"]

    print("Regenerating reference images for blur+resize tests...")

    for resize_mode in resize_modes:
        print(f"Processing resize_mode: {resize_mode}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_folder = os.path.join(tmp_dir, "images")

            # Download with current blur implementation
            download(
                input_parquet,
                input_format="parquet",
                image_size=600,
                output_folder=output_folder,
                output_format="files",
                thread_count=32,
                resize_mode=resize_mode,
                resize_only_if_bigger=False,
                bbox_col="bboxes",
            )

            # Find the output image
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file.endswith(".jpg"):
                        output_img_path = os.path.join(root, file)

                        # Copy to reference location
                        reference_path = os.path.join(test_folder, f"resize_{resize_mode}.jpg")
                        shutil.copy2(output_img_path, reference_path)
                        print(f"Updated: {reference_path}")
                        break
                else:
                    continue
                break


if __name__ == "__main__":
    regenerate_reference_images()
    print("All reference images updated!")
