"""blurrer module to blur parts of the image"""

import numpy as np
import random

import albumentations as A


class BoundingBoxBlurrer:
    """blur images based on a bounding box.

    The bounding box used is assumed to have format [x_min, y_min, x_max, y_max]
    (with elements being floats in [0,1], relative to the original shape of the
    image).
    """

    def __init__(self) -> None:
        pass

    def __call__(self, img, bbox_list):
        """Apply blurring to bboxes of an image.

        Args:
            img: The image to blur.
            bbox_list: The list of bboxes to blur.

        Returns:
            The image with bboxes blurred.
        """

        # Skip if there are no boxes to blur.
        if len(bbox_list) == 0:
            return img

        height, width = img.shape[:2]

        # Convert to float temporarily
        img = img.astype(np.float32) / 255.0

        mask = np.zeros_like(img)

        # Incorporate max diagonal from ImageNet code.
        max_diagonal = 0

        for bbox in bbox_list:
            adjusted_bbox = [
                int(bbox[0] * width),
                int(bbox[1] * height),
                int(bbox[2] * width),
                int(bbox[3] * height),
            ]

            diagonal = max(adjusted_bbox[2] - adjusted_bbox[0], adjusted_bbox[3] - adjusted_bbox[1])
            max_diagonal = max(max_diagonal, diagonal)

            # Adjusting bbox as in:
            # https://github.com/princetonvisualai/imagenet-face-obfuscation
            adjusted_bbox[0] = int(adjusted_bbox[0] - 0.1 * diagonal)
            adjusted_bbox[1] = int(adjusted_bbox[1] - 0.1 * diagonal)
            adjusted_bbox[2] = int(adjusted_bbox[2] + 0.1 * diagonal)
            adjusted_bbox[3] = int(adjusted_bbox[3] + 0.1 * diagonal)

            # Clipping for indexing.
            adjusted_bbox[0] = np.clip(adjusted_bbox[0], 0, width - 1)
            adjusted_bbox[1] = np.clip(adjusted_bbox[1], 0, height - 1)
            adjusted_bbox[2] = np.clip(adjusted_bbox[2], 0, width - 1)
            adjusted_bbox[3] = np.clip(adjusted_bbox[3], 0, height - 1)

            mask[adjusted_bbox[1] : adjusted_bbox[3], adjusted_bbox[0] : adjusted_bbox[2], ...] = 1

        sigma = 0.1 * max_diagonal
        # Use GaussianBlur transform instead of deprecated gaussian_blur function
        # blur_limit needs to be an odd integer, so convert sigma to appropriate kernel size
        kernel_size = max(3, int(2 * np.ceil(sigma) + 1))
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1

        # Set fixed seed for deterministic results
        np.random.seed(42)
        random.seed(42)

        # Use tuple format (min, max) with same value for exact kernel size
        blur_transform = A.GaussianBlur(blur_limit=(kernel_size, kernel_size), p=1.0, always_apply=True)
        blurred_img = blur_transform(image=img)["image"]
        blurred_mask = blur_transform(image=mask)["image"]

        result = img * (1 - blurred_mask) + blurred_img * blurred_mask

        # Convert back to uint8
        result = (result * 255.0).astype(np.uint8)

        return result
