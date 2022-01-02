"""resizer module handle image resizing"""

import albumentations as A
import cv2
import numpy as np
from enum import Enum
import imghdr

_INTER_STR_TO_CV2 = dict(
    nearest=cv2.INTER_NEAREST,
    linear=cv2.INTER_LINEAR,
    bilinear=cv2.INTER_LINEAR,
    cubic=cv2.INTER_CUBIC,
    bicubic=cv2.INTER_CUBIC,
    area=cv2.INTER_AREA,
    lanczos=cv2.INTER_LANCZOS4,
    lanczos4=cv2.INTER_LANCZOS4,
)


class ResizeMode(Enum):
    no = 0
    keep_ratio = 1
    center_crop = 2
    border = 3


def inter_str_to_cv2(inter_str):
    inter_str = inter_str.lower()
    if inter_str not in _INTER_STR_TO_CV2:
        raise Exception(f"Invalid option for interpolation: {inter_str}")
    return _INTER_STR_TO_CV2[inter_str]


class Resizer:
    """
    Resize images
    Expose a __call__ method to be used as a callable object

    Should be used to resize one image at a time

    Options:
        resize_mode: "no", "keep_ratio", "center_crop", "border"
        resize_only_if_bigger: if True, resize only if image is bigger than image_size
        image_size: size of the output image to resize
    """

    def __init__(
        self,
        image_size,
        resize_mode,
        resize_only_if_bigger,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        skip_reencode=False,
    ):
        self.image_size = image_size
        if isinstance(resize_mode, str):
            if resize_mode not in ResizeMode.__members__:  # pylint: disable=unsupported-membership-test
                raise Exception(f"Invalid option for resize_mode: {resize_mode}")
            resize_mode = ResizeMode[resize_mode]
        self.resize_mode = resize_mode
        self.resize_only_if_bigger = resize_only_if_bigger
        self.upscale_interpolation = inter_str_to_cv2(upscale_interpolation)
        self.downscale_interpolation = inter_str_to_cv2(downscale_interpolation)
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality]
        self.skip_reencode = skip_reencode

    def __call__(self, img_stream):
        """
        input: an image stream
        output: img_str, width, height, original_width, original_height, err
        """
        try:
            encode_needed = imghdr.what(img_stream) != "jpeg" if self.skip_reencode else True
            img_buf = np.frombuffer(img_stream.read(), np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception("Image decoding error")
            if len(img.shape) == 3 and img.shape[-1] == 4:
                # alpha matting with white background
                alpha = img[:, :, 3, np.newaxis]
                img = alpha / 255 * img[..., :3] + 255 - alpha
                img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
                encode_needed = True
            original_height, original_width = img.shape[:2]

            # resizing in following conditions
            if self.resize_mode in (ResizeMode.keep_ratio, ResizeMode.center_crop):
                downscale = min(original_width, original_height) > self.image_size
                if not self.resize_only_if_bigger or downscale:
                    interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                    img = A.smallest_max_size(img, self.image_size, interpolation=interpolation)
                    if self.resize_mode == ResizeMode.center_crop:
                        img = A.center_crop(img, self.image_size, self.image_size)
                    encode_needed = True
            elif self.resize_mode == ResizeMode.border:
                downscale = max(original_width, original_height) > self.image_size
                if not self.resize_only_if_bigger or downscale:
                    interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                    img = A.longest_max_size(img, self.image_size, interpolation=interpolation)
                    img = A.pad(
                        img, self.image_size, self.image_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                    encode_needed = True
            height, width = img.shape[:2]
            if encode_needed:
                img_str = cv2.imencode(".jpg", img, params=self.encode_params)[1].tobytes()
            else:
                img_str = img_buf.tobytes()
            return img_str, width, height, original_width, original_height, None

        except Exception as err:  # pylint: disable=broad-except
            return None, None, None, None, None, str(err)
