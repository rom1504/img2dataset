"""resizer module handle image resizing"""

import albumentations as A
import cv2
import numpy as np
from enum import Enum
import imghdr
import os

_INTER_STR_TO_CV2 = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "lanczos4": cv2.INTER_LANCZOS4,
}


class ResizeMode(Enum):
    no = 0  # pylint: disable=invalid-name
    keep_ratio = 1  # pylint: disable=invalid-name
    center_crop = 2  # pylint: disable=invalid-name
    border = 3  # pylint: disable=invalid-name
    keep_ratio_largest = 4  # pylint: disable=invalid-name


# thanks https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class SuppressStdoutStderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def inter_str_to_cv2(inter_str):
    inter_str = inter_str.lower()
    if inter_str not in _INTER_STR_TO_CV2:
        raise ValueError(f"Invalid option for interpolation: {inter_str}")
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
        encode_format="jpg",
        skip_reencode=False,
        disable_all_reencoding=False,
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
        blurrer=None,
    ):
        if encode_format not in ["jpg", "png", "webp"]:
            raise ValueError(f"Invalid encode format {encode_format}")
        if encode_format == "png":
            if encode_quality < 0 or encode_quality > 9:
                raise ValueError(
                    "For png, encode quality represents compression which"
                    f"must be between 0 and 9, got {encode_quality}"
                )

        self.image_size = image_size
        if isinstance(resize_mode, str):
            if resize_mode not in ResizeMode.__members__:  # pylint: disable=unsupported-membership-test
                raise ValueError(f"Invalid option for resize_mode: {resize_mode}")
            resize_mode = ResizeMode[resize_mode]
        self.resize_mode = resize_mode
        self.resize_only_if_bigger = resize_only_if_bigger
        self.upscale_interpolation = inter_str_to_cv2(upscale_interpolation)
        self.downscale_interpolation = inter_str_to_cv2(downscale_interpolation)
        self.encode_format = encode_format
        cv2_img_quality = None
        if encode_format == "jpg":
            cv2_img_quality = int(cv2.IMWRITE_JPEG_QUALITY)
            self.what_ext = "jpeg"
        elif encode_format == "png":
            cv2_img_quality = int(cv2.IMWRITE_PNG_COMPRESSION)
            self.what_ext = "png"
        elif encode_format == "webp":
            cv2_img_quality = int(cv2.IMWRITE_WEBP_QUALITY)
            self.what_ext = "webp"
        if cv2_img_quality is None:
            raise ValueError(f"Invalid option for encode_format: {encode_format}")
        self.encode_params = [cv2_img_quality, encode_quality]
        self.skip_reencode = skip_reencode
        self.disable_all_reencoding = disable_all_reencoding
        self.min_image_size = min_image_size
        self.max_image_area = max_image_area
        self.max_aspect_ratio = max_aspect_ratio
        self.blurrer = blurrer

    def __call__(self, img_stream, blurring_bbox_list=None):
        """
        input: an image stream, optionally a list of bounding boxes to blur.
        output: img_str, width, height, original_width, original_height, err
        """
        try:
            if self.disable_all_reencoding:
                return img_stream.read(), None, None, None, None, None
            with SuppressStdoutStderr():
                cv2.setNumThreads(1)
                img_stream.seek(0)
                encode_needed = imghdr.what(img_stream) != self.what_ext if self.skip_reencode else True
                img_stream.seek(0)
                img_buf = np.frombuffer(img_stream.read(), np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Image decoding error")
                if len(img.shape) == 3 and img.shape[-1] == 4:
                    # alpha matting with white background
                    alpha = img[:, :, 3, np.newaxis]
                    img = alpha / 255 * img[..., :3] + 255 - alpha
                    img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
                    encode_needed = True
                original_height, original_width = img.shape[:2]
                # check if image is too small
                if min(original_height, original_width) < self.min_image_size:
                    return None, None, None, None, None, "image too small"
                if original_height * original_width > self.max_image_area:
                    return None, None, None, None, None, "image area too large"
                # check if wrong aspect ratio
                if max(original_height, original_width) / min(original_height, original_width) > self.max_aspect_ratio:
                    return None, None, None, None, None, "aspect ratio too large"

                # check if resizer was defined during init if needed
                if blurring_bbox_list is not None and self.blurrer is None:
                    return None, None, None, None, None, "blurrer not defined"

                # Flag to check if blurring is still needed.
                maybe_blur_still_needed = True

                # resizing in following conditions
                if self.resize_mode in (ResizeMode.keep_ratio, ResizeMode.center_crop):
                    downscale = min(original_width, original_height) > self.image_size
                    if not self.resize_only_if_bigger or downscale:
                        interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                        img = A.smallest_max_size(img, self.image_size, interpolation=interpolation)
                        if blurring_bbox_list is not None and self.blurrer is not None:
                            img = self.blurrer(img=img, bbox_list=blurring_bbox_list)
                        if self.resize_mode == ResizeMode.center_crop:
                            img = A.center_crop(img, self.image_size, self.image_size)
                        encode_needed = True
                        maybe_blur_still_needed = False
                elif self.resize_mode in (ResizeMode.border, ResizeMode.keep_ratio_largest):
                    downscale = max(original_width, original_height) > self.image_size
                    if not self.resize_only_if_bigger or downscale:
                        interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                        img = A.longest_max_size(img, self.image_size, interpolation=interpolation)
                        if blurring_bbox_list is not None and self.blurrer is not None:
                            img = self.blurrer(img=img, bbox_list=blurring_bbox_list)
                        if self.resize_mode == ResizeMode.border:
                            img = A.pad(
                                img,
                                self.image_size,
                                self.image_size,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=[255, 255, 255],
                            )
                        encode_needed = True
                        maybe_blur_still_needed = False

                # blur parts of the image if needed
                if maybe_blur_still_needed and blurring_bbox_list is not None and self.blurrer is not None:
                    img = self.blurrer(img=img, bbox_list=blurring_bbox_list)

                height, width = img.shape[:2]
                if encode_needed:
                    img_str = cv2.imencode(f".{self.encode_format}", img, params=self.encode_params)[1].tobytes()
                else:
                    img_str = img_buf.tobytes()
                return img_str, width, height, original_width, original_height, None

        except Exception as err:  # pylint: disable=broad-except
            return None, None, None, None, None, str(err)
