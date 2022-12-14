"""blurrer module to blur parts of the image"""

import numpy as np

import albumentations as A
import functools

class BoundingBoxBlurrer:

    def __init__(
        self,
        bbox_format: str = "albumentations",
    ) -> None:
        self.bbox_format = bbox_format

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

            if self.bbox_format == "albumentations":
                adjusted_bbox = [
                    int(bbox[0] * width),
                    int(bbox[1] * height),
                    int(bbox[2] * width),
                    int(bbox[3] * height),
                ]
            elif self.bbox_format == "coco":
                adjusted_bbox = [
                    int(bbox[0] * width),
                    int(bbox[1] * height),
                    int((bbox[0]+bbox[2]) * width),
                    int((bbox[1]+bbox[3]) * height),
                ]
            else:
                raise ValueError("bounding box format not recognised")

            diagonal = max(adjusted_bbox[2] - adjusted_bbox[0], adjusted_bbox[3]-adjusted_bbox[1])
            max_diagonal = max(max_diagonal, diagonal)

            adjusted_bbox[0] = int(adjusted_bbox[0] - 0.1 * diagonal) 
            adjusted_bbox[1] = int(adjusted_bbox[1] - 0.1 * diagonal) 
            adjusted_bbox[2] = int(adjusted_bbox[2] + 0.1 * diagonal) 
            adjusted_bbox[3] = int(adjusted_bbox[3] + 0.1 * diagonal) 

            mask[adjusted_bbox[1]:adjusted_bbox[3], adjusted_bbox[0]:adjusted_bbox[2], :] = 1
        
        sigma = 0.1*max_diagonal
        ksize = int(2*np.ceil(sigma))+1
        blurred_img = A.augmentations.gaussian_blur(img, ksize=ksize, sigma=sigma)
        blurred_mask = A.augmentations.gaussian_blur(mask, ksize=ksize, sigma=sigma)

        result = img * (1-blurred_mask) + blurred_img * blurred_mask

        # Convert back to uint8
        result = (result * 255.0).astype(np.uint8)

        return result