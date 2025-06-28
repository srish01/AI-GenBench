from typing import Any, Dict, List
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F


class RandomCropIfLarge(transforms.Transform):
    def __init__(self, threshold, force_central_crop=False):
        """
        Initialize the transformation.

        :param threshold: Maximum size (height, width) above which the image will be cropped.
        :param force_central_crop: If True, the crop will be centered (deterministic).
            If False, the crop will be random.
        """
        super().__init__()
        self.threshold = threshold
        self.force_central_crop = force_central_crop

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self.make_params(flat_inputs)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = dict(
            crop_params=RandomCropIfLarge.get_crop_params(
                flat_inputs[0], self.threshold
            )
        )

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        return self.transform(inpt, params)

    def transform(self, inpt: Any, params: Dict[str, Any]):
        crop_params = params["crop_params"]
        if not crop_params:
            return inpt
        else:
            return self.apply_max_crop(inpt, crop_params)

    @staticmethod
    def get_crop_params(img, threshold, force_central_crop=False):
        height, width = F.get_size(img)

        # Crop if either dimension is greater than the threshold
        if width > threshold[1] or height > threshold[0]:
            # Compute cropping dimensions based on threshold
            crop_width = min(width, threshold[1])
            crop_height = min(height, threshold[0])

            if force_central_crop:
                # Deterministic central cropping
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
            else:

                # Randomly select the starting position for cropping
                max_x = width - crop_width
                max_y = height - crop_height

                # Ensure the cropping position is within the image boundaries
                left = 0 if max_x <= 0 else torch.randint(0, max_x, (1,)).item()
                top = 0 if max_y <= 0 else torch.randint(0, max_y, (1,)).item()

            return top, left, crop_height, crop_width
        else:
            return None

    def apply_max_crop(self, img, crop_params):
        top, left, crop_height, crop_width = crop_params
        img = F.crop(img, top, left, crop_height, crop_width)
        return img


__all__ = ["RandomCropIfLarge"]
