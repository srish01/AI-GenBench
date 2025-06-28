from torch.nn import Module
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
import random

from dataset_loading.fixed_augmentations.augmentation_utils import _any_to_pil


class RandomResizedCropVariable(Module):
    def __init__(
        self,
        min_size=256,
        max_size=512,
        scale=(0.5, 1.0),
        ratio=(0.9, 1 / 0.9),
        interpolation_options=[InterpolationMode.BILINEAR],
    ):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation_options = interpolation_options

    def forward(self, img):
        random_size = random.randint(self.min_size, self.max_size)
        random_interpolation = random.choice(self.interpolation_options)
        transform = transforms.RandomResizedCrop(
            size=random_size,
            scale=self.scale,
            ratio=self.ratio,
            interpolation=random_interpolation,
        )

        if random_interpolation not in {
            InterpolationMode.NEAREST,
            InterpolationMode.NEAREST_EXACT,
            InterpolationMode.BILINEAR,
            InterpolationMode.BICUBIC,
        }:
            # Probably LANCZOS -> make sure it's a PIL image and not a PyTorch Tensor
            img = _any_to_pil(img, manage_multicrop=False)

        return transform(img)


__all__ = [
    "RandomResizedCropVariable",
]
