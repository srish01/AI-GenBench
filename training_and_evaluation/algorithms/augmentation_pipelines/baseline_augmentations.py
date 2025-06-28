from typing import Callable, Literal, Optional, Tuple, Union
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
from albumentations.augmentations import CoarseDropout, GaussNoise

from dataset_loading.fixed_augmentations.format_adapter_dataset import (
    ComposeMixedAugmentations,
)
from lightning_data_modules.augmentation_utils.augmentations_functions import (
    data_augment_blur,
    data_augment_cmp,
    data_augment_rot90,
)


def make_baseline_train_aug(
    model_input_size: Union[int, Tuple[int, int]],
    resize_or_crop_mechanism: Literal["resize", "random_crop", "center_crop", "as_is"],
    resize_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:

    assert resize_or_crop_mechanism in {
        "resize",
        "random_crop",
        "center_crop",
        "as_is",
    }

    if resize_or_crop_mechanism == "resize":
        input_size_adaptation_transform = transforms.Resize(
            size=model_input_size, interpolation=resize_interpolation
        )
    elif resize_or_crop_mechanism == "random_crop":
        input_size_adaptation_transform = transforms.RandomCrop(
            size=model_input_size, pad_if_needed=True, padding_mode="constant"
        )
    elif resize_or_crop_mechanism == "center_crop":
        input_size_adaptation_transform = transforms.CenterCrop(size=model_input_size)
    else:
        # resize_or_crop_mechanism == "as_is"
        input_size_adaptation_transform = transforms.Identity()

    cutout = CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(1, 48),
        hole_width_range=(1, 48),
        fill=128,
        p=0.2,
    )

    gaussian_noise = GaussNoise(p=0.2)

    deterministic_transforms = ComposeMixedAugmentations(
        [
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=256,
                        scale=(0.08, 1.0),
                        ratio=(0.75, 1.0 / 0.75),
                        interpolation=resize_interpolation,
                    )
                ],
                p=0.2,
            ),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            cutout,
            gaussian_noise,
            transforms.RandomApply(
                [transforms.Lambda(lambda img: data_augment_blur(img, [0.0, 3.0]))],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda img: data_augment_cmp(
                            img, ["cv2", "pil"], list(range(30, 101))
                        )
                    )
                ],
                p=0.5,
            ),
        ]
    )

    # Rotations, flips and random crops are allowed to be non-deterministic
    non_deterministic_transforms = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.Lambda(lambda img: data_augment_rot90(img)),
                ],
                p=1.0,
            ),
            transforms.RandomHorizontalFlip(),
            input_size_adaptation_transform,
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return deterministic_transforms, non_deterministic_transforms


def make_baseline_val_aug():
    # Note: also check the mandatory_val_preprocessing() method in the datamodule!
    # The mandatory augmentations are applied before the crop and, as the name suggests,
    # are mandatory when using this pipeline in the context of the proposed benchmark.

    before_crop_transforms = transforms.Identity()
    after_crop_transforms = transforms.Compose(
        [
            # Note: resize/crop strategy is defined in the model!
            # Images arrive here already resized to the expected model input size
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return before_crop_transforms, after_crop_transforms


def make_baseline_test_aug():
    return make_baseline_val_aug()


def make_baseline_predict_aug():
    return make_baseline_val_aug()


__all__ = [
    "make_baseline_train_aug",
    "make_baseline_val_aug",
    "make_baseline_test_aug",
    "make_baseline_predict_aug",
]
