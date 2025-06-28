from typing import Callable, Optional, Sequence

from .fixed_augmentations import (
    PreAugmentedDataset,
    FormatFlags,
    ALBUMENTATIONS_PYTORCH_OUTPUT,
    PYTORCH_OUTPUT_FORMAT,
)


class AlbumentationsPyTorchDataset(PreAugmentedDataset):
    """
    A dataset that expects the augmentation function to be an albumentations transformations.

    Note: this is a subclass of :class:`PreAugmentedDataset`. Consider checking the documentation of that
    class for more information.
    """

    def __init__(
        self,
        dataset: Sequence,
        augmentation: Callable,
        augmentation_factor: int = 1,
        base_seed: Optional[int] = None,
        enable_deterministic_augmentations: bool = True,
        augmentation_format_flags: FormatFlags = ALBUMENTATIONS_PYTORCH_OUTPUT,
        post_process_format_flags: FormatFlags = PYTORCH_OUTPUT_FORMAT,
    ):
        """
        Args:
            dataset: The dataset to apply transformations to.
            augmentation: A list of transformations to apply to the dataset.
            augmentation_factor: The number of times each element in the dataset should be repeated.
            base_seed: The seed to use as a base for the random number generator, in range [0, 2^31).
            enable_deterministic_augmentations: If True, the augmentations will be deterministic (and base_seed is required).
                This means that the same transformation will be applied to the same element each time that element is retrieved.
                Defaults to False, which means that the augmentations will be random (as usual).
                If augmentation_factor > 1 and enable_deterministic_augmentations is False, this will result in different augmentations
                being applied to the same element each time it is retrieved. It would be like extending the dataset N times (not really useful...).
            augmentations_format_flags: The format flags to use for the dataset. Defaults to `ALBUMENTATIONS_TORCH_FORMAT`.
            post_process_format_flags: The format flags to use for the post-processing of the dataset. Defaults to `PYTORCH_OUTPUT_FORMAT`.
        """
        super().__init__(
            dataset=dataset,
            augmentation=augmentation,
            augmentation_factor=augmentation_factor,
            base_seed=base_seed,
            enable_deterministic_augmentations=enable_deterministic_augmentations,
            augmentation_format_flags=augmentation_format_flags,
            post_process_format_flags=post_process_format_flags,
        )


__all__ = ["AlbumentationsPyTorchDataset"]
