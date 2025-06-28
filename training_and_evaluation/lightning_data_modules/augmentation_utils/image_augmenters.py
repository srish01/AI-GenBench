from PIL.Image import Image
from torch import Tensor
from torch.nn import Module

from torchvision.transforms.v2 import Identity

from typing import Callable, List, Optional, Sequence, Union

from dataset_loading.fixed_augmentations.format_adapter_dataset import (
    ComposeMixedAugmentations,
)


class TrainImageAugmenter(Module):
    def __init__(
        self,
        mandatory_preprocessing: Optional[Callable],
        augmentation: Callable,
    ):
        super().__init__()
        self._mandatory_preprocessing = mandatory_preprocessing
        self._augmentation = augmentation

        assert mandatory_preprocessing is None or isinstance(
            mandatory_preprocessing, Callable
        )
        assert isinstance(self._augmentation, Callable)

        self._overall_augmentation: Callable
        if self._mandatory_preprocessing is not None:
            self._overall_augmentation = ComposeMixedAugmentations(
                [self._mandatory_preprocessing, self._augmentation]
            )
        else:
            self._overall_augmentation = ComposeMixedAugmentations([self._augmentation])

    def __repr__(self):
        return super().__repr__() + f"({self._overall_augmentation})"

    def __call__(self, image: Union[Image, Tensor]) -> Union[Image, Tensor]:
        return self._overall_augmentation(image)

    def set_random_seed(self, seed: int):
        self._overall_augmentation.set_random_seed(seed)


class EvaluationImageAugmenter(Module):
    def __init__(
        self,
        mandatory_preprocessing: Optional[Callable],
        pre_crop_augmentation: Callable,
        post_crop_augmentation: Callable,
        cropping_strategy: Callable,
    ):
        super().__init__()

        assert mandatory_preprocessing is None or isinstance(
            mandatory_preprocessing, Callable
        )
        assert isinstance(pre_crop_augmentation, Callable)
        assert isinstance(post_crop_augmentation, Callable)
        assert isinstance(cropping_strategy, Callable)

        mandatory_preprocessing_not_none = (
            Identity() if mandatory_preprocessing is None else mandatory_preprocessing
        )

        self._mandatory_preprocessing = ComposeMixedAugmentations(
            [mandatory_preprocessing_not_none]
        )
        self._pre_crop_augmentation = ComposeMixedAugmentations([pre_crop_augmentation])
        self._post_crop_augmentation = ComposeMixedAugmentations(
            [post_crop_augmentation]
        )
        self._cropping_strategy = cropping_strategy

    def __call__(self, image: Union[Image, Tensor]) -> List[Union[Image, Tensor]]:
        if self._mandatory_preprocessing is not None:
            image = self._mandatory_preprocessing(image)

        image = self._pre_crop_augmentation(image)

        crops = self._cropping_strategy(image)
        if not isinstance(crops, Sequence):
            crops = [crops]
        else:
            crops = list(crops)  # Ensure is mutable

        for crop_idx, crop in enumerate(crops):
            # Replace the crop in the list (in-place, to preserve memory)
            crops[crop_idx] = self._post_crop_augmentation(crop)

        return crops

    def set_random_seed(self, seed: int):
        self._mandatory_preprocessing.set_random_seed(seed)
        self._pre_crop_augmentation.set_random_seed(seed)
        self._post_crop_augmentation.set_random_seed(seed)


__all__ = [
    "TrainImageAugmenter",
    "EvaluationImageAugmenter",
]
