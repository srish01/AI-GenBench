from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
)

from algorithms.augmentation_pipelines.soft_train_augmentations import (
    make_soft_train_aug,
)
from algorithms.base_model import BaseDeepfakeDetectionModel


class SoftTrainAugmentationsModel(BaseDeepfakeDetectionModel):

    def train_augmentation(
        self,
    ) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
        return make_soft_train_aug(
            self.model_input_size,
            self.training_cropping_strategy,
        )


__all__ = ["SoftTrainAugmentationsModel"]
