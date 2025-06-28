from typing import Any, Callable, Sequence
import numpy as np

from .format_adapter_dataset import *
from .RNGManager import RNGManager
from .format_flags import FormatFlags


class DeterministicAugmentationsDataset(FormatAdapterDataset):
    """
    A dataset that applies transformations to the input data in a deterministic way (based on a starting seed).

    Transformations are applied differently to each element in the dataset, but the same transformation is applied
    to the same element each time that element is retrieved.

    This dataset is a subclass of :class:`FormatAdapterDataset`. This allows to set the desired format for the output,
    the expected input format of the augmentation function and other intermediate elements.
    """

    def __init__(
        self,
        dataset: Sequence,
        augmentation: Callable,
        base_seed: int,
        format_flags: FormatFlags = FormatFlags(),
    ):
        """
        Args:
            dataset: The dataset to apply transformations to.
            augmentation: A list of transformations to apply to the dataset.
            base_seed: The seed to use as a base for the random number generator, in range [0, 2^31).
        """

        super().__init__(
            dataset=dataset, augmentation=augmentation, format_flags=format_flags
        )

        self.base_seed: int = base_seed
        self._is_albumentation, _ = FormatAdapterDataset._detect_common_libraries(
            augmentation
        )
        self.has_set_random_seed = callable(
            getattr(augmentation, "set_random_seed", None)
        )

    def __getitem__(self, index: int) -> Any:
        fixed_seed = self.base_seed + index
        RNGManager.set_random_seeds(fixed_seed)
        self._set_non_global_seeds(fixed_seed)
        return super().__getitem__(index)

    def _set_non_global_seeds(self, new_seed: int):
        # torchvision: uses global (CPU) RNG. Already handled by RNGManager.set_random_seeds

        # Albumentations: each transform (even Compose) uses its own private RNG
        if self._is_albumentation or self.has_set_random_seed:
            try:
                # from albumentations import BaseCompose, BasicTransform
                # augmentation: Union[BasicTransform, BaseCompose]

                augmentation = self.augmentation
                augmentation.set_random_seed(new_seed)
            except Exception as e:
                pass


class NonDeterministicPostprocessDataset(FormatAdapterDataset):
    """
    A dataset that can be used to apply non-deterministic post-processing transformations to the output of a deterministic dataset.

    It works by creating its own private RNG which is then used to re-seed the global RNGs after obtaining the output of the deterministic dataset.
    """

    def __init__(
        self,
        deterministic_dataset: DeterministicAugmentationsDataset,
        postprocessing_transformation: Callable,
        format_flags: FormatFlags = FormatFlags(),
    ):
        """
        Args:
            deterministic_dataset: The deterministic dataset to apply transformations to.
            postprocessing_transformation: A list of transformations to apply to the output of the deterministic dataset.
            format_flags: The format flags to use for the dataset. Defaults to an empty (pure autodetect) FormatFlags object.

        """

        super().__init__(
            dataset=deterministic_dataset,
            augmentation=postprocessing_transformation,
            format_flags=format_flags,
        )

        self._is_albumentation, _ = FormatAdapterDataset._detect_common_libraries(
            postprocessing_transformation
        )
        self.has_set_random_seed = callable(
            getattr(postprocessing_transformation, "set_random_seed", None)
        )
        self.private_rng = None
        self._torch_is_worker_process = (
            False  # A mechanism to detect if we are in a PyTorch DataLoader worker
        )

    def __getitem__(self, index: int) -> Any:
        self._reset_seeds_if_needed()

        random_seed = int(self.private_rng.integers(0, 2**31))
        deterministic_element = self.dataset[index]

        RNGManager.set_random_seeds(random_seed)
        # Note not needed, as albumentations already use their own private (per-transformation) RNG!
        # self._set_non_global_seeds(random_seed)
        post_processed_element = self._apply_pipeline(deterministic_element)

        return post_processed_element

    def _reset_non_global_seeds(self):
        # Albumentations: each transform (even Compose) uses its own private RNG
        if self._is_albumentation or self.has_set_random_seed:
            try:
                # from albumentations import BaseCompose, BasicTransform
                # augmentation: Union[BasicTransform, BaseCompose]

                augmentation = self.augmentation
                augmentation.set_random_seed(None)
            except Exception as e:
                pass

    def _reset_seeds_if_needed(self):
        if self.private_rng is None or self._torch_check_if_loading_context_changed():
            self.private_rng = np.random.default_rng()
            self._reset_non_global_seeds()

    def _torch_check_if_loading_context_changed(self) -> bool:
        if self._torch_is_worker_process:
            return False

        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self._torch_is_worker_process = True
                return True
        except ImportError:
            pass

        return False


__all__ = ["DeterministicAugmentationsDataset", "NonDeterministicPostprocessDataset"]
