from typing import Any, Callable, Dict, Optional, Sequence
import numpy as np

from dataset_loading.fixed_augmentations.deterministic_augmentations_dataset import (
    DeterministicAugmentationsDataset,
    NonDeterministicPostprocessDataset,
)
from dataset_loading.fixed_augmentations.format_adapter_dataset import *
from dataset_loading.fixed_augmentations.format_flags import FormatFlags


MAXIMUM_SEED_VALUE = 2**31  # OpenCV RNG seed must be in range [0, 2^31)


class PreAugmentedDataset:
    """
    A dataset that can apply transformations to the input data in a deterministic way (based on a starting seed).
    This functionality must be enabled by setting `enable_deterministic_augmentations=True`.

    Given a dataset, this dataset will appear to be of size `len(dataset) * augmentation_factor` (where dataset is the input dataset).
    Each element will be available multiple times in the dataset (at indices `i, i+N, i+2N, ...` where N is the dataset size),
    each with a different transformation applied to it.

    The transformations are applied to the input data in a deterministic way, such that the same transformation
    is applied to the same element each time that element is retrieved (based on the index).

    Finally, an (optional) `non_deterministic_post_process` function can be applied to the output of the deterministic augmentation function.
    The usual use of this dataset is to apply deterministic augmentations to the input data and then apply a non-deterministic
    random crop/resize/flip/rotate to the output of the deterministic augmentations.

    This dataset uses :class:`DeterministicAugmentationsDataset` under the hood. This allows to set the desired format for the output,
    the expected input format of the augmentation function and other intermediate elements.

    Note: if augmentation_factor==1, this dataset is equivalent to a single :class:`DeterministicAugmentationsDataset`.
    If augmentation_factor==1 and enable_deterministic_augmentations==False, this dataset is equivalent to a "normal" dataset (of :class:`FormatAdapterDataset`).
    If augmentation_factor>1 and enable_deterministic_augmentations==False, this dataset is equivalent to a concatenation of `augmentation_factor` "normal" datasets.
    """

    def __init__(
        self,
        dataset: Sequence,
        augmentation: Optional[Callable],
        augmentation_factor: int = 1,
        non_deterministic_post_process: Optional[Callable] = None,
        base_seed: Optional[int] = None,
        enable_deterministic_augmentations: bool = True,
        augmentation_format_flags: FormatFlags = FormatFlags(),
        post_process_format_flags: FormatFlags = FormatFlags(),
        seed_shift: Optional[int] = 2**27,
    ):
        """
        Args:
            dataset: The dataset to apply transformations to.
            augmentation: A list of transformations to apply to the dataset.
                Note: the transformation will be called by passing the entire dataset row, not just the image.
            augmentation_factor: The number of times each element in the dataset should be repeated.
            non_deterministic_post_process: A function to apply to the output of the deterministic augmentations.
            base_seed: The seed to use as a base for the random number generator, in range [0, 2^31).
            enable_deterministic_augmentations: If True, the augmentations will be deterministic (and base_seed is required).
                This means that the same transformation will be applied to the same element each time that element is retrieved.
                Defaults to False, which means that the augmentations will be random (as usual).
                If augmentation_factor > 1 and enable_deterministic_augmentations is False, this will result in different augmentations
                being applied to the same element each time it is retrieved. It would be like extending the dataset N times (not really useful...).
            augmentation_format_flags: The format flags to use for the dataset. Defaults to an empty (pure autodetect) FormatFlags object.
            post_process_format_flags: The format flags to use for the post-processing of the dataset. Defaults to an empty (pure autodetect) FormatFlags object.
            seed_shift: The shift to apply to the base_seed for each subset. If None, the length of the dataset will be used.
        """
        self.dataset: Sequence = dataset
        self.augmentation: Optional[Callable] = augmentation
        self.non_deterministic_post_process: Optional[Callable] = (
            non_deterministic_post_process
        )
        self.augmentation_factor: int = augmentation_factor
        self.enable_deterministic_augmentations = enable_deterministic_augmentations

        if enable_deterministic_augmentations and base_seed is None:
            raise ValueError(
                "If enable_deterministic_augmentations is True, base_seed must be provided"
            )

        self.base_seed: int = base_seed if base_seed is not None else 0
        self.seed_shift: Optional[int] = seed_shift

        self._augmented_datasets = [
            self._make_subset(
                subset_index, augmentation_format_flags, post_process_format_flags
            )
            for subset_index in range(augmentation_factor)
        ]

    def __len__(self) -> int:
        return len(self.dataset) * self.augmentation_factor

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, (int, np.integer)):
            return self._augmented_datasets[index % self.augmentation_factor][
                index // self.augmentation_factor
            ]
        else:
            # Probably a string, used to access a column in the dataset (HF datasets style)
            return self._augmented_datasets[index]

    def _make_subset(
        self,
        subset_index: int,
        augmentation_format_flags: FormatFlags,
        post_process_format_flags: FormatFlags,
    ) -> "FormatAdapterDataset":
        seed_shift = (
            self.seed_shift if self.seed_shift is not None else len(self.dataset)
        )

        if self.enable_deterministic_augmentations:
            augmented_dataset = DeterministicAugmentationsDataset(
                dataset=self.dataset,
                augmentation=self.augmentation,
                base_seed=(self.base_seed + subset_index * seed_shift)
                % MAXIMUM_SEED_VALUE,
                format_flags=augmentation_format_flags,
            )

            if self.non_deterministic_post_process is not None:
                augmented_dataset = NonDeterministicPostprocessDataset(
                    deterministic_dataset=augmented_dataset,
                    postprocessing_transformation=self.non_deterministic_post_process,
                    format_flags=post_process_format_flags,
                )
        else:
            augmented_dataset = FormatAdapterDataset(
                dataset=self.dataset,
                augmentation=self.augmentation,
                format_flags=augmentation_format_flags,
            )

            if self.non_deterministic_post_process is not None:
                augmented_dataset = FormatAdapterDataset(
                    dataset=augmented_dataset,
                    augmentation=self.non_deterministic_post_process,
                    format_flags=post_process_format_flags,
                )

        return augmented_dataset


__all__ = ["PreAugmentedDataset"]


if __name__ == "__main__":
    from dataset_loading.fixed_augmentations.format_flags import PYTORCH_OUTPUT_FORMAT
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    import os
    from tqdm import tqdm

    augmentation_factor = 3
    base_seed = 42
    batch_size = 16
    num_workers = 16

    # Load MNIST dataset
    mnist_dataset = MNIST(
        root=os.path.expanduser("~/MNIST"), train=True, download=True, transform=None
    )

    tv_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(30),
        ]
    )

    # Define transformations (without flags, not very convenient)
    def transform(row: Dict[str, Any]) -> Dict[str, Any]:
        row = list(row)
        row[0] = tv_transforms(row[0])
        return tuple(row)

    # Create PreAugmentedDataset

    preaugmented_dataset = PreAugmentedDataset(
        dataset=mnist_dataset,
        augmentation=transform,
        augmentation_factor=augmentation_factor,
        base_seed=base_seed,
        enable_deterministic_augmentations=True,
    )

    permutation_indices = np.random.permutation(len(preaugmented_dataset)).tolist()
    preaugmented_dataset_shuffled = Subset(preaugmented_dataset, permutation_indices)

    print(preaugmented_dataset[0][0].shape)

    # Create DataLoader
    data_loader = DataLoader(
        preaugmented_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Check deterministic behavior
    first_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            first_pass.append(image)

    image_index = 0
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            assert np.array_equal(
                first_pass[image_index], image
            ), "Augmentations are not deterministic"
            image_index += 1

    # Dataloader for the shuffled version
    data_loader = DataLoader(
        preaugmented_dataset_shuffled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    first_pass_shuffled = [first_pass[i] for i in permutation_indices]
    image_index = 0
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            assert np.array_equal(
                first_pass_shuffled[image_index], image
            ), "Augmentations are not deterministic"
            image_index += 1

    # ----- Test with flags -----
    preaugmented_dataset = PreAugmentedDataset(
        dataset=mnist_dataset,
        augmentation=tv_transforms,
        augmentation_factor=augmentation_factor,
        base_seed=base_seed,
        augmentation_format_flags=PYTORCH_OUTPUT_FORMAT,
        enable_deterministic_augmentations=True,
    )

    print(preaugmented_dataset[0][0].shape)

    # Create DataLoader
    data_loader = DataLoader(
        preaugmented_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Check deterministic behavior
    first_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            first_pass.append(image)

    image_index = 0
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            assert np.array_equal(
                first_pass[image_index], image
            ), "Augmentations are not deterministic"
            image_index += 1

    print("Test passed: Augmentations are deterministic")

    # ----- Test postprocessing (deterministic) using mixed augmentations -----
    import albumentations as A

    mixed_augmentations = ComposeMixedAugmentations(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(30),
            A.HorizontalFlip(p=0.5),
        ]
    )

    preaugmented_dataset = PreAugmentedDataset(
        dataset=mnist_dataset,
        augmentation=mixed_augmentations,
        augmentation_factor=augmentation_factor,
        base_seed=base_seed,
        enable_deterministic_augmentations=True,
    )

    preaugmented_dataset = Subset(preaugmented_dataset, list(range(10)))

    permutation_indices = np.random.permutation(len(preaugmented_dataset)).tolist()
    preaugmented_dataset_shuffled = Subset(preaugmented_dataset, permutation_indices)

    print(preaugmented_dataset[0][0].shape)

    # Create DataLoader
    data_loader = DataLoader(
        preaugmented_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Check deterministic behavior
    first_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            first_pass.append(image)

    image_index = 0
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            assert np.array_equal(
                first_pass[image_index], image
            ), "Augmentations are not deterministic"
            image_index += 1

    # Dataloader for the shuffled version
    data_loader = DataLoader(
        preaugmented_dataset_shuffled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    first_pass_shuffled = [first_pass[i] for i in permutation_indices]
    image_index = 0
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images.numpy(force=True):
            assert np.array_equal(
                first_pass_shuffled[image_index], image
            ), "Augmentations are not deterministic"
            image_index += 1

    print("Test passed (mixed): augmentations are deterministic")

    # ----- Test postprocessing (non-deterministic) -----

    from torch.utils.data import Subset

    # Now with flags
    non_deterministic_post_process_p = 0.69
    preaugmented_dataset_with_postprocess = PreAugmentedDataset(
        dataset=mnist_dataset,
        augmentation=tv_transforms,
        augmentation_factor=augmentation_factor,
        base_seed=base_seed,
        enable_deterministic_augmentations=True,
        non_deterministic_post_process=transforms.RandomHorizontalFlip(
            p=non_deterministic_post_process_p
        ),
        post_process_format_flags=PYTORCH_OUTPUT_FORMAT,
    )

    print(preaugmented_dataset_with_postprocess[0][0].shape)

    # Create DataLoader
    data_loader = DataLoader(
        preaugmented_dataset_with_postprocess,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Check non-deterministic behavior
    first_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images:
            first_pass.append(image.numpy(force=True))

    second_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images:
            second_pass.append(image.numpy(force=True))

    equal_images_count = sum(
        np.array_equal(img1, img2) for img1, img2 in zip(first_pass, second_pass)
    )

    p_both_were_not_flipped = (1 - non_deterministic_post_process_p) ** 2
    p_both_were_flipped = non_deterministic_post_process_p**2
    expected_probability_of_equal_images = p_both_were_not_flipped + p_both_were_flipped
    print("Expected probability of equal images:", expected_probability_of_equal_images)
    print(
        "Expected number of equal images:",
        len(first_pass) * expected_probability_of_equal_images,
    )
    print(
        f"Number of equal images between two passes: {equal_images_count} / {len(first_pass)} ({equal_images_count / len(first_pass) * 100:.2f}%)"
    )

    # ----- Test postprocessing (non-deterministic) using albumentations -----
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    non_deterministic_post_process_p = 0.69
    albumentations_post_process = A.Compose(
        [A.HorizontalFlip(p=non_deterministic_post_process_p), ToTensorV2()]
    )

    preaugmented_dataset_with_postprocess = PreAugmentedDataset(
        dataset=mnist_dataset,
        augmentation=tv_transforms,
        augmentation_factor=augmentation_factor,
        base_seed=base_seed,
        enable_deterministic_augmentations=True,
        non_deterministic_post_process=albumentations_post_process,
        post_process_format_flags=PYTORCH_OUTPUT_FORMAT,
    )

    print(preaugmented_dataset_with_postprocess[0][0].shape)

    # Create DataLoader
    data_loader = DataLoader(
        preaugmented_dataset_with_postprocess,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Check non-deterministic behavior
    first_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images:
            first_pass.append(image.numpy(force=True))

    second_pass = []
    for batch in tqdm(data_loader):
        images, labels = batch
        for image in images:
            second_pass.append(image.numpy(force=True))

    equal_images_count = sum(
        np.array_equal(img1, img2) for img1, img2 in zip(first_pass, second_pass)
    )

    p_both_were_not_flipped = (1 - non_deterministic_post_process_p) ** 2
    p_both_were_flipped = non_deterministic_post_process_p**2
    expected_probability_of_equal_images = p_both_were_not_flipped + p_both_were_flipped
    print("Albumentations non-determinism test:")
    print("Expected probability of equal images:", expected_probability_of_equal_images)
    print(
        "Expected number of equal images:",
        len(first_pass) * expected_probability_of_equal_images,
    )
    print(
        f"Number of equal images between two passes: {equal_images_count} / {len(first_pass)} ({equal_images_count / len(first_pass) * 100:.2f}%)"
    )
