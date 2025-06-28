from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from dataset_loading.fixed_augmentations.augmentation_utils import (
    _is_pil_image,
    _is_torch_tensor,
    _is_numpy_array,
    _bgr_to_rgb,
    _get_image_type,
    _numpy_to_pil,
    _numpy_to_torch,
    _pil_to_numpy,
    _pil_to_torch,
    _rgb_to_bgr,
    _torch_to_numpy,
    _torch_to_pil,
)

from dataset_loading.fixed_augmentations.format_flags import *


class FormatAdapterDataset:
    """
    A utility dataset class that simplifies the conversion of different formats for input and output of augmentations.
    """

    def __init__(
        self,
        dataset: Sequence,
        augmentation: Callable[[Dict[str, Any]], Any],
        format_flags: FormatFlags = FormatFlags(),
        manage_multicrop: bool = True,
        dictionary_keys: Optional[Sequence[str]] = None,
        default_augmentation_kwarg_name: str = "image",
    ):
        """
        Args:
            dataset: The dataset to apply transformations to.
            augmentation: A list of transformations to apply to the dataset rows.
            format_flags: An instance of FormatFlags containing all format-related flags.
            manage_multicrop: If True, the dataset will manage the multicrop augmentations (which usually
                returna list of images instead of a single image). If False, the user is responsible for
                handling the multicrop augmentations by stacking the images before returning them.
            dictionary_keys: The keys to use when converting a dictionary to a list or tuple. If None, the keys
                will be autodetected if the input is a dictionary (will assume OrderedDict, which is the default
                when using Python 3.7+), or an error will be raised both dictionary_keys
                and the autodetection fail.
            default_augmentation_kwarg_name: The name of the argument to use when passing the input to an augmentation
                function that requires a named parameter. Defaults to "image" (such as in albumentations).
                Used only when providing a single image to such kind of augmentations.
        """
        self.dataset = dataset
        self._augmentation = augmentation
        self._manage_multicrop = manage_multicrop
        self._dictionary_keys = dictionary_keys
        self._default_augmentation_kwarg_name = default_augmentation_kwarg_name

        self.input_dataset_format: InputDatasetFormat = (
            format_flags.input_dataset_format
        )
        self.input_dataset_image_channels_format: InputDatasetImageChannelFormat = (
            format_flags.input_dataset_image_channels_format
        )
        self.augmentation_argument_format: AugmentationArgumentFormat = (
            format_flags.augmentation_argument_format
        )
        self.augmentation_input_format: AugmentationInputFormat = (
            format_flags.augmentation_input_format
        )
        self.augmentation_input_image_channels_format: (
            AugmentationImageChannelFormat
        ) = format_flags.augmentation_image_channel_format
        self.dataset_output_format: DatasetOutputFormat = (
            format_flags.dataset_output_format
        )
        self.dataset_image_output_format: DatasetImageOutputFormat = (
            format_flags.dataset_image_output_format
        )

        _detected_user_augmentation_format = FormatAdapterDataset._augmentation_format(
            augmentation
        )

        if self.augmentation_argument_format == AugmentationArgumentFormat.AUTODETECT:
            self.augmentation_argument_format = _detected_user_augmentation_format[0]

        if self.augmentation_input_format == AugmentationInputFormat.AUTODETECT:
            self.augmentation_input_format = _detected_user_augmentation_format[1]

        if (
            self.augmentation_input_image_channels_format
            == AugmentationImageChannelFormat.AUTODETECT
        ):
            self.augmentation_input_image_channels_format = (
                _detected_user_augmentation_format[2]
            )

        if self.augmentation_argument_format == AugmentationArgumentFormat.IMAGE_ONLY:
            self.augmentation_argument_format = (
                AugmentationArgumentFormat.IMAGE_ONLY
                | AugmentationArgumentFormat.LIST_OR_TUPLE
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, (int, np.integer)):
            element = self.dataset[index]
            augmented_element = self._apply_pipeline(element)
        else:
            return self.dataset[index]

        return augmented_element

    @property
    def transform(self):
        """
        Alias for the `augmentation` property (read-only field).
        """
        return self._augmentation

    @property
    def augmentation(self):
        """
        The augmentation function that will be applied to the dataset (read-only field).
        """
        return self._augmentation

    def _apply_pipeline(self, dataset_row: Union[List, Tuple, Dict[str, Any]]) -> Any:
        row_dictionary_keys = self._dictionary_keys
        if row_dictionary_keys is None and isinstance(dataset_row, dict):
            row_dictionary_keys = dataset_row.keys()

        if row_dictionary_keys is not None:
            row_dictionary_keys = list(row_dictionary_keys)

        # Convert the dataset row to the expected format
        dataset_row = self._prepare_input_for_augmentation(
            dataset_row, row_dictionary_keys=row_dictionary_keys
        )

        # Apply the user-defined transformation
        dataset_row = self._apply_augmentation(
            dataset_row, row_dictionary_keys=row_dictionary_keys
        )

        row_dictionary_keys = self._dictionary_keys
        if row_dictionary_keys is None and isinstance(dataset_row, dict):
            row_dictionary_keys = dataset_row.keys()

        if row_dictionary_keys is not None:
            row_dictionary_keys = list(row_dictionary_keys)

        # Convert the dataset row to the expected output format
        dataset_row = self._convert_output_dataset_format(
            dataset_row, row_dictionary_keys=row_dictionary_keys
        )

        return dataset_row

    def _prepare_input_for_augmentation(
        self, dataset_row: Any, row_dictionary_keys: Optional[List[str]] = None
    ) -> Any:
        if self.input_dataset_format == InputDatasetFormat.AUTODETECT:
            self.input_dataset_format = self._detect_element_format(dataset_row)

        # Usually "image" is the first element in the dictionary
        # Will be none if using tuples or lists (common in PyTorch datasets)
        image_dict_key = (
            row_dictionary_keys[0] if row_dictionary_keys is not None else None
        )

        # Make defensive (shallow) copy and obtain image object
        if self.input_dataset_format == InputDatasetFormat.DICTIONARY:
            assert (
                image_dict_key is not None
            ), "Dictionary keys must be provided when using dictionary format"
            dataset_row = dict(dataset_row)
            image_object = dataset_row[image_dict_key]
        else:
            dataset_row = list(dataset_row)
            image_object = dataset_row[0]

        input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
            image_object, manage_multicrop=self._manage_multicrop
        )

        if (
            self.input_dataset_image_channels_format
            == InputDatasetImageChannelFormat.AUTODETECT
        ):
            if input_is_torch or input_is_pil:
                self.input_dataset_image_channels_format = (
                    InputDatasetImageChannelFormat.RGB
                )
            else:
                raise ValueError(
                    f"Input dataset image channel format could not be autodetected for dataset row: {dataset_row.shape}, {dataset_row.dtype}"
                )

        if (
            input_is_pil
            and self.input_dataset_image_channels_format
            == InputDatasetImageChannelFormat.BGR
        ):
            raise ValueError(
                f"Input dataset image channel format is BGR, but PIL images are always treated as RGB. Dataset row: {dataset_row}"
            )

        augmentation_input_can_be_pil = (
            AugmentationInputFormat.PIL_IMAGE in self.augmentation_input_format
        )
        augmentation_input_can_be_numpy = (
            AugmentationInputFormat.NUMPY_ARRAY in self.augmentation_input_format
        )
        augmentation_input_can_be_torch = (
            AugmentationInputFormat.PYTORCH_TENSOR in self.augmentation_input_format
        )

        if input_is_pil and not augmentation_input_can_be_pil:
            if augmentation_input_can_be_torch:
                image_object = _pil_to_torch(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!
            elif augmentation_input_can_be_numpy:
                image_object = _pil_to_numpy(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!
        elif input_is_numpy and not augmentation_input_can_be_numpy:
            if augmentation_input_can_be_torch:
                image_object = _numpy_to_torch(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!
            elif augmentation_input_can_be_pil:
                image_object = _numpy_to_pil(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!
        elif input_is_torch and not augmentation_input_can_be_torch:
            if augmentation_input_can_be_numpy:
                image_object = _torch_to_numpy(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!
            elif augmentation_input_can_be_pil:
                image_object = _torch_to_pil(
                    image_object, manage_multicrop=self._manage_multicrop
                )  # OK!

        (
            augmentation_input_is_pil,
            augmentation_input_is_numpy,
            augmentation_input_is_torch,
        ) = _get_image_type(image_object, manage_multicrop=self._manage_multicrop)

        if (
            self.augmentation_input_image_channels_format
            == AugmentationImageChannelFormat.AUTODETECT
        ):
            if augmentation_input_is_pil or augmentation_input_is_torch:
                self.augmentation_input_image_channels_format = (
                    AugmentationImageChannelFormat.RGB
                )
            else:
                raise ValueError(
                    f"Augmentation input image channel format could not be autodetected for dataset row: {dataset_row}"
                )

        if (
            augmentation_input_is_pil
            and self.augmentation_input_image_channels_format
            == AugmentationImageChannelFormat.BGR
        ):
            raise ValueError(
                f"Augmentation input image channel format is BGR, but PIL images are always treated as RGB. Dataset row: {dataset_row}"
            )

        if (
            self.input_dataset_image_channels_format
            == AugmentationImageChannelFormat.RGB
        ) and (
            self.augmentation_input_image_channels_format
            == AugmentationImageChannelFormat.BGR
        ):
            image_object = _rgb_to_bgr(
                image_object, manage_multicrop=self._manage_multicrop
            )
        elif (
            self.input_dataset_image_channels_format
            == AugmentationImageChannelFormat.BGR
        ) and (
            self.augmentation_input_image_channels_format
            == AugmentationImageChannelFormat.RGB
        ):
            image_object = _bgr_to_rgb(
                image_object, manage_multicrop=self._manage_multicrop
            )

        if self.input_dataset_format == InputDatasetFormat.DICTIONARY:
            dataset_row[image_dict_key] = image_object
        else:
            # Note: we previously shallow-copied and converted it to a (mutable) list
            dataset_row[0] = image_object

        # Adapt the format (dictionary or list) to the format expected by the augmentation function (AugmentationArgumentFormat)
        if AugmentationArgumentFormat.AUTODETECT in self.augmentation_argument_format:
            # Could not detect the format, let's pass it as-is
            pass
        elif AugmentationArgumentFormat.IMAGE_ONLY in self.augmentation_argument_format:
            # Managed later by the _apply_augmentation method
            pass
        elif AugmentationArgumentFormat.DICTIONARY in self.augmentation_argument_format:
            if self.input_dataset_format == InputDatasetFormat.LIST_OR_TUPLE:
                # Elements of index >= len(row_dictionary_keys) are discarded!
                assert (
                    row_dictionary_keys is not None
                ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"

                dataset_row = dict(
                    zip(row_dictionary_keys, dataset_row[: len(row_dictionary_keys)])
                )
        elif (
            AugmentationArgumentFormat.LIST_OR_TUPLE
            in self.augmentation_argument_format
        ):
            if self.input_dataset_format == InputDatasetFormat.DICTIONARY:
                # Other elements of the input dictionary are discarded!

                assert (
                    row_dictionary_keys is not None
                ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"

                dataset_row = [dataset_row[key] for key in row_dictionary_keys]

        return dataset_row

    def _apply_augmentation(
        self, dataset_row: Any, row_dictionary_keys: Optional[List[str]] = None
    ) -> Any:
        # Usually "image" is the first element in the dictionary
        # Will be none if using tuples or lists (common in PyTorch datasets)
        image_dict_key = (
            row_dictionary_keys[0] if row_dictionary_keys is not None else None
        )

        if AugmentationArgumentFormat.AUTODETECT in self.augmentation_argument_format:
            return self.augmentation(dataset_row)
        elif AugmentationArgumentFormat.IMAGE_ONLY in self.augmentation_argument_format:
            if isinstance(dataset_row, dict):
                assert (
                    image_dict_key is not None
                ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"

                if (
                    AugmentationArgumentFormat.DICTIONARY
                    in self.augmentation_argument_format
                ):
                    dataset_row[image_dict_key] = self.augmentation(
                        image=dataset_row[image_dict_key]
                    )[image_dict_key]
                elif (
                    AugmentationArgumentFormat.LIST_OR_TUPLE
                    in self.augmentation_argument_format
                ):
                    dataset_row[image_dict_key] = self.augmentation(
                        dataset_row[image_dict_key]
                    )
                else:
                    assert False, "Invalid AugmentationArgumentFormat"
            elif isinstance(dataset_row, list):
                if (
                    AugmentationArgumentFormat.DICTIONARY
                    in self.augmentation_argument_format
                ):
                    input_dict = {
                        self._default_augmentation_kwarg_name: dataset_row[0],
                    }
                    dataset_row[0] = self.augmentation(**input_dict)[
                        self._default_augmentation_kwarg_name
                    ]
                else:
                    dataset_row[0] = self.augmentation(dataset_row[0])
            elif isinstance(dataset_row, tuple):
                if (
                    AugmentationArgumentFormat.DICTIONARY
                    in self.augmentation_argument_format
                ):
                    input_dict = {
                        self._default_augmentation_kwarg_name: dataset_row[0],
                    }
                    dataset_row = (
                        self.augmentation(**input_dict)[
                            self._default_augmentation_kwarg_name
                        ],
                        *dataset_row[1:],
                    )
                else:
                    dataset_row = (
                        self.augmentation(dataset_row[0]),
                        *dataset_row[1:],
                    )
            else:
                raise ValueError(f"Dataset row format not supported: {dataset_row}")
        elif self.augmentation_argument_format == AugmentationArgumentFormat.DICTIONARY:
            return self.augmentation(**dataset_row)
        elif (
            self.augmentation_argument_format
            == AugmentationArgumentFormat.LIST_OR_TUPLE
        ):
            return self.augmentation(*dataset_row)

        return dataset_row

    def _convert_output_dataset_format(
        self, dataset_row: Any, row_dictionary_keys: Optional[List[str]] = None
    ) -> Any:
        # Usually "image" is the first element in the dictionary
        # Will be none if using tuples or lists (common in PyTorch datasets)
        image_dict_key = (
            row_dictionary_keys[0] if row_dictionary_keys is not None else None
        )

        row_is_dict = isinstance(dataset_row, dict)
        row_is_sequence = isinstance(dataset_row, (list, tuple))
        if row_is_dict:
            assert (
                image_dict_key is not None
            ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"
            image_object = dataset_row[image_dict_key]
        else:
            image_object = dataset_row[0]

        # Adapt considering DatasetImageOutputFormat
        input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
            image_object, manage_multicrop=self._manage_multicrop
        )

        if (
            self.dataset_image_output_format
            == DatasetImageOutputFormat.AS_TRANFORM_OUTPUT
        ):
            pass
        elif self.dataset_image_output_format == DatasetImageOutputFormat.PIL_IMAGE:
            if input_is_numpy:
                image_object = _numpy_to_pil(
                    image_object, manage_multicrop=self._manage_multicrop
                )
            elif input_is_torch:
                image_object = _torch_to_pil(
                    image_object, manage_multicrop=self._manage_multicrop
                )
        elif (
            self.dataset_image_output_format == DatasetImageOutputFormat.PYTORCH_TENSOR
        ):
            if input_is_numpy:
                image_object = _numpy_to_torch(
                    image_object, manage_multicrop=self._manage_multicrop
                )
            elif input_is_pil:
                image_object = _pil_to_torch(
                    image_object, manage_multicrop=self._manage_multicrop
                )
        elif self.dataset_image_output_format == DatasetImageOutputFormat.NUMPY_ARRAY:
            if input_is_pil:
                image_object = _pil_to_numpy(
                    image_object, manage_multicrop=self._manage_multicrop
                )
            elif input_is_torch:
                image_object = _torch_to_numpy(
                    image_object, manage_multicrop=self._manage_multicrop
                )

        if row_is_dict:
            dataset_row[image_dict_key] = image_object
        else:
            dataset_row = list(dataset_row)  # Make mutable (in case it was a tuple)
            dataset_row[0] = image_object

        # Adapt considering DatasetOutputFormat
        if self.dataset_output_format == DatasetOutputFormat.AS_TRANFORM_OUTPUT:
            return dataset_row
        elif self.dataset_output_format == DatasetOutputFormat.DICTIONARY:
            if row_is_dict:
                return dataset_row
            elif row_is_sequence:
                assert (
                    row_dictionary_keys is not None
                ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"
                return dict(
                    zip(row_dictionary_keys, dataset_row[: len(row_dictionary_keys)])
                )
            else:
                raise ValueError(f"Dataset row format not supported: {dataset_row}")
        elif self.dataset_output_format == DatasetOutputFormat.TUPLE:
            if row_is_dict:
                assert (
                    row_dictionary_keys is not None
                ), "Dictionary keys must be provided when using dictionary format (autodetection failed)"

                return tuple([dataset_row[key] for key in row_dictionary_keys])
            elif row_is_sequence:
                return tuple(dataset_row)
            else:
                raise ValueError(f"Dataset row format not supported: {dataset_row}")

    @staticmethod
    def _detect_element_format(element: Any) -> InputDatasetFormat:
        if isinstance(element, dict):
            return InputDatasetFormat.DICTIONARY
        elif isinstance(element, (list, tuple)):
            return InputDatasetFormat.LIST_OR_TUPLE
        else:
            raise ValueError(f"Element format could not be autodetected: {element}")

    @staticmethod
    def _detect_common_libraries(augmentation):
        is_albumentation = False
        is_torchvision = False

        if isinstance(augmentation, ComposeMixedAugmentations):
            is_albumentation = False
            is_torchvision = True
            return is_albumentation, is_torchvision

        try:
            try:
                from albumentations.core.serialization import Serializable
            except ImportError:
                from albumentations import Serializable

            is_albumentation = is_torchvision or isinstance(augmentation, Serializable)
        except ImportError:
            pass

        try:
            from torchvision.transforms import (
                Compose,
                ToTensor,
                PILToTensor,
                ToPILImage,
                Lambda,
            )
            from torch.nn import Module

            is_torchvision = is_torchvision or isinstance(
                augmentation,
                (Compose, Module, ToTensor, PILToTensor, ToPILImage, Lambda),
            )
        except ImportError:
            pass

        try:
            from torchvision.transforms.v2 import Transform

            is_torchvision = is_torchvision or isinstance(augmentation, Transform)
        except ImportError:
            pass

        if is_albumentation and is_torchvision:
            # Ooops, we have a problem with the detection mechanism (assume it's a custom augmentation)
            is_albumentation = False
            is_torchvision = False

        return is_albumentation, is_torchvision

    @staticmethod
    def _augmentation_format(
        augmentation,
    ) -> Tuple[
        AugmentationArgumentFormat,
        AugmentationInputFormat,
        AugmentationImageChannelFormat,
    ]:
        is_albumentation, is_torchvision = (
            FormatAdapterDataset._detect_common_libraries(augmentation=augmentation)
        )

        if is_albumentation:
            return (
                AugmentationArgumentFormat.IMAGE_ONLY
                | AugmentationArgumentFormat.DICTIONARY,
                AugmentationInputFormat.NUMPY_ARRAY,
                AugmentationImageChannelFormat.RGB,
            )
        elif is_torchvision:
            return (
                AugmentationArgumentFormat.IMAGE_ONLY,
                AugmentationInputFormat.PIL_IMAGE
                | AugmentationInputFormat.PYTORCH_TENSOR,
                AugmentationImageChannelFormat.RGB,
            )
        else:
            return (
                AugmentationArgumentFormat.AUTODETECT,
                AugmentationInputFormat.AUTODETECT,
                AugmentationImageChannelFormat.AUTODETECT,
            )


class ComposeMixedAugmentations:
    """
    A class that behaves similar to torchvision and Albumentations Compose, but allows to mix
    torchvision and Albumentations transformations in the same pipeline, while providing the
    set_random_seed method needed to ensure deterministic augmentations.

    Note: do NOT wrap albumentations transformations in a Lambda transform, as it will break the
    set_random_seed functionality. This class already takes care of the torchvision/albumentations
    transformations compatibility.
    """

    def __init__(self, transforms: Sequence, manage_multicrop: bool = True):
        self.transforms = transforms
        self.manage_multicrop = manage_multicrop
        self.transform_is_albumentations = (
            [  # TODO: has_set_random_seed should already cover albumentations
                FormatAdapterDataset._detect_common_libraries(t)[0]
                for t in self.transforms
            ]
        )
        self.transforms_has_set_random_seed = [
            callable(getattr(t, "set_random_seed", None)) for t in self.transforms
        ]

    def __call__(self, image):
        for t, is_albumentations in zip(
            self.transforms, self.transform_is_albumentations
        ):
            input_image = image
            input_is_pil = _is_pil_image(
                input_image, manage_multicrop=self.manage_multicrop
            )
            input_is_torch = _is_torch_tensor(
                input_image, manage_multicrop=self.manage_multicrop
            )
            input_is_numpy = _is_numpy_array(
                input_image, manage_multicrop=self.manage_multicrop
            )

            if is_albumentations:
                if input_is_pil:
                    input_image = _pil_to_numpy(
                        input_image, manage_multicrop=self.manage_multicrop
                    )
                elif input_is_torch:
                    input_image = _torch_to_numpy(
                        input_image, manage_multicrop=self.manage_multicrop
                    )
                else:
                    assert (
                        input_is_numpy
                    ), "Albumentations only accepts images as NumPy arrays (OpenCV images)"

                image = t(image=input_image)["image"]
            else:
                if input_is_numpy:
                    # Better not convert from NumPy to PIL.
                    # It may happen that the image values are [0, 1], or the dtype is not uint8.
                    # In that case, it's better to convert it to a PyTorch tensor.
                    input_image = _numpy_to_torch(
                        input_image, manage_multicrop=self.manage_multicrop
                    )
                else:
                    assert (
                        input_is_torch or input_is_pil
                    ), "Torchvision only accepts images as PyTorch tensors or PIL images"

                image = t(input_image)

        image_is_numpy = _is_numpy_array(image, manage_multicrop=self.manage_multicrop)

        # Ensure the output is in torchvision format (PIL or PyTorch tensor)
        if image_is_numpy:
            image = _numpy_to_torch(image, manage_multicrop=self.manage_multicrop)

        return image

    def set_random_seed(self, seed: int):
        for t, is_albumentations, has_set_random_seed in zip(
            self.transforms,
            self.transform_is_albumentations,
            self.transforms_has_set_random_seed,
        ):
            if has_set_random_seed or is_albumentations:
                t.set_random_seed(seed)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


__all__ = [
    "FormatAdapterDataset",
    "ComposeMixedAugmentations",
]


def _test_torchvision_transforms():
    from torchvision import transforms
    from PIL import Image

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample torchvision transform
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [(image, 0, "generator")]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset, augmentation=transform, format_flags=FormatFlags()
    )

    # Apply the transform and print the result
    transformed_image, label, generator = adapter_dataset[0]
    print(type(transformed_image), transformed_image.shape, label, generator)

    # Save image
    transformed_image = _torch_to_numpy(transformed_image, False)
    transformed_image = _numpy_to_pil(transformed_image, False)
    transformed_image.save("transformed_image_torchvision_flipped.jpg")


def _test_torchvision_already_tensor():
    from torchvision import transforms
    from PIL import Image

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample torchvision transform
    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.RandomVerticalFlip(p=1.0),
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [(image, 0, "generator")]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset,
        augmentation=transform,
        format_flags=FormatFlags(
            augmentation_input_format=AugmentationInputFormat.PYTORCH_TENSOR,
            dataset_image_output_format=DatasetImageOutputFormat.PIL_IMAGE,
        ),
    )

    # Apply the transform and print the result
    transformed_image, label, generator = adapter_dataset[0]
    print(type(transformed_image), transformed_image.size, label, generator)

    # Save image
    transformed_image.save("transformed_image_torchvision_flipped_already_tensor.jpg")


def _test_albumentations_transforms():
    import albumentations as A
    from PIL import Image
    import numpy as np

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample albumentations transform
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomRotate90(p=1.0),
            # A.Normalize()
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [{"image": image, "label": 0, "generator": "StyleGANv2"}]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset,
        augmentation=transform,
        format_flags=FormatFlags(
            dataset_image_output_format=DatasetImageOutputFormat.PYTORCH_TENSOR,
        ),
    )

    # Apply the transform and print the result
    transformed_row = adapter_dataset[0]
    print(
        type(transformed_row["image"]),
        transformed_row["image"].shape,
        transformed_row["label"],
        transformed_row["generator"],
    )

    # Save image
    transformed_image = transformed_row["image"]
    transformed_image = transformed_image.permute(1, 2, 0).numpy()
    transformed_image = (transformed_image * 255).astype(np.uint8)
    transformed_image = Image.fromarray(transformed_image)
    transformed_image.save("transformed_image_albumentations_albumentations_rotate.jpg")


def _test_albumentations_as_is_output():
    import albumentations as A
    from PIL import Image
    import numpy as np

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample albumentations transform
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ToGray(p=1.0),
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [{"image": image, "label": 0, "generator": "StyleGANv2"}]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset, augmentation=transform, format_flags=FormatFlags()
    )

    # Apply the transform and print the result
    transformed_row = adapter_dataset[0]
    print(
        type(transformed_row["image"]),
        transformed_row["image"].shape,
        transformed_row["label"],
        transformed_row["generator"],
    )

    # Save image
    transformed_image = transformed_row["image"]
    assert isinstance(transformed_image, np.ndarray)
    transformed_image = Image.fromarray(transformed_image)
    transformed_image.save("transformed_image_albumentations_grayscale_as_is.jpg")


def _test_prebaked_torch_definition():
    import torchvision.transforms.v2 as transforms
    from torchvision.datasets import MNIST
    import numpy as np
    from torch.utils.data import DataLoader
    import os
    import torch

    # Define transformations

    transform = transforms.Compose(
        [
            transforms.RGB(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomRotation(30),
        ]
    )

    # Load MNIST dataset
    mnist_dataset = MNIST(
        root=os.path.expanduser("~/MNIST"), train=True, download=True, transform=None
    )

    # Create FormatAdapterDataset
    preaugmented_dataset = FormatAdapterDataset(
        mnist_dataset, transform, format_flags=TORCHVISION_FORMAT
    )

    # Create DataLoader
    batch_size = 16
    data_loader = DataLoader(
        preaugmented_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # Check deterministic behavior
    for batch in data_loader:
        images, labels = batch
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        assert images.shape[1:] == (3, 28, 28)
        assert labels.shape[0] == images.shape[0]


def _test_prebaked_albumentations_torch_definition():
    import albumentations as A
    from PIL import Image
    import numpy as np
    import torch

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample albumentations transform
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomFog(p=1.0),
            # A.Normalize()
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [{"image": image, "label": 0, "generator": "StyleGANv2"}]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset,
        augmentation=transform,
        format_flags=ALBUMENTATIONS_PYTORCH_OUTPUT,
    )

    # Apply the transform and print the result
    transformed_row = adapter_dataset[0]
    assert isinstance(transformed_row, tuple)
    assert isinstance(transformed_row[0], torch.Tensor)
    assert transformed_row[0].shape == (3, 256, 256)
    assert transformed_row[0].max().item() <= 1.0
    assert transformed_row[0].min().item() >= 0.0
    assert isinstance(transformed_row[1], int)


def _test_prebaked_albumentations_definition():
    import albumentations as A
    from PIL import Image
    import numpy as np

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample albumentations transform
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomFog(p=1.0),
            # A.Normalize()
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [{"image": image, "label": 0, "generator": "StyleGANv2"}]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset,
        augmentation=transform,
        format_flags=PURE_ALBUMENTATIONS,
    )

    # Apply the transform and print the result
    transformed_row = adapter_dataset[0]
    assert isinstance(transformed_row, dict)
    assert isinstance(transformed_row["image"], np.ndarray)
    assert transformed_row["image"].shape == (256, 256, 3)
    assert transformed_row["image"].max() <= 255
    assert (
        transformed_row["image"] > 1
    ).sum() > 0  # Not all values are between 0 and 1 (because OpenCV images should be in range [0, 255])
    assert transformed_row["image"].min() >= 0
    assert isinstance(transformed_row["label"], int)


def _test_compose_mixed_augmentations():
    import albumentations as A
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    # Load a sample image
    image_path = "image_3.jpg"
    image = Image.open(image_path)

    # Define a sample mixed transform
    mixed_transform = ComposeMixedAugmentations(
        [
            A.Resize(256, 256),
            transforms.RandomHorizontalFlip(p=1.0),
            A.CoarseDropout(p=1.0, min_holes=3, max_holes=6),
        ]
    )

    # Create a FormatAdapterDataset instance
    dataset = [{"image": image, "label": 0, "generator": "StyleGANv2"}]
    adapter_dataset = FormatAdapterDataset(
        dataset=dataset,
        augmentation=mixed_transform,
        format_flags=FormatFlags(
            dataset_image_output_format=DatasetImageOutputFormat.PYTORCH_TENSOR,
        ),
    )

    # Apply the transform and print the result
    transformed_row = adapter_dataset[0]
    print(
        type(transformed_row[0]),
        transformed_row[0].shape,
        transformed_row[1],
        transformed_row[2],
    )

    # Save image
    transformed_image = transformed_row[0]
    transformed_image = transformed_image.permute(1, 2, 0).numpy()
    transformed_image = (transformed_image * 255).astype(np.uint8)
    transformed_image = Image.fromarray(transformed_image)
    transformed_image.save("transformed_image_mixed_augmentations.jpg")


if __name__ == "__main__":
    _test_torchvision_transforms()
    _test_torchvision_already_tensor()
    _test_albumentations_transforms()
    _test_albumentations_as_is_output()

    _test_prebaked_torch_definition()
    _test_prebaked_albumentations_torch_definition()
    _test_prebaked_albumentations_definition()

    _test_compose_mixed_augmentations()
