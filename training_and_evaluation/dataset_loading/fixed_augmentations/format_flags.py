from dataclasses import dataclass
import dataclasses
from enum import Enum, Flag, auto


class InputDatasetFormat(Enum):
    """
    An enum that represents the different formats that the dataset rows can be in.
    These options are mutually exclusive.
    """

    AUTODETECT = 0
    DICTIONARY = (
        1  # Usually from a HF Dataset. Expected keys: "image", "label", "generator"
    )
    LIST_OR_TUPLE = (
        2  # Usually from a PyTorch Dataset. Expected order: [image, label, generator]
    )


class InputDatasetImageChannelFormat(Enum):
    """
    An enum that represents the different channels order that the image data can be in.
    These options are mutually exclusive.

    When using an AUTODETECT format PIL inputs will be treated as RGB, but tensor NumPy arrays
    will raise an error if the channel format is AUTODETECT. If the input is a PyTorch tensor,
    then the channel format will be autodetected as RGB.
    """

    AUTODETECT = 0
    RGB = 1
    BGR = 2


class AugmentationArgumentFormat(Flag):
    """
    An enum that represents the different formats that the dataset row must be fed to the transformation function.
    These options are NOT mutually exclusive. For instance, if your transform function accepts both a dictionary
    and a list, you can use `DICTIONARY | LIST_OR_TUPLE` (if list/tuple, we suppose list[0] is the image,
    list[1] is the binary label, list[2] is the generator name).

    When using AUTODETECT, the function will try to use the first format that is
    compatible with the transformation (by checking the transformation format).

    Note: when using IMAGE_ONLY alone, it is supposed to be IMAGE_ONLY | LIST_OR_TUPLE.
    If you need to pass only the image but as named parameters (albumentations), use IMAGE_ONLY | DICTIONARY.
    """

    AUTODETECT = auto()
    DICTIONARY = auto()
    LIST_OR_TUPLE = auto()
    IMAGE_ONLY = auto()


class AugmentationInputFormat(Flag):
    """
    An enum that represents the different formats that the image data must be fed to the transformation in.
    These options are NOT mutually exclusive. For instance, if your transform function accepts both a PIL image
    and a PyTorch tensor, you can use `PIL_IMAGE | PYTORCH_TENSOR`.

    When using AUTODETECT, the function will try to use the first format that is
    compatible with the transformation (by checking the transformation format).
    This automatic detection mechanism may fail if the transformation is a custom one.
    """

    AUTODETECT = auto()
    PIL_IMAGE = (
        auto()
    )  # PIL Image (RGB, can be converted to numpy with pixels in range [0, 255], shape HxWxC). Common input format for torchvision augmentations.
    PYTORCH_TENSOR = (
        auto()
    )  # PyTorch tensor (RGB, pixels in range [0, 1], shape CxHxW). Common output format for torchvision, not really common as input (usually PIL).
    NUMPY_ARRAY = (
        auto()
    )  # NumPy array (RGB, pixels in range [0, 255], shape HxWxC). Commonly used as input and output of albumentations (OpenCV).


class AugmentationImageChannelFormat(Enum):
    """
    An enum that represents the different channels order that the image data can be in.
    These options are mutually exclusive.

    When using AUTODETECT, PIL inputs will be passed as-is, NumPy arrays will raise an error (
    because it's not possible to know the channel format), while PyTorch tensors will
    be considered as RGB (as torchvision usually handles images in this format).
    """

    AUTODETECT = 0
    RGB = 1  # torchvision, Albumentations (note that even albumentations uses RGB! https://albumentations.ai/docs/faq/#why-do-you-call-cv2cvtcolorimage-cv2color_bgr2rgb-in-your-examples)
    BGR = 2


class DatasetOutputFormat(Enum):
    """
    An enum that represents the different formats the outputs must be.
    These options are mutually exclusive.
    """

    AS_TRANFORM_OUTPUT = 0
    DICTIONARY = 1
    TUPLE = 2  # Usual for a PyTorch codebase


class DatasetImageOutputFormat(Enum):
    """
    An enum that represents the different formats the image outputs must be.
    These options are mutually exclusive.
    """

    AS_TRANFORM_OUTPUT = 0
    PIL_IMAGE = 1  # PIL Image (RGB, can be converted to numpy with pixels in range [0, 255], shape HxWxC). Common output for a dataset, uncommon for an augmented image. NOT USABLE TO TRAIN A PYTORCH MODEL!
    PYTORCH_TENSOR = 2  # PyTorch tensor (RGB, pixels in range [0, 1], shape CxHxW). Common format of an image augmented using torchvision.
    NUMPY_ARRAY = 3  # NumPy array (RGB, pixels in range [0, 255], shape HxWxC). Commonly used in albumentations (OpenCV). NOT USABLE TO TRAIN A PYTORCH MODEL!


# Note: there is no DatasetOutputImageChannelFormat because we suppose that
# augmentations will prepare the image in the format expected by the model.
# If the model expects a specific format, the user should use a final transformation
# to convert the image to the expected channel format.
# By default, the output image will follow AugmentationImageChannelFormat channel format.


@dataclass
class FormatFlags:
    input_dataset_format: InputDatasetFormat = InputDatasetFormat.AUTODETECT
    input_dataset_image_channels_format: InputDatasetImageChannelFormat = (
        InputDatasetImageChannelFormat.AUTODETECT
    )
    augmentation_argument_format: AugmentationArgumentFormat = (
        AugmentationArgumentFormat.AUTODETECT
    )
    augmentation_input_format: AugmentationInputFormat = (
        AugmentationInputFormat.AUTODETECT
    )
    augmentation_image_channel_format: AugmentationImageChannelFormat = (
        AugmentationImageChannelFormat.AUTODETECT
    )
    dataset_output_format: DatasetOutputFormat = DatasetOutputFormat.AS_TRANFORM_OUTPUT
    dataset_image_output_format: DatasetImageOutputFormat = (
        DatasetImageOutputFormat.AS_TRANFORM_OUTPUT
    )


PYTORCH_OUTPUT_FORMAT = FormatFlags(
    dataset_output_format=DatasetOutputFormat.TUPLE,
    dataset_image_output_format=DatasetImageOutputFormat.PYTORCH_TENSOR,
)

TORCHVISION_FORMAT = dataclasses.replace(
    PYTORCH_OUTPUT_FORMAT,
    augmentation_argument_format=AugmentationArgumentFormat.IMAGE_ONLY,
    augmentation_input_format=AugmentationInputFormat.PIL_IMAGE
    | AugmentationInputFormat.PYTORCH_TENSOR,
    augmentation_image_channel_format=AugmentationImageChannelFormat.RGB,
)

ALBUMENTATIONS_PYTORCH_OUTPUT = dataclasses.replace(
    PYTORCH_OUTPUT_FORMAT,
    augmentation_argument_format=(
        AugmentationArgumentFormat.IMAGE_ONLY | AugmentationArgumentFormat.DICTIONARY
    ),
    augmentation_input_format=AugmentationInputFormat.NUMPY_ARRAY,
    augmentation_image_channel_format=AugmentationImageChannelFormat.RGB,
)

PURE_ALBUMENTATIONS = dataclasses.replace(
    ALBUMENTATIONS_PYTORCH_OUTPUT,
    dataset_output_format=DatasetOutputFormat.DICTIONARY,
    dataset_image_output_format=DatasetImageOutputFormat.AS_TRANFORM_OUTPUT,  # or NUMPY_ARRAY, should be the same
)


__all__ = [
    "InputDatasetFormat",
    "InputDatasetImageChannelFormat",
    "AugmentationArgumentFormat",
    "AugmentationInputFormat",
    "AugmentationImageChannelFormat",
    "DatasetOutputFormat",
    "DatasetImageOutputFormat",
    "FormatFlags",
    "PYTORCH_OUTPUT_FORMAT",
    "TORCHVISION_FORMAT",
    "ALBUMENTATIONS_PYTORCH_OUTPUT",
    "PURE_ALBUMENTATIONS",
]
