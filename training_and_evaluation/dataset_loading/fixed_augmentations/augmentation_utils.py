from typing import Any, List, Tuple, Union
from PIL.Image import Image
from PIL import Image as ImageModule
import numpy as np


def _is_multicrop(image_object: Any) -> bool:
    return isinstance(image_object, (list, tuple))


def _get_image_type(
    image_object: Any, manage_multicrop: bool
) -> Tuple[bool, bool, bool]:
    input_is_pil = _is_pil_image(image_object, manage_multicrop=manage_multicrop)
    input_is_numpy = (not input_is_pil) and _is_numpy_array(
        image_object, manage_multicrop=manage_multicrop
    )
    input_is_torch = (not (input_is_pil or input_is_numpy)) and _is_torch_tensor(
        image_object, manage_multicrop=manage_multicrop
    )

    assert (
        input_is_pil or input_is_numpy or input_is_torch
    ), f"Input dataset image format could not be autodetected for dataset image: {image_object}"

    return input_is_pil, input_is_numpy, input_is_torch


def _rgb_to_bgr(image_object, manage_multicrop: bool):
    if manage_multicrop and _is_multicrop(image_object):
        return [_rgb_to_bgr(image, manage_multicrop=False) for image in image_object]

    input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
        image_object, manage_multicrop=manage_multicrop
    )

    if input_is_pil:
        # Should never happen!
        raise NotImplementedError(
            "PIL (implicit RGB) to BGR conversion not implemented"
        )
    elif input_is_numpy:
        return image_object[..., ::-1]
    else:  #  implicit: input_is_torch
        # https://discuss.pytorch.org/t/torch-tensor-variable-from-rgb-to-bgr/18955/2
        return image_object[..., [2, 1, 0]]


def _bgr_to_rgb(image_object, manage_multicrop: bool):
    if manage_multicrop and _is_multicrop(image_object):
        return [_bgr_to_rgb(image, manage_multicrop=False) for image in image_object]

    input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
        image_object, manage_multicrop=manage_multicrop
    )

    if input_is_pil:
        # May happen if the image was initially loaded with OpenCV and then converted to PIL as-is without caring about
        # the channel order.  We need to explicitly swap the channels!
        # https://stackoverflow.com/a/4661652
        b, g, r = image_object.split()
        image_object = ImageModule.merge("RGB", (r, g, b))
        return image_object
    elif input_is_numpy:
        return image_object[..., ::-1]
    else:  #  implicit: input_is_torch
        # https://discuss.pytorch.org/t/torch-tensor-variable-from-rgb-to-bgr/18955/2
        return image_object[..., [2, 1, 0]]


def _is_torch_tensor(element: Any, manage_multicrop: bool) -> bool:
    if manage_multicrop and _is_multicrop(element):
        return all(_is_torch_tensor(image, manage_multicrop=False) for image in element)

    try:
        import torch

        return isinstance(element, torch.Tensor)
    except ImportError:
        return False


def _is_pil_image(element: Any, manage_multicrop: bool) -> bool:
    if manage_multicrop and _is_multicrop(element):
        return all(_is_pil_image(image, manage_multicrop=False) for image in element)
    return isinstance(element, Image)


def _is_numpy_array(element: Any, manage_multicrop: bool) -> bool:
    if manage_multicrop and _is_multicrop(element):
        return all(_is_numpy_array(image, manage_multicrop=False) for image in element)
    return isinstance(element, np.ndarray)


def _pil_to_torch(image: Union[Image, List[Image]], manage_multicrop: bool) -> Any:
    if manage_multicrop and _is_multicrop(image):
        return [_pil_to_torch(x, manage_multicrop=False) for x in image]

    from torchvision.transforms.functional import to_tensor

    return to_tensor(image)


def _pil_to_numpy(
    image: Union[Image, List[Image]], manage_multicrop: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    if manage_multicrop and _is_multicrop(image):
        return [_pil_to_numpy(x, manage_multicrop=False) for x in image]

    return np.array(image)


def _numpy_to_torch(image: np.ndarray, manage_multicrop: bool) -> Any:
    if manage_multicrop and _is_multicrop(image):
        return [_numpy_to_torch(x, manage_multicrop=False) for x in image]

    from torchvision.transforms.functional import to_tensor

    try:
        return to_tensor(image)
    except ValueError:
        # Fix negative stride error
        return to_tensor(image.copy())


def _numpy_to_pil(
    image: np.ndarray, manage_multicrop: bool
) -> Union[Image, List[Image]]:
    if manage_multicrop and _is_multicrop(image):
        return [_numpy_to_pil(x, manage_multicrop=False) for x in image]

    return ImageModule.fromarray(image)


def _torch_to_pil(image: Any, manage_multicrop: bool) -> Union[Image, List[Image]]:
    if manage_multicrop and _is_multicrop(image):
        return [_torch_to_pil(x, manage_multicrop=False) for x in image]

    from torchvision.transforms.functional import to_pil_image

    return to_pil_image(image)


def _torch_to_numpy(image: Any, manage_multicrop: bool) -> np.ndarray:
    # He has spoken: https://discuss.pytorch.org/t/convert-image-tensor-to-numpy-image-array/22887/3
    # But with a twist: we need to bring it to range [0, 255] and convert it to uint8

    if manage_multicrop and _is_multicrop(image):
        return [_torch_to_numpy(x, manage_multicrop=False) for x in image]

    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _any_to_numpy(image: Any, manage_multicrop: bool) -> np.ndarray:
    if manage_multicrop and _is_multicrop(image):
        return [_any_to_numpy(x, manage_multicrop=False) for x in image]

    input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
        image_object=image, manage_multicrop=False
    )

    if input_is_pil:
        image = _pil_to_numpy(image=image, manage_multicrop=False)
    elif input_is_numpy:
        pass
    else:  #  implicit: input_is_torch
        image = _torch_to_numpy(image=image, manage_multicrop=False)

    return image


def _any_to_pil(image: Any, manage_multicrop: bool) -> Image:
    if manage_multicrop and _is_multicrop(image):
        return [_any_to_pil(x, manage_multicrop=False) for x in image]

    input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
        image_object=image, manage_multicrop=False
    )

    if input_is_pil:
        pass
    elif input_is_numpy:
        image = _numpy_to_pil(image=image, manage_multicrop=False)
    else:  #  implicit: input_is_torch
        image = _torch_to_pil(image=image, manage_multicrop=False)

    return image


def _any_to_torch(image: Any, manage_multicrop: bool) -> Any:
    if manage_multicrop and _is_multicrop(image):
        return [_any_to_torch(x, manage_multicrop=False) for x in image]

    input_is_pil, input_is_numpy, input_is_torch = _get_image_type(
        image_object=image, manage_multicrop=False
    )

    if input_is_pil:
        image = _pil_to_torch(image=image, manage_multicrop=False)
    elif input_is_numpy:
        image = _numpy_to_torch(image=image, manage_multicrop=False)
    else:  #  implicit: input_is_torch
        pass

    return image


def _image_to_format(image: Any, image_type: Any, manage_multicrop: bool = True):
    if manage_multicrop and _is_multicrop(image):
        return [
            _image_to_format(x, image_type=image_type, manage_multicrop=False)
            for x in image
        ]

    image_is_pil, image_is_numpy, image_is_torch = image_type
    if image_is_pil:
        return _any_to_pil(image, manage_multicrop=False)
    elif image_is_numpy:
        return _any_to_numpy(image, manage_multicrop=False)
    else:
        assert image_is_torch
        return _any_to_torch(image, manage_multicrop=False)


__all__ = [
    "_get_image_type",
    "_rgb_to_bgr",
    "_bgr_to_rgb",
    "_is_torch_tensor",
    "_is_pil_image",
    "_is_numpy_array",
    "_pil_to_torch",
    "_pil_to_numpy",
    "_numpy_to_torch",
    "_numpy_to_pil",
    "_torch_to_pil",
    "_torch_to_numpy",
    "_any_to_numpy",
    "_any_to_pil",
    "_any_to_torch",
    "_image_to_format",
]
