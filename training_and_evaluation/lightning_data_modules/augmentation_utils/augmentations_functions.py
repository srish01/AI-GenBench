from io import BytesIO
import random
from typing import Tuple, Union
import numpy as np

from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import functional as F


from PIL import Image
from scipy.ndimage import gaussian_filter

from dataset_loading.fixed_augmentations.augmentation_utils import (
    _get_image_type,
    _any_to_numpy,
    _image_to_format,
)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def data_augment_blur(img, blur_sig):
    input_img_format = _get_image_type(img, manage_multicrop=False)
    img = _any_to_numpy(img, manage_multicrop=False)
    sig = sample_continuous(blur_sig)
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sig)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sig)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sig)
    img = _image_to_format(img, input_img_format)

    return img


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def cv2_jpg(img, compress_val):
    import cv2

    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


cmp_dict = {
    "cv2": cv2_jpg,
    "cv2_jpg": cv2_jpg,
    "pil": pil_jpg,
    "pil_jpg": pil_jpg,
}


def cmp_from_key(img, compress_val, key):
    return cmp_dict[key](img, compress_val)


def data_augment_cmp(img, cmp_method, cmp_qual):
    input_img_format = _get_image_type(img, manage_multicrop=False)
    img = _any_to_numpy(img, manage_multicrop=False)
    method = sample_discrete(cmp_method)
    qual = sample_discrete(cmp_qual)
    img = cmp_from_key(img, qual, method)
    img = _image_to_format(img, input_img_format)

    return img


def data_augment_rot90(img):
    angle = sample_discrete([0, 90, 180, 270])
    return F.rotate(img, angle, expand=True)


def ensure_croppable(
    img, crop_size: Union[int, Tuple[int, int]], interpolation=InterpolationMode.BICUBIC
):
    crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    minimum_h = crop_size[0] // 2
    minimum_w = crop_size[1] // 2
    height, width = F.get_size(img)

    if width < minimum_w or height < minimum_h:
        img = F.resize(img, [minimum_h, minimum_w], interpolation=interpolation)
    return img


__all__ = [
    "data_augment_blur",
    "data_augment_cmp",
    "data_augment_rot90",
    "ensure_croppable",
]


if __name__ == "__main__":
    # Demonstrates that RandomCrop crashes if the image is smaller than the crop size // 2
    # (which means it only executes when images are very small!)

    # This is a minimal example to reproduce the error

    from torchvision.transforms.v2 import RandomCrop, CenterCrop

    image = Image.new("RGB", (100, 100), color="red")
    image = F.to_tensor(image)

    # This will raise an error
    try:
        RandomCrop(size=(224, 224), pad_if_needed=True, padding_mode="symmetric")(image)
    except:
        print("Error raised")

    # This will not raise an error
    image_minresized = ensure_croppable(
        image, (224, 224), interpolation=InterpolationMode.BILINEAR
    )
    RandomCrop(size=(224, 224), pad_if_needed=True, padding_mode="symmetric")(
        image_minresized
    )
    print("No error raised")

    # Central crop doesn't raise an error
    res = CenterCrop(
        size=(224, 224),
    )(image)
    print("No error raised")

    # Save image
    res = F.to_pil_image(res)
    res.save("test_image.jpg")
    print("Image saved")

    # Random crop with non-symmetric padding doesn't raise an error
    res = RandomCrop(size=(224, 224), pad_if_needed=True, padding_mode="constant")(
        image
    )
    print("No error raised")

    # Save image
    res = F.to_pil_image(res)
    res.save("test_image2.jpg")
    print("Image saved")
