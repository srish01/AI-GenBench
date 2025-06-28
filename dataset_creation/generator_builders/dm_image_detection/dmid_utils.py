from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Tuple, Union
from PIL import Image
import re

from dataset_utils.common_utils import (
    RowDict,
    check_max_samples,
    check_needed_samples,
    is_image_valid,
    prepare_image,
)
from generator_builders.deepfake_dataset_builder import AvailableFile
from generator_builders.generator_utils import DatasetContext, process_generator
from dataset_utils.file_extraction import MultiSourceFilesIterator
from generator_builders.imagenet_class_names import IMAGENET_CLASS_ID_TO_STR


DMID_AVAILABLE_SUBSETS = [
    "latent_diffusion_class2image",
    "latent_diffusion_noise2image_bedrooms",
    "latent_diffusion_noise2image_churches",
    "latent_diffusion_noise2image_FFHQ",
    "latent_diffusion_text2img_set0",
    "latent_diffusion_text2img_set1",
    "latent_diffusion_text2img_set2",
]

DMID_NOISE2IMAGE_SUBSETS = [
    "latent_diffusion_noise2image_bedrooms",
    "latent_diffusion_noise2image_churches",
    "latent_diffusion_noise2image_FFHQ",
]

DMID_TEXT2IMAGE_SUBSETS = [
    "latent_diffusion_text2img_set0",
    "latent_diffusion_text2img_set1",
    "latent_diffusion_text2img_set2",
]

DMID_CLASS2IMAGE_SUBSETS = [
    "latent_diffusion_class2image",
]


# Examples:
# latent-diffusion_class2img-imagenet_class-0_000002.png, output 0
# latent-diffusion_class2img-imagenet_class-914_045723.png, output 914
# Use named groups to extract the class id
CLASS_ID_REGEX = re.compile(r"^.*class2img-imagenet_class-(?P<class_id>\d+)_\d+\.png$")


def dmid_generator(
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    subset_type: Literal["text2img", "noise2img", "class2img"],
    dmit_split: str,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]
    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            try:
                img = Image.open(fpath)
            except Exception as e:
                print(f"Problematic image: {fpath} produce error {e}")
                continue

            if not is_image_valid(img):
                print(f"Disarded image: {fpath}")
                continue

            generator = "Latent Diffusion"

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            context = DatasetContext()
            context.add_values(
                {
                    "subset_type": lambda: subset_type,
                    "class_id": lambda: int(
                        _match_regex(CLASS_ID_REGEX, fpath.name, "class_id")
                    ),
                    "class2img_description": lambda ctx=context: IMAGENET_CLASS_ID_TO_STR[
                        ctx["class_id"]
                    ],
                    "noise2img_description": lambda: fpath.parent.name.split("_")[-1],
                }
            )
            values = ["conditioning", "description"]
            final_values = process_generator(
                generator_config, generator, values, context
            )
            conditioning = final_values["conditioning"]
            description = final_values["description"]

            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": description,
                "positive_prompt": "",
                "negative_prompt": "",
                "conditioning": conditioning,
                "origin_dataset": f"DMimageDetection/{dmit_split}",
                "paired_real_images": [],
            }


# biggan_998_531493.png, output 998
BIGGAN_CLASS_ID_REGEX = re.compile(r"^biggan_(?P<class_id>\d+)_\d+\.png$")

# DALL·E 2022-08-13 08.58.24 - A black Honda motorcycle parked in front of a garage.png, output "A black Honda motorcycle parked in front of a garage"
# DALL·E 2022-10-23 19.09.29 - A purple flower is in a watering can on the window sill_358629.png, output "A purple flower is in a watering can on the window sill"
# DALL·E 2022-10-23 19.30.55.png, output ""
DALLE_2_PROMPT_RE = re.compile(r"^DALL·E.+-.+(- (?P<prompt>.*?)(_\d*)?)?\.png$")

# ann000000000414.png, output "000000000414"
COCO2017_ANNOTATION_ID_RE = re.compile(r"^ann(?P<annotation_id>\d+)\.png$")

# img000000001425.png, output "000000001425"
COCO2017_IMAGE_ID_RE = re.compile(r"^img(?P<image_id>\d+)\.png$")


def dmid_test_generator(
    label: int,
    coco_val2017_caption: List[Dict[str, Any]],
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]
    caption_id_to_data = {
        int(annotation["id"]): (annotation["image_id"], annotation["caption"])
        for annotation in coco_val2017_caption
    }
    image_id_to_data = defaultdict(list)
    for annotation in coco_val2017_caption:
        image_id_to_data[int(annotation["image_id"])].append(
            (annotation["id"], annotation["caption"])
        )

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = _dmid_test_get_generator_name(fpath)

            if not check_needed_samples(max_samples, generator, lock):
                continue

            try:
                img = Image.open(fpath)
            except Exception as e:
                print(f"Problematic image: {fpath} produce error {e}")
                continue

            if not is_image_valid(img):
                print(f"Disarded image: {fpath}")
                continue

            conditioning = None
            description = None
            positive_prompt = None
            paired_real_images = []

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            context = DatasetContext()
            context.add_values(
                {
                    "class_id": lambda: _imagenet_class(
                        BIGGAN_CLASS_ID_REGEX, fpath.name
                    )[0],
                    "biggan_description": lambda: _imagenet_class(
                        BIGGAN_CLASS_ID_REGEX, fpath.name
                    )[1],
                    "dall-e_2_positive_prompt": lambda: _match_regex(
                        DALLE_2_PROMPT_RE, fpath.name, "prompt"
                    ),
                    "coco_annotation": lambda: _match_regex(
                        COCO2017_ANNOTATION_ID_RE, fpath.name, "annotation_id"
                    ),
                    "coco_paired_image_id": lambda ctx=context: caption_id_to_data[
                        int(ctx["coco_annotation"])
                    ][0],
                    "coco_positive_prompt": lambda ctx=context: caption_id_to_data[
                        int(ctx["coco_annotation"])
                    ][1],
                    "conditioning_type": lambda: fpath.parent.name.split("_")[-2],
                    "paired_real_set": lambda: fpath.parent.name.split("_")[-1],
                    "noise2image_description": lambda ctx=context: ctx[
                        "paired_real_set"
                    ].removeprefix("LSUN"),
                    "segm2image_coco_image_id": lambda: int(
                        _match_regex(COCO2017_IMAGE_ID_RE, fpath.name, "image_id")
                    ),
                    "segm2image_captions": lambda ctx=context: image_id_to_data[
                        ctx["segm2image_coco_image_id"]
                    ],
                    "segm2image_description": lambda ctx=context: " ".join(
                        [caption for _, caption in ctx["segm2image_captions"]]
                    ),
                    "progan_description": lambda: fpath.name.split("_")[0],
                    "stylegan_dataset": lambda: fpath.parent.name.split("_")[-2],
                    "stylegan_dataset_without_prefix": lambda ctx=context: ctx[
                        "stylegan_dataset"
                    ].removeprefix("lsun"),
                    "imagenet_class_id": lambda: _imagenet_class(
                        CLASS_ID_REGEX, fpath.name
                    )[0],
                    "imagenet_description": lambda: _imagenet_class(
                        CLASS_ID_REGEX, fpath.name
                    )[1],
                    "vqgan_class_id": lambda: fpath.name.split("_")[0],
                    "vqgan_description": lambda ctx=context: IMAGENET_CLASS_ID_TO_STR[
                        int(ctx["vqgan_class_id"])
                    ],
                }
            )
            values = [
                "conditioning",
                "description",
                "positive_prompt",
                "paired_real_image",
            ]
            final_values = process_generator(
                generator_config, generator, values, context
            )
            conditioning = final_values["conditioning"]
            description = final_values["description"]
            positive_prompt: str = final_values["positive_prompt"]
            positive_prompt = positive_prompt.replace("(1)", "")
            if final_values["paired_real_image"]:
                paired_real_images.append(final_values["paired_real_image"])
            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": description,
                "positive_prompt": positive_prompt,
                "negative_prompt": "",
                "conditioning": conditioning,
                "origin_dataset": "DMimageDetection/test",
                "paired_real_images": paired_real_images,
            }


def _imagenet_class(regex: re.Pattern, input: str) -> Tuple[int, str]:
    class_id = int(_match_regex(regex, input, "class_id"))
    assert class_id in IMAGENET_CLASS_ID_TO_STR
    description = IMAGENET_CLASS_ID_TO_STR[class_id]
    return class_id, description


def _match_regex(
    regex: re.Pattern, input: str, group_name: str, allow_empty=True
) -> str:
    match = regex.match(input)
    if match is None:
        raise ValueError(f"Could not match the regex {regex} with the input {input}")

    result = match.group(group_name)
    if result is None and not allow_empty:
        raise ValueError(
            f"Could not find the group {group_name} in the regex match {match}"
        )

    if result is None:
        result = ""

    return result


TEST_FOLDER_TO_NAME = {
    "dalle-mini": "DALL-E Mini",
    "glide": "Glide",
    "biggan": "BigGAN",
    "eg3d": "EG3D",
    "guided-diffusion": "ADM",
    "latent-diffusion": "Latent Diffusion",
    "taming-transformers": "VQGAN",
    "progan": "ProGAN",
    "stylegan2": "StyleGAN2",
    "stylegan3": "StyleGAN3",
}


def _dmid_test_get_generator_name(fpath: Union[str, Path]) -> str:
    fpath = Path(fpath)
    if fpath.parent.name == "dalle_2":
        return "DALL-E 2"

    if fpath.parent.name.startswith("stable_diffusion"):
        return "Stable Diffusion"

    generator_name = fpath.parent.name.split("_")[0]

    return TEST_FOLDER_TO_NAME[generator_name]


__all__ = [
    "DMID_AVAILABLE_SUBSETS",
    "DMID_NOISE2IMAGE_SUBSETS",
    "DMID_TEXT2IMAGE_SUBSETS",
    "DMID_CLASS2IMAGE_SUBSETS",
    "dmid_generator",
]
