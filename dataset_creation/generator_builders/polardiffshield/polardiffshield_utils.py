import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Union
from PIL import Image
from dataset_utils.common_utils import (
    RowDict,
    check_max_samples,
    check_needed_samples,
    is_image_valid,
    prepare_image,
)
from dataset_utils.file_extraction import MultiSourceFilesIterator
from generator_builders.deepfake_dataset_builder import AvailableFile


POLARDIFFSHIELD_FOLDER_TO_NAMES = {
    "Dall-e-2": "DALL-E 2",
    "Dall-e-3": "DALL-E 3",
    "firefly1": "Firefly 1",
    "firefly2": "Firefly 2",
    "Glide": "Glide",
    "midjourney5.0": "Midjourney 5",
    "midjourney5.1": "Midjourney 5.1",
    "midjourney5.2": "Midjourney 5.2",
    "Stable-Diffusion1-1": "Stable Diffusion 1.1",
    "Stable-Diffusion1-2": "Stable Diffusion 1.2",
    "Stable-Diffusion1-3": "Stable Diffusion 1.3",
    "Stable-Diffusion1-4": "Stable Diffusion 1.4",
    "Stable-Diffusion-2-1": "Stable Diffusion 2.1",
    "Stable-Diffusion-XL": "Stable Diffusion XL 1.0",
}

POLARDIFFSHIELD_MODEL_DATA_NAMES = ["md ver", "sd 1.x model", "ff modelVersion"]


def polardiffshield_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = polardiffshield_get_generator_name(root_path, fpath)

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

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            prompt = polardiffshield_get_image_data(fpath)["prompt"]
            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": "",
                "positive_prompt": prompt,
                "negative_prompt": "",
                "conditioning": "text",
                "origin_dataset": "Polardiffshield",
                "paired_real_images": [],
            }


def polardiffshield_get_generator_name(root_path: Path, fpath: Union[str, Path]) -> str:
    fpath = Path(fpath)
    folder_name: str = polardiffshield_get_folder_name(root_path, fpath)
    model_data: Dict[str, Any] = polardiffshield_get_image_data(fpath)
    model: str = ""
    for key, value in model_data.items():
        if key in POLARDIFFSHIELD_MODEL_DATA_NAMES:
            model = str(value)
            break
    return POLARDIFFSHIELD_FOLDER_TO_NAMES[folder_name + model]


def polardiffshield_get_folder_name(root_path: Path, fpath: Path) -> str:
    return fpath.relative_to(root_path).parts[0]


def polardiffshield_get_image_data(fpath: Path) -> Dict[str, Any]:
    json_path = Path(str(fpath).split(".")[0] + ".json")
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


__all__ = [
    "POLARDIFFSHIELD_FOLDER_TO_NAMES",
    "polardiffshield_generator",
    "polardiffshield_get_generator_name",
    "polardiffshield_get_folder_name",
    "polardiffshield_get_image_data",
]
