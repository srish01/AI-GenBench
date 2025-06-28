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
from generator_builders.generator_utils import DatasetContext, process_generator

USED_FAKE_IMAGES_FOLDER = [
    "dalle3",
    "dalle3_addition",
    "dalle3_addition2",
    "journeydb",
    "wikiart_stylegan",
    "sdxl_faces_fake",
    "sdxl_paintings_fake",
    "sdxl_photos_fake",
    "sd_faces_fake",
    "sd_paintings_fake",
    "sd_photos_fake",
]

IMAGINET_FOLDER_TO_NAMES = {
    "dalle3": "DALL-E 3",
    "dalle3_addition": "DALL-E 3",
    "dalle3_addition2": "DALL-E 3",
    "journeydb": "Midjourney",
    "wikiart_stylegan": "StyleGAN3",
    "sdxl_faces_fake": "Stable Diffusion XL 1.0",
    "sdxl_paintings_fake": "Stable Diffusion XL 1.0",
    "sdxl_photos_fake": "Stable Diffusion XL 1.0",
    "sd_faces_fake": "Stable Diffusion 2.1",
    "sd_paintings_fake": "Stable Diffusion 2.1",
    "sd_photos_fake": "Stable Diffusion 2.1",
}


def imaginet_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = imaginet_get_generator_name(root_path, fpath)

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

            context = DatasetContext()
            context.add_values(
                {
                    "description": lambda: imaginet_get_folder_name(
                        root_path, fpath
                    ).split("_")[1],
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
                "origin_dataset": "Imaginet",
                "paired_real_images": [],
            }


def imaginet_get_generator_name(
    root_path: Union[str, Path], fpath: Union[str, Path]
) -> str:
    folder_name: str = imaginet_get_folder_name(root_path, fpath)
    return IMAGINET_FOLDER_TO_NAMES[folder_name]


def imaginet_get_folder_name(
    root_path: Union[str, Path], fpath: Union[str, Path]
) -> str:
    return Path(fpath).relative_to(Path(root_path)).parts[0]


__all__ = [
    "IMAGINET_FOLDER_TO_NAMES",
    "imaginet_generator",
    "imaginet_get_generator_name",
    "imaginet_get_folder_name",
]
