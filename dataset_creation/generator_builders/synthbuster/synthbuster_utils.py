import csv
from pathlib import Path
from typing import Generator, List
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

SYNTHBUSTER_GENERATORS = {
    "dalle2": "DALL-E 2",
    "dalle3": "DALL-E 3",
    "firefly": "Firefly",
    "glide": "Glide",
    "midjourney-v5": "Midjourney 5",
    "stable-diffusion-1-3": "Stable Diffusion 1.3",
    "stable-diffusion-1-4": "Stable Diffusion 1.4",
    "stable-diffusion-2": "Stable Diffusion 2",
    "stable-diffusion-xl": "Stable Diffusion XL 1.0",
}


def synthbuster_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    prompts = dict()
    metadata_path = root_path / "prompts.csv"
    with metadata_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts[row["image name (matching Raise-1k)"]] = row["Prompt"]

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = SYNTHBUSTER_GENERATORS[fpath.parent.name]

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

            raise_paired_file = fpath.stem
            prompt = prompts[raise_paired_file]

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
                "origin_dataset": "Synthbuster",
                "paired_real_images": [f"RAISE/{raise_paired_file}"],
            }


__all__ = [
    "SYNTHBUSTER_GENERATORS",
    "synthbuster_generator",
]
