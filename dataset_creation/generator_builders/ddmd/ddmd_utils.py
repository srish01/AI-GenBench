from pathlib import Path
from typing import Generator, List, Union
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

DMD_FOLDER_TO_NAMES = {
    "ADM": "ADM",
    "DDPM": "DDPM",
    "Diff-ProjectedGAN": "Diffusion GAN (ProjectedGAN)",
    "Diff-StyleGAN2": "Diffusion GAN (StyleGAN2)",
    "LDM": "Latent Diffusion",
    "ProGAN": "ProGAN",
    "ProjectedGAN": "ProjectedGAN",
    "StyleGAN": "StyleGAN1",  # Supposed
}


def dmd_generator(
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

            generator = dmd_get_generator_name(fpath)

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

            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": "bedroom",
                "positive_prompt": "",
                "negative_prompt": "",
                "conditioning": "noise",
                "origin_dataset": "DDMD",
                "paired_real_images": [],
            }


def dmd_get_generator_name(fpath: Union[str, Path]) -> str:
    model: str = Path(fpath).parts[-2]
    return DMD_FOLDER_TO_NAMES[model]


__all__ = [
    "DMD_FOLDER_TO_NAMES",
    "dmd_generator",
    "dmd_get_generator_name",
]
