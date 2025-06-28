from pathlib import Path
from typing import Dict, Generator, List, Union
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
from generator_builders.generator_utils import create_dict_from_csv


SFHQ_T2I_FOLDER_TO_NAMES = {
    "FLUX1_schnell": "FLUX 1 Schnell",
    "FLUX1_dev": "FLUX 1 Dev",
    "FLUX1_pro": "FLUX 1 Pro",
    "SDXL": "Stable Diffusion XL 1.0",
    "DALLE3": "DALL-E 3",
}


def sfhq_t2i_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]
    prompt_dict: Dict[str, str] = create_dict_from_csv(
        root_path / "SFHQ_T2I_dataset.csv", value_id=2
    )
    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id
            generator = sfhq_t2i_get_generator_name(fpath)

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

            prompt = prompt_dict[fpath.parts[-1]]
            yield {
                "image": img,
                "width": img.width,
                "height": img.height,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": "face",
                "positive_prompt": prompt,
                "negative_prompt": "",
                "conditioning": "text",
                "origin_dataset": "SFHQ-T2I",
                "paired_real_images": [],
            }


def sfhq_t2i_get_generator_name(fpath: Union[str, Path]) -> str:
    file_split = Path(fpath).parts[-1].split("_")
    if len(file_split) == 3:
        gen = file_split[0]
    elif len(file_split) == 4:
        gen = file_split[0] + "_" + file_split[1]
    else:
        raise ValueError(f"Unexpected file name: {fpath}")
    return SFHQ_T2I_FOLDER_TO_NAMES[gen]


__all__ = [
    "SFHQ_T2I_FOLDER_TO_NAMES",
    "sfhq_t2i_generator",
]
