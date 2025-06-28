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
from generator_builders.generator_utils import (
    DatasetContext,
    create_dict_from_csv,
    process_generator,
)


AEROBLADE_FOLDER_TO_NAMES = {
    "midjourney-v4": "Midjourney 4",
    "midjourney-v5": "Midjourney 5",
    "midjourney-v5-1": "Midjourney 5.1",
    "CompVis-stable-diffusion-v1-1-ViT-L-14-openai": "Stable Diffusion 1.1",
    "runwayml-stable-diffusion-v1-5-ViT-L-14-openai": "Stable Diffusion 1.5",
    "stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k": "Stable Diffusion 2.1",
}

PROMPT_FOLDER_BASE_PATH = "data/raw/prompts"


def aeroblade_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    prompt_dict1 = create_dict_from_csv(
        root_path / PROMPT_FOLDER_BASE_PATH / "ViT-L-14-openai.csv"
    )
    prompt_dict2 = create_dict_from_csv(
        root_path / PROMPT_FOLDER_BASE_PATH / "ViT-H-14-laion2b_s32b_b79k.csv"
    )

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = aeroblade_get_generator_name(root_path, fpath)

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
                    "img_id": lambda: str(fpath.parts[-1].split(".")[0]),
                    "prompt1": lambda ctx=context: prompt_dict1[ctx["img_id"]],
                    "prompt2": lambda ctx=context: prompt_dict2[ctx["img_id"]],
                }
            )
            values = ["positive prompt"]
            final_values = process_generator(
                generator_config, generator, values, context
            )
            prompt = final_values["positive prompt"]
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
                "origin_dataset": "Aeroblade",
                "paired_real_images": [],
            }


def aeroblade_get_generator_name(root_path: Path, fpath: Union[str, Path]) -> str:
    folder_name: str = aeroblade_get_folder_name(root_path, fpath)
    return AEROBLADE_FOLDER_TO_NAMES[folder_name]


def aeroblade_get_folder_name(root_path: Path, fpath: Union[str, Path]) -> str:
    return Path(fpath).relative_to(root_path).parts[-2]


__all__ = [
    "AEROBLADE_FOLDER_TO_NAMES",
    "aeroblade_generator",
    "aeroblade_get_generator_name",
    "aeroblade_get_folder_name",
]
