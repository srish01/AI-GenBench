from pathlib import Path
from typing import Any, Dict, Generator, List
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


FOLDER_GENERATOR_NAMES_MAPPING = {
    "biggan": "BigGAN",
    "crn": "Cascaded Refinement Networks",
    "cyclegan": "CycleGAN",
    "gaugan": "GauGAN",
    "imle": "IMLE",
    "progan": "ProGAN",
    "stargan": "StarGAN",
    "stylegan": "StyleGAN1",
    "stylegan2": "StyleGAN2",
}

CYCLEGAN_MAPPING = {
    "apple": "orange2apple",
    "horse": "zebra2horse",
    "orange": "apple2orange",
    "summer": "winter2summer",
    "winter": "summer2winter",
    "zebra": "horse2zebra",
}


def forensynths_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    fs_split: str,
    is_test: bool,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            relative_path = fpath.relative_to(root_path)
            if is_test:
                generator = relative_path.parts[0]
                generator = FOLDER_GENERATOR_NAMES_MAPPING[generator]
            else:
                generator = "ProGAN"

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
                    "parent_parent_name": lambda: fpath.parent.parent.name,
                    "translation_type": lambda ctx=context: CYCLEGAN_MAPPING[
                        ctx["parent_parent_name"]
                    ],
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
                "origin_dataset": f"Forensynths/{fs_split}",
                "paired_real_images": [],
            }


__all__ = ["forensynths_generator"]
