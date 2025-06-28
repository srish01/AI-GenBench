from pathlib import Path
from typing import Generator, List, Union
from datasets import Dataset, DatasetDict, load_from_disk

from dataset_utils.common_utils import (
    RowDict,
    check_max_samples,
    is_image_valid,
    prepare_image,
)
from generator_builders.deepfake_dataset_builder import AvailableFile


ELSA_D3_GENERATOR_NAMES = [
    "DeepFloyd IF",
    "Stable Diffusion 1.4",
    "Stable Diffusion 2.1",
    "Stable Diffusion XL 1.0",
]


def load_elsa_d3(output_path: Path, num_proc=16) -> DatasetDict:
    print("Loading ELSA_D3 dataset...")
    dataset = load_from_disk(str(output_path))

    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict for ELSA_D3, got {type(dataset)}")

    return dataset


def elsa_d3_split_row(
    dataset: Dataset,
    label: int,
    convert_to_jpeg: bool,
    elsa_split: str,
    considered_generators: Union[int, List[int]],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    if isinstance(considered_generators, int):
        considered_generators = [considered_generators]

    columns_to_keep = [
        "id",
        "positive_prompt",
        "negative_prompt",
    ]

    for gen_id in considered_generators:
        columns_to_keep.append(f"image_gen{gen_id}")

    dataset = dataset.select_columns(columns_to_keep)

    for file_info in filelist:
        file_id = file_info.file_id
        row_index, gen_id = file_info.file_path.split("/")[1:3]
        row_index = int(row_index)
        gen_id = int(gen_id)

        row = dataset[row_index]
        img = row[f"image_gen{gen_id}"]

        if not is_image_valid(img):
            print(f"broken img for generator {gen_id} at row {row_index}")
            continue

        if not check_max_samples(max_samples, gen_id, lock):
            continue

        # Convert to RGB (and JPEG if required)
        img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

        yield {
            "image": img,
            "height": img.height,
            "width": img.width,
            "label": label,
            "generator": ELSA_D3_GENERATOR_NAMES[gen_id],
            "file_id": file_id,
            "description": "",
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
            "conditioning": "text",
            "origin_dataset": f"ELSA_D3/{elsa_split}",
            "paired_real_images": [f'LAION-400M/{row["id"]}'],
        }


__all__ = [
    "ELSA_D3_GENERATOR_NAMES",
    "load_elsa_d3",
    "elsa_d3_split_row",
]
