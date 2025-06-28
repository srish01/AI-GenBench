from pathlib import Path
from dataset_paths import parse_dataset_paths
from collections import Counter
from typing import Dict, Optional
from datasets import DatasetDict, Dataset, concatenate_datasets

from custom_features.large_image import LargeImage

from dataset_utils.common_utils import (
    PathAlike,
    check_dataset_format,
)

import datasets
from datasets import load_from_disk

datasets.features.features.register_feature(LargeImage, "LargeImage")

from fake_part_generation import make_fake_dataset
from real_part_generation import make_real_dataset


# For more info on local.cfg format, see dataset_paths.py
# Example local.cfg:
# [paths]
# input_datasets_path = /deepfake
# output_path = ~/deepfake_benchmark_output
# tmp_cache_dir = ~/deepfake_benchmark_output_cache
# intermediate_outputs_path = ~/deepfake_benchmark_intermediate_outputs
# drct = "/different_folder/DRCT-2M"
# imagenet "/classic_datasets/imagenet"


def main():
    all_paths = parse_dataset_paths("local.cfg")
    final_dataset_path: Path = all_paths["output_path"] / "ai_gen_bench_v1.0.0"
    num_proc = 8
    make_jpeg_dataset = False
    # Note: the expected hash may vary depending on the seed used
    expected_hash_fake_set: Optional[str] = (
        "1262938e987ae3db94c273453cbbb8f4411626b71abe4474444431c86f97d425"
    )
    expected_hash_real_set: Optional[str] = (
        "93e24fd8c2231f5696ab6a7f3a37489bd22077a49ef70ebdf4713cfa1037ef75"
    )

    # --- END OF CONFIGURATION ---

    unified_fake_dataset_path: Optional[Path] = None
    unified_real_dataset_path: Optional[Path] = None

    pre_selected_fake_image_ids_lists: Optional[Dict[str, PathAlike]] = None
    pre_selected_real_image_ids_lists: Optional[Dict[str, PathAlike]] = None

    if expected_hash_fake_set:
        pre_selected_fake_image_ids_lists = {
            "train": all_paths["intermediate_outputs_path"]
            / f"fake_complete_{expected_hash_fake_set}-train_file_ids.txt",
            "validation": all_paths["intermediate_outputs_path"]
            / f"fake_complete_{expected_hash_fake_set}-validation_file_ids.txt",
        }
        unified_fake_dataset_path = all_paths["intermediate_outputs_path"] / (
            "fake_complete_" + expected_hash_fake_set
        )
    if expected_hash_real_set:
        pre_selected_real_image_ids_lists = {
            "train": all_paths["intermediate_outputs_path"]
            / f"real_complete_{expected_hash_real_set}-train_file_ids.txt",
            "validation": all_paths["intermediate_outputs_path"]
            / f"real_complete_{expected_hash_real_set}-validation_file_ids.txt",
        }
        unified_real_dataset_path = all_paths["intermediate_outputs_path"] / (
            "real_complete_" + expected_hash_real_set
        )

    if final_dataset_path.exists():
        raise RuntimeError(
            f"Final dataset already exists at {final_dataset_path}. Please remove it before running the script."
        )

    # max_generators_in_dataset = 26
    samples_per_gen = 5000

    if unified_fake_dataset_path is not None and unified_fake_dataset_path.exists():
        print(f"Using existing fake dataset at {unified_fake_dataset_path}")
        unified_fake_dataset = load_from_disk(unified_fake_dataset_path)
    else:
        unified_fake_dataset, unified_fake_dataset_path = make_fake_dataset(
            all_paths=all_paths,
            num_proc=num_proc,
            samples_per_generator=samples_per_gen,
            convert_to_jpeg=make_jpeg_dataset,
            pre_selected_image_ids_lists=pre_selected_fake_image_ids_lists,
        )
    assert isinstance(unified_fake_dataset, DatasetDict)

    for split in unified_fake_dataset.keys():
        print(f"Split {split} contains {len(unified_fake_dataset[split])} samples.")
        check_dataset_format(unified_fake_dataset[split])
        count_images_per_generator(unified_fake_dataset[split])

    if unified_real_dataset_path is not None and unified_real_dataset_path.exists():
        print(f"Using existing real dataset at {unified_fake_dataset_path}")
        unified_real_dataset = load_from_disk(unified_real_dataset_path)
    else:
        unified_real_dataset = make_real_dataset(
            all_paths=all_paths,
            reference_fake_dataset=unified_fake_dataset,
            num_proc=num_proc,
            seed=1234,
            convert_to_jpeg=make_jpeg_dataset,
            pre_selected_image_ids_lists=pre_selected_real_image_ids_lists,
        )

    complete_dataset = DatasetDict(
        {
            "train": concatenate_datasets(
                [unified_fake_dataset["train"], unified_real_dataset["train"]]
            ),
            "validation": concatenate_datasets(
                [unified_fake_dataset["validation"], unified_real_dataset["validation"]]
            ),
        }
    )

    print("Saving the complete dataset to disk...")
    complete_dataset.save_to_disk(
        final_dataset_path,
        num_proc=num_proc if make_jpeg_dataset else None,
        max_shard_size="500MB",
    )


def count_images_per_generator(
    dataset: Dataset, print_to_stdout: bool = True
) -> Counter:
    dataset = dataset.select_columns(["generator"])
    generator_counts = Counter([("real" if not x else x) for x in dataset["generator"]])

    if print_to_stdout:
        for generator_name, count in generator_counts.items():
            print(f"  {generator_name}: {count} elements")

    return generator_counts


if __name__ == "__main__":
    main()
