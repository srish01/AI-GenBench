import datasets
from custom_features.large_image import LargeImage

datasets.features.features.register_feature(LargeImage, "LargeImage")

from pathlib import Path
from dataset_paths import parse_simple_dataset_paths
from collections import Counter
from typing import Dict, Optional
from datasets import (
    DatasetDict,
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from dataset_preparation.finalize_dataset import finalize_dataset
from dataset_utils.common_utils import (
    PathAlike,
    check_dataset_format,
)

from real_part_generation import make_real_dataset


# For more info on local_simple.cfg format, see dataset_paths.py
# Example local_simple.cfg:
# [paths]
# input_datasets_path = /deepfake
# output_path = ~/deepfake_benchmark_output
# tmp_cache_dir = ~/deepfake_benchmark_output_cache
# intermediate_outputs_path = ~/deepfake_benchmark_intermediate_outputs
# imagenet = /datasets/imagenet
# coco = /datasets/coco
# raise = /datasets/RAISE_all
# laion400m_elsad3_real_train = /deepfake/simple_laion400m_elsad3_real_train_arrow
# laion400m_elsad3_real_validation = /deepfake/simple_laion400m_elsad3_real_validation_arrow


def main():
    all_paths = parse_simple_dataset_paths("local_simple.cfg")
    final_dataset_path: Path = all_paths["output_path"] / "ai_gen_bench_v1.0.0"
    num_proc = 8
    make_jpeg_dataset = False

    # If you already built the real part, put the path here:
    unified_real_dataset_path: Optional[Path] = None

    pre_selected_real_image_ids_lists: Dict[str, PathAlike] = {
        "train": "resources/train_real_file_ids.txt",
        "validation": "resources/validation_real_file_ids.txt",
    }

    # --- END OF CONFIGURATION ---
    unified_fake_dataset: DatasetDict
    unified_real_dataset: DatasetDict

    unified_fake_dataset = load_dataset("lrzpellegrini/AI-GenBench-fake_part")

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
            allow_laion_spare_images=True,  # Absolutely needed to ensure all users are able to create the dataset
        )

    unified_real_dataset = finalize_dataset(unified_real_dataset)

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
