import argparse
import warnings
from datasets import load_from_disk, DatasetDict
import datasets
from datasets.features import Image
from dataset_utils.common_utils import check_dataset_format
from custom_features.large_image import LargeImage
from pathlib import Path

# Register the LargeImage type
datasets.features.features.register_feature(LargeImage, "LargeImage")


def finalize_and_save_dataset(dataset_path, output_path):
    dataset = load_from_disk(dataset_path)
    dataset = finalize_dataset(dataset)

    print(f"Saving finalized dataset {dataset_path} to {output_path}...")
    dataset.save_to_disk(output_path)
    return dataset


def finalize_dataset(dataset):
    """
    Finalize the dataset by casting the 'image' column to Image type.
    If the dataset is a DatasetDict, it processes each split separately.
    """
    if isinstance(dataset, datasets.DatasetDict):
        splits = dict()
        for split in dataset.keys():
            split_dataset = dataset[split]
            split_dataset = split_dataset.cast_column("image", Image())
            splits[split] = split_dataset

        dataset = DatasetDict(splits)
    else:
        dataset = dataset.cast_column("image", Image())

    return dataset


def needs_finalization(dataset):
    """
    Check if the dataset needs finalization.
    This is determined by checking if the 'image' column is of type LargeImage.
    """
    if isinstance(dataset, datasets.DatasetDict):
        for split in dataset.values():
            if isinstance(split.features["image"], LargeImage):
                return True
    else:
        if isinstance(dataset.features["image"], LargeImage):
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument(
        "--dataset_path", type=str, help="Path to the HuggingFace dataset"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the finalized dataset"
    )

    args = parser.parse_args()
    if Path(args.output_path).exists():
        warnings.warn(
            f"Output path {args.output_path} already exists. Just checking its format!"
        )
    else:
        finalize_and_save_dataset(args.dataset_path, args.output_path)

    print("Checking final dataset format...")
    dataset = load_from_disk(args.output_path)
    check_dataset_format(dataset["train"])
    check_dataset_format(dataset["validation"])
