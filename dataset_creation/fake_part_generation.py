import warnings
from datasets import load_from_disk

import ruamel.yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datasets import DatasetDict

from dataset_utils.common_utils import (
    PathAlike,
    hash_dataset,
    load_image_ids_filelists,
)


from dataset_generation_builders import (
    build_drct_dataset,
    build_genimage_dataset,
    build_dmd_test_dataset,
    build_sfhq_t2i_dataset,
    build_polardiffshield_dataset,
    build_aeroblade_dataset,
    build_elsa_d3_datasets,
    build_dmid_train_valid_datasets,
    build_dmid_test_dataset,
    build_forensynths_trainval_datasets,
    build_forensynths_test_dataset,
    build_artifact_dataset,
    build_synthbuster_dataset,
    build_imaginet_dataset,
)

from dataset_utils.common_utils import (
    PIL_setup,
    check_dataset_format,
    make_train_val_splits,
    merge_all_datasets_and_splits,
    sample_dataset_by_generator,
)


def make_fake_dataset(
    all_paths: Dict[str, Path],
    samples_per_generator: int = 5000,
    num_proc: int = 1,
    convert_to_jpeg: bool = True,
    pre_selected_image_ids_lists: Optional[Dict[str, PathAlike]] = None,
) -> Tuple[DatasetDict, Path]:
    fake_datasets = []
    yaml = ruamel.yaml.YAML()
    PIL_setup()

    configuration_files_path = (
        Path(__file__).resolve().parent / "config_origin_datasets"
    )

    pre_selected_image_ids: Optional[List[str]] = None
    pre_selected_image_ids_dict: Dict[str, List[str]] = dict()
    if pre_selected_image_ids_lists is not None:
        pre_selected_image_ids_dict, expected_hash = load_image_ids_filelists(
            pre_selected_image_ids_lists
        )

        candidate_path = (
            all_paths["intermediate_outputs_path"] / f"fake_complete_{expected_hash}"
        )
        if candidate_path.exists():
            print(f"Using cached dataset {expected_hash}.")
            return load_from_disk(candidate_path), candidate_path

        # Flatten dict to list
        pre_selected_image_ids = []
        for split, file_ids in pre_selected_image_ids_dict.items():
            pre_selected_image_ids.extend(file_ids)

    builder_arguments = {
        "paths": all_paths,
        "configuration_files_path": configuration_files_path,
        "yaml": yaml,
        "convert_to_jpeg": convert_to_jpeg,
        "num_proc": num_proc,
        "pre_selected_image_ids": pre_selected_image_ids,
    }

    fake_datasets.append(build_drct_dataset(**builder_arguments))
    fake_datasets.append(build_genimage_dataset(**builder_arguments))
    fake_datasets.append(build_dmd_test_dataset(**builder_arguments))
    fake_datasets.append(build_sfhq_t2i_dataset(**builder_arguments))
    fake_datasets.append(build_polardiffshield_dataset(**builder_arguments))
    fake_datasets.append(build_aeroblade_dataset(**builder_arguments))
    fake_datasets.extend(build_elsa_d3_datasets(**builder_arguments))
    fake_datasets.extend(build_dmid_train_valid_datasets(**builder_arguments))
    fake_datasets.append(build_dmid_test_dataset(**builder_arguments))
    fake_datasets.extend(build_forensynths_trainval_datasets(**builder_arguments))
    fake_datasets.append(build_forensynths_test_dataset(**builder_arguments))
    fake_datasets.append(build_artifact_dataset(**builder_arguments))
    fake_datasets.append(build_synthbuster_dataset(**builder_arguments))
    fake_datasets.append(build_imaginet_dataset(**builder_arguments))

    print("Merging all datasets...")
    complete_dataset = merge_all_datasets_and_splits(fake_datasets)

    print(f"Complete dataset contains {len(complete_dataset)} samples.")
    check_dataset_format(complete_dataset)
    # visualize_dataset_rows(complete_dataset)

    # Images per generator
    gen_names, gen_count = np.unique(complete_dataset["generator"], return_counts=True)
    print("Images per generator:")
    for gen_name, gen_count in zip(gen_names, gen_count):
        print(f"{gen_name}: {gen_count}")

    if pre_selected_image_ids is None:
        sampled_subset = sample_dataset_by_generator(
            complete_dataset,
            samples_per_generator=samples_per_generator,
            seed=1234,
            stratify_by_origin_dataset=True,
            check_has_all_generators=True,
        )

        # print(f"Sampled subset contains {len(sampled_subset)} samples.")
        # visualize_dataset_rows(sampled_subset)

        dataset_train, dataset_valid = make_train_val_splits(
            sampled_subset, int(len(sampled_subset) * 0.80), seed=1234
        )

        # visualize_dataset_rows(dataset_train)
        # visualize_dataset_rows(dataset_valid)
        unified_fake_dataset = DatasetDict(
            {
                "train": dataset_train,
                "validation": dataset_valid,
            }
        )
    else:
        subsets = dict()
        id_to_index = {x: i for i, x in enumerate(complete_dataset["file_id"])}
        for split, ids_to_take in pre_selected_image_ids_dict.items():
            split_indices = [id_to_index[x] for x in ids_to_take]

            if len(ids_to_take) != len(split_indices):
                raise RuntimeError(
                    f"Not all selected images were found in the dataset. "
                    f"Expected {len(ids_to_take)}, found {len(split_indices)}."
                )

            subsets[split] = complete_dataset.select(split_indices)
        unified_fake_dataset = DatasetDict(subsets)

    dataset_hash = hash_dataset(unified_fake_dataset)
    dataset_hash_digest = dataset_hash.hexdigest()
    final_dataset_path = (
        all_paths["intermediate_outputs_path"] / f"fake_complete_{dataset_hash_digest}"
    )

    if final_dataset_path.exists():
        warnings.warn(
            f"Dataset {dataset_hash_digest} already exists. Will try to re-use it."
        )
        return load_from_disk(final_dataset_path), final_dataset_path

    unified_fake_dataset.save_to_disk(
        final_dataset_path,
        num_proc=num_proc,
        max_shard_size="500MB",
    )

    for split in unified_fake_dataset.keys():
        file_ids = unified_fake_dataset[split]["file_id"]
        file_ids_path = (
            all_paths["intermediate_outputs_path"]
            / f"fake_complete_{dataset_hash_digest}-{split}_file_ids.txt"
        )
        if file_ids_path.exists():
            print(
                f"File IDs file {file_ids_path} already exists. Will not create it again."
            )
            continue
        with open(file_ids_path, "w") as f:
            for file_id in file_ids:
                f.write(f"{file_id}\n")

    return unified_fake_dataset, final_dataset_path


__all__ = [
    "make_fake_dataset",
]
