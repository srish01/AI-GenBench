from functools import partial
import warnings

from collections import defaultdict
import itertools
import shutil
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import numpy as np
from datasets import DatasetDict, NamedSplit, Dataset, load_from_disk
from PIL import Image as ImageModule

from dataset_utils.common_utils import (
    PathAlike,
    cached_dataset_dict,
    hash_dataset,
    load_image_ids_filelists,
)


from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    RowDict,
    RowDictPath,
    check_dataset_format,
    check_max_samples,
    create_shared_max_samples,
    is_image_valid,
    prepare_image,
)


# Real images generators
from real_builders.coco_image_builder import COCODatasetManager
from real_builders.imagenet_image_builder import ImagenetDatasetManager
from real_builders.laion_elsa_subset_images_builder import LaionELSAD3Builder
from real_builders.real_images_manager import RealImagesManager
from real_builders.raise_image_builder import RaiseDatasetManager


def make_real_dataset(
    all_paths: Dict[str, Path],
    reference_fake_dataset: DatasetDict,
    num_proc: int = 1,
    seed: int = 1234,
    convert_to_jpeg: bool = True,
    pre_selected_image_ids_lists: Optional[Dict[str, PathAlike]] = None,
    allow_laion_spare_images: bool = False,
) -> DatasetDict:
    split_dataset: Dataset
    result_real_datasets: Dict[Union[str, NamedSplit], Dataset] = dict()

    # Define the order of the splits
    split_order = [
        x
        for x in ["train", "validation", "test"]
        if x in set(reference_fake_dataset.keys())
    ]
    other_splits = sorted(
        [x for x in reference_fake_dataset.keys() if x not in set(split_order)]
    )
    split_order += other_splits

    pre_selected_image_ids_dict: Optional[Dict[str, List[str]]] = None
    if pre_selected_image_ids_lists is not None:
        pre_selected_image_ids_dict, expected_hash = load_image_ids_filelists(
            pre_selected_image_ids_lists
        )

        cached_set, cached_set_path, dataset_key = cached_dataset_dict(
            cache_path=all_paths["intermediate_outputs_path"],
            selected_image_ids=pre_selected_image_ids_dict,
            directory_prefix="real_complete_",
        )
        if cached_set is not None:
            print(f"Using cached dataset {dataset_key}.")
            return load_from_disk(cached_set_path)

    real_images_manager = _make_real_datasets_manager(
        all_paths=all_paths, num_proc=num_proc, convert_to_jpeg=convert_to_jpeg
    )

    (
        selected_images_per_split,
        aligned_images_per_split,
        pre_selected_images_per_split,
    ) = _select_real_images(
        reference_fake_dataset=reference_fake_dataset,
        real_images_manager=real_images_manager,
        pre_selected_image_ids_dict=pre_selected_image_ids_dict,
        split_order=split_order,
        seed=seed,
        allow_laion_spare_images=allow_laion_spare_images,
    )

    # Finally, generate the actual datasets
    for split in split_order:
        print(f"Generating real images dataset for split {split}...")
        split_dataset = reference_fake_dataset[split]
        split_real_images = selected_images_per_split[split]
        split_aligned_images = aligned_images_per_split[split]
        split_pre_selected_images = pre_selected_images_per_split[split]

        priority_images = split_pre_selected_images.union(split_aligned_images)

        # Shuffle all_real_images to distribute "big" images
        # Note: this doesn't change the actual content of the dataset,
        # but it changes the order in which the images are processed and so
        # the index of the images in the dataset.
        split_real_images = sorted(split_real_images)
        np.random.seed(seed)
        np.random.shuffle(split_real_images)

        gen_definition = {
            "generators": {
                0: {
                    "max samples": len(split_dataset),
                }
            }
        }

        n_required_images, lock = create_shared_max_samples(gen_definition)

        print(
            f"Creating real dataset for split {split}, with {len(split_real_images)} candidate images, {len(split_dataset)} required images."
        )

        images_defs = []
        for image_id in split_real_images:
            images_defs.append(real_images_manager.get_image(image_id))

        if set(split_real_images) == priority_images:
            # Performance optimization:
            # If all images are priority images, we don't need have a separate set
            # (makes from_generator startup faster).
            # Usually happens when using a pre-made file list
            priority_images = set()

        split_cache_dir = all_paths["tmp_cache_dir"] / f"real_cache_{split}"
        if split_cache_dir.exists():
            _cleanup_cache(split_cache_dir)

        # Register cleanup handler
        atexit.register(_cleanup_cache, path=split_cache_dir)

        real_set = Dataset.from_generator(
            generator=partial(
                _real_dataset_generator,
                n_required_images,
                lock,
                convert_to_jpeg,
                priority_images,
            ),
            features=DEEPFAKE_DATASET_FEATURES,
            num_proc=num_proc,
            gen_kwargs={
                "selected_images": images_defs,
            },
            cache_dir=str(all_paths["tmp_cache_dir"] / f"real_cache_{split}"),
        )

        assert isinstance(real_set, Dataset)

        print(f"Real dataset size: {len(real_set)}")
        if len(real_set) != len(split_dataset):
            raise RuntimeError(
                f"Real dataset size ({len(real_set)}) does not match fake dataset size ({len(split_dataset)})"
            )
        else:
            print(f"Real dataset size matches fake dataset size ({len(real_set)}). OK!")

        check_dataset_format(real_set)
        result_real_datasets[split] = real_set

    final_real_dataset = DatasetDict(result_real_datasets)
    final_hash_key = hash_dataset(final_real_dataset).hexdigest()
    final_dataset_folder = f"real_complete_{final_hash_key}"
    final_real_dataset.save_to_disk(
        all_paths["intermediate_outputs_path"] / final_dataset_folder,
        num_proc=num_proc,
        max_shard_size="500MB",  # Mitigation for huggingface/datasets#5717
    )

    for split in final_real_dataset.keys():
        file_ids = final_real_dataset[split]["file_id"]
        # Save the file ids to a text file
        file_ids_path = (
            all_paths["intermediate_outputs_path"]
            / f"real_complete_{final_hash_key}-{split}_file_ids.txt"
        )
        if file_ids_path.exists():
            print(
                f"File IDs file {file_ids_path} already exists. Will not create it again."
            )
        else:
            with open(file_ids_path, "w") as f:
                for file_id in file_ids:
                    f.write(f"{file_id}\n")

    return final_real_dataset


def _cleanup_cache(path: PathAlike):
    """
    Cleanup the cache directory.
    """
    path = Path(path)
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path, ignore_errors=True)


def _make_real_datasets_manager(
    all_paths,
    num_proc: int = 1,
    convert_to_jpeg: bool = True,
):

    coco_train = COCODatasetManager(
        root_path=all_paths["coco"] / "train2017",
        captions_json=all_paths["coco"] / "captions_train2017.json",
        split_name="train",
        convert_to_jpeg=convert_to_jpeg,
        num_proc=num_proc,
    )

    coco_val = COCODatasetManager(
        root_path=all_paths["coco"] / "val2017",
        captions_json=all_paths["coco"] / "captions_val2017.json",
        split_name="val",
        convert_to_jpeg=convert_to_jpeg,
        num_proc=num_proc,
    )

    laion_dataset_manager = LaionELSAD3Builder(
        elsa_d3_dataset_path=all_paths.get("elsa_d3", None),
        root_paths=(
            all_paths["laion400m_elsad3_real_train"],
            all_paths["laion400m_elsad3_real_validation"],
        ),
        convert_to_jpeg=convert_to_jpeg,
        num_proc=num_proc,
    )

    imagenet_manager_train = ImagenetDatasetManager(
        all_paths["imagenet"] / "train",
        "train",
        convert_to_jpeg=convert_to_jpeg,
    )

    imagenet_manager_val = ImagenetDatasetManager(
        all_paths["imagenet"] / "val",
        "val",
        convert_to_jpeg=convert_to_jpeg,
    )

    raise_manager = RaiseDatasetManager(
        all_paths["raise"] / "RAISE_all_TIF", convert_to_jpeg=convert_to_jpeg
    )

    real_images_manager = RealImagesManager(
        [
            coco_train,
            coco_val,
            laion_dataset_manager,
            imagenet_manager_train,
            imagenet_manager_val,
            raise_manager,
        ]
    )
    return real_images_manager


def _select_real_images(
    reference_fake_dataset: DatasetDict,
    real_images_manager: RealImagesManager,
    pre_selected_image_ids_dict: Optional[Dict[str, List[str]]],
    split_order: List[str],
    seed: int,
    use_pre_selected_only: bool = True,
    allow_laion_spare_images: bool = False,
):
    pre_selected_images_per_split: Dict[str, Set[str]] = defaultdict(set)
    aligned_images_per_split: Dict[str, Set[str]] = defaultdict(set)
    selected_images_per_split: Dict[str, Set[str]] = defaultdict(set)
    selected_images_so_far: Set[str] = set()

    # Use pre-selected image ids if provided
    if pre_selected_image_ids_dict is not None:
        for split in split_order:
            if split not in pre_selected_image_ids_dict:
                warnings.warn(
                    f"Split {split} not found in pre-selected image ids dictionary."
                )
                continue

            pre_selected_image_ids = pre_selected_image_ids_dict[split]
            if len(pre_selected_image_ids) == 0:
                warnings.warn(
                    f"Split {split} has no pre-selected image ids. Skipping... but you should check this!"
                )
                continue

            print(
                f"Looking for {len(pre_selected_image_ids)} pre-selected images for split {split}..."
            )

            # Check if the pre-selected image ids are valid
            found_image_ids = real_images_manager.available_images(
                among=pre_selected_image_ids
            )

            if set(found_image_ids) != set(pre_selected_image_ids):
                warnings.warn(
                    f"Not all pre-selected image ids were found among real images datasets. "
                    f"Expected {len(pre_selected_image_ids)}, found {len(found_image_ids)}. "
                    f"This is common for scraped dataset (LAION), but not for others... "
                    "The remaining images will be selected randomly."
                )

            # Discard images that were already allocated to other splits
            pre_selected_image_ids = [
                image_id
                for image_id in found_image_ids
                if image_id not in selected_images_so_far
            ]
            print(
                f"Pre-selected {len(pre_selected_image_ids)} images for {split} split (after excluding those already taken or not found)."
            )
            pre_selected_images_per_split[split].update(pre_selected_image_ids)
            selected_images_per_split[split].update(pre_selected_image_ids)
            selected_images_so_far.update(pre_selected_image_ids)

    if use_pre_selected_only:
        if not allow_laion_spare_images:
            print("Using pre-selected images only. Skipping further selection.")
            return (
                selected_images_per_split,
                aligned_images_per_split,
                pre_selected_images_per_split,
            )
        else:
            print(
                "All the pre-selected (from filelist). Selecting spare images from LAION."
            )
            real_images_manager = real_images_manager.keep_builders(
                builder_prefixes=["LAION-400M"]
            )

    # Select aligned images
    for split in split_order:
        print(f"Selecting aligned images for split {split}...")
        split_dataset = reference_fake_dataset[split]
        # Give priority to aligned images (which are usually the real images with the same description, segmentation mask, etc.)
        aligned_real_images_nested: List[List[str]] = split_dataset[
            "paired_real_images"
        ]
        aligned_real_images: Set[str] = set(
            itertools.chain(*aligned_real_images_nested)
        )

        # Images (by prefix):
        searched_by_prefix = defaultdict(int)
        for image_id in aligned_real_images:
            prefix = image_id.split("/")[0]
            searched_by_prefix[prefix] += 1

        print("Looking for images from:")
        for prefix, count in searched_by_prefix.items():
            print(f"{prefix}: {count}")

        # Check if those images exist
        aligned_images_ids = sorted(
            real_images_manager.available_images(among=aligned_real_images)
        )
        print(
            f"Found {len(aligned_images_ids)} aligned images for {split} split (among {len(aligned_real_images)} possible ones)."
        )

        # Not found images (by prefix):
        not_found_images = aligned_real_images - set(aligned_images_ids)
        not_found_by_prefix = defaultdict(int)
        for image_id in not_found_images:
            prefix = image_id.split("/")[0]
            not_found_by_prefix[prefix] += 1

        print("Not found images by prefix:")
        for prefix, count in not_found_by_prefix.items():
            print(f"{prefix}: {count}")

        # Discard images that were already allocated to other splits
        aligned_images_ids = [
            image_id
            for image_id in aligned_images_ids
            if image_id not in selected_images_so_far
        ]

        print(
            f"Taken {len(aligned_images_ids)} aligned images for {split} split (after excluding those already taken)."
        )

        aligned_images_per_split[split].update(aligned_images_ids)
        selected_images_per_split[split].update(aligned_images_ids)
        selected_images_so_far.update(aligned_images_ids)
        print()

    # Then, pick complementary images
    for split in split_order:
        print(f"Selecting complementary images for split {split}...")
        split_dataset = reference_fake_dataset[split]
        images_current_split = selected_images_per_split[split]

        # Select complementary images
        # Consider the len of fake dataset + 40% minus the aligned images
        # Those additional images are included as many of them are not valid
        # (corrupted, decompression bombs, too small, too big) and we need to
        # have enough candidate images to fill the dataset.
        n_to_select = int(len(split_dataset) * 1.4 - len(images_current_split))
        complementary_images = real_images_manager.select_random_images(
            num_images=n_to_select, seed=seed, excluding=selected_images_so_far
        )

        print(
            f"Found {len(complementary_images)} complementary images for {split} split."
        )

        complementary_by_prefix = defaultdict(int)
        for image_id in complementary_images:
            prefix = image_id.split("/")[0]
            complementary_by_prefix[prefix] += 1

        print("Complementary images by prefix:")
        for prefix, count in complementary_by_prefix.items():
            print(f"{prefix}: {count}")
        print()

        selected_images_per_split[split].update(complementary_images)
        # assert len(selected_images_per_split[split]) >= len(split_dataset), (
        #     f"Picked real images number ({len(selected_images_per_split[split])}) is smaller than the fake dataset size ({len(split_dataset)})"
        #     f" for split {split}. "
        # )

        selected_images_so_far.update(complementary_images)

    return (
        selected_images_per_split,
        aligned_images_per_split,
        pre_selected_images_per_split,
    )


def _real_dataset_generator(
    n_required_images,
    lock,
    convert_to_jpeg: bool,
    priority_image_ids: Set[str],
    selected_images: List[RowDictPath],
):
    # split_aligned_images have priority over others
    img_def: RowDictPath
    result: RowDict

    priority_images = [
        img_def
        for img_def in selected_images
        if img_def["file_id"] in priority_image_ids
    ]

    other_images = [
        img_def
        for img_def in selected_images
        if img_def["file_id"] not in priority_image_ids
    ]

    for img_def in priority_images + other_images:
        img = None
        try:
            fpath = img_def["image"]
            if isinstance(fpath, (str, Path)):
                img = ImageModule.open(fpath)
            else:
                ref_dset, ref_index = fpath
                img = ref_dset[ref_index]["image"]
        except:
            continue

        if not is_image_valid(img):
            # print(f"Problematic image: {img_def['filename']}, size: {img.size}")
            continue

        # Convert to RGB (and JPEG if required)
        img = prepare_image(img, convert_to_jpeg=convert_to_jpeg, lazy_load=True)

        if not check_max_samples(n_required_images, 0, lock):
            break

        result = {
            "image": img,
            "height": img.height,
            "width": img.width,
        }

        for key in img_def.keys():
            if key == "image":
                continue
            result[key] = img_def[key]

        yield result


__all__ = ["make_real_dataset"]
