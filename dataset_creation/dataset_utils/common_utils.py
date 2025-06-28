from collections import defaultdict
from multiprocessing import Manager
from pathlib import Path
import pickle
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    Self,
)
from typing import TypedDict
from collections.abc import Buffer
import warnings
import PIL
import datasets
from datasets import (
    Dataset,
    ClassLabel,
    Value,
    Features,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
import numpy as np
from custom_features.large_image import LargeImage
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split

import hashlib


class HASH(Protocol):
    @property
    def digest_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def name(self) -> str: ...
    def copy(self) -> Self: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def update(self, obj: Buffer, /) -> None: ...


PathAlike = Union[str, Path]

REAL_IMAGES_LABEL = 0
FAKE_IMAGES_LABEL = 1

IMAGE_MIN_SIZE = 200
IMG_MAX_PIXELS = 49766400


DEEPFAKE_DATASET_FEATURES = Features(
    {
        "image": LargeImage(),
        "height": Value(dtype="uint32"),
        "width": Value(dtype="uint32"),
        "label": ClassLabel(num_classes=2, names=["real", "fake"]),
        "generator": Value(dtype="large_string"),
        "file_id": Value(dtype="large_string"),
        "description": Value(dtype="large_string"),
        "positive_prompt": Value(dtype="large_string"),
        "negative_prompt": Value(dtype="large_string"),
        "conditioning": Value(dtype="large_string"),
        "origin_dataset": Value(dtype="large_string"),
        "paired_real_images": datasets.features.LargeList(
            datasets.Value(dtype="large_string")
        ),
    }
)


class RowDict(TypedDict):
    image: Image.Image
    height: int
    width: int
    label: int
    generator: str
    file_id: str
    description: str
    positive_prompt: str
    negative_prompt: str
    conditioning: str
    origin_dataset: str
    paired_real_images: List[str]


class RowDictPath(TypedDict):
    image: Union[Path, Tuple[Dataset, int]]
    label: int
    generator: str
    file_id: str
    description: str
    positive_prompt: str
    negative_prompt: str
    conditioning: str
    origin_dataset: str
    paired_real_images: List[str]


ALLOWED_GENERATOR_NAMES: Set[str] = {
    "ADM",
    "BigGAN",
    "CIPS",
    "Cascaded Refinement Networks",
    "CycleGAN",
    "DALL-E 2",
    "DALL-E 3",
    "DALL-E Mini",
    "DDPM",
    "DeepFloyd IF",
    "Denoising Diffusion GAN",
    "Diffusion GAN (ProjectedGAN)",
    "Diffusion GAN (StyleGAN2)",
    "EG3D",
    "FaceSynthetics",
    "Firefly",
    "Firefly 1",
    "Firefly 2",
    "FLUX 1 Dev",
    "FLUX 1 Pro",
    "FLUX 1 Schnell",
    "GANformer",
    "GauGAN",
    "Glide",
    "IMLE",
    "LaMa",
    "Latent Diffusion",
    "MAT",
    "Midjourney",
    "Midjourney 4",
    "Midjourney 5",
    "Midjourney 5.1",
    "Midjourney 5.2",
    "Palette",
    "ProGAN",
    "ProjectedGAN",
    "SN-PatchGAN",
    "Stable Diffusion",
    "Stable Diffusion 1.1",
    "Stable Diffusion 1.2",
    "Stable Diffusion 1.3",
    "Stable Diffusion 1.4",
    "Stable Diffusion 1.5",
    "Stable Diffusion 2",
    "Stable Diffusion 2.1",
    "Stable Diffusion Turbo",
    "Stable Diffusion XL 1.0",
    "Stable Diffusion XL Turbo",
    "StarGAN",
    "StyleGAN1",
    "StyleGAN2",
    "StyleGAN2 (SFHQ)",
    "StyleGAN3",
    "VQ-Diffusion",
    "VQGAN",
    "Wukong",
}


def check_dataset_format(dataset: Dataset):
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"dataset must be an instance of datasets.Dataset, not {type(dataset)}"
        )

    columns = set(dataset.column_names)
    expected_columns = set(DEEPFAKE_DATASET_FEATURES.keys())
    assert (
        expected_columns == columns
    ), f"Columns are {columns} (missing columns: {expected_columns - columns}, extra columns: {columns - expected_columns})"

    generator_column_values = set(dataset["generator"])
    if "" in generator_column_values:
        generator_column_values.remove("")
    if not generator_column_values.issubset(ALLOWED_GENERATOR_NAMES):
        unallowed = generator_column_values - ALLOWED_GENERATOR_NAMES
        raise ValueError(
            f"generator column values must be a subset of {ALLOWED_GENERATOR_NAMES}. Unexpected: {unallowed}"
        )

    # Try loading the first and last rows
    check_row_format(dataset[0])
    check_row_format(dataset[-1])

    # Check if the dataset doesn't contain duplicate filenames
    file_ids = dataset["file_id"]
    if len(file_ids) != len(set(file_ids)):
        n_duplicates = len(file_ids) - len(set(file_ids))
        raise ValueError(
            f"The dataset contains {n_duplicates} duplicate filenames. Please ensure all filenames are unique."
        )


def check_row_format(row):
    if not isinstance(row, dict):
        raise ValueError("row must be a dictionary")

    if not isinstance(row["image"], Image.Image):
        raise ValueError("image must be a PIL Image")

    if not isinstance(row["height"], int):
        raise ValueError("height must be an integer")

    if not isinstance(row["width"], int):
        raise ValueError("width must be an integer")

    if not isinstance(row["label"], int):
        raise ValueError("label must be an integer")

    if not row["label"] in {REAL_IMAGES_LABEL, FAKE_IMAGES_LABEL}:
        raise ValueError(
            f"label must be either {REAL_IMAGES_LABEL} or {FAKE_IMAGES_LABEL}"
        )

    if not isinstance(row["generator"], str):
        raise ValueError("generator must be a string")

    if not isinstance(row["file_id"], str):
        raise ValueError("file_id must be a string")

    if not isinstance(row["description"], str):
        raise ValueError("description must be a string")

    if not isinstance(row["positive_prompt"], str):
        raise ValueError("positive_prompt must be a string")

    if not isinstance(row["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")

    if not isinstance(row["conditioning"], str):
        raise ValueError("is_noise2image must be a boolean")

    if not isinstance(row["origin_dataset"], str):
        raise ValueError("dataset must be a string")

    if not isinstance(row["paired_real_images"], list):
        raise ValueError("paired_real_images must be a list")

    for img in row["paired_real_images"]:
        if not isinstance(img, str):
            raise ValueError("paired_real_images must be a list of strings")


def prepare_image(
    image: Image.Image,
    convert_to_jpeg: bool,
    convert_to_rgb: bool = True,
    jpeg_quality: int = 95,
    lazy_load: bool = False,
) -> Image.Image:
    # If converting to JPEG, we must convert to RGB
    if convert_to_jpeg:
        convert_to_rgb = True

    if not lazy_load:
        image.load()
    image.info.pop("xmp", None)
    # Convert to RGB if needed and handle transparency (transparency to white background)
    if convert_to_rgb and image.mode != "RGB":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))

        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")

    # Convert to JPEG if required
    if convert_to_jpeg:
        with BytesIO() as f:
            image.save(f, format="JPEG", quality=jpeg_quality)
            f.seek(0)
            image = Image.open(f)
            image.load()

    return image


def save_image_ids(pkl_file_path, ids):
    Path(pkl_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(ids, f)


def load_image_ids(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        ids = pickle.load(f)
    return ids


def make_empty_dataset():
    return Dataset.from_dict(
        {
            "image": [],
            "height": [],
            "width": [],
            "label": [],
            "generator": [],
            "file_id": [],
            "description": [],
            "positive_prompt": [],
            "negative_prompt": [],
            "conditioning": [],
            "origin_dataset": [],
            "paired_real_images": [],
        },
        features=DEEPFAKE_DATASET_FEATURES,
    )


def sort_paths(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=lambda x: str(x))


def merge_all_datasets_and_splits(
    datasets: List[Union[Dataset, DatasetDict]],
) -> Dataset:
    all_sets_flat = []
    for dataset in datasets:
        if isinstance(dataset, Dataset):
            all_sets_flat.append(dataset)
        else:
            for split in dataset:
                all_sets_flat.append(dataset[split])

    return concatenate_datasets(all_sets_flat)


def sample_dataset_by_generator(
    dataset: Dataset,
    samples_per_generator: int,
    seed: int = 1234,
    stratify_by_origin_dataset: bool = True,
    check_has_all_generators: bool = False,
):
    check_dataset_format(dataset)

    available_dataset_generators: Set[str] = set(dataset["generator"])
    if "" in available_dataset_generators:
        available_dataset_generators.remove("")
    if check_has_all_generators:
        if available_dataset_generators != ALLOWED_GENERATOR_NAMES:
            missing = ALLOWED_GENERATOR_NAMES - available_dataset_generators
            raise ValueError(
                f"generator column values must conver all generators: {ALLOWED_GENERATOR_NAMES}. Missing: {missing}."
            )

    # indices = list(range(len(dataset)))
    generators: List[str] = list(dataset["generator"])
    file_ids: List[str] = list(dataset["file_id"])
    origin_datasets: List[str] = list(dataset["origin_dataset"])

    generator_datasets = []

    # Also this "sorted" is necessary to ensure deterministic sampling
    ordered_dataset_generators = sorted(available_dataset_generators)
    discarded_generators = []
    for gen in ordered_dataset_generators:
        indices = [i for i in range(len(dataset)) if generators[i] == gen]

        # Sort indices by file_id
        # This is important to ensure deterministic sampling: elements may not be
        # already sorted by file_id in the dataset
        indices.sort(key=lambda i: file_ids[i])

        stratify = [origin_datasets[i] for i in indices]

        if len(indices) < samples_per_generator:
            discarded_generators.append(gen)
            continue
        elif len(indices) == samples_per_generator:
            subset_indices = indices
        elif (
            stratify_by_origin_dataset
            and np.unique(stratify, return_counts=True)[1].min() < 2
        ):
            warnings.warn(
                f"Generator {gen} has only one sample for some of the origin datasets"
            )

            # Take that single element from those generators
            subset_indices = []
            unique_gens, counts = np.unique(stratify, return_counts=True)
            for gen, count in zip(unique_gens, counts):
                if count == 1:
                    subset_indices.append(indices[stratify.index(gen)])

            # Remove the selected indices from the list
            indices, stratify = zip(
                *[(i, s) for i, s in zip(indices, stratify) if i not in subset_indices]
            )

            subset_indices.extend(
                train_test_split(
                    indices,
                    stratify=stratify,
                    train_size=samples_per_generator - len(subset_indices),
                    random_state=seed,
                )[0]
            )
        elif stratify_by_origin_dataset:
            subset_indices, _ = train_test_split(
                indices,
                stratify=stratify,
                train_size=samples_per_generator,
                random_state=seed,
            )
        else:
            subset_indices, _ = train_test_split(
                indices,
                train_size=samples_per_generator,
                random_state=seed,
            )

        generator_datasets.append(dataset.select(subset_indices))

    if len(discarded_generators) > 0:
        warnings.warn(
            f"Discarded generators: {discarded_generators}. "
            f"Not enough samples (minimum {samples_per_generator} required)."
        )

    return concatenate_datasets(generator_datasets)


def make_train_val_splits(
    dataset: Dataset,
    train_samples: int,
    seed: int = 1234,
    stratify_by_generator: bool = True,
):
    check_dataset_format(dataset)

    if stratify_by_generator:
        stratify = list(dataset["generator"])
    else:
        stratify = None

    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        train_size=train_samples,
        stratify=stratify,
        random_state=seed,
    )

    return dataset.select(train_indices), dataset.select(test_indices)


X = TypeVar("X")
Y = TypeVar("Y")


def saturating_choice_count(
    n_values: int, values: List[X], discriminators: List[Y], seed: Optional[int] = None
) -> Tuple[Dict[Y, List[X]], Dict[Y, int]]:
    if seed is not None:
        random.seed(seed)

    values_per_discriminator: Dict[Y, List[X]] = defaultdict(list)
    for value, discriminator in zip(values, discriminators):
        values_per_discriminator[discriminator].append(value)

    base_n = n_values // len(values_per_discriminator.keys())

    n_values_per_discriminator: Dict[Y, int] = dict()

    for builder_key, disc_values in values_per_discriminator.items():
        n_values_per_discriminator[builder_key] = min(base_n, len(disc_values))

    remaining_values = n_values - sum(n_values_per_discriminator.values())
    remainder_discriminators = [
        builder_idx
        for builder_idx, images in values_per_discriminator.items()
        if len(images) > n_values_per_discriminator[builder_idx]
    ]

    while remaining_values > 0 and len(remainder_discriminators) > 0:
        if remaining_values > len(remainder_discriminators):
            for builder_idx in remainder_discriminators:
                n_values_per_discriminator[builder_idx] += 1
                remaining_values -= 1

            remainder_discriminators = [
                builder_idx
                for builder_idx in remainder_discriminators
                if len(values_per_discriminator[builder_idx])
                > n_values_per_discriminator[builder_idx]
            ]
        else:
            builder_idx = random.choice(remainder_discriminators)
            n_values_per_discriminator[builder_idx] += 1
            remaining_values -= 1
            if (
                len(values_per_discriminator[builder_idx])
                == n_values_per_discriminator[builder_idx]
            ):
                remainder_discriminators.remove(builder_idx)

    if remaining_values > 0:
        warnings.warn(
            f"Could not select {remaining_values} values from the available disciminators."
        )

    return values_per_discriminator, n_values_per_discriminator


def saturating_balanced_choice(
    n_values: int,
    values: List[X],
    discriminators: List[Y],
    seed: Optional[int] = None,
    choice_fn: Optional[Callable[[Y, List[X], int], List[X]]] = None,
) -> Tuple[List[X], List[Y]]:
    if len(values) < n_values:
        return values, discriminators

    if choice_fn is None:
        choice_fn = lambda x, y, z: random.sample(y, z)

    values_per_discriminator, n_values_per_disciminator = saturating_choice_count(
        n_values, values, discriminators, seed
    )

    if seed is not None:
        random.seed(seed + 1)
    selected_images: List[X] = []
    selected_discriminators: List[Y] = []
    for builder, n_images in n_values_per_disciminator.items():
        selected_images.extend(
            choice_fn(builder, values_per_discriminator[builder], n_images)
        )
        selected_discriminators.extend([builder] * n_images)

    return selected_images, selected_discriminators


def is_image_valid(image: Image.Image) -> bool:
    """
    Checks if an image has both its width and height greater than or equal to a minimum size.

    Parameters:
        image (Image.Image): The PIL Image object to be checked.

    Returns:
        bool: True if the image has both dimensions >= IMAGE_MIN_SIZE, False otherwise.
    """
    try:
        width, height = image.size
        return width >= IMAGE_MIN_SIZE and height >= IMAGE_MIN_SIZE
    except Exception as e:
        raise ValueError(f"An error occurred while processing the image: {e}")


def create_shared_max_samples(input_dict: Dict[str, Any]):
    """
    Creates a shared dictionary with 'max samples' for each generator
    and a lock for thread-safe operations.

    Parameters:
        input_dict (dict): A dictionary with a 'generators' key mapping to generator details.

    Returns:
        Tuple[DictProxy, Lock]: A shared dictionary proxy containing 'max samples' for each generator
                                and a manager-controlled lock.
    """
    manager = Manager()
    generators = input_dict.get("generators", {})
    shared_max_samples = manager.dict(
        {key: value["max samples"] for key, value in generators.items()}
    )
    lock = manager.Lock()
    return shared_max_samples, lock


def check_max_samples(max_samples, generator, lock):
    """
    Thread-safely checks and decrements the sample count for a generator.

    Args:
        max_samples (dict): Shared dictionary of generator names and sample counts.
        generator (str): Generator to check and update.
        lock (Lock): Lock for thread-safe access.

    Returns:
        bool: `True` if decremented successfully, `False` if there are already enough samples.
    """
    with lock:
        if max_samples[generator] == 0:
            return False
        max_samples[generator] -= 1
        return True


def check_needed_samples(max_samples, generator, lock):
    """
    Thread-safely checks the sample count for a generator without decrementing.

    Args:
        max_samples (dict): Shared dictionary of generator names and sample counts.
        generator (str): Generator to check and update.
        lock (Lock): Lock for thread-safe access.

    Returns:
        bool: `True` if samples are needed for the generator, `False` otherwise.
    """
    with lock:
        return max_samples[generator] > 0


def are_all_generated(max_samples, lock):
    """
    Thread-safely checks if all generators have been used.

    Args:
        max_samples (dict): Shared dictionary of generator names and sample counts.
        lock (Lock): Lock for thread-safe access.

    Returns:
        bool: `True` if all generators have been used, `False` otherwise.
    """
    all_generated = True
    with lock:

        # return all(value == 0 for value in max_samples.values())
        for generator, value in max_samples.items():
            if value > 0:
                warnings.warn(
                    f"Generator {generator} has {value} samples left. "
                    "Possible reasons: i) 'samples' and/or 'max samples' was not set correctly in the config; "
                    "ii) the generator contains too many invalid/corrupted images; "
                    "iii) you are missing images in the dataset folder."
                )
                all_generated = False

    return all_generated


def PIL_setup():
    """
    Add a limit to the max nubmer of pixels in an image.
    """
    PIL.Image.MAX_IMAGE_PIXELS = IMG_MAX_PIXELS // 2


def update_hasher_image_ids(selected_image_ids: Iterable[str], hasher: HASH):
    # Concat all strings and convert them to bytes
    all_ids = "".join(sorted(selected_image_ids)).encode()
    hasher.update(all_ids)


def hash_image_ids(selected_image_ids: Iterable[str]) -> str:
    """
    Returns a hex digest of the image IDs.

    Note: image IDs are internally sorted to ensure deterministic hashing.
    """
    hasher = _default_hasher()
    update_hasher_image_ids(selected_image_ids, hasher)
    return hasher.hexdigest()


def cached_dataset_dict(
    cache_path: Path,
    selected_image_ids: Mapping[str, Iterable[str]],
    directory_prefix: str = "real_",
) -> Tuple[Optional[DatasetDict], Optional[Path], str]:
    """
    Checks if a dataset with the given image IDs is already cached.
    If it is, returns the cached dataset and its path.
    If not, returns None and a key for the dataset.
    """
    ids_hash = hash_file_ids_dict(selected_image_ids)
    selected_image_ids_set: Dict[str, Set[str]] = {
        k: set(v) for k, v in selected_image_ids.items()
    }

    dataset_key = f"{directory_prefix}{ids_hash}"
    del selected_image_ids

    if cache_path.exists():
        for candidate_dataset_path in cache_path.iterdir():
            if (
                candidate_dataset_path.is_dir()
                and candidate_dataset_path.name.startswith(directory_prefix)
            ):
                try:
                    dataset_dict: DatasetDict = load_from_disk(candidate_dataset_path)
                    assert isinstance(dataset_dict, DatasetDict)

                    # Check if all splits have the required image IDs
                    all_splits_valid = True
                    for split, image_ids in selected_image_ids_set.items():
                        if split not in dataset_dict:
                            all_splits_valid = False
                            break
                        images_in_dataset_list = dataset_dict[split]["file_id"]
                        images_in_dataset_set = set(images_in_dataset_list)

                        if not image_ids.issubset(images_in_dataset_set):
                            all_splits_valid = False
                            break

                    if all_splits_valid:
                        print(
                            f"Pre-selected image IDs are already in the dataset {candidate_dataset_path.name}"
                        )

                        # The previous method loads the entire row, use indices instead:
                        cached_dataset_dict = dict()
                        for split in selected_image_ids_set.keys():
                            split_indices = [
                                i
                                for i, file_id in enumerate(
                                    dataset_dict[split]["file_id"]
                                )
                                if file_id in selected_image_ids_set[split]
                            ]
                            cached_dataset_dict[split] = dataset_dict[split].select(
                                split_indices
                            )
                            assert (
                                cached_dataset_dict[split]["file_id"]
                                == selected_image_ids_set[split]
                            )

                        return (
                            DatasetDict(cached_dataset_dict),
                            candidate_dataset_path,
                            candidate_dataset_path.name,
                        )
                except:
                    pass

    return None, None, dataset_key


def cached_dataset(
    cache_path: Path,
    selected_image_ids: Iterable[str],
    directory_prefix: str = "real_",
) -> Tuple[Optional[Dataset], Optional[Path], str]:
    selected_image_ids = set(selected_image_ids)
    ids_hash = hash_image_ids(selected_image_ids)
    dataset_key = f"{directory_prefix}{ids_hash}"

    if cache_path.exists():
        for candidate_dataset_path in cache_path.iterdir():
            if (
                candidate_dataset_path.is_dir()
                and candidate_dataset_path.name.startswith(directory_prefix)
            ):
                try:
                    dataset: Dataset = load_from_disk(candidate_dataset_path)
                    assert isinstance(dataset, Dataset)
                    images_in_dataset_list = dataset["file_id"]
                    images_in_dataset_set = set(images_in_dataset_list)

                    if set(selected_image_ids).issubset(images_in_dataset_set):
                        print(
                            f"Pre-selected image IDs are already in the dataset {candidate_dataset_path.name}"
                        )

                        subset_indices = [
                            i
                            for i, file_id in enumerate(images_in_dataset_list)
                            if file_id in selected_image_ids
                        ]
                        cached_dataset = dataset.select(subset_indices)

                        assert set(cached_dataset["file_id"]) == selected_image_ids
                        return (
                            cached_dataset,
                            candidate_dataset_path,
                            candidate_dataset_path.name,
                        )
                except:
                    pass

    return None, None, dataset_key


def hash_dataset(
    dataset: Union[DatasetDict, Dataset], hasher: Optional[HASH] = None
) -> HASH:
    """
    Returns a hasher already updated with the dataset content.

    Note: dataset split names and file IDs are internally sorted to ensure deterministic hashing.
    """
    if hasher is None:
        hasher = _default_hasher()

    if isinstance(dataset, DatasetDict):
        splits = sorted(dataset.keys())
        split: str
        for split in splits:
            hasher.update(split.encode())
            hash_dataset(dataset[split], hasher)
    else:
        update_hasher_image_ids(dataset["file_id"], hasher)

    return hasher


def hash_file_ids_dict(
    file_ids_dict: Mapping[str, Iterable[str]], hasher: Optional[HASH] = None
) -> HASH:
    if hasher is None:
        hasher = _default_hasher()

    splits = sorted(file_ids_dict.keys())
    split: str
    for split in splits:
        hasher.update(split.encode())
        update_hasher_image_ids(file_ids_dict[split], hasher)

    return hasher


def _default_hasher() -> HASH:
    return hashlib.sha256()


def load_image_ids_filelists(
    pre_selected_image_ids_lists: Dict[str, PathAlike],
) -> Tuple[Dict[str, List[str]], str]:
    """
    Load the image IDs from the file lists (one for each split)
    and return a dictionary with the splits.

    It also returns a hash of the file IDs, which is used to
    identify the dataset in the cache.
    """
    pre_selected_image_ids_dict: Dict[str, List[str]] = dict()
    for split, file_ids_list in pre_selected_image_ids_lists.items():
        pre_selected_image_ids_dict[split] = list(
            x for x in Path(file_ids_list).read_text().splitlines()
        )

    file_ids_hash = hash_file_ids_dict(pre_selected_image_ids_dict)

    return pre_selected_image_ids_dict, file_ids_hash.hexdigest()


__all__ = [
    "REAL_IMAGES_LABEL",
    "FAKE_IMAGES_LABEL",
    "DEEPFAKE_DATASET_FEATURES",
    "RowDict",
    "PathAlike",
    "check_dataset_format",
    "check_row_format",
    "prepare_image",
    "save_image_ids",
    "load_image_ids",
    "make_empty_dataset",
    "sort_paths",
    "merge_all_datasets_and_splits",
    "sample_dataset_by_generator",
    "make_train_val_splits",
    "saturating_choice_count",
    "saturating_balanced_choice",
    "is_image_valid",
    "create_shared_max_samples",
    "check_max_samples",
    "are_all_generated",
    "PIL_setup",
    "update_hasher_image_ids",
    "hash_image_ids",
    "cached_dataset_dict",
    "cached_dataset",
    "hash_dataset",
    "hash_file_ids_dict",
    "load_image_ids_filelists",
]
