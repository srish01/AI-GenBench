from abc import ABC, abstractmethod
from pathlib import Path
import json
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    TypedDict,
    NamedTuple,
)
import warnings
from dataset_utils.common_utils import (
    PathAlike,
    cached_dataset,
    check_dataset_format,
)
from datasets import (
    Dataset,
    load_from_disk,
)
import weakref
import shutil

from generator_builders.generator_utils import subset_size_with_sampling_per_generator
from dataset_utils.common_utils import hash_image_ids


class IntermediateDatasetPaths(TypedDict):
    output_path: Path
    cache_dir: Optional[Path]


class AvailableFile(NamedTuple):
    file_id: str
    file_path: str


def _no_op(x):
    return x


class DeepfakeDatasetBuilder(ABC):
    def __init__(
        self,
        input_path: PathAlike,
        output_path: PathAlike,
        tmp_cache_dir: Optional[PathAlike] = None,
        subset_size: int = -1,
        generator_config: Optional[Dict[str, Any]] = None,
        generator_samples_mapping: Callable[[str], str] = _no_op,
        convert_to_jpeg: bool = False,
        seed: int = 1234,
        num_proc: int = 1,
        check_already_done_marker: bool = True,
        cleanup_cache_on_exit: bool = True,
        pre_selected_image_ids: Optional[Iterable[str]] = None,
    ):
        self.input_path: Path = Path(input_path)
        self.output_path: Path = Path(output_path)
        self.tmp_cache_dir: Optional[Path] = (
            Path(tmp_cache_dir) if tmp_cache_dir is not None else None
        )
        self.generator_config: Dict[str, Any] = (
            generator_config
            if generator_config is not None
            else self.__get_default_generator_config(subset_size)
        )
        self.generator_samples_mapping: Callable[[str], str] = generator_samples_mapping
        self.convert_to_jpeg: bool = convert_to_jpeg
        self.seed: int = seed
        self.num_proc: int = num_proc
        self.check_already_done_marker: bool = check_already_done_marker
        self._cleanup_cache_on_exit: bool = cleanup_cache_on_exit
        self._pre_selected_image_ids: Optional[List[str]] = (
            None if pre_selected_image_ids is None else list(pre_selected_image_ids)
        )
        self._are_pre_selected_image_ids_filtered: bool = False
        self._pre_made_dataset: Optional[Dataset] = None
        self._pre_made_dataset_path: Optional[Path] = None

        # Samples for each generator
        self.samples_per_generator: Dict[str, int] = {
            generator: info.get("samples", 0)
            for generator, info in self.generator_config.get("generators", {}).items()
        }
        sampling = self.generator_config.get("sampling", {})

        # Specifies if samples per generator are used instead of subset size
        self.use_samples_per_generator: bool = (
            not sampling.get("take_all", False) and sampling.get("samples") is None
        )

        self._available_files: Optional[List[AvailableFile]] = None

        self.indices_groups = None
        if sampling.get("take_all", False):
            # Take all the samples from the dataset
            self.subset_size = self._dataset_max_len()
        elif "samples" in sampling:
            # Take a specific number of samples from the dataset
            self.subset_size = sampling["samples"]
        else:
            # Take a specific number of samples from each generator
            self.indices_groups = self._make_indices_groups()
            if self.indices_groups is None:
                raise ValueError(
                    "No indices groups found while using per-generator sampling. "
                    "Please simplement _make_indices_groups appropriately."
                )
            generator_mapping = self.__map_groups_to_generator(self.indices_groups)
            self.subset_size = subset_size_with_sampling_per_generator(
                self.samples_per_generator,
                generator_mapping,
            )

        # Datasets
        self._result_dataset: Optional[Dataset] = None

        # Cleanup handler (weakref)
        self._cleanup_handler: Optional[weakref.finalize] = None

        # Selected subset indices
        self._selected_subset_indices: Optional[List[int]] = None

    # --- TO IMPLEMENT/OVERRIDE ---
    @abstractmethod
    def _make_dataset(self, output_path: Path, selected_indices: List[int]): ...

    @abstractmethod
    def _make_available_files_list(self) -> List[AvailableFile]: ...

    @abstractmethod
    def _is_valid_file_id(self, file_id: str) -> bool: ...

    def _make_indices_groups(self) -> Optional[Dict[str, List[int]]]:
        return None

    # -----------------------------

    def _dataset_max_len(self) -> int:
        return len(self.available_files)

    @property
    def available_files(self) -> List[AvailableFile]:
        """
        A list of available files in the dataset, sorted by file_id.
        """
        if self._available_files is not None:
            return self._available_files

        self._available_files = self._load_filelist_cache()
        if self._available_files is None:
            self._available_files = self._make_available_files_list()
            if len(self._available_files) == 0:
                raise ValueError(
                    "No available files found. Please check your dataset configuration (especially paths)."
                )

            # Check file IDs
            file_id_unique = set()
            for file in self._available_files:
                if file.file_id in file_id_unique:
                    raise ValueError(
                        f"Duplicate file_id found: {file.file_id}. Please check your dataset configuration/implementation."
                    )
                if not self._is_valid_file_id(file.file_id):
                    raise ValueError(
                        f"Invalid file_id found: {file.file_id}. Please check your dataset implementation."
                    )
                file_id_unique.add(file.file_id)

            # Sort by file_id
            self._available_files.sort(key=lambda x: x.file_id)
            self._save_filelist_cache(self._available_files)

        return self._available_files

    def _load_filelist_cache(self) -> Optional[List[AvailableFile]]:
        available_files_cache = self.output_path / "available_files.json"
        if not available_files_cache.exists():
            return None

        with open(available_files_cache, "r") as f:
            filelist = json.load(f)

        return [
            AvailableFile(
                file_id=file[0],
                file_path=f"{str(self.input_path)}{file[1]}",
            )
            for file in filelist
        ]

    def _save_filelist_cache(self, filelist: List[AvailableFile]):
        available_files_cache = self.output_path / "available_files.json"
        available_files_cache.parent.mkdir(parents=True, exist_ok=True)

        if available_files_cache.exists():
            return

        filelist_without_prefix_path = [
            (
                file.file_id,
                file.file_path.removeprefix(str(self.input_path)),
            )
            for file in filelist
        ]
        with open(available_files_cache, "w") as f:
            json.dump(filelist_without_prefix_path, f)

    @property
    def result_dataset(self) -> Dataset:
        return self._manage_make_subset()

    @property
    def cleanup_cache_on_exit(self) -> bool:
        return self._cleanup_cache_on_exit

    @cleanup_cache_on_exit.setter
    def cleanup_cache_on_exit(self, value: bool):
        self._cleanup_cache_on_exit = value
        self._register_cleanup_handler()

    @property
    def selected_subset_indices(self) -> List[int]:
        """
        The indices of the images in the dataset that were selected for the subset.

        Indices are sorted in ascending order.
        """
        if self._selected_subset_indices is not None:
            return list(self._selected_subset_indices)

        selected_indices = sorted(self._select_subset_indices())

        if len(selected_indices) == 0:
            raise ValueError(
                "Selected indices are empty. Please check your dataset configration (especially paths)."
            )
        self._selected_subset_indices = list(selected_indices)
        return selected_indices

    def __map_groups_to_generator(
        self, indices_groups_dict: Dict[str, List[int]]
    ) -> dict[str, List[int]]:
        return {
            self.generator_samples_mapping(k): v for k, v in indices_groups_dict.items()
        }

    def __get_default_generator_config(self, subset_size: int) -> Dict[str, Any]:
        return {
            "sampling": {
                "take_all": True if subset_size == -1 else False,
                "samples": subset_size,
            }
        }

    def cleanup(self):
        self._result_dataset = None
        _cleanup_intermediates(self._dataset_cleanup_paths())

    def check_already_done(self, path: Path) -> bool:
        return (Path(path) / ".successfully_built").exists()

    def mark_as_done(self, path: Path):
        (Path(path) / ".successfully_built").touch(exist_ok=True)

    def _make_seed(self) -> int:
        return self.seed

    def _init_seed(self, selected_seed: int):
        random.seed(selected_seed)

    def _check_pre_selected_image_ids_dataset(self) -> Optional[Dataset]:
        if self.pre_selected_image_ids is None:
            return None

        if self._pre_made_dataset is not None:
            return self._pre_made_dataset

        if not self.output_path.exists():
            return None

        cached_dataset_obj, cached_dataset_path, _ = cached_dataset(
            self.output_path, self.pre_selected_image_ids, directory_prefix="subset_"
        )

        if cached_dataset_obj is not None:
            assert cached_dataset_path is not None
            print(
                f"Found a dataset with the pre-selected image IDs at {cached_dataset_path}"
            )
            self._pre_made_dataset = cached_dataset_obj
            self._pre_made_dataset_path = cached_dataset_path
            return self._pre_made_dataset

        print("Could not find a dataset with the pre-selected image IDs.")
        return None

    def _make_dataset_paths(self) -> IntermediateDatasetPaths:
        return IntermediateDatasetPaths(
            output_path=self.output_path / self._dataset_dir_name(),
            cache_dir=self.tmp_cache_dir,
        )

    def _dataset_dir_name(self) -> str:
        selected_indices = self.selected_subset_indices
        selected_image_ids = [self.available_files[i].file_id for i in selected_indices]

        # Hash the concatenated string
        subset_hash = hash_image_ids(selected_image_ids)
        return "subset_" + subset_hash

    def _dataset_cleanup_paths(self) -> Union[Path, Iterable[Path]]:
        dataset_paths: Dict[str, Optional[Path]] = dict(self._make_dataset_paths())

        # We don't want to cleanup the final dataset
        del dataset_paths["output_path"]

        return list(x for x in dataset_paths.values() if x is not None)

    def _register_cleanup_handler(self):
        if (not self.cleanup_cache_on_exit) and self._cleanup_handler is not None:
            self._cleanup_handler.detach()
            self._cleanup_handler = None
        elif self.cleanup_cache_on_exit and self._cleanup_handler is None:
            paths_to_clean = self._dataset_cleanup_paths()
            self._cleanup_handler = weakref.finalize(
                self, _cleanup_intermediates, paths_to_clean
            )

    def _filter_pre_selected_image_ids(self):
        if (
            self._pre_selected_image_ids is None
            or self._are_pre_selected_image_ids_filtered
        ):
            return

        self._pre_selected_image_ids = [
            x for x in self._pre_selected_image_ids if self._is_valid_file_id(x)
        ]
        self._are_pre_selected_image_ids_filtered = True

        if len(self._pre_selected_image_ids) == 0:
            raise RuntimeError(
                "No valid image IDs apply to this dataset. "
                "Please check your dataset configuration (especially paths)."
            )

    @property
    def pre_selected_image_ids(self) -> Optional[List[str]]:
        if self._pre_selected_image_ids is None:
            return None
        self._filter_pre_selected_image_ids()
        return list(self._pre_selected_image_ids)

    def _select_subset_indices(self) -> List[int]:
        seed = self._make_seed()
        self._init_seed(seed)
        max_len = self._dataset_max_len()
        if self.pre_selected_image_ids is not None:
            # A list of IDs has been passed

            # Get the indices of the selected image IDs
            selected_indices = [
                i
                for i, file in enumerate(self.available_files)
                if file.file_id in self.pre_selected_image_ids
            ]

            if len(selected_indices) != len(self.pre_selected_image_ids):
                raise ValueError(
                    "Some of the required image IDs were not found among available images. "
                    "Please check your dataset configuration (especially paths)."
                )

            assert set(
                self.available_files[i].file_id for i in selected_indices
            ) == set(self.pre_selected_image_ids)

            return selected_indices
        elif max_len == 0:
            return []
        elif self.subset_size < 0 or self.subset_size >= max_len:
            return list(range(max_len))
        else:
            indices_groups_dict = (
                self._make_indices_groups()
                if self.indices_groups is None
                else self.indices_groups
            )
            if indices_groups_dict is not None:
                selected_indices = []
                # Stratify
                group_names = list(indices_groups_dict.keys())
                indices_groups = list(indices_groups_dict.values())

                if self.use_samples_per_generator:
                    new_indices_groups_dict = self.__map_groups_to_generator(
                        indices_groups_dict
                    )
                    if not set(new_indices_groups_dict.keys()).issubset(
                        self.samples_per_generator.keys()
                    ):
                        # NOTE This case is used when it is not possible (or is not very smart) to specify samples per generator
                        self.use_samples_per_generator = False
                        return self._select_subset_indices()

                    for generator, samples in self.samples_per_generator.items():
                        n_taken = min(samples, len(new_indices_groups_dict[generator]))
                        selected_indices.extend(
                            random.sample(
                                new_indices_groups_dict[generator],
                                n_taken,
                            )
                        )
                        print(f"from {generator} took {n_taken} images")
                    print("Overall selected images:", len(selected_indices))
                    return selected_indices

                n_groups = len(indices_groups)
                base_subgroup_size = self.subset_size // n_groups
                subgroups_allocation = [base_subgroup_size] * n_groups
                for subgroup_idx in range(n_groups):
                    subgroups_allocation[subgroup_idx] = min(
                        len(indices_groups[subgroup_idx]), base_subgroup_size
                    )

                subgroup_size = sum(subgroups_allocation)
                subgroups_remainder = self.subset_size - subgroup_size

                allocated_something = True
                non_saturated_groups = [
                    idx
                    for idx in range(n_groups)
                    if len(indices_groups[idx]) > subgroups_allocation[idx]
                ]
                while subgroups_remainder > 0 and allocated_something:
                    allocated_something = False
                    sorted_non_saturated_groups = sorted(
                        non_saturated_groups,
                        key=lambda idx: (
                            len(indices_groups[idx])
                            - subgroups_allocation[subgroup_idx]
                        ),
                        reverse=True,
                    )
                    for subgroup_idx in sorted_non_saturated_groups:
                        if subgroups_remainder == 0:
                            break
                        if (
                            len(indices_groups[subgroup_idx])
                            == subgroups_allocation[subgroup_idx]
                        ):
                            non_saturated_groups.remove(subgroup_idx)
                            continue
                        allocated_something = True
                        subgroups_allocation[subgroup_idx] += 1
                        subgroups_remainder -= 1

                for subgroup_idx in range(n_groups):
                    print(
                        f"Selected {subgroups_allocation[subgroup_idx]} indices from subgroup {group_names[subgroup_idx]}"
                    )
                    selected_indices.extend(
                        random.sample(
                            indices_groups[subgroup_idx],
                            subgroups_allocation[subgroup_idx],
                        )
                    )
                return selected_indices
            else:
                return random.sample(range(max_len), self.subset_size)

    def _manage_make_subset(self) -> Dataset:
        result_dataset: Dataset

        if self._result_dataset is not None:
            return self._result_dataset

        pre_made_dataset = self._check_pre_selected_image_ids_dataset()
        if pre_made_dataset is not None:
            self._result_dataset = pre_made_dataset
            return pre_made_dataset

        self._register_cleanup_handler()

        paths = self._make_dataset_paths()
        ref_path = paths["output_path"]

        if not self.check_already_done(ref_path):
            _check_not_exists(ref_path)
            selected_indices = self.selected_subset_indices
            self._make_dataset(ref_path, selected_indices)
        else:
            print(f"Dataset already exists at {str(ref_path)}")

        result_dataset = load_from_disk(ref_path)
        assert isinstance(result_dataset, Dataset)
        check_dataset_format(result_dataset)

        self.mark_as_done(ref_path)

        self._result_dataset = result_dataset
        return result_dataset


def _cleanup_intermediates(paths):
    if paths is None:
        return

    if isinstance(paths, Path):
        paths = [paths]

    for path in paths:
        path = Path(path)
        # print(f"Cleaning up {str(path)}")

        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)


def _check_not_exists(path: PathAlike):
    if Path(path).exists():
        raise FileExistsError(f"Path {str(path)} already exists")


__all__ = ["IntermediateDatasetPaths", "DeepfakeDatasetBuilder", "AvailableFile"]
