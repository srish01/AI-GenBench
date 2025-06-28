from collections import defaultdict
import csv
from functools import partial
from typing import Any, Dict, List, Optional
import re

from generator_builders.deepfake_dataset_builder import (
    AvailableFile,
    DeepfakeDatasetBuilder,
)
from datasets import Dataset
from pathlib import Path
from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    FAKE_IMAGES_LABEL,
    PathAlike,
    are_all_generated,
    create_shared_max_samples,
    sort_paths,
)
from generator_builders.artifact.artifact_utils import (
    artifact_test_generator,
    artifact_test_get_generator_name,
)

# Example:
# Artifact/big_gan/big/biggan-data/0/img000000
# That is: Artifact/<generator_name>/<a certain number of intermediate folders>/<filename>
FILE_ID_FORMAT = re.compile(
    r"^Artifact/(?P<generator_name>[A-Za-z\-\._\d ]+)(?:/[^/]+)*/(?P<filename>[^/]+)$"
)


class Artifact_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        output_path: PathAlike,
        generator_config: Dict[str, Any],
        convert_to_jpeg: bool = False,
        tmp_cache_dir: Optional[PathAlike] = None,
        num_proc: int = 16,
        seed: int = 1234,
        check_already_done_marker: bool = True,
        cleanup_cache_on_exit: bool = True,
        pre_selected_image_ids: Optional[List[str]] = None,
    ):
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            tmp_cache_dir=tmp_cache_dir,
            generator_config=generator_config,
            convert_to_jpeg=convert_to_jpeg,
            seed=seed,
            num_proc=num_proc,
            check_already_done_marker=check_already_done_marker,
            cleanup_cache_on_exit=cleanup_cache_on_exit,
            pre_selected_image_ids=pre_selected_image_ids,
        )

    def _make_dataset(self, output_path: Path, indices: List[int]):
        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]
        max_samples, lock = create_shared_max_samples(self.generator_config)

        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            generator=partial(
                artifact_test_generator,
                self.input_path,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                self.generator_config,
                max_samples,
                lock,
            ),
            features=DEEPFAKE_DATASET_FEATURES,
            num_proc=self.num_proc,
            gen_kwargs={"filelist": selected_files},
            cache_dir=(
                str(self.tmp_cache_dir) if self.tmp_cache_dir is not None else None
            ),
        )

        assert isinstance(subset, Dataset)
        assert are_all_generated(
            max_samples, lock
        ), "The required number of images was not generated."

        # Save the dataset to disk
        print("Saving the fake images dataset to disk:", str(output_path))
        subset.save_to_disk(output_path, num_proc=self.num_proc, max_shard_size="500MB")

    def _make_indices_groups(self) -> Optional[Dict[str, List[int]]]:
        image_list = self.available_files
        # Indices by generator
        indices_dict = defaultdict(list)
        for idx, available_file in enumerate(image_list):
            indices_dict[
                artifact_test_get_generator_name(
                    self.input_path, available_file.file_path
                )
            ].append(idx)

        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _artifact_image_list(self.input_path):
            file_path = str(file_path)
            result.append(
                AvailableFile(
                    file_path=file_path,
                    file_id=self._make_file_id(file_path),
                )
            )
        return result

    def _make_file_id(self, file_path: str) -> str:
        fpath = Path(file_path).with_suffix("")
        relative_path = fpath.relative_to(self.input_path)
        return f"Artifact/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        return bool(FILE_ID_FORMAT.match(file_id))


def _artifact_image_list(folder: Path) -> List[Path]:
    all_paths = []
    for folder in list(folder.iterdir()):
        if not folder.is_dir():
            continue

        current_paths = set()

        # Read CSV (with header) and create a dict filename -> target
        csv_metadata_path = folder / "metadata.csv"
        with open(csv_metadata_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = folder / row["image_path"]
                if img_path in current_paths:
                    raise ValueError(
                        f'Duplicate filename in {csv_metadata_path}: {row["filename"]}.'
                    )
                if int(row["target"]) != 0:
                    current_paths.add(img_path)

        all_paths.extend(current_paths)

    return sort_paths(all_paths)


__all__ = ["Artifact_Builder"]
