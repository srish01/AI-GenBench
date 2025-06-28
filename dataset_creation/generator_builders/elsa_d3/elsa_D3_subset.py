from functools import partial
from typing import Any, Dict, List, Literal, Optional
import re

from generator_builders.deepfake_dataset_builder import (
    AvailableFile,
    DeepfakeDatasetBuilder,
)
from datasets import DatasetDict, Dataset, Split, NamedSplit
from pathlib import Path
from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    FAKE_IMAGES_LABEL,
    PathAlike,
    are_all_generated,
    create_shared_max_samples,
)
from generator_builders.elsa_d3.elsa_utils import (
    elsa_d3_split_row,
)
from generator_builders.elsa_d3.elsa_utils import load_elsa_d3

# Example:
# ELSA_D3/train/row0/image_gen1
FILE_ID_FORMAT = re.compile(
    r"^ELSA_D3\/(?P<split_name>train|validation)\/row(?P<row_id>\d+)\/image_gen(?P<generator_id>\d+)$"
)


class ELSA_D3_Subset_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        origin_split: NamedSplit,
        output_path: PathAlike,
        generator_id: Literal[0, 1, 2, 3],
        generator_config: Dict[str, Any] = None,
        subset_size: int = -1,
        max_samples: int = -1,
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
            subset_size=subset_size,
            convert_to_jpeg=convert_to_jpeg,
            seed=seed,
            num_proc=num_proc,
            check_already_done_marker=check_already_done_marker,
            cleanup_cache_on_exit=cleanup_cache_on_exit,
            pre_selected_image_ids=pre_selected_image_ids,
        )

        self.origin_split: NamedSplit = origin_split
        self.generator_id: Literal[0, 1, 2, 3] = generator_id

        # Datasets
        self._elsa_d3_dataset: Optional[DatasetDict] = None
        self.max_samples: int = max_samples

    @property
    def elsa_d3_dataset(self) -> DatasetDict:
        if self._elsa_d3_dataset is None:
            self._elsa_d3_dataset = load_elsa_d3(
                self.input_path, num_proc=self.num_proc
            )

        return self._elsa_d3_dataset

    def _make_dataset(self, output_path: Path, indices: List[int]):
        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]

        # Load/download full dataset
        elsa_dataset_train: Dataset = self.elsa_d3_dataset[self.origin_split]
        gen_definition = {
            "generators": {
                self.generator_id: {
                    "max samples": self.max_samples,
                }
            }
        }

        max_samples, lock = create_shared_max_samples(gen_definition)

        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            partial(
                elsa_d3_split_row,
                elsa_dataset_train,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                "train" if self.origin_split == Split.TRAIN else "valid",
                self.generator_id,
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

    def _make_available_files_list(self) -> List[AvailableFile]:
        dataset: Dataset = self.elsa_d3_dataset[self.origin_split]
        result = []
        for row_idx in range(len(dataset)):
            file_id = (
                f"ELSA_D3/{self.origin_split}/row{row_idx}/image_gen{self.generator_id}"
            )
            result.append(
                AvailableFile(
                    file_path=f"{self.origin_split}/{row_idx}/{self.generator_id}",
                    file_id=file_id,
                )
            )
        return result

    def _is_valid_file_id(self, file_id: str) -> bool:
        matched = FILE_ID_FORMAT.match(file_id)
        if matched is None:
            return False

        # Check split and generator id
        split_name = matched.group("split_name")
        generator_id = int(matched.group("generator_id"))
        if split_name != self.origin_split:
            return False
        if generator_id != self.generator_id:
            return False
        return True


__all__ = ["ELSA_D3_Subset_Builder"]
