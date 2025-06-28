from collections import defaultdict
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
from generator_builders.forensynths.forensynths_utils import forensynths_generator

# Example:
# Forensynths/progan_train/airplane/1_fake/00000
# that is: Forensynths/progan_<split>/<object_class>/1_fake/<filename>
FILE_ID_FORMAT = re.compile(
    r"^Forensynths\/progan_(?P<split_name>train|valid)\/(?P<object_class>[a-z]+)\/1_fake\/(?P<filename>[A-Za-z \-\_\d]+)$"
)


class Forensynths_TrainVal_Subset_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        origin_split: str,
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
        self.origin_split: str = origin_split

    def _make_dataset(self, output_path: Path, indices: List[int]):
        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]
        max_samples, lock = create_shared_max_samples(self.generator_config)

        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            generator=partial(
                forensynths_generator,
                self.input_path,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                self.generator_config,
                max_samples,
                lock,
                self.origin_split,
                False,
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
        # Indices by folder
        indices_dict = defaultdict(list)
        for idx, available_file in enumerate(image_list):
            fpath = Path(available_file.file_path)
            indices_dict[str(fpath.parent.parent.name)].append(idx)
        else:
            return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _forensynths_train_val_image_list(self.input_path):
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
        return f"Forensynths/progan_{self.origin_split}/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        matched = FILE_ID_FORMAT.match(file_id)
        if matched is None:
            return False

        split_name = matched.group("split_name")

        # Check if the split name is valid
        if split_name != self.origin_split:
            return False

        return True


def _forensynths_train_val_image_list(folder: Path) -> List[Path]:
    return sort_paths([x for x in folder.rglob("*.png") if x.parent.name == "1_fake"])


__all__ = ["Forensynths_TrainVal_Subset_Builder"]
