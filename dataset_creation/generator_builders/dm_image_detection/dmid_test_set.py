from collections import defaultdict
from functools import partial
import json
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
from generator_builders.dm_image_detection.dmid_utils import (
    _dmid_test_get_generator_name,
    dmid_test_generator,
)

DALLE2_FILEPATH_FORMAT = re.compile(
    r"^dalle_2\/(?P<file_prefix>DALL·E \d{4}-\d\d-\d\d \d\d\.\d\d\.\d\d)( - .+)?$"
)

# Example:
# DMimageDetection/test/biggan_256/biggan_000_302875
# that is: DMimageDetection/<split>/<subset>/<filename>
FILE_ID_FORMAT = re.compile(
    r"^DMimageDetection\/test\/(?P<generator_name>[A-Za-z\_\-\d]+)\/(?P<file_id>[A-Za-z·\ \-\_\.\d]+)$"
)


class DMID_TestSubset_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        coco2017_val_captions_path: PathAlike,
        output_path: PathAlike,
        generator_config: Dict[str, Any],
        subset_size: int = -1,
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
        self.coco2017_val_captions_path = Path(coco2017_val_captions_path)
        self.num_proc: int = num_proc

    def _make_dataset(self, output_path: Path, indices: List[int]):
        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]
        max_samples, lock = create_shared_max_samples(self.generator_config)

        print("Loading the COCO 2017 validation captions...")
        # Read json
        with open(self.coco2017_val_captions_path, "r") as f:
            coco_captions = json.load(f)["annotations"]

        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            generator=partial(
                dmid_test_generator,
                FAKE_IMAGES_LABEL,
                coco_captions,
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
                _dmid_test_get_generator_name(available_file.file_path)
            ].append(idx)

        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        # The dataset contains duplicate (same hash) files
        unique_ids = set()
        for file_path in sort_paths(self.input_path.rglob("*.png")):
            file_path = str(file_path)
            file_id = self._make_file_id(file_path)
            if file_id in unique_ids:
                print("Duplicate file ID found, skipping:", file_id)
                continue

            unique_ids.add(file_id)
            result.append(
                AvailableFile(
                    file_path=file_path,
                    file_id=file_id,
                )
            )
        return result

    def _make_file_id(self, file_path: str) -> str:
        fpath = Path(file_path).with_suffix("")
        relative_path = fpath.relative_to(self.input_path)
        if relative_path.parts[0] == "dalle_2":
            # DALL·E 2 specific case
            match = DALLE2_FILEPATH_FORMAT.match(str(relative_path))
            if not match:
                raise ValueError(
                    f"File path {relative_path} does not match the expected DALL·E 2 format."
                )
            file_id = match.group("file_prefix")
            if file_id[-8:] == "19.07.49":
                if "820316" in relative_path.name:
                    file_id += "_1"
            if file_id[-8:] in {"19.06.08", "19.12.25"}:
                if "(1)" in relative_path.name:
                    file_id += "_1"

            return f"DMimageDetection/test/dalle_2/{file_id}"

        return f"DMimageDetection/test/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        return bool(FILE_ID_FORMAT.match(file_id))


__all__ = ["DMID_TestSubset_Builder"]
