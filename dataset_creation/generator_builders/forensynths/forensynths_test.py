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
from generator_builders.forensynths.forensynths_utils import (
    FOLDER_GENERATOR_NAMES_MAPPING,
    forensynths_generator,
)

# Example:
# Forensynths/test/biggan/1_fake/00000701
# that is: Forensynths/test/<generator_name>/<optional_class_folder>/1_fake/<filename>https://it.overleaf.com/project/67c5b47ca58e5183230bbf72
FILE_ID_FORMAT = re.compile(
    r"^Forensynths\/test\/(?P<generator_name>[a-zA-Z0-9_]+)\/([a-zA-Z0-9]+\/)?1_fake\/(?P<filename>[A-Za-z \:\-\_\d]+)$"
)


class Forensynths_Test_Builder(DeepfakeDatasetBuilder):
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
            generator_samples_mapping=lambda x: FOLDER_GENERATOR_NAMES_MAPPING[x],
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
                forensynths_generator,
                self.input_path,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                self.generator_config,
                max_samples,
                lock,
                "test",
                True,
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
            rel_path = fpath.relative_to(self.input_path)
            # The first subfolder relative to input path
            sub_folder = rel_path.parts[0]
            indices_dict[sub_folder].append(idx)

        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _forensynths_test_image_list(self.input_path):
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
        return f"Forensynths/test/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        matched = FILE_ID_FORMAT.match(file_id)

        if matched is None:
            return False

        generator_name = matched.group("generator_name")
        if generator_name in {"deepfake", "seeingdark", "san", "whichfaceisreal"}:
            return False

        return True


def _forensynths_test_image_list(folder: Path) -> List[Path]:
    pngs = [x for x in folder.rglob("*.png") if x.parent.name == "1_fake"]

    jpegs = [x for x in folder.rglob("*.jpeg") if x.parent.name == "1_fake"]

    # There actually are bmp files, but those belong to "san/0_real" (which we ignore)

    unfiltered_paths = sort_paths(pngs + jpegs)
    result_paths = []
    for fpath in unfiltered_paths:
        rel_path = fpath.relative_to(folder)
        # The first subfolder relative to input path
        sub_folder = rel_path.parts[0]
        if sub_folder in {"deepfake", "seeingdark", "san", "whichfaceisreal"}:
            continue
        result_paths.append(fpath)
    return result_paths


__all__ = ["Forensynths_Test_Builder"]
