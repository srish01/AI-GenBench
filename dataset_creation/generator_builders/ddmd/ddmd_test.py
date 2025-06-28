from collections import defaultdict
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from tqdm import tqdm
from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    FAKE_IMAGES_LABEL,
    PathAlike,
    are_all_generated,
    create_shared_max_samples,
    sort_paths,
)
from generator_builders.deepfake_dataset_builder import (
    AvailableFile,
    DeepfakeDatasetBuilder,
)
from datasets import Dataset

from generator_builders.ddmd.ddmd_utils import (
    DMD_FOLDER_TO_NAMES,
    dmd_generator,
    dmd_get_generator_name,
)

# Example:
# DDMD/diffusion_model_deepfakes_lsun_bedroom/test/ADM/000000
# that is, DDMD/<subfolder>/test/<generator_name>/<file_id>
FILE_ID_FORMAT = re.compile(
    r"^DDMD\/(?P<subfolder>[A-Za-z \-_\.\d]+)\/test\/(?P<generator_name>[A-Za-z \-\._\d]+)\/(?P<filename>[A-Za-z \-_\d]+)$"
)


class Ddmd_test_Builder(DeepfakeDatasetBuilder):
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
                dmd_generator,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
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

        print("Saving the fake images dataset to disk:", str(output_path))
        subset.save_to_disk(output_path, num_proc=self.num_proc, max_shard_size="500MB")

    def _make_indices_groups(self) -> Optional[Dict[str, List[int]]]:
        image_list = self.available_files
        indices_dict = defaultdict(list)
        for idx, fpath in enumerate(image_list):
            indices_dict[dmd_get_generator_name(fpath.file_path)].append(idx)
        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _dmd_image_list(self.input_path):
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
        return f"DDMD/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        return bool(FILE_ID_FORMAT.match(file_id))


def _dmd_image_list(folder: Path) -> List[Path]:
    image_paths = []

    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            for subsubfolder in subfolder.iterdir():
                if (
                    subsubfolder.parts[-1] == "test"
                ):  # Limit the images to 10k per generator
                    for file in subsubfolder.rglob("*.png"):
                        if (
                            file.parts[-2] in DMD_FOLDER_TO_NAMES.keys()
                        ):  # Only used generators
                            image_paths.append(file)

    return sort_paths(image_paths)


__all__ = ["Ddmd_test_Builder"]
