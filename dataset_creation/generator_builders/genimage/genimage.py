from collections import defaultdict
from functools import partial
import re
from typing import Any, Dict, List, Optional
import random

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
)
from dataset_utils.file_extraction import list_archive_files
from generator_builders.genimage.genimage_utils import (
    FOLDER_NAME_TO_GEN_NAME,
    FOLDER_NAME_TO_SUBFOLDER_NAME,
    genimage_generator,
)


VALID_FILENAME_REGEX = re.compile(r".+[^\/]+\.[^\/]+$")
# Example:
# GenImage/ADM/train/ai/0_adm_0
FILE_ID_FORMAT = re.compile(
    r"^GenImage\/(?P<generator_name>[A-Za-z \-\.\d]+)\/(?P<split_name>train|val)\/ai\/(?P<filename>[A-Za-z \-\_\d]+)$"
)


class GenImage_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        output_path: PathAlike,
        generator_config: Any,
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
            generator_samples_mapping=lambda x: FOLDER_NAME_TO_GEN_NAME[
                x.split("/")[-1]
            ],
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

        # Shuffle (deterministically) to balance the load across processes
        random.seed(self.seed)
        random.shuffle(selected_files)

        subset = Dataset.from_generator(
            generator=partial(
                genimage_generator,
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
        # Indices by folder
        indices_dict = defaultdict(list)
        for idx, available_file in enumerate(image_list):
            fpath = available_file.file_path
            if "!" in fpath:
                # root_folder/<generator_folder_name>/imagenet_ai_0508_adm.zip!train/ai/image.jpg
                generator_folder_path = Path(fpath.split("!")[0]).parent
                generator_folder_name = generator_folder_path.name
            else:
                # root_folder/<generator_folder_name>/imagenet_ai_0508_adm/train/ai/image.jpg

                generator_folder_path = Path(fpath)
                for _ in range(4):
                    generator_folder_path = generator_folder_path.parent
                generator_folder_name = generator_folder_path.name

            # Ensures generator_folder_name is in the dictionary
            _ = FOLDER_NAME_TO_GEN_NAME[generator_folder_name]

            indices_dict[str(generator_folder_path)].append(idx)

        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _genimage_image_list_zip(self.input_path):
            result.append(
                AvailableFile(
                    file_path=file_path,
                    file_id=self._make_file_id(file_path),
                )
            )
        return result

    def _make_file_id(self, file_path: str) -> str:
        fpath = Path(file_path).with_suffix("")
        intermediate_parent = fpath.parent.parent.parent
        if intermediate_parent.name.startswith("imagenet"):
            generator = str(intermediate_parent.parent.name)
        else:
            generator = str(intermediate_parent.name)
        generator = FOLDER_NAME_TO_GEN_NAME[generator]

        internal_path = fpath.relative_to(intermediate_parent)
        return f"GenImage/{generator}/{internal_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        return bool(FILE_ID_FORMAT.match(file_id))


def _genimage_image_list_zip(folder: Path) -> List[str]:
    all_images: List[str] = []
    for generator_folder_name in FOLDER_NAME_TO_GEN_NAME:
        subfolder_name = FOLDER_NAME_TO_SUBFOLDER_NAME[generator_folder_name]
        candidate_folders = [
            folder / generator_folder_name / subfolder_name,
            folder / generator_folder_name / (subfolder_name + ".zip"),
        ]

        for candidate_folder in candidate_folders:
            if candidate_folder.exists():
                break
        else:
            raise ValueError(
                f"Missing generator folder or zip file in {folder}: {generator_folder_name}"
            )

        if candidate_folder.suffix == ".zip":
            all_images.extend(
                list_archive_files(candidate_folder, include_archive_path=True)
            )
        else:
            all_images.extend(
                [str(x) for x in candidate_folder.rglob("*") if x.is_file()]
            )

    all_images = [
        x
        for x in all_images
        if ("/ai/" in x or "\\ai\\" in x)
        and VALID_FILENAME_REGEX.match(x)
        and "__MACOSX" not in x
        and not x.endswith(".svg")
    ]

    return sorted(all_images)


__all__ = ["GenImage_Builder"]
