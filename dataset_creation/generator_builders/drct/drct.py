from collections import defaultdict
from functools import partial
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    FAKE_IMAGES_LABEL,
    PathAlike,
    are_all_generated,
    create_shared_max_samples,
)
from generator_builders.deepfake_dataset_builder import (
    AvailableFile,
    DeepfakeDatasetBuilder,
)
from datasets import Dataset

from dataset_utils.file_extraction import list_archive_files
from generator_builders.drct.drct_utils import FOLDER_NAME_TO_DRCT_NAME, DRCT_generator

# Example:
# DRCT/sd-turbo/train2017/000000000078
FILE_ID_FORMAT = re.compile(
    r"^DRCT/(?P<generator_name>[^/]+)/(?P<subset>train2017|val2017)/(?P<image_id>\d{12})$",
)


class DRCT_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        output_path: PathAlike,
        coco2017_val_captions_path: PathAlike,
        coco2017_train_captions_path: PathAlike,
        generator_config: Any,
        convert_to_jpeg: bool = False,
        tmp_cache_dir: Optional[PathAlike] = None,
        num_proc: int = 16,
        seed: int = 1234,
        check_already_done_marker: bool = True,
        cleanup_cache_on_exit: bool = True,
        pre_selected_image_ids: Optional[List[str]] = None,
    ):
        self.coco_val = coco2017_val_captions_path
        self.coco_train = coco2017_train_captions_path
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
        print("Loading the COCO 2017 train captions...")
        # Read json
        with open(self.coco_train, "r") as f:
            coco_train_captions = json.load(f)["annotations"]

        print("Loading the COCO 2017 validation captions...")
        # Read json
        with open(self.coco_val, "r") as f:
            coco_val_captions = json.load(f)["annotations"]

        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]
        max_samples, lock = create_shared_max_samples(self.generator_config)
        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            generator=partial(
                DRCT_generator,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                max_samples,
                lock,
                coco_train_captions,
                coco_val_captions,
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
                # File inside a zip file
                generator_name = FOLDER_NAME_TO_DRCT_NAME[
                    str(Path(fpath.split("!")[1]).parts[0])
                ]
            else:
                # File inside a folder
                generator_name = FOLDER_NAME_TO_DRCT_NAME[str(Path(fpath).parts[-3])]
            indices_dict[generator_name].append(idx)

        return indices_dict

    def _make_available_files_list(self) -> List[AvailableFile]:
        result = []
        for file_path in _drct_image_list_zip(self.input_path):
            result.append(
                AvailableFile(
                    file_path=file_path,
                    file_id=self._make_file_id(file_path),
                )
            )
        return result

    def _make_file_id(self, file_path: str) -> str:
        fpath = Path(file_path).with_suffix("")
        intermediate_path = fpath.parent.parent
        intermediate_path_zip_or_folder_name = str(intermediate_path.name)
        if "!" in intermediate_path_zip_or_folder_name:
            intermediate_path_zip_or_folder_name = (
                intermediate_path_zip_or_folder_name.split("!")[1]
            )

        relative_path = (
            f"{intermediate_path.parent.name}/{fpath.relative_to(intermediate_path)}"
        )
        file_id = relative_path.removeprefix("images/").removeprefix(
            "images\\"
        )  # The DRCT dataset may be contained in an "images" folder (we remove it as it's redundant)
        file_id = f"{intermediate_path_zip_or_folder_name}/{file_id}"
        return f"DRCT/{file_id}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        return bool(FILE_ID_FORMAT.match(file_id))


def _drct_image_list_zip(folder: Path) -> List[str]:
    all_images: List[str] = []

    for generator_folder_name in FOLDER_NAME_TO_DRCT_NAME.keys():
        candidate_folders = [
            folder / generator_folder_name,
            folder / "images" / generator_folder_name,
            folder / (generator_folder_name + ".zip"),
            folder / "images" / (generator_folder_name + ".zip"),
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

    all_images = [x for x in all_images if not x.endswith(".json")]

    return sorted(all_images)


__all__ = ["DRCT_Builder"]
