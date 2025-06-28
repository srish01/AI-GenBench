from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import OrderedDict

from dataset_utils.common_utils import (
    REAL_IMAGES_LABEL,
    PathAlike,
    RowDictPath,
)
from datasets import Dataset, load_from_disk, DatasetDict

from generator_builders.elsa_d3.elsa_utils import load_elsa_d3
from real_builders.real_images_builder import RealImagesBuilder


class LaionELSAD3Builder(RealImagesBuilder):

    def __init__(
        self,
        root_paths: Tuple[PathAlike, PathAlike],
        convert_to_jpeg: bool = False,
        num_proc: int = 1,
        elsa_d3_dataset_path: Optional[PathAlike] = None,
    ):
        self.elsa_d3_dataset_path: Optional[Path] = (
            Path(elsa_d3_dataset_path) if elsa_d3_dataset_path else None
        )
        self.root_paths: List[Path] = [Path(x) for x in root_paths]
        self.convert_to_jpeg: bool = convert_to_jpeg
        self.num_proc: int = num_proc

        self._arrow_datasets: Optional[List[Dataset]] = None
        self._elsa_d3_dataset: Optional[DatasetDict] = None

        self._laion_id_to_dataset: Optional[
            Dict[str, Tuple[Dataset, int, Dataset, int]]
        ] = None

        self._available_images = None

    @property
    def elsa_d3_dataset(self) -> DatasetDict:
        if self._elsa_d3_dataset is None:
            if self.elsa_d3_dataset_path is None:
                raise ValueError(
                    "The provided LAION subsets do not feature a id, which means the elsa_d3_dataset_path must be provided to load ELSA D3 dataset to obtain file ids."
                )
            self._elsa_d3_dataset = load_elsa_d3(
                self.elsa_d3_dataset_path, num_proc=self.num_proc
            )

            for split in self._elsa_d3_dataset:
                self._elsa_d3_dataset[split] = self._elsa_d3_dataset[
                    split
                ].select_columns(
                    [
                        "id",
                        "original_prompt",
                    ]
                )

        return self._elsa_d3_dataset

    @property
    def laion_datasets(self) -> List[Dataset]:
        if self._arrow_datasets is None:
            self._arrow_datasets = [load_from_disk(str(x)) for x in self.root_paths]
        return self._arrow_datasets

    @property
    def laion_train_subset(self) -> Dataset:
        return self.laion_datasets[0]

    @property
    def laion_val_subset(self) -> Dataset:
        return self.laion_datasets[1]

    @property
    def laion_id_to_dataset(self) -> Dict[str, Tuple[Dataset, int, Dataset, int]]:
        if self._laion_id_to_dataset is None:
            expected_prefix = self.get_prefix()
            self._laion_id_to_dataset = {}
            for split_idx, split_name in enumerate(["train", "validation"]):
                laion_subset = self.laion_datasets[split_idx]

                # Note: filename is the legacy name for the file_id column
                subset_filenames: List[str] = (
                    laion_subset["file_id"]
                    if "file_id" in laion_subset.column_names
                    else laion_subset["filename"]
                )

                # If all filenames start with the expected_prefix...
                if all(
                    filename.startswith(expected_prefix)
                    for filename in subset_filenames
                ):
                    for idx, filename in enumerate(subset_filenames):
                        laion_id = filename.split("/")[1]
                        self._laion_id_to_dataset[laion_id] = (
                            laion_subset,
                            idx,
                            laion_subset,
                            idx,
                        )
                else:
                    elsa_dataset = self.elsa_d3_dataset[split_name]
                    laion_ids: List[str] = elsa_dataset["id"]
                    for idx, filename in enumerate(subset_filenames):
                        fake_img_index = int(filename.split(".")[0])
                        self._laion_id_to_dataset[laion_ids[fake_img_index]] = (
                            laion_subset,
                            idx,
                            elsa_dataset,
                            fake_img_index,
                        )

            self._laion_id_to_dataset = OrderedDict(
                sorted(self._laion_id_to_dataset.items(), key=lambda x: x[0])
            )

        return self._laion_id_to_dataset

    def get_prefix(self) -> str:
        return f"LAION-400M"

    def get_builder_name(self) -> str:
        return f"LAION-400M (ELSA_D3 subset)"

    def available_images(self) -> Iterable[str]:
        if self._available_images is not None:
            return list(self._available_images)

        result = []
        prefix = self.get_prefix()
        for laion_id in self.laion_id_to_dataset.keys():
            result.append(f"{prefix}/{laion_id}")

        self._available_images = result
        return list(result)

    def get_image(
        self,
        image_id: str,
    ) -> RowDictPath:
        prefix = self.get_prefix()
        id_prefix = image_id.split("/")[0]
        if id_prefix != prefix:
            raise ValueError(f"Invalid image_id: {image_id}")

        laion_id = image_id.split("/")[1]
        image_def = self.laion_id_to_dataset[laion_id]
        description: str
        metadata_row = image_def[2][image_def[3]]
        if "description" in metadata_row:
            description = metadata_row["description"]
        else:
            description = metadata_row["original_prompt"]

        return {
            "image": (image_def[0], image_def[1]),
            "label": REAL_IMAGES_LABEL,
            "generator": "",
            "file_id": f"{prefix}/{laion_id}",
            "description": description,
            "positive_prompt": "",
            "negative_prompt": "",
            "conditioning": "",
            "origin_dataset": prefix,
            "paired_real_images": [],
        }


if __name__ == "__main__":

    laion_elsa_subset = LaionELSAD3Builder(
        "/deepfake/ELSA_D3_offline",
        (
            "/deepfake/aaa/laion400m_elsad3_real_train_arrow",
            "/deepfake/aaa/laion400m_elsad3_real_validation_arrow",
        ),
        convert_to_jpeg=False,
        num_proc=1,
    )

    check_exist = laion_elsa_subset.available_images()
    print("N images:", len(check_exist))

    dataset = laion_elsa_subset.get_image(list(check_exist)[200])
    print(dataset)


__all__ = ["LaionELSAD3Builder"]
