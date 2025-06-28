from functools import partial
from typing import Any, Dict, List, Literal, Optional
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
    dmid_generator,
)

# Example:
# DMimageDetection/train/latent_diffusion_noise2image_FFHQ/guided-diffusion_noise2img-ffhq256_sample_00000000
# that is: DMimageDetection/<split>/<subset>/<filename>
FILE_ID_FORMAT = re.compile(
    r"^DMimageDetection\/(?P<split_name>train|valid)\/(?P<subset_name>[A-Za-z\-\_\.\d]+)\/(?P<filename>[A-Za-z\-\_\d]+)$"
)


class DMID_TrainValSubset_Builder(DeepfakeDatasetBuilder):
    def __init__(
        self,
        input_path: PathAlike,
        origin_split: str,
        output_path: PathAlike,
        generator_config: Dict[str, Any],
        subsets: List[str],
        subset_type: Literal["text2img", "noise2img", "class2img"],
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
        self.origin_split: str = origin_split
        self.subsets: List["str"] = sorted(subsets)
        self.subset_type: Literal["text2img", "noise2img", "class2img"] = subset_type

    def _make_dataset(self, output_path: Path, indices: List[int]):
        filelist_complete = self.available_files
        selected_files = [filelist_complete[i] for i in indices]
        max_samples, lock = create_shared_max_samples(self.generator_config)

        print("Generating the fake images dataset...")
        subset = Dataset.from_generator(
            generator=partial(
                dmid_generator,
                FAKE_IMAGES_LABEL,
                self.convert_to_jpeg,
                self.generator_config,
                max_samples,
                lock,
                self.subset_type,
                self.origin_split,
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
        files = []
        for subset in self.subsets:
            files.extend((self.input_path / subset).glob("*.png"))

        files = sort_paths(files)
        result = []
        for file in files:
            file_path = str(file)
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
        return f"DMimageDetection/{self.origin_split}/{relative_path}"

    def _is_valid_file_id(self, file_id: str) -> bool:
        matched = FILE_ID_FORMAT.match(file_id)
        if matched is None:
            return False

        # Check self.origin_split
        split_name = matched.group("split_name")
        if split_name != self.origin_split:
            return False

        #  Check if filename contains the subset_type
        filename = matched.group("filename")
        if self.subset_type not in filename:
            return False

        # Check if subset is in any of subsets
        subset_name = matched.group("subset_name")
        if subset_name not in self.subsets:
            return False

        return True


__all__ = ["DMID_TrainValSubset_Builder"]
