from typing import Optional

from dataset_utils.common_utils import PathAlike
from generator_builders.dm_image_detection.dmid_train_valid_subset import (
    DMID_TrainValSubset_Builder,
)
from generator_builders.dm_image_detection.dmid_utils import DMID_CLASS2IMAGE_SUBSETS


class DMID_Class2Image_Builder(DMID_TrainValSubset_Builder):
    def __init__(
        self,
        input_path: PathAlike,
        origin_split: str,
        output_path: PathAlike,
        subset_size: int = -1,
        convert_to_jpeg: bool = False,
        tmp_cache_dir: Optional[PathAlike] = None,
        num_proc: int = 16,
        seed: int = 1234,
        check_already_done_marker: bool = True,
        cleanup_cache_on_exit: bool = True,
    ):
        super().__init__(
            input_path=input_path,
            origin_split=origin_split,
            output_path=output_path,
            subsets=DMID_CLASS2IMAGE_SUBSETS,
            subset_type="class2img",
            subset_size=subset_size,
            convert_to_jpeg=convert_to_jpeg,
            tmp_cache_dir=tmp_cache_dir,
            num_proc=num_proc,
            seed=seed,
            check_already_done_marker=check_already_done_marker,
            cleanup_cache_on_exit=cleanup_cache_on_exit,
        )


__all__ = ["DMID_Class2Image_Builder"]
