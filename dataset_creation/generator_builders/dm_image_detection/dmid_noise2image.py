from typing import Any, Dict, List, Optional

from dataset_utils.common_utils import PathAlike
from generator_builders.dm_image_detection.dmid_train_valid_subset import (
    DMID_TrainValSubset_Builder,
)
from generator_builders.dm_image_detection.dmid_utils import DMID_NOISE2IMAGE_SUBSETS


class DMID_Noise2Image_Builder(DMID_TrainValSubset_Builder):
    def __init__(
        self,
        input_path: PathAlike,
        origin_split: str,
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
            origin_split=origin_split,
            output_path=output_path,
            generator_config=generator_config,
            subsets=DMID_NOISE2IMAGE_SUBSETS,
            subset_type="noise2img",
            subset_size=subset_size,
            convert_to_jpeg=convert_to_jpeg,
            tmp_cache_dir=tmp_cache_dir,
            num_proc=num_proc,
            seed=seed,
            check_already_done_marker=check_already_done_marker,
            cleanup_cache_on_exit=cleanup_cache_on_exit,
            pre_selected_image_ids=pre_selected_image_ids,
        )


__all__ = ["DMID_Noise2Image_Builder"]
