from io import BytesIO
from typing import Any, Dict, Generator, List

from PIL import Image
from dataset_utils.common_utils import (
    RowDict,
    check_max_samples,
    check_needed_samples,
    is_image_valid,
    prepare_image,
)
from dataset_utils.file_extraction import MultiSourceFilesIterator
from generator_builders.deepfake_dataset_builder import AvailableFile


FOLDER_NAME_TO_DRCT_NAME = {
    "stable-diffusion-v1-4": "Stable Diffusion 1.4",
    "stable-diffusion-v1-5": "Stable Diffusion 1.5",
    "stable-diffusion-2-1": "Stable Diffusion 2.1",
    "sd-turbo": "Stable Diffusion Turbo",
    "stable-diffusion-xl-base-1.0": "Stable Diffusion XL 1.0",
    "sdxl-turbo": "Stable Diffusion XL Turbo",
}


def DRCT_generator(
    label: int,
    convert_to_jpeg: bool,
    max_samples,
    lock,
    coco_train_captions: List[Dict[str, Any]],
    coco_val_captions: List[Dict[str, Any]],
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    caption_train_image_id = {
        int(annotation["image_id"]): annotation["caption"]
        for annotation in coco_train_captions
    }

    caption_val_image_id = {
        int(annotation["image_id"]): annotation["caption"]
        for annotation in coco_val_captions
    }

    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths, batch_size=64) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            intermediate_path = fpath.parent.parent
            intermediate_path_zip_or_folder_name = str(intermediate_path.name)
            generator = FOLDER_NAME_TO_DRCT_NAME[intermediate_path_zip_or_folder_name]

            if not check_needed_samples(max_samples, generator, lock):
                continue

            try:
                with open(fpath, "rb") as fh:
                    buf = BytesIO(fh.read())
                img = Image.open(buf)
                img.load()
            except Exception as e:
                print(f"Problematic image: {file_info.file_path} produce error {e}")
                continue

            if not is_image_valid(img):
                print(f"Disarded image: {fpath}")
                continue

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            # All images should be paired with a coco image from train or val
            coco_image_folder = fpath.parent.name
            coco_image_name = fpath.name.split(".")[0]
            coco_image_id = int(coco_image_name)
            positive_prompt = ""
            paired_image = []
            try:
                if coco_image_folder == "train2017":
                    positive_prompt = caption_train_image_id[coco_image_id]
                    paired_image.append(f"COCO2017_train/{coco_image_name}")
                elif coco_image_folder == "val2017":
                    positive_prompt = caption_val_image_id[coco_image_id]
                    paired_image.append(f"COCO2017_val/{coco_image_name}")
            except:
                print("Wrong pair: ", file_info.file_path)
                continue

            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": "",
                "positive_prompt": positive_prompt,
                "negative_prompt": "",
                "conditioning": "text",
                "origin_dataset": "DRCT",
                "paired_real_images": paired_image,
            }


__all__ = ["DRCT_generator", "FOLDER_NAME_TO_DRCT_NAME"]
