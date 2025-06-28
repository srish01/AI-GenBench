from pathlib import Path
from typing import Any, Dict, Generator, List, Union
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
from generator_builders.imagenet_class_names import IMAGENET_CLASS_ID_TO_STR
from generator_builders.generator_utils import (
    DatasetContext,
    process_generator,
)


ARTIFACT_REAL_IMAGES_FOLDERS = [
    "afhq",
    "celebahq",
    "coco",
    "ffhq",
    "imagenet",
    "landscape",
    "lsun",
    "metfaces",
]

ARTIFACT_MIXED_IMAGES_FOLDERS = [
    "cycle_gan",
]

FOLDER_TO_NAMES = {
    "afhq": "AFHQ",
    "afhq_v2": "AFHQv2",
    "celebahq": "CelebAHQ",
    "coco": "COCO2017",  # NOTE: in metadata.csv there is a field stating if it comes from train, valid or test
    "ffhq": "FFHQ",
    "imagenet": "ImageNet",
    "landscape": "Landscape",
    "lsun": "LSUN",
    "metfaces": "MetFaces",
    "cycle_gan": "CycleGAN",
    "big_gan": "BigGAN",
    "cips": "CIPS",
    "ddpm": "DDPM",
    "denoising_diffusion_gan": "Denoising Diffusion GAN",
    "diffusion_gan": "Diffusion GAN",
    "face_synthetics": "FaceSynthetics",
    "gansformer": "GANformer",
    "gau_gan": "GauGAN",
    "generative_inpainting": "SN-PatchGAN",
    "glide": "Glide",
    "lama": "LaMa",
    "latent_diffusion": "Latent Diffusion",
    "mat": "MAT",
    "palette": "Palette",
    "pro_gan": "ProGAN",
    "projected_gan": "ProjectedGAN",
    "sfhq": "StyleGAN2 (SFHQ)",
    "stable_diffusion": "Stable Diffusion",
    "star_gan": "StarGAN",
    "stylegan1": "StyleGAN1",
    "stylegan2": "StyleGAN2",
    "stylegan3": "StyleGAN3",
    "taming_transformer": "VQGAN",
    "vq_diffusion": "VQ-Diffusion",
}


def glide_conditioning(data):
    fpath = data["fpath"]
    if fpath.parent.name == "text2img" or fpath.parent.parent.name.endswith("-t2i"):
        return "text"
    elif fpath.parent.name == "inpainting" or fpath.parent.parent.name.endswith("-in"):
        return "inpainting/text"
    else:
        raise ValueError(
            f"Can't detect Glide conditioning type for {fpath.parent.name}"
        )


def artifact_test_generator(
    root_path: Path,
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = artifact_test_get_generator_name(root_path, fpath)

            if not check_needed_samples(max_samples, generator, lock):
                continue

            try:
                img = Image.open(fpath)
            except Exception as e:
                print(f"Problematic image: {fpath} produce error {e}")
                continue

            if not is_image_valid(img):
                print(f"Disarded image: {fpath}")
                continue

            relative_path = fpath.relative_to(root_path)
            conditioning = None
            description = None
            positive_prompt = None
            paired_real_images = []

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            context = DatasetContext()
            context.add_values(
                {
                    "parent_name": lambda: fpath.parent.name,
                    "parent_parent_name": lambda: fpath.parent.parent.name,
                    "cips_description": lambda ctx=context: ctx[
                        "parent_name"
                    ].removeprefix("cips-"),
                    "class_id": lambda ctx=context: ctx["parent_name"],
                    "cyclegan_description": lambda ctx=context: ctx[
                        "parent_name"
                    ].split("2")[-1],
                    "diffusiongan_description": lambda ctx=context: ctx[
                        "parent_parent_name"
                    ]
                    .removeprefix("lsun-")
                    .removesuffix("-data"),
                    "ganformer_description": lambda ctx=context: ctx[
                        "parent_name"
                    ].removesuffix("_images"),
                    "stargan_face_modifier": lambda ctx=context: ctx["parent_name"]
                    .replace("_", " ")
                    .lower(),
                    "stablediffusion_face_modifier": lambda ctx=context: ctx[
                        "parent_name"
                    ].lower(),
                    "stylegan2_description": lambda: relative_path.parts[1].split("-")[
                        0
                    ],
                    "stylegan3_description": lambda ctx=context: ctx[
                        "parent_name"
                    ].split("-")[-2],
                    "vqgan_base": lambda: relative_path.parts[1],
                    "imagenet_class": lambda: IMAGENET_CLASS_ID_TO_STR[
                        int(fpath.parent.name)
                    ],
                }
            )
            data = {
                "fpath": fpath,
            }
            values = ["conditioning", "description", "positive_prompt"]
            final_values = process_generator(
                generator_config, generator, values, context, data
            )
            conditioning = final_values["conditioning"]
            description = final_values["description"]
            positive_prompt = final_values["positive_prompt"]
            assert generator is not None
            assert description is not None
            assert conditioning is not None
            assert positive_prompt is not None
            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": description,
                "positive_prompt": positive_prompt,
                "negative_prompt": "",
                "conditioning": conditioning,
                "origin_dataset": "Artifact",
                "paired_real_images": paired_real_images,
            }


def artifact_test_get_generator_name(
    root_path: Union[str, Path], fpath: Union[str, Path]
):
    fpath = Path(fpath)
    generator_folder_name = _artifact_extract_folder_name(root_path, fpath)
    generator_name = FOLDER_TO_NAMES[generator_folder_name]
    if generator_folder_name == "diffusion_gan":
        subgenerator = fpath.parent.name.split("-")[1]
        generator_name = f"{generator_name} ({subgenerator})"

    return generator_name


def artifact_test_get_real_dataset_name(
    root_path: Union[str, Path], fpath: Union[str, Path]
):
    fpath = Path(fpath)
    dataset_folder_name = _artifact_extract_folder_name(root_path, fpath)

    if dataset_folder_name == "afhq":
        subfolder = fpath.relative_to(root_path).parts[2]
        return FOLDER_TO_NAMES[subfolder]  # Either 'afhq' or 'afhq_v2'

    return FOLDER_TO_NAMES[dataset_folder_name]


def _artifact_extract_folder_name(
    root_path: Union[str, Path], fpath: Union[str, Path]
) -> str:
    # The path may be in the form of:
    # <root_path>/<generator_name>/<various_subfolders>/<image_name>
    # We want to extract the generator_name from the path
    # We assume that the generator_name is the first subfolder in the path

    # Get the relative path
    rel_path = Path(fpath).relative_to(Path(root_path))

    # Get the first subfolder
    generator_name = rel_path.parts[0]

    return generator_name


__all__ = [
    "ARTIFACT_REAL_IMAGES_FOLDERS",
    "ARTIFACT_MIXED_IMAGES_FOLDERS",
    "artifact_test_get_generator_name",
    "artifact_test_generator",
]
