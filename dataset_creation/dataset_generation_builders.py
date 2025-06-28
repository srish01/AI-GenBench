import itertools
from typing import Dict, Any, List, Optional
from datasets import Split
import ruamel.yaml
from pathlib import Path

from generator_builders.elsa_d3.elsa_utils import ELSA_D3_GENERATOR_NAMES
from generator_builders.dm_image_detection import (
    DMID_Noise2Image_Builder,
    DMID_TestSubset_Builder,
    DMID_Text2Image_Builder,
)
from generator_builders.elsa_d3 import ELSA_D3_Subset_Builder
from generator_builders.forensynths import (
    Forensynths_Test_Builder,
    Forensynths_TrainVal_Subset_Builder,
)
from generator_builders.genimage import GenImage_Builder
from generator_builders.aeroblade import Aeroblade_Builder
from generator_builders.drct.drct import DRCT_Builder
from generator_builders.generator_utils import add_loader
from generator_builders.artifact.artifact_utils import glide_conditioning
from generator_builders.artifact import Artifact_Builder
from generator_builders.ddmd import Ddmd_test_Builder
from generator_builders.imaginet import Imaginet_Builder
from generator_builders.polardiffshield import Polardiffshield_Builder
from generator_builders.sfhq_t2i import SFHQT2I_Builder
from generator_builders.synthbuster import Synthbuster_Builder


def open_config(
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    file_name: str,
) -> Dict[str, Any]:
    return yaml.load((configuration_files_path / f"{file_name}.yaml").open("r"))


def build_drct_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading DRCT subset...")
    builder = DRCT_Builder(
        input_path=paths["drct"],
        output_path=paths["intermediate_outputs_path"] / "DRCT",
        coco2017_val_captions_path=paths["coco"] / "captions_val2017.json",
        coco2017_train_captions_path=paths["coco"] / "captions_train2017.json",
        generator_config=open_config(configuration_files_path, yaml, "drct_feature"),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "DRCT",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"DRCT dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_genimage_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading GenImage subset...")
    builder = GenImage_Builder(
        input_path=paths["genimage"],
        output_path=paths["intermediate_outputs_path"] / "GenImage",
        generator_config=open_config(
            configuration_files_path, yaml, "genimage_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "GenImage",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"GenImage dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_dmd_test_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading Ddmd test subset...")
    builder = Ddmd_test_Builder(
        input_path=paths["tddmd"],
        output_path=paths["intermediate_outputs_path"] / "Ddmd_test",
        generator_config=open_config(
            configuration_files_path, yaml, "dmd_test_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Dmd_test",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Ddmd test dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_sfhq_t2i_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading SFHQ-T2I subset...")
    builder = SFHQT2I_Builder(
        input_path=paths["sfhq-t2i"],
        output_path=paths["intermediate_outputs_path"] / "SFHQ-T2I",
        generator_config=open_config(
            configuration_files_path, yaml, "sfhq_t2i_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "SFHQ-T2I",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"SFHQ-T2I dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_polardiffshield_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading Polardiffshield subset...")
    builder = Polardiffshield_Builder(
        input_path=paths["polardiffshield"],
        output_path=paths["intermediate_outputs_path"] / "Polardiffshield",
        generator_config=open_config(
            configuration_files_path, yaml, "polardiffshield_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Polardiffshield",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Polardiffshiled dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_aeroblade_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print("Loading Aeroblade subset...")
    builder = Aeroblade_Builder(
        input_path=paths["aeroblade"],
        output_path=paths["intermediate_outputs_path"] / "Aeroblade",
        generator_config=open_config(
            configuration_files_path, yaml, "aeroblade_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Aeroblade",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Aeroblade dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_elsa_d3_datasets(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    ELSA_generators: Dict[str, Dict[str, int]] = open_config(
        configuration_files_path, yaml, "elsa_feature"
    )["generators"]
    datasets = []
    for split, generator_id in itertools.product(
        [Split.TRAIN, Split.VALIDATION], range(4)
    ):
        generator_name = ELSA_D3_GENERATOR_NAMES[generator_id]
        train_val_str = "train" if split == Split.TRAIN else "validation"
        suffix = f"{generator_name}_{train_val_str}"

        print(f"Loading ELSA_D3 subset ({generator_name}/{train_val_str})...")
        builder = ELSA_D3_Subset_Builder(
            input_path=paths["elsa_d3"],
            origin_split=split,
            output_path=paths["intermediate_outputs_path"] / f"ELSA_D3_{suffix}",
            generator_id=generator_id,
            tmp_cache_dir=paths["tmp_cache_dir"] / f"ELSA_D3_{suffix}",
            subset_size=ELSA_generators[generator_name]["samples"],
            max_samples=ELSA_generators[generator_name]["max samples"],
            convert_to_jpeg=convert_to_jpeg,
            num_proc=num_proc,
            seed=1234,
            pre_selected_image_ids=pre_selected_image_ids,
        )
        print(
            f"ELSA_D3 ({generator_name}/{train_val_str}) dataset contains {len(builder.result_dataset)} samples."
        )
        datasets.append(builder.result_dataset)
    return datasets


def build_dmid_train_valid_datasets(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    datasets = []
    for dmid_split in ["train", "valid"]:
        print(f"Loading DMID subset (Noise2Image/{dmid_split})...")
        builder = DMID_Noise2Image_Builder(
            input_path=paths["dmid_trainingset"] / dmid_split,
            origin_split=dmid_split,
            output_path=paths["intermediate_outputs_path"]
            / f"DMimageDetection_Noise2Image_{dmid_split}",
            generator_config=open_config(
                configuration_files_path, yaml, "dmid_train_valid_feature"
            ),
            convert_to_jpeg=convert_to_jpeg,
            tmp_cache_dir=paths["tmp_cache_dir"]
            / f"DMimageDetection_Noise2Image_{dmid_split}",
            num_proc=num_proc,
            seed=1234,
            pre_selected_image_ids=pre_selected_image_ids,
        )
        print(
            f"DMID (Noise2Image/{dmid_split}) dataset contains {len(builder.result_dataset)} samples."
        )
        datasets.append(builder.result_dataset)

        print(f"Loading DMID subset (Text2Image/{dmid_split})...")
        builder = DMID_Text2Image_Builder(
            input_path=paths["dmid_trainingset"] / dmid_split,
            origin_split=dmid_split,
            output_path=paths["intermediate_outputs_path"]
            / f"DMimageDetection_Text2Image_{dmid_split}",
            generator_config=open_config(
                configuration_files_path, yaml, "dmid_train_valid_feature"
            ),
            convert_to_jpeg=convert_to_jpeg,
            tmp_cache_dir=paths["tmp_cache_dir"]
            / f"DMimageDetection_Text2Image_{dmid_split}",
            num_proc=num_proc,
            seed=1234,
            pre_selected_image_ids=pre_selected_image_ids,
        )
        print(
            f"DMID (Text2Image/{dmid_split}) dataset contains {len(builder.result_dataset)} samples."
        )
        datasets.append(builder.result_dataset)
    return datasets


def build_dmid_test_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print(f"Loading DMID subset (Test) subset...")
    builder = DMID_TestSubset_Builder(
        input_path=paths["dmid_testset"],
        coco2017_val_captions_path=paths["coco"] / "captions_val2017.json",
        output_path=paths["intermediate_outputs_path"] / "DMimageDetection_Test",
        generator_config=open_config(
            configuration_files_path, yaml, "dmid_test_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "DMimageDetection_Test",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"DMID (Test) dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_forensynths_trainval_datasets(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    datasets = []
    for forensynths_split, subset_suffix in zip(["train", "valid"], ["train", "val"]):
        print(f"Loading Forensynths subset ({forensynths_split})...")
        builder = Forensynths_TrainVal_Subset_Builder(
            input_path=paths["forensynths"] / f"progan_{subset_suffix}",
            origin_split=forensynths_split,
            output_path=paths["intermediate_outputs_path"]
            / f"Forensynths_{forensynths_split}",
            generator_config=open_config(
                configuration_files_path,
                yaml,
                f"forensynths_{forensynths_split}_feature",
            ),
            convert_to_jpeg=convert_to_jpeg,
            tmp_cache_dir=paths["tmp_cache_dir"] / f"Forensynths_{forensynths_split}",
            num_proc=num_proc,
            seed=1234,
            pre_selected_image_ids=pre_selected_image_ids,
        )
        print(
            f"Forensynths dataset ({forensynths_split}) contains {len(builder.result_dataset)} samples."
        )
        datasets.append(builder.result_dataset)
    return datasets


def build_forensynths_test_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print(f"Loading Forensynths subset (Test)...")
    builder = Forensynths_Test_Builder(
        input_path=paths["forensynths"] / "test",
        output_path=paths["intermediate_outputs_path"] / "Forensynths_Test",
        generator_config=open_config(
            configuration_files_path, yaml, "forensynths_test_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Forensynths_Test",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Forensynths dataset (Test) contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_artifact_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    add_loader(
        yaml,
        "!artifact",
        {
            "glide_conditioning": glide_conditioning,
        },
    )
    print(f"Loading Artifact subset...")
    builder = Artifact_Builder(
        input_path=paths["artifact"],
        output_path=paths["intermediate_outputs_path"] / "Artifact",
        generator_config=open_config(
            configuration_files_path, yaml, "artifact_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Artifact",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Artifact dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_synthbuster_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print(f"Loading Synthbuster subset...")
    builder = Synthbuster_Builder(
        input_path=paths["synthbuster"],
        output_path=paths["intermediate_outputs_path"] / "Synthbuster",
        generator_config=open_config(
            configuration_files_path, yaml, "synthbuster_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Synthbuster",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Synthbuster dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


def build_imaginet_dataset(
    paths: Dict[str, Path],
    configuration_files_path: Path,
    yaml: ruamel.yaml.YAML,
    convert_to_jpeg: bool,
    num_proc: int,
    pre_selected_image_ids: Optional[List[str]] = None,
):
    print(f"Loading Imaginet subset...")
    builder = Imaginet_Builder(
        input_path=paths["imaginet"],
        output_path=paths["intermediate_outputs_path"] / "Imaginet",
        generator_config=open_config(
            configuration_files_path, yaml, "imaginet_feature"
        ),
        convert_to_jpeg=convert_to_jpeg,
        tmp_cache_dir=paths["tmp_cache_dir"] / "Imaginet",
        num_proc=num_proc,
        seed=1234,
        pre_selected_image_ids=pre_selected_image_ids,
    )
    print(f"Imaginet dataset contains {len(builder.result_dataset)} samples.")
    return builder.result_dataset


__all__ = [
    "build_drct_dataset",
    "build_genimage_dataset",
    "build_dmd_test_dataset",
    "build_sfhq_t2i_dataset",
    "build_polardiffshield_dataset",
    "build_aeroblade_dataset",
    "build_elsa_d3_datasets",
    "build_dmid_train_valid_datasets",
    "build_dmid_test_dataset",
    "build_forensynths_trainval_datasets",
    "build_forensynths_test_dataset",
    "build_artifact_dataset",
    "build_synthbuster_dataset",
    "build_imaginet_dataset",
]
