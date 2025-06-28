from collections import OrderedDict
from pathlib import Path

# The local.cfg or local_simple.cfg files are used to specify the paths for input datasets and output directories.
# There are some mandatory paths that must be set in the local.cfg file, and some optional ones.
# The mandatory paths are: input_datasets_path, output_path, tmp_cache_dir, and intermediate_outputs_path.
# You can optionally set the paths for the datasets: (see datasets_default_folder_names in the code).
# If you do not set the paths for these datasets, they will be searched in the input_datasets_path directory.

# Example local.cfg content:
# [paths]
# input_datasets_path = /deepfake
# output_path = ~/deepfake_benchmark_output
# tmp_cache_dir = ~/deepfake_benchmark_output_cache
# intermediate_outputs_path = ~/deepfake_benchmark_intermediate_outputs
# drct = "/different_folder/DRCT-2M"
# imagenet "/classic_datasets/imagenet"


def parse_dataset_paths(config_file="local.cfg"):
    default_working_paths = [
        ("input_datasets_path", "/deepfake"),
        ("output_path", "~/deepfake_benchmark_output"),
        ("tmp_cache_dir", "~/deepfake_benchmark_output_cache"),
        ("intermediate_outputs_path", "~/deepfake_benchmark_intermediate_outputs"),
    ]

    defaults: OrderedDict[str, Path] = OrderedDict()
    for key, default_path in default_working_paths:
        defaults[key] = Path(default_path).expanduser().resolve()

    datasets_default_folder_names = {
        "drct": "DRCT-2M",
        "tddmd": "TDDMD",
        "sfhq-t2i": "SFHQ-T2I",
        "polardiffshield": "polardiffshield",
        "aeroblade": "aeroblade",
        "elsa_d3": "ELSA_D3_offline",
        "dmid_trainingset": "DMimageDetection_latent_diffusion_trainingset",
        "dmid_testset": "DMimageDetection_TestSet",
        "forensynths": "forensynths",
        "artifact": "artifact",
        "genimage": "GenImage",
        "synthbuster": "synthbuster",
        "imaginet": "imaginet",
        "coco": "coco",
        "imagenet": "imagenet",
        "raise": "RAISE_all",
        "laion400m_elsad3_real_train": "laion400m_elsad3_real_train_arrow",
        "laion400m_elsad3_real_validation": "laion400m_elsad3_real_validation_arrow",
    }

    result: OrderedDict = OrderedDict()
    for key, folder_name in defaults.items():
        result[key] = Path(folder_name).expanduser().resolve()

    if Path(config_file).exists():
        # Read paths from local.cfg
        from configparser import ConfigParser

        config = ConfigParser()
        config.read(config_file)

        for key in result.keys():
            result[key] = Path(config["paths"][key]).expanduser().resolve()

        for key, folder_name in datasets_default_folder_names.items():
            if key in config["paths"]:
                result[key] = (Path(config["paths"][key])).expanduser().resolve()

    input_datasets_path = result["input_datasets_path"]

    for key, folder_name in datasets_default_folder_names.items():
        if key not in result:
            result[key] = input_datasets_path / folder_name

    any_error = False
    print("Using paths:")
    for key, path in result.items():
        print(f"  {key}: {path}", end="")
        if not path.exists():
            print(" (does not exist!)")

            if key in {
                "input_datasets_path",
                "output_path",
                "tmp_cache_dir",
                "intermediate_outputs_path",
            }:
                continue
            any_error = True
        else:
            print()

    if any_error:
        raise FileNotFoundError("Some paths do not exist (see output)")

    return result


def parse_simple_dataset_paths(config_file="local_simple.cfg"):
    default_working_paths = [
        ("input_datasets_path", "/deepfake"),
        ("output_path", "~/deepfake_benchmark_output"),
        ("tmp_cache_dir", "~/deepfake_benchmark_output_cache"),
        ("intermediate_outputs_path", "~/deepfake_benchmark_intermediate_outputs"),
    ]

    defaults: OrderedDict[str, Path] = OrderedDict()
    for key, default_path in default_working_paths:
        defaults[key] = Path(default_path).expanduser().resolve()

    datasets_default_folder_names = {
        "coco": "coco",
        "imagenet": "imagenet",
        "raise": "RAISE_all",
        "laion400m_elsad3_real_train": "simple_laion400m_elsad3_real_train_arrow",
        "laion400m_elsad3_real_validation": "simple_laion400m_elsad3_real_validation_arrow",
    }

    result: OrderedDict = OrderedDict()
    for key, folder_name in defaults.items():
        result[key] = Path(folder_name).expanduser().resolve()

    if Path(config_file).exists():
        # Read paths from local.cfg
        from configparser import ConfigParser

        config = ConfigParser()
        config.read(config_file)

        for key in result.keys():
            result[key] = Path(config["paths"][key]).expanduser().resolve()

        for key, folder_name in datasets_default_folder_names.items():
            if key in config["paths"]:
                result[key] = (Path(config["paths"][key])).expanduser().resolve()

    input_datasets_path = result["input_datasets_path"]

    for key, folder_name in datasets_default_folder_names.items():
        if key not in result:
            result[key] = input_datasets_path / folder_name

    any_error = False
    print("Using paths:")
    for key, path in result.items():
        print(f"  {key}: {path}", end="")
        if not path.exists():
            print(" (does not exist!)")

            if key in {
                "input_datasets_path",
                "output_path",
                "tmp_cache_dir",
                "intermediate_outputs_path",
            }:
                continue
            any_error = True
        else:
            print()

    if any_error:
        raise FileNotFoundError("Some paths do not exist (see output)")

    return result


__all__ = ["parse_dataset_paths"]
