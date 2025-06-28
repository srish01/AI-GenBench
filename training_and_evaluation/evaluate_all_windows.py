import click
from pathlib import Path
import subprocess
from typing import List, Optional, Union

import yaml
import jsonargparse


TYPE = "probe_resize"
MODEL_NAME = "RN50_clip"
ADDITIONAL_ARGS = []
BENCHMARK_CONFIG = (
    f"training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml"
)
MODEL_CONFIG = f"training_configurations/{MODEL_NAME}/{MODEL_NAME}_{TYPE}.yaml"

NEW_POSTFIX = "re_evaluation"


def run_command(
    checkpoint_path,
    window_id,
    logger_step,
    benchmark_config,
    model_config,
    output_experiment_name,
    additional_args,
):
    print(
        "Running command with checkpoint path",
        checkpoint_path,
        "and window id",
        window_id,
    )
    main_script = "lightning_main.py"

    command = [
        "python",
        main_script,
        "validate",
        "--config",
        str(benchmark_config),
        "--config",
        str(model_config),
        "--experiment_info.experiment_id",
        str(output_experiment_name),
        "--experiment_info.sliding_windows_definition.current_window",
        str(window_id),
        "--model.base_weights",
        str(checkpoint_path),
        *additional_args,
    ]

    if logger_step >= 0:
        command += ["--experiment_info.logging_initial_step", str(logger_step)]

    print("Command:", " ".join(command))

    subprocess.run(command, check=True)


@click.command()
@click.argument("experiment_id")
@click.argument("additional_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--experiment_data_path",
    default="experiments_data",
    help="Path to the experiment data folder",
)
@click.option(
    "--benchmark_config",
    default=BENCHMARK_CONFIG,
    help="Path to the benchmark configuration for evaluation",
)
@click.option(
    "--model_config", default=MODEL_CONFIG, help="Path to the model configuration"
)
@click.option(
    "--output_experiment_name",
    default=None,
    help="Name for the output experiment. Defaults to experiment_id_re_evaluation",
)
@click.option("--max_window_id", default=8, help="Maximum window ID")
@click.option(
    "--aggregate",
    is_flag=True,
    help="Aggregate all TensorBoard event files in the last window folder",
)
def evaluate_multiwindow(
    experiment_id: str,
    additional_args: List[str],
    experiment_data_path: Union[str, Path],
    benchmark_config: Union[str, Path],
    model_config: Union[str, Path],
    output_experiment_name: Optional[str],
    max_window_id: int,
    aggregate: bool,
):
    if output_experiment_name is None:
        output_experiment_name = f"{experiment_id}_{NEW_POSTFIX}"
    experiment_data_path = Path(experiment_data_path).expanduser()
    benchmark_config = Path(benchmark_config).expanduser()
    model_config = Path(model_config).expanduser()

    assert (
        experiment_data_path.exists()
    ), f"Experiment path {experiment_data_path} not found"
    assert benchmark_config.exists(), f"Benchmark config {benchmark_config} not found"
    assert model_config.exists(), f"Model config {model_config} not found"

    for window_id in range(0, max_window_id + 1):
        print("Evaluating window", window_id)

        experiment_folder = experiment_data_path / f"experiment_{experiment_id}"
        assert (
            experiment_folder.exists()
        ), f"Experiment folder {experiment_folder} not found"

        window_folder = experiment_folder / f"window_{window_id}"
        assert window_folder.exists(), f"Window folder {window_folder} not found"

        checkpoints_folder = window_folder / "checkpoints"
        assert (
            checkpoints_folder.exists()
        ), f"Checkpoints folder {checkpoints_folder} not found"

        checkpoint_file = _find_checkpoint_file(checkpoints_folder)
        assert (
            checkpoint_file is not None
        ), f"No checkpoint found in {checkpoints_folder}"
        print("Checkpoint file:", checkpoint_file)

        global_step = -1
        try:
            with open(window_folder / "step_tracker", "r") as f:
                global_step = int(f.read())
                assert global_step >= 0
        except:
            pass

        run_command(
            checkpoint_file,
            window_id,
            global_step,
            benchmark_config,
            model_config,
            output_experiment_name,
            list(additional_args),
        )

    print("Evaluation finished for all windows")
    if not aggregate:
        return 0

    print("Moving TensorBoard event files into the last window folder...")

    save_dir = _get_tensorboard_save_dir([benchmark_config, model_config])
    assert save_dir is not None, "Could not find save_dir in the model config"
    save_dir = Path(save_dir).expanduser()
    results_path = save_dir / f"experiment_{output_experiment_name}"
    assert results_path.exists(), f"Results path {results_path} not found"

    all_tfevent_files = list(results_path.glob("**/*tfevents*"))
    all_tfevent_files = [x.resolve() for x in all_tfevent_files if x.is_file()]
    print("Found", len(all_tfevent_files), "tfevent files in folder", results_path)

    if len(all_tfevent_files) == 0:
        raise RuntimeError("No TensorBoard event files found in the results folder")

    last_window_folder = (results_path / f"window_{max_window_id}").resolve()
    all_tfevent_files = [
        x for x in all_tfevent_files if not x.is_relative_to(last_window_folder)
    ]
    last_tfevent_files = list(last_window_folder.glob("**/*tfevents*"))
    last_tfevent_files = [x.resolve() for x in last_tfevent_files if x.is_file()]

    for tfevent_file in all_tfevent_files:
        # Copy by keeping timestamps
        subprocess.run(
            ["cp", "-p", str(tfevent_file), str(last_window_folder)], check=True
        )

        window_id = tfevent_file.parent.name.removeprefix("window_")

        # Rename by adding the ".win_<number>" suffix
        new_name = tfevent_file.name + f".win_{window_id}"

        subprocess.run(
            [
                "mv",
                str(last_window_folder / tfevent_file.name),
                str(last_window_folder / new_name),
            ],
            check=True,
        )

        print("Moved", tfevent_file, "to", last_window_folder)

    for tfevent_file in last_tfevent_files:
        # Add the ".win_<number>" suffix
        window_id = tfevent_file.parent.name.removeprefix("window_")
        new_name = tfevent_file.name + f".win_{window_id}"

        subprocess.run(
            [
                "mv",
                str(tfevent_file),
                str(last_window_folder / new_name),
            ],
            check=True,
        )

        print("Renamed", tfevent_file, "to", new_name)

    # Remove all tfevents
    for tfevent_file in all_tfevent_files:
        tfevent_file.unlink()


def _find_checkpoint_file(candidate_checkpoint_dir: Path) -> Optional[Path]:
    if not candidate_checkpoint_dir.exists():
        return None

    checkpoint_files = list(candidate_checkpoint_dir.glob("*.ckpt"))
    if len(checkpoint_files) == 0:
        return None

    # Order by creation time
    checkpoint_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    return checkpoint_files[0]


def _get_tensorboard_save_dir(configuration_yamls: List[Path]) -> Optional[Path]:
    save_dir = None
    for config_file in configuration_yamls:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logger_config = config.get("trainer", {}).get("logger", None)
            if not logger_config:
                continue

            # If logger_config is a dict, make it a list for uniformity
            if not isinstance(logger_config, list):
                logger_config = [logger_config]

            # Iterate over loggers to find a save_dir with tfevent files
            for logger in logger_config:
                init_args = logger.get("init_args", {})
                candidate_save_dir = init_args.get("save_dir", None)
                if candidate_save_dir:
                    tfevent_files = list(Path(candidate_save_dir).glob("**/*tfevents*"))
                    tfevent_files = [x.resolve() for x in tfevent_files if x.is_file()]
                    if len(tfevent_files) > 0:
                        save_dir = candidate_save_dir
                        break
            if save_dir:
                break

    if save_dir:
        save_dir = Path(save_dir).expanduser()
    return save_dir


if __name__ == "__main__":
    evaluate_multiwindow()
