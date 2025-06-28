"""
This script can be used to evaluate the predictions of a model on the sliding windows
benchmark protocol.

By default it computes the same metrics logged by the default set of metrics
in the training script, but you can add your own metrics by implementing adapting the code a bit.

Predictions will be taken from the npz files in the experiments folder.
By default, the "validate" ones will be taken, but you can change this by
by changing the _find_predictions_file function.

The output is a set of plots and a set of metrics printed to the console.
The script is designed to be run from the command line. Check the arguments of the
evaluate_multiwindow_predictions for more details!
"""

from collections import defaultdict
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import click

from evaluation_scripts.evaluate_predictions_utils_legacy import (
    evaluate_predictions_legacy_format,
)

PREDICTIONS_FILE_FORMAT = re.compile(r"predictions_validate_epoch=\d+-step=(\d+)\.npz")


@click.command()
@click.argument(
    "experiment_path",
    type=click.Path(exists=True),
    default="/home/lorenzo/Desktop/VCS/deepfake-benchmark/experiments_logs/RN50_clip_tune_resize/experiment_0",
)
@click.option(
    "--computed_on",
    type=click.Choice(["immediate_future", "growing", "growing_whole", "past"]),
    default="immediate_future",
)
@click.option("--use_pairing", type=bool, default=True)
@click.option("--results_plot_dir", type=click.Path(), default="./eval_plots")
def evaluate_multiwindow_predictions(
    experiment_path,
    computed_on,
    use_pairing,
    results_plot_dir,
):
    metrics_timeline = defaultdict(list)

    results_plot_dir = Path(results_plot_dir)
    results_plot_dir.mkdir(exist_ok=True, parents=True)

    experiment_path = Path(experiment_path)
    for window_id in range(9):
        print("----- Window", window_id, "-----")
        # Load the npz file

        predictions_file = _find_predictions_file(
            experiment_path / f"window_{window_id}"
        )

        window_metrics = evaluate_predictions_legacy_format(
            predictions_file,
            computed_on,
            window_id,
            threshold=0.5,
            use_pairing=use_pairing,
            unknown_model_predictions_converter=convert_unknown_model_predictions,
            save_plots_dir=results_plot_dir,
        )

        # Append metrics to the timeline
        for metric_name, metric_value in window_metrics.items():
            metrics_timeline[metric_name].append(metric_value)

        print(f"Overall Accuracy: {window_metrics['accuracy']:.4f}")
        print(f"Overall Balanced Accuracy: {window_metrics['balanced_accuracy']:.4f}")
        print(f"Overall Precision: {window_metrics['precision']:.4f}")
        print(f"Overall Recall: {window_metrics['recall']:.4f}")
        print(f"Overall F1 Score: {window_metrics['f1']:.4f}")
        print(f"Overall True Negative Rate (TNR): {window_metrics['tnr']:.4f}")
        print(f"Average Precision: {window_metrics['average_precision']:.4f}")
        print(f"ROC AUC Score: {window_metrics['roc_auc']:.4f}")

    # Print and plot metrics timeline
    print("----- Metrics Timeline -----")
    for metric_name, metric_values in metrics_timeline.items():
        if metric_name == "counts_per_generator":
            continue

        print(f"{metric_name}: {metric_values}")
        plt.figure()
        plt.plot(metric_values, marker=".")
        plt.title(f"{metric_name} Timeline")
        plt.xlabel("Window")
        plt.ylabel(metric_name)
        plt.savefig(results_plot_dir / f"timeline_{computed_on}_{metric_name}.png")
        plt.close()


def convert_unknown_model_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    generator_names_to_ids: np.ndarray,
) -> np.ndarray:
    print(
        "This happens because your model was trained on a different set of generators than the evaluation set",
        file=sys.stderr,
    )
    print(
        "Don't worry, just implement convert_unknown_model_predictions at the bottom of the script (where the following exception is raised)",
        file=sys.stderr,
    )
    raise ValueError(
        "You should implement a custom conversion from your model output to binary scores"
    )

    # In most cases, your model is trained to predict the generator ID directly, which means that 0 is usually the real class and the rest are fake classes
    # In that case you predictions shape is [num_samples, num_generators+1] (+1 because of the real class).

    # Ensure predictions are binary class labels (0 for real, 1 for fake)
    # Your code here ...

    # return binary_classification_scores


def _find_predictions_file(sliding_window_results_path: Path):
    # The "_all" file is the last one for each window
    predictions_file = sliding_window_results_path / "predictions_validate_all.npz"

    if predictions_file.exists():
        return predictions_file

    # Otherwise, we take the last one (the one with the highest step)
    max_step = -1
    selected_file = None

    for file in sliding_window_results_path.glob("*.npz"):
        match = PREDICTIONS_FILE_FORMAT.match(file.name)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                selected_file = file

    if selected_file is None:
        raise FileNotFoundError(
            f"No predictions file found in {sliding_window_results_path}"
        )

    return selected_file


if __name__ == "__main__":
    evaluate_multiwindow_predictions()
