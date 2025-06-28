from typing import Callable, List, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    det_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)
from scipy.special import softmax


def evaluate_predictions_legacy_format(
    predictions_file: Union[str, Path],
    computed_on: str,
    window_id: int,
    threshold: float = 0.5,
    use_pairing: bool = True,
    unknown_model_predictions_converter: Optional[Callable] = None,
    save_plots_dir: Optional[Union[str, Path]] = None,
):
    data = np.load(predictions_file)
    predictions: np.ndarray = data["prediction"]
    ground_truth: np.ndarray = data["generator"]
    generator_id_to_name: np.ndarray = data["generator_id_to_name"]
    generator_id_to_name_list: List[str] = generator_id_to_name.tolist()
    pairing: np.ndarray = data["pairing"]

    windows_timeline: List[List[str]] = data["windows_timeline"].tolist()

    set_windows_order = [
        set([generator_id_to_name_list.index(gen_name) for gen_name in window])
        for window in windows_timeline
    ]

    generators_to_consider = _considered_generators(
        computed_on, set_windows_order, window_id
    )

    # Ensure predictions are binary class labels (0 for real, 1 for fake)
    if predictions.ndim > 1:
        if predictions.shape[1] == 1:
            # Already a binary prediction, just stored as (N, 1) -> flatten to be (N,)
            predictions = predictions.flatten()
        elif predictions.shape[1] == 2:
            # A 2-class prediction, stored as (N, 2) -> convert to binary using a softmax, take the fake (class 1) score
            predictions = softmax(predictions, axis=1)
            predictions = predictions[:, 1]
        elif predictions.shape[1] == len(generator_id_to_name):
            # Multiclass prediction, stored as (N, num_classes) -> convert to labels using argmax
            # NOTE: you may want to use a different mechanism here...
            predictions = np.argmax(predictions, axis=1)
        else:
            if unknown_model_predictions_converter is None:
                raise ValueError(
                    f"Unexpected prediction shape {predictions.shape} for model predictions. "
                    "Please provide a custom `unknown_model_predictions_converter` function to handle this case."
                )
            predictions = unknown_model_predictions_converter(
                predictions, ground_truth, generator_id_to_name
            )

    mask = np.zeros_like(ground_truth, dtype=bool)
    for gen_id in generators_to_consider:
        mask |= ground_truth == gen_id

    if use_pairing:
        # Only take real images (generator 0) paired with considered generators
        for i, generator_id, paired_generator_id in zip(
            range(len(ground_truth)), ground_truth, pairing
        ):
            if generator_id != 0:
                continue

            assert paired_generator_id != -1

            if paired_generator_id in generators_to_consider:
                mask[i] = True
    else:
        mask |= ground_truth == 0

    predictions = predictions[mask]
    ground_truth = ground_truth[mask]

    # Print number of images per generator (from the ground truth)
    # unique_generators, counts = np.unique(ground_truth, return_counts=True)
    # for generator_id, count in zip(unique_generators, counts):
    #     print(f"Generator {generator_id}: {count} images")

    # Print number of real and fake images
    real_images = np.sum(ground_truth == 0)
    fake_images = np.sum(ground_truth != 0)
    print(f"Real images: {real_images}")
    print(f"Fake images: {fake_images}")

    # Convert continuous predictions to binary (0 for real, 1 for fake) using a threshold
    binary_predictions = (predictions >= threshold).astype(int)

    # Convert ground truth to binary (0 for real, 1 for fake)
    ground_truth_binary = (ground_truth != 0).astype(int)

    # Calculate overall metrics
    accuracy = accuracy_score(ground_truth_binary, binary_predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth_binary, binary_predictions)
    precision = precision_score(ground_truth_binary, binary_predictions)
    recall = recall_score(ground_truth_binary, binary_predictions)
    f1 = f1_score(ground_truth_binary, binary_predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth_binary, binary_predictions).ravel()
    tnr = tn / (tn + fp)
    average_precision = average_precision_score(ground_truth_binary, predictions)
    roc_auc = roc_auc_score(ground_truth_binary, predictions)

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1 Score: {f1:.4f}")
    print(f"Overall True Negative Rate (TNR): {tnr:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    if save_plots_dir is not None:
        # Plot Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            ground_truth_binary, predictions
        )
        plt.figure()
        plt.plot(recall_curve, precision_curve, marker=".")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(Path(save_plots_dir) / f"precision_recall_curve_{window_id}.png")
        plt.close()

        # Plot DET curve
        fpr, fnr, _ = det_curve(ground_truth_binary, predictions)
        plt.figure()
        plt.plot(fpr, fnr, marker=".")
        plt.title("DET Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.savefig(Path(save_plots_dir) / f"det_curve_{window_id}.png")
        plt.close()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(ground_truth_binary, predictions)
        plt.figure()
        plt.plot(fpr, tpr, marker=".")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(Path(save_plots_dir) / f"roc_curve_{window_id}.png")
        plt.close()

    return {
        "n_real_images": real_images,
        "n_fake_images": fake_images,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "counts_per_generator": np.unique(ground_truth, return_counts=True),
    }


def _considered_generators(compute_mechanism: str, windows_order, current_window: int):
    generators_to_consider = None
    if compute_mechanism == "immediate_future":
        next_window_idx = min(current_window + 1, len(windows_order) - 1)
        generators_to_consider = set(windows_order[next_window_idx])

    elif compute_mechanism == "growing":
        generators_to_consider = set()
        for window in range(current_window + 1):
            generators_to_consider.update(windows_order[window])

    elif compute_mechanism == "growing_whole":
        generators_to_consider = set()
        for window in range(min(current_window + 2, len(windows_order))):
            generators_to_consider.update(windows_order[window])

    elif compute_mechanism == "past":
        generators_to_consider = set()
        for window in range(current_window):
            generators_to_consider.update(windows_order[window])
    else:
        raise ValueError(f"Invalid compute mechanism: {compute_mechanism}")

    return generators_to_consider


__all__ = [
    "evaluate_predictions_legacy_format",
]
