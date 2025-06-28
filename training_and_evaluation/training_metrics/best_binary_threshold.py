from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

import numpy as np
from sklearn.metrics import f1_score


class _BestBinaryThreshold(Metric):
    def __init__(self, metric="f1", thresholds=None, dist_sync_on_step=False):
        """
        Computes the best threshold that yields the highest metric (accuracy or F1) score.

        Args:
            metric (str): Either 'accuracy' or 'f1'.
            thresholds (torch.Tensor, optional): A 1D tensor of thresholds to evaluate.
                Defaults to torch.linspace(0, 1, 1001).
            dist_sync_on_step (bool, optional): If True, synchronizes metric state across processes at each
                forward() before returning the value at the step. Useful in distributed settings.
                Recommended: False, as this metric should be synchronized and computed only at the
                end of an epoch.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metric_option = metric
        if thresholds is None:
            thresholds = torch.linspace(0, 1, 1001)
        self.thresholds = thresholds

        self.probabilities: List[Tensor]
        self.labels: List[Tensor]

        self.add_state("probabilities", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Accumulate predictions and true binary targets.

        Args:
            preds (torch.Tensor): Tensor of predicted probabilities.
            target (torch.Tensor): Tensor of ground truth binary labels (0 or 1).
        """
        self.probabilities.append(preds)
        self.labels.append(target)

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        result = tensor.detach()

        if result.device.type == "cuda":
            result = result.cpu()

        if result.is_floating_point() and result.dtype != torch.float32:
            # Manages bfloat16, which is not automatically converted to numpy (raises an error!)
            # https://stackoverflow.com/a/78128663
            result = result.float()

        return result.numpy()

    def compute(self):
        """
        Computes the best threshold and the score for the chosen metric over accumulated data.

        Returns:
            best_threshold: the threshold that yields the highest score for the specified metric.
        """
        probs = dim_zero_cat(self.probabilities)
        labels = dim_zero_cat(self.labels)
        labels = self.to_numpy(labels)

        best_threshold = 0.0
        best_score = -1.0

        for threshold in self.thresholds:
            # Convert threshold to a float if needed
            th = threshold.item() if isinstance(threshold, torch.Tensor) else threshold
            preds_bin = self.to_numpy(probs >= th).astype(int)

            if self.metric_option == "accuracy":
                score = (preds_bin == labels).mean()
            elif self.metric_option == "f1":
                score = f1_score(labels, preds_bin, zero_division=0)
            else:
                raise ValueError("Unsupported metric. Use 'accuracy' or 'f1'.")
            if score > best_score:
                best_score = score
                best_threshold = th

        return best_threshold


class BestBinaryThresholdF1(_BestBinaryThreshold):
    def __init__(self, thresholds=None, dist_sync_on_step=False):
        super().__init__(
            metric="f1", thresholds=thresholds, dist_sync_on_step=dist_sync_on_step
        )


class BestBinaryThresholdAccuracy(_BestBinaryThreshold):
    def __init__(self, thresholds=None, dist_sync_on_step=False):
        super().__init__(
            metric="accuracy",
            thresholds=thresholds,
            dist_sync_on_step=dist_sync_on_step,
        )


__all__ = [
    "BestBinaryThresholdF1",
    "BestBinaryThresholdAccuracy",
]


# Example usage:
if __name__ == "__main__":
    metric = BestBinaryThresholdF1()
    # Dummy predictions and labels
    preds = torch.tensor([0.2, 0.4, 0.6, 0.8]).to(torch.bfloat16)
    labels = torch.tensor([0, 0, 1, 1]).to(torch.bfloat16)
    metric.update(preds, labels)
    print("Best threshold and score:", metric.compute())
