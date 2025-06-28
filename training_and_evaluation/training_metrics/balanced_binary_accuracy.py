from typing import Optional, Union, Sequence
from torch import Tensor
from torchmetrics.classification import BinaryStatScores

from torchmetrics.functional.classification.accuracy import (
    _safe_divide,
)


class BalancedBinaryAccuracy(BinaryStatScores):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()

        # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        acc_l = _safe_divide(tp, tp + fn)
        acc_r = _safe_divide(tn, tn + fp)

        return (acc_l + acc_r) / 2

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax=None):
        return self._plot(val, ax)


__all__ = [
    "BalancedBinaryAccuracy",
]
