from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
import torch
from torch import Tensor


class MultiStageBasePredictionWriter(BasePredictionWriter):
    """
    A custom version of the BasePredictionWriter that also stores predictions for "validate" and "test".
    """

    def __init__(
        self,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        super().__init__(write_interval)

        # stage -> dataloader_idx -> (predictions, batch_indices)
        self._stage_storage: Dict[str, Dict[int, List[Tensor]]] = defaultdict(
            _predictions_dict_factory
        )

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        return super().setup(trainer, pl_module, stage)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        stage: str,
    ) -> None:
        """Override with the logic to write a single batch."""
        raise NotImplementedError()

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        stage: str,
    ) -> None:
        """Override with the logic to write all batches."""
        raise NotImplementedError()

    def merge_multiprocess_predictions(
        self,
        stage: str,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ):
        pass

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.interval.on_batch:
            return
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "predict"
        )

    def on_predict_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.interval.on_epoch:
            return
        self.write_on_epoch_end(
            trainer, pl_module, trainer.predict_loop.predictions, "predict"
        )
        self.merge_multiprocess_predictions("predict", trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
            "validate",
            trainer.validate_loop,
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._on_epoch_end(trainer, pl_module, "validate")
        self.merge_multiprocess_predictions("validate", trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
            "test",
            trainer.test_loop,
        )

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._on_epoch_end(trainer, pl_module, "test")
        self.merge_multiprocess_predictions("test", trainer, pl_module)

    def _on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        stage: str,
        stage_loop: _EvaluationLoop,
    ):
        self._store_prediction_data(
            stage_loop, trainer, batch_idx, dataloader_idx, prediction, stage
        )
        if not self.interval.on_batch:
            return
        self.write_on_batch_end(
            trainer, pl_module, prediction, batch, batch_idx, dataloader_idx, stage
        )

    def _on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ):
        if not self.interval.on_epoch:
            return
        epoch_predictions = self.get_epoch_predictions(stage)
        self.write_on_epoch_end(trainer, pl_module, epoch_predictions, stage)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._cleanup_stage_data("validate")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._cleanup_stage_data("test")

    def get_epoch_predictions(
        self, stage: str
    ) -> Union[List[Tensor], List[List[Tensor]]]:
        stage_storage = self._stage_storage[stage]
        dataloader_keys = sorted(stage_storage.keys())
        num_dataloaders = len(dataloader_keys)

        epoch_predictions: List[List[Tensor]] = []
        for dataloader_idx in dataloader_keys:
            epoch_predictions.append(stage_storage[dataloader_idx])

        return epoch_predictions[0] if num_dataloaders == 1 else epoch_predictions

    def _store_prediction_data(
        self,
        loop: _EvaluationLoop,
        trainer: "pl.Trainer",
        batch_idx: int,
        dataloader_idx: int,
        predictions: STEP_OUTPUT,
        stage: str,
    ):
        self._stage_storage[stage][dataloader_idx].append(
            move_data_to_device(predictions, torch.device("cpu"))
        )

    def _cleanup_stage_data(self, stage: str):
        del self._stage_storage[stage]


def _predictions_dict_factory():
    stage_storage = defaultdict(list)
    return stage_storage


__all__ = ["MultiStageBasePredictionWriter"]
