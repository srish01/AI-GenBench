from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from abc import ABC, abstractmethod
import lightning as L
import torch
from torch.nn import Module
from torch import Tensor
from PIL.Image import Image
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryRecall,
    BinaryPrecision,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryF1Score,
    BinarySpecificity,
)
from lightning.pytorch.cli import OptimizerCallable

from algorithms.model_factory_registry import ModelFactoryRegistry
from training_metrics.metrics_manager import SUPPORTED_METRICS, GeneratorKey, StageT
from training_metrics import (
    BalancedBinaryAccuracy,
    MetricsManager,
    MetricsManagerSlidingWindow,
)

from torch.nn import Identity

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


class AbstractBaseDeepfakeDetectionModel(L.LightningModule, ABC):
    """
    An abstract base class for deepfake detection models.

    This class provides a basic structure for deepfake detection models
    that can be used to train and evaluate deepfake detection models.

    You can also use the baseline implementation, BaseDeepfakeDetectionModel,
    as a starting point to implement your own model.
    """

    def __init__(
        self,
        model_name: str,
        optimizer: OptimizerCallable,
        scheduler: str,
        model_input_size: Union[int, Tuple[int, int]],
        classification_threshold: float = 0.5,
        base_weights: Optional[Path] = None,
        logging_initial_step: Optional[int] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if model_input_size is not None:
            if isinstance(model_input_size, int):
                model_input_size = (model_input_size, model_input_size)
            elif len(model_input_size) != 2:
                raise ValueError(
                    "model_input_size must be none, an int, or a tuple of 2 ints."
                )

        self.model_name: str = model_name
        self.classification_threshold: float = classification_threshold
        self.optimizer_factory: OptimizerCallable = optimizer
        self.scheduler_name: str = scheduler
        self.model_input_size: Tuple[int, int] = model_input_size

        # Note: Checkpoint weights are different from base weights.
        # Checkpoint weights will be loaded during setup(), if any are defined.
        self.base_weights: Optional[Path] = base_weights
        self.metric_managers: List[MetricsManager] = []
        self.logging_initial_step: int = (
            logging_initial_step if logging_initial_step is not None else 0
        )
        self.model_args: Dict[str, Any] = model_args if model_args is not None else {}

        self.model: Module = Identity()

    def load_previous_checkpoint(self, checkpoint_path: Path):
        print("Loading previous experiment checkpoint from", str(checkpoint_path))
        try:
            self.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["state_dict"]
            )
        except Exception as e:
            print("Error loading checkpoint", e)
            print(
                "The checkpoint may contain the weights of self.model instead of self. Trying to load them directly."
            )
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def setup(self, stage: str):
        self.make_model()
        if self.base_weights is not None:
            self.load_previous_checkpoint(self.base_weights)

        self.adapt_model_for_training()

        found_metric_manager = False
        for callback in self.trainer.callbacks:
            if isinstance(callback, MetricsManager):
                found_metric_manager = True
                self.metric_managers.append(callback)

        if not found_metric_manager:
            raise RuntimeError("MetricsManager callback(s) not found!")

        self.register_metrics()

    def make_model(self) -> Module:
        """
        Create the model instance and stores it in self.model.

        This method is called during setup().
        This method can be overridden by subclasses to customize the model creation.
        """
        arguments = {
            "model_name": self.model_name,
            "pretrained": True,
            "num_classes": 1,  # Default to binary classification (1 output neuron)
        }

        # Merge model arguments
        arguments.update(self.model_args)

        model, factory_name = ModelFactoryRegistry().make_model(**arguments)

        if self.trainer.is_global_zero:
            print(
                f"Model {arguments['model_name']} created using factory {factory_name}"
            )

        self.model = model
        return model

    def register_metrics(self):
        """
        Register the metrics to be used during training and evaluation.

        This method is called during setup().

        This method can be overridden by subclasses to register the
        appropriate metrics for the model.

        It is recommened to call this method (that is, in subclasses,
        as "super().register_metrics()") to ensure that the default metrics
        are registered as well.

        Losses are not managed here: they are logged directly in the
        training/validation/test_step methods.
        """
        self._managers_set_metrics(
            stage="fit",  # Training only
            generator_id="*",  # All examples
            metrics=BalancedBinaryAccuracy(),  # A single metric or a MetricCollection
            epoch_only=False,  # Log with both step and epoch granularity
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics(
            stage=["test", "validate"],  # During evaluation only
            generator_id=None,  # A different plot for each generator
            metrics=BinaryRecall(),
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics(
            stage=["test", "validate"],  # During evaluation only
            generator_id="*",  # All examples
            metrics=BalancedBinaryAccuracy(),
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            # Applies to all metric managers
        )

        self._managers_set_metrics(
            stage=None,  # All stages
            generator_id="*",  # All examples
            metrics=[
                BinaryRecall(),  # a.k.a. Sensitivity or True Positive Rate
                BinaryPrecision(),
                BinarySpecificity(),  # a.k.a. True Negative Rate
                BinaryAveragePrecision(),
                BinaryAUROC(),
                BinaryF1Score(),
            ],
            epoch_only=True,  # Log only at epoch, not step
            # Applies to all metric managers
        )

    def _managers_set_metrics(
        self,
        *,
        stage: Optional[Union[StageT, List[StageT]]],
        generator_id: Optional[Union[GeneratorKey, List[GeneratorKey]]],
        metrics: SUPPORTED_METRICS,
        metrics_group: str = "default",
        epoch_only: bool = False,
        sliding_window_only: bool = False,
        non_sliding_window_only: bool = False,
        check_stage: bool = True,
    ):
        """
        Utility to set the metrics in the appropriate managers.
        Usually called in the register_metrics() method.
        It is recommended to not customize this method.

        Apart from check_stage, which should be left to True, you should think carefully
        about the values of all the other parameters to obtain the desired behavior.

        Consider checking the `register_metrics()` method for an example of how to use this method.

        Args:
            stage: The stage(s) for which the metrics are set. Can be a single stage, a list of stages,
                or None to apply themetrics to all stages.
            generator_id: The generator ID(s) for which the metrics are set. Can be a single ID, a list of IDs,
                an '*' to register metrics that compute a single plot for all examples, or None to compute the metrics
                separatedly for each generator.
            metrics: The metric(s) to set. Can be a single metric, a list of metrics, or a MetricCollection.
            metrics_group: The group name for the metrics.
            epoch_only: If True, the metrics will be logged only at epoch level, not at step level.
            sliding_window_only: If True, the given metrics will be applied only to sliding-window metric managers
                (the ones that compute metrics for each sliding window separately).
            non_sliding_window_only: If True, only non-sliding window metric managers will be used.
            check_stage: If True, the stage will be checked before setting the metrics.
        """
        assert not (sliding_window_only and non_sliding_window_only)
        for manager in self.metric_managers:
            if sliding_window_only and not isinstance(
                manager, MetricsManagerSlidingWindow
            ):
                continue

            if non_sliding_window_only and isinstance(
                manager, MetricsManagerSlidingWindow
            ):
                continue

            manager.set_metrics(
                stage=stage,
                generator_id=generator_id,
                metrics=metrics,
                metrics_group=metrics_group,
                epoch_only=epoch_only,
                check_stage=check_stage,
            )

    def adapt_model_for_training(self):
        """
        Adapts the model for training.

        This method is called:
        - during the setup of the model, after the model
            has been created and the base weights have been loaded, or
        - after the validation/test, before returning to training.

        Defaults to setting the model in train mode and setting
        requires_grad to True for all the model parameters.

        Can be overridden by subclasses to implement custom logic.
        """
        self.train()
        self.requires_grad_(True)

    def adapt_model_for_evaluation(
        self, stage_name: Literal["validation", "test", "predict"]
    ):
        """
        Adapts the model for evaluation.

        This method is called before starting the validation/test/predict stage.
        The name of the stage being started is passed as an argument.

        Defaults to setting the model in eval mode (note: setting requires_grad is
        usually not needed as Lightning runs the evaluation step using torch.no_grad/inference_mode)

        Can be overridden by subclasses to implement custom logic.
        """
        self.eval()

    @abstractmethod
    def train_augmentation(
        self,
    ) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
        """
        A function that returns the augmentation for the training set.

        Important note: it should return 1 or 2 Callable values: the first callable is the deterministic part of the augmentation,
        the second (optional) one is the non-deterministic part of the augmentation (random rotation, crop, etcetera).

        If you only need purely-deterministic augmentations, only return one callable.
        """
        ...

    @abstractmethod
    def val_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the validation set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_val_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        ...

    @abstractmethod
    def test_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the test set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_test_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        ...

    @abstractmethod
    def predict_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the prediction set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_predict_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        ...

    @abstractmethod
    def make_val_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple views/crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the validation_step() method.
        As a reference, consider that BaseDeepfakeDetectionModel class implements the fusion mechanism,
        so you can use that implementation if you want.

        Each returned crop will be augmented with the second callable returned by the the val_augmentation() method.

        Hint: this function can also be used to implement a plain resize.
        """
        ...

    @abstractmethod
    def make_test_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the test_step() method.
        As a reference, consider that BaseDeepfakeDetectionModel class implements the fusion mechanism,
        so you can use that implementation if you want.

        Each returned crop will be augmented with the second callable returned by the the test_augmentation() method.

        Hint: this function can also be used to implement a plain resize.
        """
        ...

    @abstractmethod
    def make_predict_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the predict_step() method.
        As a reference, consider that BaseDeepfakeDetectionModel class implements the fusion mechanism,
        so you can use that implementation if you want.

        Each returned crop will be augmented with the second callable returned by the the predict_augmentation() method.

        Hint: this function can also be used to implement a plain resize.
        """
        ...

    def to(self, *args, **kwargs):
        # Preamble for rookies: in Pytorch, the to() method is NOT applied
        # recursively to submodules :). Instead, Pytorch runs an absurd
        # "apply this function" to all modules in the hierarchy.
        #
        # This is a problem when working with submodules with custom to() methods
        # (for instance, used to ensure that a frozen, hidden, backbone is moved
        # to the correct device).
        #
        # A custom to() method such as this is needed to ensure that the to()
        # method of self.model is called. This, of course, only covers the
        # custom to() method of self.model. Custom to() methods of other
        # submodules in the hierarchy are not covered.

        super().to(*args, **kwargs)
        # This means that self.model will "changed" by to() both when
        # super.to is called, and when self.model.to is called.
        # This is usually not a problem...
        self.model.to(*args, **kwargs)
        return self

    def _update_metrics(self, predictions, labels, generator_ids, identifiers):
        """
        Update the metrics in the appropriate managers.
        """
        for manager in self.metric_managers:
            manager.update(predictions, labels, generator_ids, identifiers)

    # ---------- OVERRIDDEN LIGHTNING CALLBACKS ----------
    def on_test_model_train(self) -> None:
        """
        This is a Lightning callback which is called to set the model in
        training mode after being used for the "test" stage.

        We here redirect this call to adapt_model_for_training().
        """
        self.adapt_model_for_training()

    def on_validation_model_train(self) -> None:
        """
        This is a Lightning callback which is called to set the model in
        training mode after being used for the "validation" stage.

        We here redirect this call to adapt_model_for_training().
        """
        self.adapt_model_for_training()

    def on_validation_model_eval(self) -> None:
        """
        This is a Lightning callback which is called to set the model in
        evaluation mode before being used for the "validation" stage.

        We here redirect this call to adapt_model_for_evaluation().
        """
        self.adapt_model_for_evaluation("validation")

    def on_test_model_eval(self) -> None:
        """
        This is a Lightning callback which is called to set the model in
        evaluation mode before being used for the "test" stage.

        We here redirect this call to adapt_model_for_evaluation().
        """
        self.adapt_model_for_evaluation("test")

    def on_predict_model_eval(self) -> None:
        """
        This is a Lightning callback which is called to set the model in
        evaluation mode before being used for the "predict" stage.

        We here redirect this call to adapt_model_for_evaluation().
        """
        self.adapt_model_for_evaluation("predict")

    def log(self, *args, **kwargs):
        if self.trainer._results is None or "step" not in self.trainer._results:
            super().log(
                name="step",
                value=self.global_step + self.logging_initial_step,
                sync_dist=False,
            )
        return super().log(*args, **kwargs)

    def log_window_id(self, stage: str):
        datamodule: "DeepfakeDetectionDatamodule" = self.trainer.datamodule
        current_window: int = datamodule.sliding_windows_definition.current_window
        if not isinstance(current_window, int):
            pass

        self.log(f"{stage}_window", current_window, on_epoch=True, sync_dist=False)

    def on_train_epoch_end(self) -> None:
        self.log_window_id("fit")

    def on_validation_epoch_end(self) -> None:
        self.log_window_id("validate")

    def on_test_epoch_end(self) -> None:
        self.log_window_id("test")

    # def on_predict_epoch_end(self) -> None:
    #     self.log_window_id('predict')


__all__ = ["AbstractBaseDeepfakeDetectionModel"]
