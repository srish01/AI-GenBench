import os
import warnings
from lightning.pytorch.cli import SaveConfigCallback
from lightning import Trainer, LightningModule
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.loggers import WandbLogger


class SaveConfigCallbackSlidingWindows(SaveConfigCallback):
    """
    Callback to save the configuration of a sliding window training.

    This is a custom child class of `SaveConfigCallback` that is specifically designed
    to save the configuration in the experiment data folder instead of the logging directory.
    """

    def __init__(
        self,
        *args,
        config_save_dir: str,
        overwrite: bool = True,
        save_config_to_wandb: bool = True,
        config_filename: str = "experiment_config.yaml",
        **kwargs,
    ):
        super().__init__(
            *args, overwrite=overwrite, config_filename=config_filename, **kwargs
        )
        self.config_save_dir = config_save_dir
        self.save_config_to_wandb = save_config_to_wandb

    def make_config_file_path(self) -> str:
        return os.path.join(self.config_save_dir, self.config_filename)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Implement to save the config in some other place additional to the standard log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.

        """

        config_path = self.make_config_file_path()
        fs = get_filesystem(self.config_save_dir)

        if not self.overwrite:
            file_exists = fs.isfile(config_path)
            if file_exists:
                warnings.warn(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. "
                    "The configuration file will not be overwritten."
                )
                return

        fs.makedirs(self.config_save_dir, exist_ok=True)
        self.parser.save(
            self.config,
            config_path,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )

    def save_to_wandb(self, trainer: Trainer) -> bool:
        if not trainer.is_global_zero:
            return False

        if not self.save_config_to_wandb:
            return False

        fs = get_filesystem(self.config_save_dir)
        config_path = self.make_config_file_path()
        parent_dir = os.path.dirname(config_path)

        if not fs.isfile(config_path):
            return False

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.save(
                    config_path,
                    parent_dir,
                    policy="now",
                )

        return True

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self.save_to_wandb(trainer)

    def on_validate_start(self, trainer: Trainer, pl_module: LightningModule):
        self.save_to_wandb(trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self.save_to_wandb(trainer)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule):
        self.save_to_wandb(trainer)


__all__ = ["SaveConfigCallbackSlidingWindows"]
