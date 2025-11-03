import os
from pathlib import Path

import lightning.pytorch.cli as pl_cli
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger


class SaveConfigCallback(pl_cli.SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
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
        logger = trainer.logger

        if isinstance(logger, WandbLogger):
            import wandb

            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                )

                metadata = {"original_path": Path(config_path), "original_filename": Path(config_path).name}
                base_name = logger.name.replace("/", "_")
                name = f"config-{base_name}-{logger.version}"
                artifact = wandb.Artifact(name=name, type="pl_config", metadata=metadata)
                artifact.add_file(config_path, name="config.yaml")
                logger.experiment.log_artifact(artifact)
        else:
            super().save_config(trainer, pl_module, stage)
