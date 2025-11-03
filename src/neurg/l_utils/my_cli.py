from typing import Any, Callable, Optional, Type, Union

from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI


from .callbacks import MaxEpochStopping
from .callbacks.save_config import SaveConfigCallback
from .gpu_sel import get_free_gpu
from .my_trainer import Trainer


class MyLightningCLI(LightningCLI):
    """Implementation of a configurable command line tool for pytorch-lightning."""

    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            parser_kwargs={"parser_mode": "omegaconf"},
            save_config_callback=SaveConfigCallback,
            save_config_kwargs={"save_to_log_dir": False},
            trainer_class=Trainer,
            run=False,
        )
        default_kwargs.update(kwargs)
        # self.run = default_kwargs.get("run", True)
        super().__init__(*args, **default_kwargs)

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

        import torch

        default_n_threads = torch.get_num_threads()
        torch.set_num_threads(min(8, default_n_threads))

        if self.config['trainer']['accelerator'] == 'gpu':
            devices = self.config['trainer']['devices'] or 1
            if isinstance(devices, int):
                free_gpus = get_free_gpu(devices)
                self.config['trainer']['devices'] = free_gpus

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.
        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}

        # tb_logger = MyTBLogger(**self.config["logger"])
        # if trainer_config["logger"] is None or trainer_config["logger"] is True:
        #     trainer_config["logger"] = tb_logger
        # self.txt_logger = tb_logger.txt_logger
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def add_arguments_to_parser(self, parser):
        from .utils.util_log import get_time_str

        parser.add_argument("--logger.save_dir", default="lightning_logs")
        parser.add_argument("--logger.version", default="v")
        parser.add_argument("--logger.time_start", default=f"{get_time_str()}")
        parser.add_argument("--logger.name", default="exp")
        parser.link_arguments("logger.name", "trainer.logger.init_args.name")

        parser.add_argument("--eval", default=False)
        parser.add_argument("--ckpt_path", default=None)

        parser.add_lightning_class_args(MaxEpochStopping, "my_max_epoch_stopping")
        parser.add_argument("--max_epochs", type=int, default=1000)
        parser.link_arguments("max_epochs", "my_max_epoch_stopping.max_epochs")

    def before_fit(self):
        pass
        # send_email(address=self.config["notification_email"], message="trainer.fit starting")

    def after_fit(self):
        pass
        # send_email(address=self.config["notification_email"], message="trainer.fit finished")


def cli_fit_test(
    model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
    datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
    test=True,
):
    cli = MyLightningCLI(model_class, datamodule_class)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    if test:
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
