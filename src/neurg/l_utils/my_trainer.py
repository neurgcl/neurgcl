"""Trainer to automate the training."""

from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from .loggers import MyWbLogger


class Trainer(pl.Trainer):
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. note:: You must call this on all processes. Failing to do so will cause your program to stall forever.

         .. code-block:: python

             def training_step(self, batch, batch_idx):
                 img = ...
                 save_img(img, self.trainer.log_dir)

        """
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], (TensorBoardLogger, CSVLogger, MyWbLogger)):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath
