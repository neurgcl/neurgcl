# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Info Logging
^^^^^^^^^^^^^^

Monitor a metric and stop training when it stops improving.

"""
import sys
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.trainer.states import RunningStage

from ..utils.util_log import logger_config, sort_dict


class InfoLogging(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._logger_info = None
        self._logger_df = None

        self._start_time: Dict[RunningStage, Optional[float]] = {stage: None for stage in RunningStage}
        self._end_time: Dict[RunningStage, Optional[float]] = {stage: None for stage in RunningStage}
        self._offset = 0

    def log_info(self, *args, **kwargs):
        if self._logger_info is not None:
            self._logger_info(*args, **kwargs)

    def log_df(self, *args, **kwargs):
        if self._logger_df is not None:
            self._logger_df(*args, **kwargs)

    # def log_info(self, msg, on_exp=True, on_ver=True, stacklevel=2, plainout=False):
    #     find_logger = False
    #     if self.txt_logger is not None and on_ver:
    #         find_logger = True
    #         self.txt_logger.info(msg, stacklevel=stacklevel, plainout=plainout)
    #     if self.exp_logger is not None and on_exp:
    #         find_logger = True
    #         self.exp_logger.info(msg, stacklevel=stacklevel, plainout=plainout)
    #     if not find_logger:
    #         print(msg)
    #     self.last_df = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.logger = getattr(trainer, "logger", None)
        if self.logger is not None:
            self._logger_info: print = getattr(self.logger, "log_info", None)
            if self._logger_info is None:
                self._logger_loginfo = logger_config(__name__)
                self._logger_info = partial(self._logger_loginfo.info, stacklevel=2)
            self._logger_df = getattr(self.logger, "log_df", None)

        # fmt: off
        try:
            if stage == "fit":
                ds = trainer.fit_loop._data_source.dataloader().dataset
            elif stage == "validate":
                ds = trainer.validate_loop._data_source.dataloader().dataset
            elif stage == "test":
                ds = trainer.test_loop._data_source.dataloader().dataset
            elif stage == "predict":
                ds = trainer.predict_loop._data_source.dataloader().dataset
            self.log_info(f"[setup {stage}] len(dataset): {len(ds)}")
        except:
            pass
        # fmt: on

    def start_time(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        """Return the start time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._start_time[stage]

    def end_time(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        """Return the end time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._end_time[stage]

    def time_elapsed(self, stage: str = RunningStage.TRAINING) -> float:
        """Return the time elapsed for a particular stage (in seconds)"""
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offset if stage == RunningStage.TRAINING else 0
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    def on_fit_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        pass
        # self.log_info(sys._getframe().f_code.co_name)

    def on_fit_end(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        pass
        # self.log_info(sys._getframe().f_code.co_name)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_info(sys._getframe().f_code.co_name)

        self._start_time[RunningStage.TRAINING] = time.monotonic()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        stage = RunningStage.TRAINING
        self._end_time[stage] = time.monotonic()
        duration = self.time_elapsed(stage)
        self.log_info(f"{sys._getframe().f_code.co_name}, duration: {duration} s")

        info = {f"{stage}/duration": duration}
        metrics = sort_dict(deepcopy(trainer.logged_metrics))
        info.update(metrics)

        last_info = {
            "best_model_score": trainer.checkpoint_callback.best_model_score,
            "best_model_path": trainer.checkpoint_callback.best_model_path,
        }
        info.update(last_info)
        self.log_df(info, rowshow=False)
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.log_metrics({"duration/train": duration})

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
        # self._start_time[RunningStage.VALIDATING] = time.monotonic()
        # self.log_info(sys._getframe().f_code.co_name)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
        # self._end_time[RunningStage.VALIDATING] = time.monotonic()
        # self.log_info(sys._getframe().f_code.co_name)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._start_time[RunningStage.TESTING] = time.monotonic()
        self.log_info(f"{sys._getframe().f_code.co_name}")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        stage = RunningStage.TESTING
        self._end_time[stage] = time.monotonic()
        self.log_info(f"{sys._getframe().f_code.co_name}, duration: {self.time_elapsed(stage)} s")

        info = {f"{stage}/duration": self.time_elapsed(stage)}
        metrics = sort_dict(deepcopy(trainer.logged_metrics))
        info.update(metrics)
        self.log_df(info, rowshow=False)

    # def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     self.log_info(f"{time.time()} on_train_epoch_start")

    # def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     self.log_info(f"{time.time()} on_train_epoch_end")

    # def on_train_batch_start(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    # ) -> None:
    #     self.log_info(f"{time.time()} on_train_batch_start")

    # def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int) -> None:
    #     self.log_info(f"{time.time()} on_train_batch_end")

    # def on_validation_batch_start(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    # ) -> None:
    #     self.log_info(f"{time.time()} on_validation_batch_start")

    # def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int) -> None:
    #     self.log_info(f"{time.time()} on_validation_batch_end")

    # def on_advance_end(self) -> None:
    #     """Hook to be called each time after :attr:`advance` is called."""
    #     self.log_info(f"{time.time()} on_advance_end")
    # def on_save_checkpoint(
    #         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    #     ) -> None:
    #     self.log_info("on_save_checkpoint")
