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
Max Epoch Stopping
^^^^^^^^^^^^^^

Monitor a metric and stop training when it stops improving.

"""
import logging
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message

log = logging.getLogger(__name__)


class MaxEpochStopping(Callback):
    r"""
    Stop training when the epochs of current experiment reach a limitation.

    Args:
        max_epochs:
        verbose: verbosity mode.
        mode:
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        log_rank_zero_only: When set ``True``, logs the status of the max epoch stopping callback only for rank 0 process.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"abs"`` or ``"rel"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import MaxEpochStopping
        >>> max_epoch_stopping = MaxEpochStopping()
        >>> trainer = Trainer(callbacks=[max_epoch_stopping])
    """

    mode_dict = {"abs": "", "rel": ""}

    def __init__(
        self,
        max_epochs: int = 100,
        verbose: bool = False,
        mode: str = "abs",
        log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.cur_exp_max_epoch = max_epochs
        self.verbose = verbose
        self.mode = mode

        self.stopped_epoch = 0
        self.cur_exp_epoch = 0  # current exp epoch count

        self.log_rank_zero_only = log_rank_zero_only

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

    @property
    def state_key(self) -> str:
        return self._generate_state_key(mode=self.mode)

    def state_dict(self) -> Dict[str, Any]:
        return {"stopped_epoch": self.stopped_epoch, "cur_exp_epoch": self.cur_exp_epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.cur_exp_epoch = state_dict["cur_exp_epoch"]
        if self.mode == "rel":
            self.cur_exp_max_epoch += self.cur_exp_epoch

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip_check(trainer):
            return
        self._run_max_epoch_stopping_check(trainer)

    def _run_max_epoch_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the max epoch stopping condition is met and if so tells the trainer to stop the training."""

        if trainer.fast_dev_run:  # short circuit if metric not present
            return

        should_stop, reason = self._evaluate_stopping_criteria(trainer)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria(self, trainer: "pl.Trainer") -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.cur_exp_epoch + 1 >= self.cur_exp_max_epoch:
            should_stop = True
            reason = (
                f"Stop at [cur exp epoch - cur total epoch]: {self.cur_exp_epoch} - {trainer.current_epoch}"
                "Signaling Trainer to stop."
            )
        self.cur_exp_epoch += 1

        return should_stop, reason

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
