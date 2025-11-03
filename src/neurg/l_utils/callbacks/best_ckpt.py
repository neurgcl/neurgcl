import os
import shutil
from typing import Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torch import Tensor


class BestCkpt(ModelCheckpoint):
    def _update_best_and_save(
        self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        ModelCheckpoint._update_best_and_save(self, current, trainer, monitor_candidates)
        self.to_yaml()
        filepath = self.format_checkpoint_name(monitor_candidates, "best")

        if not hasattr(self, 'previous_best_model_path') or self.previous_best_model_path != self.best_model_path:
            shutil.copyfile(self.best_model_path, filepath)
            self.previous_best_model_path = self.best_model_path

    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
            if self._last_checkpoint_saved != filepath:
                rel_path = os.path.relpath(self._last_checkpoint_saved, os.path.dirname(filepath))
                self._link_checkpoint(trainer, rel_path, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)
