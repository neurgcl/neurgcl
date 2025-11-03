import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import loggers as pl_loggers
from wandb.sdk.lib import runid

from ..utils.util_log import get_time_str_mdt

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class MyWbLogger(pl_loggers.WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            checkpoint_name,
            **kwargs,
        )
        # self.txt_logger = logger_config(__name__)
        # self.exp_logger = None
        self._wandb_init['id'] = self._wandb_init['id'] or (get_time_str_mdt() + '-' + str(runid.generate_id(3)))
        self._id = self._wandb_init.get("id")

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

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        return os.path.join(self._save_dir, self._project)

    @property
    def name(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        return self._name

    @property
    def log_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """

        d = os.path.join(self.save_dir, self.name, self.experiment.id)
        d = os.path.expandvars(d)
        d = os.path.expanduser(d)
        return d
