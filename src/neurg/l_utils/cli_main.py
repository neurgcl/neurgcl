import os
import sys

import numpy as np
from l_utils.my_cli import MyLightningCLI
from lightning.pytorch.cli import ArgsType
from lightning.pytorch.trainer.states import TrainerStatus


def inDebug():
    if 'PYDEVD_LOAD_VALUES_ON_DEMAND' in os.environ:
        return True
    if sys.gettrace():
        return True
    return False


def cli_main(args: ArgsType = None):
    if inDebug():
        base_config = '--config=config/lsdf.yaml'
        if len(sys.argv) == 1:
            sys.argv += [base_config]
        sys.argv += ['--logger.name=debug', '--max_epochs=100']

    print(f"sys.argv: {sys.argv}")

    cli = MyLightningCLI()

    config_eval = cli.config["eval"]
    train_flag = not (config_eval is not None and config_eval != False)

    logger = cli.trainer.logger
    log_info = logger.log_info if hasattr(logger, "log_info") else print
    ckpt_path = cli.config["ckpt_path"]
    if ckpt_path:
        log_info(f"Load ckpt at: {ckpt_path}")

    if train_flag:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
        log_info(f"trainer.state.status: {cli.trainer.state.status}")

        res_test = cli.trainer.test(cli.model, ckpt_path="last", datamodule=cli.datamodule)
        log_info(f"trainer.state.status: {cli.trainer.state.status}")
        if cli.trainer.state.status == TrainerStatus.INTERRUPTED:
            exit(0)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    cli_main()
