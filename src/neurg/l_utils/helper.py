from typing import List

import torch
from lightning.pytorch.loops import _EvaluationLoop
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT
from torch import Tensor

from .utils.util_log import clear_irc_color


def format_pl_results(results: List[_OUT_DICT], stage: str) -> None:
    from contextlib import redirect_stdout
    from io import StringIO

    with redirect_stdout(StringIO()) as out:
        _EvaluationLoop._print_results(results, stage)

    out.seek(0)
    ret = out.read()
    out.close()

    ret = clear_irc_color(ret)
    return ret


def concat_batchs(batchs):
    batch0 = batchs[0]
    if isinstance(batch0, tuple):
        return [torch.concat(list(map(lambda x: x[i], batchs))) for i in range(len(batch0))]
    return torch.concat(batchs)


def dict2float(dic):
    d1 = {}
    for k, val in dic.items():
        if isinstance(val, Tensor):
            d1[k] = val.item() if val.numel() == 1 else val.tolist()
        elif isinstance(val, dict):
            d1[k] = dict2float(val)
        else:
            d1[k] = val
    return d1
