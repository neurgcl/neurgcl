from functools import partial
from typing import Any, Callable, List, Optional, Tuple, overload

import numpy as np
import torch
from lightning.fabric.utilities.apply_func import apply_to_collection, move_data_to_device
from torch import Tensor
from torch._prims_common import DeviceLikeType

_dtype = torch.dtype


def _from_numpy(value: np.ndarray, device: DeviceLikeType) -> Tensor:
    return torch.from_numpy(value).to(device)


CONVERSION_DTYPES: List[Tuple[Any, Callable[[Any, Any], Tensor]]] = [
    # bool -> uint8 as bool -> torch.bool triggers RuntimeError: Unsupported data type for NCCL process group
    (bool, partial(torch.tensor, dtype=torch.uint8)),
    (int, partial(torch.tensor, dtype=torch.int)),
    (float, partial(torch.tensor, dtype=torch.float)),
    (np.ndarray, _from_numpy),
]


def to_tensor(
    data: Any,
    dtype: Optional[_dtype] = None,
    device: Optional[DeviceLikeType] = None,
) -> Any:
    # convert non-tensors
    if dtype is None:
        for src_dtype, conversion_func in CONVERSION_DTYPES:
            data = apply_to_collection(data, src_dtype, conversion_func, device=device)
    else:
        for src_dtype in [bool, int, float]:
            data = apply_to_collection(data, src_dtype, partial(torch.tensor, dtype=dtype, device=device))
        for src_dtype in [np.ndarray]:
            data = apply_to_collection(data, src_dtype, lambda x: torch.from_numpy(x).to(dtype=dtype, device=device))

    if device is not None:
        return move_data_to_device(data, device)
    return data


def to_device(
    data: Any,
    device: DeviceLikeType,
) -> Any:
    return move_data_to_device(data, device)
