import time

import numpy as np
import torch

from .csv_logger import CSVLogger


def test_eq(ret1, ret2, atol=3e-7):
    for i in range(len(ret1)):
        if isinstance(ret1, torch.Tensor):
            assert torch.allclose(ret1[i], ret2[i], atol=atol)


def bench_torfunc_args_to_dic(
    fn,
    args,
    prefix=None,
    clone=False,
    warmup=100,
    repeat=100,
    device="cpu",
    csv_logger: CSVLogger = None,
    arg_dict=None,
    max_time=None,
    name=None,
):
    # unit: us
    fw_latencies = []
    timeout = None
    if device != "cpu":
        for _ in range(warmup):
            torch.cuda.synchronize()
            ref = fn(*args)
            torch.cuda.synchronize()

        for _ in range(repeat):
            torch.cuda.synchronize()
            fw_begin = time.perf_counter()
            ref = fn(*args)
            torch.cuda.synchronize()
            fw_end = time.perf_counter()
            fw_latencies.append(fw_end - fw_begin)
    else:
        for _ in range(warmup):
            ref = fn(*args)

        if max_time is not None:
            timeout = time.perf_counter() + max_time
        cnt_repeat = 0
        for _ in range(repeat):
            fw_begin = time.perf_counter()
            if timeout is not None and fw_begin > timeout:
                break
            ref = fn(*args)
            fw_end = time.perf_counter()
            fw_latencies.append(fw_end - fw_begin)
    fw_latencies = np.array(fw_latencies)
    avg_fw_latency = fw_latencies.mean() * 10**6
    avg_fw_latency_std = fw_latencies.std() * 10**6

    if hasattr(fn, "__name__"):
        fn_name = str(fn.__name__)
    elif hasattr(fn, "name"):
        fn_name = str(fn.name)
    else:
        fn_name = "unknown"
    if hasattr(fn, "_torchdynamo_orig_callable"):
        compile = 1
    else:
        compile = 0

    if isinstance(fn, torch.jit.ScriptFunction):
        script = 1
    else:
        script = 0

    if prefix is None:
        prefix = ""

    if prefix == "":
        fn_name_full = fn_name
    else:
        fn_name_full = prefix + "_" + fn_name

    dic = {
        "prefix": prefix,
        "fn_name": fn_name,
        "fn_name_full": fn_name_full,
        "device": str(device),
        "warmup": warmup,
        "repeat": len(fw_latencies),
        "dt_mean": fw_latencies.mean() * 10**6,
        "dt_std": fw_latencies.std() * 10**6,
        "dt_max": fw_latencies.max() * 10**6,
        "dt_min": fw_latencies.min() * 10**6,
        "dt_50th": np.median(fw_latencies) * 10**6,
        "compile": compile,
        "script": script,
    }
    if name is not None:
        dic["name"] = name
    if arg_dict is not None:
        dic.update(arg_dict)

    if csv_logger:
        csv_logger.log_dict(dic)
    print(
        name or fn_name_full,
        ",\tdt= " + f"{avg_fw_latency:.3f}" + f",\t[std: {avg_fw_latency_std:.3f}, 50%: {dic['dt_50th']:.3f}]" + " us",
    )
    return dic
