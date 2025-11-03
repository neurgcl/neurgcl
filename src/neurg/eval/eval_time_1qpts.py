import time
from copy import deepcopy
from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

from neurg.l_utils.torch.torch_helper import try_cuda
from neurg.my_utils.util_log import get_time_str
from neurg.my_utils.util_sweep import sweep_dict_to_list
from neurg.my_utils.util_time_measure import bench_torfunc_args_to_dic
from qp_ik.dist_fun.qp_net import get_qp_net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="nsdf")
    # parser.add_argument('--method', type=str, default='neurg')
    # parser.add_argument('--method', type=str, default='jsdf')
    # parser.add_argument('--method', type=str, default='rdf')
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--fwd', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--grad', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    if args.method == "fcl":
        args.grad = False

    l_record_args = ['method', 'precision']

    cfg = {
        'time_str': get_time_str(),
        'device': str(try_cuda()),
        'dis_type': '1qpt',  # point to each link distances
    }
    for k in l_record_args:
        if hasattr(args, k):
            cfg[k] = getattr(args, k)

    warmup = 1000
    repeat = 10000
    
    sweep = {
        'gpu_type': [f'{torch.cuda.get_device_name(0)}'],
        'device': ['cuda'],
        'batch_size': [1, 10, 100, 1000, 10000, 50000, 100000],
    }

    if args.batch_size is not None:
        sweep['batch_size'] = [args.batch_size]

    l_param = sweep_dict_to_list(sweep)

    l_log_data = []
    with torch.no_grad():
        for param in tqdm(l_param):
            if args.method in ["nsdf"]:
                net_param = {}
            else:
                net_param = {}

            device = param['device']
            batch_sz = param['batch_size']

            rnet = get_qp_net(cfg['method'], device=device, **net_param)
            cfg['method_w_param'] = rnet.rnet.get_method_w_param()

            q = (torch.rand((7)).to(device) - 0.5) * 2 * (torch.pi * 2)
            pts = (torch.rand((batch_sz, 3)).to(device) - 0.5) * 2 * (torch.pi * 2)

            bench_func = partial(
                bench_torfunc_args_to_dic, warmup=warmup, repeat=repeat, device=device, max_time=60 * 3
            )

            if args.fwd:
                ret = bench_func(rnet.cal_Gamma_all, (q, pts))
                item = deepcopy(param)
                item.update(ret)
                item.update(cfg)
                item['func_type'] = 'fwd'
                l_log_data.append(item)
                time.sleep(1)

            if args.grad:
                ret = bench_func(rnet.cal_Gamma_g_all, (q, pts))
                item = deepcopy(param)
                item.update(ret)
                item.update(cfg)
                item['func_type'] = 'fwd_g'
                l_log_data.append(item)
                time.sleep(1)

    df = pd.DataFrame(l_log_data)
    show_cols = ["method_w_param", "func_type", "batch_size", "dt_mean", "dt_std"]
    val_cols = ["dt_mean", "dt_std"]
    df[val_cols] = (df[val_cols]/1000).round(3)  # unit: ms

    print("-"*40)
    print("unit: ms")
    print(df[show_cols])
    print("done!")
