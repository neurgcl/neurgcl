from copy import deepcopy
import itertools
import time
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from neurg.my_utils.util_sweep import sweep_dict_to_list
from neurg.my_utils.util_time_measure import bench_torfunc_args_to_dic
from neurg.utils.model_helper import get_robot_net


# config = yaml.load(open(exp_dir + '/config.yaml', 'r'), Loader=yaml.FullLoader)
# hid_dim = config['model']['init_args']['hid_dim']
# hid_size = config['model']['init_args']['hid_size']
# actv_type = config['model']['init_args']['actv_type']
# batch_sz = 1
# device = torch.device('cuda')
# x = torch.rand((batch_sz, 3), dtype=torch.float32, device=device)
# model = model.to(device)
# y = model(x)


if __name__ == '__main__':
    # sweep = {
    #     'link_name': ['link0'],
    #     'hid_dim': [32, 64, 128],
    #     'hid_size': [2, 3, 4],
    #     'actv_type': ['relu', 'softplus'],
    #     'batch_size': [1, 10, 100, 1000, 10000, 100000],
    #     'device': ['cpu', 'cuda'],
    # }

    sweep = {
        'link_name': ['link0'],
        'hid_dim': [32],
        'hid_size': [3],
        'actv_type': ['softplus'],
        'batch_size': [1, 10, 100, 1000, 10000, 100000],
        'device': ['cuda'],
    }
    l_param = sweep_dict_to_list(sweep)


    repeat = 10000

    with torch.no_grad():
        l_log_data = []
        for param in tqdm(l_param):
            device = param['device']
            batch_size = param['batch_size']

            link_name = param['link_name']
            net_param = {
                "hid_dim": param['hid_dim'],
                "hid_sz": param['hid_size'],
                "actv_type": param['actv_type'],
            }

            rnet = get_robot_net("nsdf", device=device, **net_param)
            lnet = rnet.get_link_net(link_name)

            x = torch.rand((batch_size, 3), device=device)
            lnet = lnet.to(device=device)
            ret = bench_torfunc_args_to_dic(lnet.l_dist, (x,), repeat=repeat, device=device)
            item = deepcopy(param)
            n_param_one_mlp = sum([_param.nelement() for _param in lnet.parameters()])
            item["param_sz"] = n_param_one_mlp
            item["method_w_param"] = rnet.get_method_w_param()
            item.update(ret)
            l_log_data.append(item)

            df = pd.DataFrame(l_log_data)
            show_cols = ["method_w_param", "param_sz","batch_size", "device", "dt_mean", "dt_std"]
            val_cols = ["dt_mean", "dt_std"]
            df[val_cols] = (df[val_cols]/1000).round(3) # unit: ms
            df = df[show_cols]
            df = df.rename(columns={'dt_mean': 'time_mean(ms)', 'dt_std': 'time_std(ms)'})
            print(df)
