from functools import partial
import pandas as pd
import torch

from neurg.api.nsdf_model import NSDFLModel, NSDFModel
from neurg.cfg import ROBOT_CKPT_PATH
from neurg.eval.eval_helper import analysis_link_sdf_grad_err
from neurg.l_utils.torch.to_tensor import to_tensor
from neurg.l_utils.torch.torch_helper import try_cuda
from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.l_utils.torch.xy_dataset import batch_predict
from neurg.my_utils.util_file import load_pickle
from neurg.my_utils.util_log import get_time_str

"""
Evalulate the SDF gradient error for points on the link surface for each NeuG

  link_name            MGE
0     link0  0.075 ± 0.112
1     link1  0.074 ± 0.088
2     link2  0.071 ± 0.085
3     link3  0.083 ± 0.096
4     link4  0.085 ± 0.095
5     link5  0.072 ± 0.081
6     link6  0.096 ± 0.099
7     link7  0.129 ± 0.122
8      hand  0.129 ± 0.117
9       all  0.090 ± 0.103
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_grad", type=bool, default=True)
    parser.add_argument("--data_file", type=str, default=f'logs/nsdf/dataset/surface_normal/100000.pkl')
    parser.add_argument("--robot_ckpt", type=str, default=ROBOT_CKPT_PATH)
    args = parser.parse_args()

    cfg = {
        'time_str': get_time_str(),
        'device': str(try_cuda()),
        'dis_type': 'pt-link',
        'test_ds': 'ours_col',
        'batch_size': 100000,
        'scale': 1000,
    }
    cfg.update(vars(args))

    device = cfg["device"]
    batch_size = cfg["batch_size"]
    scale = cfg["scale"]

    l_link_name = [f"link{i}" for i in range(8)] + ["hand"]

    repeat = 10000
    neurg = NSDFModel(robot_ckpt_path=args.robot_ckpt)

    d_normal = load_pickle(args.data_file)
    d_normal = to_tensor(d_normal, dtype=torch.float32)

    l_y_gts = []
    l_y_preds = []
    l_log_data = []
    with torch.no_grad():
        for link_name in l_link_name + ['all']:
            if link_name != 'all':
                lnet = neurg.get_link_net(link_name)

                pts = d_normal[link_name]['pts']
                grads_gt = d_normal[link_name]['normals']

                dl = SliceBatchLoader(pts, batch_size=batch_size)
                sdfs_pred, grads_pred = batch_predict(partial(lnet.l_dist, ret_grad=True), dl, device)

                y_gts = torch.cat([torch.zeros([grads_gt.shape[0], 1]), grads_gt], dim=1)
                y_preds = torch.cat([sdfs_pred, grads_pred], dim=1)

                l_y_gts.append(y_gts)
                l_y_preds.append(y_preds)
            else:
                y_gts = torch.cat(l_y_gts, dim=0)
                y_preds = torch.cat(l_y_preds, dim=0)

            infos = analysis_link_sdf_grad_err(y_preds, y_gts, scale=cfg['scale'], partition=False)
            logdata = dict(
                link_name=link_name,
                **cfg,
            )
            logdata.update(infos)
            l_log_data.append(logdata)

        df = pd.DataFrame(l_log_data)
        val_cols = ["Grad/mean", "Grad/std"]
        df[val_cols] = df[val_cols].round(3)
        df["MGE"] = df.apply(lambda row: f"{row['Grad/mean']:.3f} ± {row['Grad/std']:.3f}", axis=1)
        show_cols = ["link_name", "MGE"]
        df_show = df[show_cols]
        print(df_show)
