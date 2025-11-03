import pandas as pd
import torch

from neurg.api.nsdf_model import NSDFLModel, NSDFModel
from neurg.cfg import DATA_DIR, ROBOT_CKPT_PATH
from neurg.eval.eval_helper import analysis_link_error
from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.l_utils.torch.xy_dataset import xy_predict
from neurg.my_utils.util_log import get_time_str
from neurg.net.l_model_linksdf import LinkSdfDataset

"""
Evalulate the SDF error for each NeuG on our testing dataset

  link_name          MDE(Unit: mm)
0     link0  0.41 ± 0.37
1     link1  0.39 ± 0.36
2     link2  0.35 ± 0.33
3     link3  0.37 ± 0.33
4     link4  0.36 ± 0.33
5     link5  0.39 ± 0.34
6     link6  0.42 ± 0.37
7     link7  0.43 ± 0.35
8      hand  0.52 ± 0.43
9       all  0.40 ± 0.36
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--robot_ckpt", type=str, default=ROBOT_CKPT_PATH)
    args = parser.parse_args()

    data_dir = args.data_dir
    robot_ckpt_path = args.robot_ckpt

    time_str = get_time_str()

    cfg = {"device": "cuda", "batch_size": 100000, "scale": 1000}


    l_link_name = [f"link{i}" for i in range(8)] + ["hand"]

    repeat = 10000
    neurg = NSDFModel(robot_ckpt_path=robot_ckpt_path)

    l_y_gts = []
    l_y_preds = []
    l_log_data = []

    torch.set_grad_enabled(False)
    for link_name in l_link_name + ["all"]:
        if link_name != "all":
            neug = neurg.get_link_net(link_name)

            ds_test = LinkSdfDataset(
                data_dir=data_dir,
                link_name=link_name,
                stage="test",
                normalize=False,
            )
            dl = SliceBatchLoader(ds_test, batch_size=cfg["batch_size"])
            _, y_gts, y_preds = xy_predict(neug.l_dist, dl, cfg["device"])
            l_y_gts.append(y_gts)
            l_y_preds.append(y_preds)
        else:
            y_gts = torch.cat(l_y_gts, dim=0)
            y_preds = torch.cat(l_y_preds, dim=0)

        infos = analysis_link_error(
            y_preds,
            y_gts,
            link_name=link_name,
            time_str=time_str,
            partition=False,
            scale=cfg["scale"]
        )
        l_log_data.append(infos)

    df = pd.DataFrame(l_log_data)
    val_cols = ["l1/mean", "l1/std"]
    df[val_cols] = df[val_cols].round(2)
    df["MDE"] = df.apply(
        lambda row: f"{row['l1/mean']:.2f} ± {row['l1/std']:.2f}", axis=1
    )
    show_cols = ["link_name", "MDE"]
    df_show = df[show_cols]
    print(df_show)
