import os

import numpy as np
import trimesh
from tqdm import tqdm

from neurg.net.l_model_linksdf import LinkSdfDataset
from neurg.my_utils.util_file import load_yaml, write_pickle, write_yaml
from neurg.my_utils.util_log import get_time_str
from neurg.my_utils.sdf.util_sdf import cal_sdf_grad_via_trimesh
from neurg.utils.data import DatasetCfg

"""
Generate gradient dataset for the testset of LinkSdfDataset
"""

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--thresh_surface", default=1e-5)
    args = parser.parse_args()
    version = get_time_str()

    data_dir = args.data_dir
    thresh_surface = args.thresh_surface
    # ----------------------------

    if data_dir is None:
        param = load_yaml("config/lsdf_fast_train.yaml")
        data_dir = param["data"]["init_args"]["data_dir"]

    ds_cfg = DatasetCfg.read_json(os.path.join(data_dir, "config.json"))
    mesh_dir = ds_cfg.mesh_dir
    link_meshs = ds_cfg.link_meshs
    link_names = ds_cfg.get_link_names()

    out_dir = os.path.join(mesh_dir, "../full_grad")

    d_grad = {}
    for i, link_name in enumerate(tqdm(link_names)):
        ds = LinkSdfDataset(
            data_dir=data_dir, link_name=link_name, stage="test", normalize=False
        )
        pts = ds.data_all[:, :3].astype(np.float64)
        dists = ds.data_all[:, 3].astype(np.float64)
        mesh = trimesh.load_mesh(f"{mesh_dir}/{link_meshs[i]}")

        grads = cal_sdf_grad_via_trimesh(mesh, pts, thresh_surface)
        pts_normals = {
            "x": np.array(pts).astype(np.float32),
            "y": np.array(dists).astype(np.float32),
            "grad": np.array(grads).astype(np.float32),
        }
        d_grad[link_name] = pts_normals

    ds_normal_cfg = ds_cfg.to_dict()
    ds_normal_cfg["data_dir"] = str(data_dir)

    out_cfg_path = out_dir + "/config.yaml"
    out_data_path = out_dir + "/d_grad.pkl"

    os.makedirs(out_dir, exist_ok=True)
    write_yaml(out_cfg_path, ds_normal_cfg)
    write_pickle(out_data_path, d_grad)
