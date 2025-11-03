import os

import numpy as np
import trimesh
from jsdf.l_jsdf import JData, JsdfDataset

from neurg.kinetics.cu_kin import get_fk_model
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.torch_tf import b_inv_mat44, b_pair_tf_pt
from neurg.my_utils.util_file import  load_yaml, write_pickle
from neurg.my_utils.util_log import get_time_str
from neurg.my_utils.sdf.util_sdf import cal_sdf_grad_via_trimesh
from neurg.utils.data import DatasetCfg

"""
Generate gradient dataset for the testset of JsdfDataset
"""

if __name__ == '__main__':
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

    out_dir = os.path.join(PATH_ROOT, "data/baselines/jsdf")

    ds_cfg = DatasetCfg.read_json(os.path.join(data_dir, "config.json"))
    mesh_dir = ds_cfg.mesh_dir
    link_meshs = ds_cfg.link_meshs
    link_names = ds_cfg.get_link_names()
    
    jdata = JData(device="cuda", data_dir=PATH_ROOT + f'/data/baselines/jsdf/data_mesh_roscol.mat')
    jdata.y = jdata.y * 0.01
    ds_test = JsdfDataset(jdata, 'test')

    # convert JSDF data to local-link frames' points
    kin_model = get_fk_model(ee_link='panda_hand')
    np_dists = ds_test.y.cpu().numpy()
    q = ds_test.x[:, :7]    # (47025,7)
    pts = ds_test.x[:, 7:]  # (47025,3)

    l_T_w2links = kin_model.fk_link_T44(q).contiguous() # (n_sample, L, 4, 4)
    l_T_links2w = b_inv_mat44(l_T_w2links)  
    pts_local = b_pair_tf_pt(l_T_links2w, pts)   # (n_sample, L, 3)
    np_pts = pts_local.cpu().numpy()

    N, L, _ = np_pts.shape

    l_grad = []
    for i in range(L):
        mesh = trimesh.load_mesh(mesh_dir + "/" + link_meshs[i])
        pts = np_pts[:, i].astype(np.float64)
        grads = cal_sdf_grad_via_trimesh(mesh, pts, thresh_surface)
        l_grad.append(grads)
    l_grad = np.stack(l_grad, axis=1)

    d_grad = {'x': ds_test.x.cpu(), 'y': ds_test.y.cpu(), 'grad': l_grad}

    os.makedirs(out_dir, exist_ok=True)
    data_path = out_dir + '/d_test_grad.pkl'
    write_pickle(data_path, d_grad)
