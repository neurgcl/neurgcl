import os
import re

import pandas as pd
from my_utils.util_apitable import dst_bulk_update_or_create, get_dst

from bl_sim_env.utils import np_tf
import numpy as np
from bl_sim_env.utils.util_file import load_pickle, detect_walk_files
from qp_ik.cfg import DEV_WS_DIR
from qp_ik.utils.robot_kin import kin_model

info_path = DEV_WS_DIR +  '/logs/qpik/dyna_ball_01172341/250118_000150_jsdf_traj_info.pkl'

def extract_path_length(d_info):
    if isinstance(d_info, str):
        d_info = load_pickle(d_info)
    elif isinstance(d_info, dict):
        pass
    else:
        raise TypeError("d_info must be a string or dict")

    l_q = d_info['q']
    _, T_ees = kin_model.np_fk_links_ee(l_q)
    l_pos=T_ees[:,:3,3]
    delta_ee_pos = np.diff(l_pos, axis=0)
    ee_move = np.linalg.norm(delta_ee_pos, axis=1).sum()
    delta_q = np.diff(l_q, axis=0)
    joint_move = np.deg2rad(np.abs(delta_q).sum())
    return ee_move, joint_move

def get_version_key(info_path):
    version_key=None
    folder, filename = os.path.split(info_path)
    # '250117_234140_jsdf_traj_info.pkl' - > '250117_234140_jsdf
    try:
        version_key = re.match("(.*)_traj_info\.pkl", filename).groups()[0]
    except:
        pass
    return version_key

def extract_move_distance_in_folder(dir_path:str, dst_name:str):
    l_files = detect_walk_files(dir_path, re_pattern=".*" + "traj_info\.pkl$")
    dst = get_dst(dst_name)
    l_d_out = []
    # info_path = l_files[0]
    for info_path in l_files:
        version_key = get_version_key(info_path)
        d_info = load_pickle(info_path)
        ee_move, joint_move = extract_path_length(d_info)
        d_out = {
            'exp_ver_w_method': version_key,
            'ee_move': ee_move,
            'joint_move': joint_move
        }
        l_d_out.append(d_out)

    df = pd.DataFrame(l_d_out)
    df.set_index("exp_ver_w_method", inplace=True)
    dst_bulk_update_or_create(dst, df)
    return l_files

if __name__ == '__main__':
    # extract_move_distance_in_folder(DEV_WS_DIR +  '/logs/qpik/dyna_ball_01172341', 'eval_qp_dyna_ball')
    extract_move_distance_in_folder(DEV_WS_DIR +  '/logs/qpik/shelf_01161400', 'eval_shelf')
    # from jsonargparse import CLI
    # CLI(extract_move_distance_in_folder)