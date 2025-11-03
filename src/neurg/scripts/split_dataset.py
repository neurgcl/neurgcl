import argparse
import json
import os

import _pickle as pickle
import numpy as np
import trimesh

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import write_json, write_pickle
from neurg.my_utils.util_log import get_time_str

"""
shuffle and split raw data into train/test/val
add normal for pts on the mesh surface
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default=PATH_ROOT / 'logs/dataset/raw')
    parser.add_argument('--out_dir', type=str, default=PATH_ROOT / 'logs/dataset/splited')
    args = parser.parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    
    config = json.loads(open(raw_dir + '/config.json', 'r').read())
    link_meshs = config['link_meshs']
    config['version'] = get_time_str()

    write_json(out_dir + '/config.json', config)

    for link_mesh in link_meshs:
        link_name = os.path.splitext(link_mesh)[0]
        link_dir = raw_dir + '/' + link_name
        with open(f"{link_dir}/{link_name}.pkl", 'rb') as f:
            dic_data = pickle.load(f)

        for k in dic_data.keys():
            if k[:3] == 'pts':
                dic_data[k] = dic_data[k].reshape(-1, 4)

        # ['pts_inside', 'pts_outside', 'pts_on_sur', 'pts_near_sur', 'pts_bbox_pad_fixed', 'pts_bbox_largest', 'pts_on_sur_idx']

        # add normal
        dic_data['idx_pts_on_sur'] = dic_data['pts_on_sur_idx'].reshape(-1)
        dic_data.pop('pts_on_sur_idx')

        mesh = trimesh.load(config['mesh_dir'] + '/' + link_mesh)
        fids = dic_data['idx_pts_on_sur']
        dic_data['normal_pts_on_sur'] = mesh.face_normals[fids]

        for k in dic_data.keys():
            if k.startswith('pts'):
                print(f"{k} shape: {dic_data[k].shape}")

        for k in dic_data.keys():
            if k in ['pts_inside', 'pts_outside', 'pts_near_sur', 'pts_bbox_pad_fixed', 'pts_bbox_largest']:
                d_len = len(dic_data[k])
                idx_random = np.random.permutation(d_len)
                dic_data[k] = dic_data[k][idx_random]
            elif k == 'pts_on_sur':
                d_len = len(dic_data[k])
                idx_random = np.random.permutation(d_len)
                dic_data[k] = dic_data[k][idx_random]

                dic_data['normal_pts_on_sur'] = dic_data['normal_pts_on_sur'][idx_random]
                dic_data['idx_pts_on_sur'] = dic_data['idx_pts_on_sur'][idx_random]

        train_ratio = 0.9
        test_ratio = 0.05
        val_ratio = 1 - train_ratio - test_ratio

        d_train = {}
        d_test = {}
        d_val = {}

        for k in dic_data.keys():
            if k[:3] == 'pts' or k[:3] == 'idx' or k[:5] == 'normal':
                len_train = int(len(dic_data[k]) * train_ratio)
                len_test = int(len(dic_data[k]) * test_ratio)
                # len_val = len(dic_data[k]) - len_train - len_test
                d_train[k] = dic_data[k][:len_train]
                d_test[k] = dic_data[k][len_train : len_train + len_test]
                d_val[k] = dic_data[k][len_train + len_test :]
            else:
                d_train[k] = dic_data[k]
                d_test[k] = dic_data[k]
                d_val[k] = dic_data[k]

        o_dir = out_dir + '/train'
        os.makedirs(o_dir, exist_ok=True)
        write_pickle(o_dir + '/' + link_name + '.pkl', d_train)

        o_dir = out_dir + '/test'
        os.makedirs(o_dir, exist_ok=True)
        write_pickle(o_dir + '/' + link_name + '.pkl', d_test)

        o_dir = out_dir + '/val'
        os.makedirs(o_dir, exist_ok=True)
        write_pickle(o_dir + '/' + link_name + '.pkl', d_val)
