import os

import numpy as np
import torch
from tqdm import tqdm
import trimesh

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import detect_walk_files, load_pickle, write_yaml
from neurg.my_utils.util_log import get_time_str
from neurg.my_utils.util_sweep import sweep_dict_to_list
from neurg.net.l_model_linksdf import LinkSdfLModel

def simplify_dict_val_type(d):
    d1 = {}
    for k, v in d.items():
        val = v
        if isinstance(v, torch.Tensor):
            val = v.tolist()
        elif isinstance(v, str):
            val = v
        elif isinstance(v, dict):
            val = simplify_dict_val_type(v)
        if isinstance(val, (int, float, list, type(None), str, dict)):
            d1[k] = val
    return d1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='nsdf')
    args = parser.parse_args()
    model_name = args.model_name
    
    cpkt_root = PATH_ROOT + f'/lightning_logs/{model_name}'
    dataset_dir = PATH_ROOT + f'/logs/{model_name}/dataset'
    out_dir = PATH_ROOT + f'/logs/{model_name}/link_infos'

    sweep = {
        'actv_type': ['softplus'],
        'hid_dim': [32],
        'hid_sz': [3],
    }
    l_param = sweep_dict_to_list(sweep)

    for param in tqdm(l_param):
        actv_type = param['actv_type']
        hid_dim = param['hid_dim']
        hid_sz = param['hid_sz']

        ckpt_dir = cpkt_root + f'/{actv_type}/coll/{hid_dim}x{hid_sz}/'
        l_file = detect_walk_files(ckpt_dir, re_pattern='.*best.ckpt$')

        data_dir = os.path.join(dataset_dir, "splited/train")
        mesh_dir = os.path.join(dataset_dir, 'meshes')

        version = get_time_str()
        d_link_info = {}
        for f in l_file:
            suffix = f[len(ckpt_dir) :]
            link_name = suffix.split('/')[0]
            if os.path.exists(f"{mesh_dir}/{link_name}.obj"):
                link_mesh = trimesh.load(f"{mesh_dir}/{link_name}.obj")
            else:
                link_mesh = trimesh.load(f"{mesh_dir}/{link_name}.stl")

            rel_path = os.path.relpath(f, start=PATH_ROOT)
            data = load_pickle(data_dir + f"/{link_name}.pkl")
            bbox = data['bbox_largest']
            bbox_extents = bbox[1] - bbox[0]
            bbox_center = bbox[0] + bbox_extents / 2.0

            assert np.allclose(bbox_center, link_mesh.bounding_box.centroid)
            bbox_raw = link_mesh.bounding_box.bounds
            vertices = link_mesh.vertices - bbox_center
            bbox_radius = np.max(np.linalg.norm(vertices, axis=1))
            bbox_radius_padded = max(bbox_radius * 1.1, bbox_radius + 0.1)

            ckpt = torch.load(f, map_location='cpu')
            lmodel = LinkSdfLModel.load_from_checkpoint(f, map_location='cpu')
            model = lmodel.model

            d = {
                'bbox_largest': torch.tensor(bbox),
                'bbox_extents': torch.tensor(bbox_extents),
                'bbox_raw': torch.tensor(bbox_raw),
                'bbox_center': torch.tensor(bbox_center),
                'bbox_radius': torch.tensor(bbox_radius),
                'bbox_radius_padded': torch.tensor(bbox_radius_padded),
                'ckpt_path': rel_path,
                'model': model,
                'version': version,
            }
            d_link_info[link_name] = d

        out_file = out_dir + f'/{actv_type}_coll_{hid_dim}x{hid_sz}.pth'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save(d_link_info, out_file)
        print(f"save to {out_file}")

        yaml_d = simplify_dict_val_type(d_link_info)
        write_yaml(out_file.replace('.pth', '.yaml'), yaml_d)
