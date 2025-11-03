import argparse
from tqdm import tqdm
import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
import skimage
import pyrender
import torch

from my_utils.config import PATH_ROOT
from my_utils.util_file import detect_walk_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', default=PATH_ROOT + "/data/raw/ros_mesh/collision", type=str)
    parser.add_argument('--out_dir', default=PATH_ROOT + "/data/baselines/rdf/data/raw_coll", type=str)
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    out_dir = args.out_dir

    dir_all = os.path.join(out_dir, 'all')
    dir_train = os.path.join(out_dir, 'train')
    dir_test = os.path.join(out_dir, 'test')
    dir_val = os.path.join(out_dir, 'val')

    mesh_files = detect_walk_files(mesh_dir, '.stl')

    for mf in tqdm(mesh_files):
        mesh_name = mf.split('/')[-1].split('.')[0]
        mesh = trimesh.load(mf)

        center = mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

        n_sample = 500000
        # sample points near surface (as same as deepSDF)
        near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            number_of_points=n_sample,
            surface_point_method='scan',
            sign_method='normal',
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=100,
            min_size=0.015,
            return_gradients=False,
        )
        # # sample points randomly within the bounding box [-1,1]
        random_points = np.random.rand(n_sample, 3) * 2.0 - 1.0
        random_sdf = mesh_to_sdf.mesh_to_sdf(
            mesh,
            random_points,
            surface_point_method='scan',
            sign_method='normal',
            bounding_radius=None,
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=100,
        )

        near_points = near_points * scale + center
        random_points = random_points * scale + center
        near_sdf = near_sdf * scale
        random_sdf = random_sdf * scale

        # save data
        data = {
            'near_points': near_points,
            'near_sdf': near_sdf,
            'random_points': random_points,
            'random_sdf': random_sdf,
            'center': center,
            'scale': scale,
        }
        if not os.path.exists(dir_all):
            os.makedirs(dir_all)
        np.save(os.path.join(dir_all, f'voxel_128_{mesh_name}.npy'), data)

        train_ratio = 0.9
        test_ratio = 0.05
        val_ratio = 1 - train_ratio - test_ratio

        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(n_sample, generator=generator).tolist()
        idx_map = {
            'train': indices[: int(n_sample * train_ratio)],
            'test': indices[int(n_sample * train_ratio) : int(n_sample * (train_ratio + test_ratio))],
            'val': indices[int(n_sample * (train_ratio + test_ratio)) :],
        }

        for k in idx_map:
            dir_k = os.path.join(out_dir, f'{k}')
            if not os.path.exists(dir_k):
                os.makedirs(dir_k)
            data = {
                'near_points': near_points[idx_map[k]],
                'near_sdf': near_sdf[idx_map[k]],
                'random_points': random_points[idx_map[k]],
                'random_sdf': random_sdf[idx_map[k]],
                'center': center,
                'scale': scale,
            }
            np.save(os.path.join(dir_k, f'voxel_128_{mesh_name}.npy'), data)
