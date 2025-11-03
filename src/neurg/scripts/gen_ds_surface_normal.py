import os
import numpy as np
from tqdm import tqdm
import trimesh

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import detect_walk_files, write_pickle, write_yaml
from neurg.my_utils.util_log import get_time_str


def sample_patch_surface_points(mesh, num_samples):
    points, fids = trimesh.sample.sample_surface_even(mesh, num_samples)
    normals = mesh.face_normals[fids]
    return points, normals


"""
Generate gradient dataset for the points in each link mesh surface
"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, default=PATH_ROOT / 'logs/nsdf/dataset/meshes')
    parser.add_argument('--n_sample', type=int, default=1024)
    args = parser.parse_args()
    
    version = get_time_str()
    mesh_dir = args.mesh_dir
    n_sample = args.n_sample
    out_dir = os.path.join(mesh_dir, "../surface_normal")

    link_files = detect_walk_files(mesh_dir, '(obj|stl)')
    link_fname = [os.path.split(f)[-1] for f in link_files]

    d_surface = {}

    confs = {
        'mesh_dir': str(os.path.relpath(mesh_dir, PATH_ROOT)),
        'link_fname': link_fname,
    }
    
    os.makedirs(out_dir, exist_ok=True)
    write_yaml(out_dir + '/config.yaml', confs)

    for i in tqdm(range(0, len(link_files))):
        link_file = link_files[i]
        link_name = os.path.splitext(os.path.split(link_file)[-1])[0]
        mesh = trimesh.load(link_file)

        pts, grads = sample_patch_surface_points(mesh, n_sample)
        pts_grads = {'pts': np.array(pts), 'normals': np.array(grads)}
        d_surface[link_name] = pts_grads

    data_path = out_dir + f'/{n_sample}.pkl'
    write_pickle(data_path, d_surface)

