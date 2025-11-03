import os

import trimesh

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import detect_walk_files

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=PATH_ROOT + '/data/urdf/bullet/franka_panda/meshes/collision_ros_stl')
    parser.add_argument('--out_dir', type=str, default=PATH_ROOT+"/data/out/meshes/collision_ros_obj")
    parser.add_argument('--surfix', type=str, default="obj")
    args = parser.parse_args()

    in_dir = args.in_dir
    # surfix = 'obj'
    # surfix = 'stl'
    surfix = args.surfix
    out_dir = args.out_dir or (in_dir + f'_{surfix}_trimesh')

    os.makedirs(out_dir, exist_ok=True)
    # files = detect_walk_files(in_dir, '.dae ')
    files = detect_walk_files(in_dir, '(obj|stl|dae)')
    for f in files:
        mesh = trimesh.load(f)
        fname, ext = os.path.splitext(f)
        _, basename = os.path.split(fname)
        if surfix == 'obj':
            mesh.export(f'{out_dir}/{basename}.{surfix}', include_texture=False)
        else:
            mesh.export(f'{out_dir}/{basename}.{surfix}')
