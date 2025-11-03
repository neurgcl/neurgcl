import os
import shutil

from neurg.kinetics.cu_kin import (
    get_fk_model_from_robot_yaml_file,
    update_get_col_mesh_paths_from_robot_model,
)
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import write_json


def extract_col_mesh_paths_from_robot_yaml(robot_yaml):
    if not os.path.isabs(robot_yaml):
        robot_yaml = os.path.join(PATH_ROOT, robot_yaml)
    kin = get_fk_model_from_robot_yaml_file(robot_yaml)
    col_mesh_paths = update_get_col_mesh_paths_from_robot_model(kin)
    return col_mesh_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_yaml",
        type=str,
        default=PATH_ROOT + "/data/curobo_cfg/franka_nofinger_tcp.yml",
    )
    parser.add_argument(
        "--out_dir", type=str, default=PATH_ROOT + "/logs/dataset/meshes"
    )
    args = parser.parse_args()

    col_mesh_paths = extract_col_mesh_paths_from_robot_yaml(args.robot_yaml)

    # save to out_dir --------------------
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    link_meshs = []
    for k, src_path in col_mesh_paths.items():
        dst_path = os.path.join(args.out_dir, os.path.basename(src_path))
        shutil.copyfile(src_path, dst_path)
        link_meshs.append(os.path.basename(dst_path))

    ds_cfg = dict(mesh_dir=args.out_dir, link_meshs=link_meshs)
    write_json(os.path.join(args.out_dir, "config.json"), ds_cfg)
