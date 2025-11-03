#!/bin/bash
set -ex

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")
PROJECT_ROOT=$(cd "$(dirname "${SHELL_FOLDER}")";pwd)
source ${SHELL_FOLDER}/env_vars.sh

cd ${PROJECT_ROOT}

dataset_dir=logs/nsdf/dataset

# 1. Extract meshes from URDF file
python src/neurg/scripts/extract_mesh_from_urdf.py --robot_yaml=data/curobo_cfg/franka_nofinger_tcp.yml --out_dir=${dataset_dir}/meshes

# 2. Generate point clouds from meshes
python src/neurg/scripts/gen_dataset.py --mesh_dir=${dataset_dir}/meshes --out_dir=${dataset_dir}/raw

# 3. Split dataset into train/val/test sets
python src/neurg/scripts/split_dataset.py --raw_dir=${dataset_dir}/raw --out_dir=${dataset_dir}/splited

# 4. generate training commands for all links
python src/neurg/scripts/gen_train_cmds.py --data_dir=${dataset_dir}/splited --out_file=logs/cmd_train_all.sh --log_name=nsdf

# 5. Generate surface normals for all meshes, used for self-collision and evaluation purposes
python src/neurg/scripts/gen_ds_surface_normal.py --mesh_dir=${dataset_dir}/meshes --n_sample=1024
python src/neurg/scripts/gen_ds_surface_normal.py --mesh_dir=${dataset_dir}/meshes --n_sample=100000
