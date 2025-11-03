#!/bin/bash
set -e

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")
PROJECT_ROOT=$(cd "$(dirname "${SHELL_FOLDER}")";pwd)

source ${SHELL_FOLDER}/env_vars.sh

# Download JSDF dataset
gdown --fuzzy https://drive.google.com/file/d/1mPoNOGheYUPHUwbMYdmvsy1ilU-bLs3b/view?usp=sharing -O data/baselines/jsdf/data_mesh_roscol.mat
