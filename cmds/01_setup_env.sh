#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

conda env create -f env.yml -n neurg
conda activate neurg

# conda create -n neurg python=3.10.18 "setuptools<80" -c conda-forge
# conda activate neurg
# conda install -y trimesh=4.5 rtree 'numpy<2' scipy matplotlib==3.8.4 pandas opencv tabulate "setuptools<80" -c conda-forge
# conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install -r requirements.txt

# install cuRobo according to its install instruction: https://curobo.org/get_started/1_install_instructions.html
sudo apt -y install git-lfs
mkdir libs && cd libs
git clone https://github.com/NVlabs/curobo.git -b v0.7.7
cd curobo
pip install -e . --no-build-isolation
# This will take 20 minutes to install.
