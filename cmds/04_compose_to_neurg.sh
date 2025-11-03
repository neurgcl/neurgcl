#!/bin/bash
set -e

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")
PROJECT_ROOT=$(cd "$(dirname "${SHELL_FOLDER}")";pwd)

source ${SHELL_FOLDER}/env_vars.sh

cd ${PROJECT_ROOT}
python src/neurg/scripts/compose_to_neurg.py --model_name=nsdf