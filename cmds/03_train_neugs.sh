#!/bin/bash
set -e

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")
PROJECT_ROOT=$(cd "$(dirname "${SHELL_FOLDER}")";pwd)

source ${SHELL_FOLDER}/env_vars.sh

cd ${PROJECT_ROOT}
bash logs/cmd_train_all.sh