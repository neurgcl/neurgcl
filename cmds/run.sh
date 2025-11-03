#!/bin/bash

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")

source ${SHELL_FOLDER}/env_vars.sh

python src/qp_ik/demo/dyna_ball.py
