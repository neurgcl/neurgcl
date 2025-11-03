#!/bin/bash

SHELL_FOLDER=$(dirname "$(realpath "${BASH_SOURCE:-$0}")")
PROJECT_ROOT=$(cd "$(dirname "${SHELL_FOLDER}")";pwd)

# if IN_MY_ENV exist and IN_MY_ENV == $SHELL_FOLDER, then return
if [ ! -z $IN_MY_ENV ] && [ $IN_MY_ENV = $SHELL_FOLDER ]; then
    return
fi
export IN_MY_ENV=${PROJECT_ROOT}

export PYTHONPATH=$PROJECT_ROOT/src/baselines:$PROJECT_ROOT/src:$PYTHONPATH
export DEV_WS_DIR=$PROJECT_ROOT
