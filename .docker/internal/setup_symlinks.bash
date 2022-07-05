#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"

## Setup symbolic link for entrypoint
ln -s "${SCRIPT_DIR}/entrypoint.bash" "${HOME}/entrypoint.bash"

## Setup symbolic link for bash aliases
ln -s "${SCRIPT_DIR}/.bash_aliases" "${HOME}/.bash_aliases"

## Setup symbolic links to important directories and files of `drl_grasping`
ln -s "${WS_SRC_DIR}/drl_grasping/hyperparams" "${HOME}/hyperparams"
ln -s "${WS_SRC_DIR}/drl_grasping/examples" "${HOME}/examples"
ln -s "${WS_SRC_DIR}/drl_grasping/scripts" "${HOME}/scripts"
ln -s "${WS_SRC_DIR}/drl_grasping/launch" "${HOME}/launch"
ln -s "${WS_SRC_DIR}/drl_grasping/drl_grasping/envs/__init__.py" "${HOME}/envs.py"
