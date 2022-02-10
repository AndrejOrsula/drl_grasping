#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HOST_SETUP_DIR="${SCRIPT_DIR}/_host_setup"

source "${HOST_SETUP_DIR}/setup.bash"

DOCKER_RUN_DRL_GRASPING_CMD=(
    "${SCRIPT_DIR}/run.bash"
    andrejorsula/drl_grasping
)

echo -e "\033[1;30m${DOCKER_RUN_DRL_GRASPING_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_RUN_DRL_GRASPING_CMD[*]}
