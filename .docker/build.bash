#!/usr/bin/env bash

TAG="andrejorsula/drl_grasping"

if [ "${#}" -gt "0" ]; then
    if [[ "${1}" != "-"* ]]; then
        TAG="${TAG}:${1}"
        BUILD_ARGS=${*:2}
    else
        BUILD_ARGS=${*:1}
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

DOCKER_BUILD_CMD=(
    docker build
    "${PROJECT_DIR}"
    --tag "${TAG}"
    "${BUILD_ARGS}"
)

echo -e "\033[1;30m${DOCKER_BUILD_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_BUILD_CMD[*]}
