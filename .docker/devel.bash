#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

DOCKER_HOME="/root"
DOCKER_WS_DIR="${DOCKER_HOME}/ws"
DOCKER_WS_SRC_DIR="${DOCKER_WS_DIR}/src"
DOCKER_TARGET_SRC_DIR="${DOCKER_WS_SRC_DIR}/$(basename "${PROJECT_DIR}")"

echo -e "\033[2;37mDevelopment volume: ${PROJECT_DIR} -> ${DOCKER_TARGET_SRC_DIR}\033[0m" | xargs

exec "${SCRIPT_DIR}/run.bash" -v "${PROJECT_DIR}:${DOCKER_TARGET_SRC_DIR}" "${@}"
