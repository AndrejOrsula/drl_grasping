#!/usr/bin/env bash
# This script setups git hooks for this repository.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

pip install --user pre-commit &&
cd "${REPO_DIR}" &&
pre-commit install &&
cd - || exit
