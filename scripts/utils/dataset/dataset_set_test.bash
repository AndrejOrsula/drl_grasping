#!/usr/bin/env bash

COLLECTION_OWNER="googleresearch"
OWNER_DIR="${HOME}/.ignition/fuel/fuel.ignitionrobotics.org/${COLLECTION_OWNER}"
MODELS_TEST_DIR="models_test"

if [[ -d "${OWNER_DIR}/${MODELS_TEST_DIR}" ]]; then
    ln -sTf "${OWNER_DIR}/${MODELS_TEST_DIR}" "${OWNER_DIR}/models" && \
        echo "Info: All models under '${OWNER_DIR}/${MODELS_TEST_DIR}' are now symbolically linked to '${OWNER_DIR}/models'."
else
    echo >&2 "Error: Directory '${OWNER_DIR}/${MODELS_TEST_DIR}' does not exist. Setting of training dataset skipped. Please make sure to download the dataset first."
fi
