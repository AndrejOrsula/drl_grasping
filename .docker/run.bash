#!/usr/bin/env bash

TAG="andrejorsula/drl_grasping"

if [ "${#}" -gt "0" ]; then
    if [[ $(docker images --format "{{.Tag}}" "${TAG}") =~ (^|[[:space:]])${1}($|[[:space:]]) || $(wget -q https://registry.hub.docker.com/v2/repositories/${TAG}/tags -O - | grep -Poe '(?<=(\"name\":\")).*?(?=\")') =~ (^|[[:space:]])${1}($|[[:space:]]) ]]; then
        # Use the first argument as a tag is such tag exists either locally or on the remote registry
        TAG="${TAG}:${1}"
        CMD=${*:2}
    else
        CMD=${*:1}
    fi
fi

## GPU
# Enable GPU either via NVIDIA Container Toolkit or NVIDIA Docker (depending on Docker version)
if dpkg --compare-versions "$(docker version --format '{{.Server.Version}}')" gt "19.3"; then
    GPU_OPT="--gpus all"
else
    GPU_OPT="--runtime nvidia"
fi

## GUI
# To enable GUI, make sure processes in the container can connect to the x server
XAUTH=/tmp/.docker.xauth
if [ ! -f ${XAUTH} ]; then
    touch ${XAUTH}
    chmod a+r ${XAUTH}

    XAUTH_LIST=$(xauth nlist "${DISPLAY}")
    if [ -n "${XAUTH_LIST}" ]; then
        # shellcheck disable=SC2001
        XAUTH_LIST=$(sed -e 's/^..../ffff/' <<<"${XAUTH_LIST}")
        echo "${XAUTH_LIST}" | xauth -f ${XAUTH} nmerge -
    fi
fi
# GUI-enabling volumes
GUI_VOLUMES=(
    "${XAUTH}:${XAUTH}"
    "/tmp/.X11-unix:/tmp/.X11-unix"
    "/dev/input:/dev/input"
)
# GUI-enabling environment variables
GUI_ENVS=(
    XAUTHORITY="${XAUTH}"
    QT_X11_NO_MITSHM=1
    DISPLAY="${DISPLAY}"
)

## Additional volumes
CUSTOM_VOLUMES=()
# Synchronize timezone with host
CUSTOM_VOLUMES+=("/etc/localtime:/etc/localtime:ro")
# Persistent storage of logs
CUSTOM_VOLUMES+=("${PWD}/drl_grasping_training_docker:/root/drl_grasping_training")

## Additional environment variables
CUSTOM_ENVS=()
# Synchronize ROS_DOMAIN_ID with host
if [ -n "${ROS_DOMAIN_ID}" ]; then
    CUSTOM_ENVS+=("ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
# Synchronize IGN_PARTITION with host
if [ -n "${IGN_PARTITION}" ]; then
    CUSTOM_ENVS+=("IGN_PARTITION=${IGN_PARTITION}")
fi

DOCKER_RUN_CMD=(
    docker run
    --interactive
    --tty
    --rm
    --network host
    --ipc host
    --privileged
    --security-opt "seccomp=unconfined"
    "${GUI_VOLUMES[@]/#/"--volume "}"
    "${GUI_ENVS[@]/#/"--env "}"
    "${GPU_OPT}"
    "${CUSTOM_VOLUMES[@]/#/"--volume "}"
    "${CUSTOM_ENVS[@]/#/"--env "}"
    "${TAG}"
    "${CMD}"
)

echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
