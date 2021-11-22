#!/usr/bin/env bash

if [ ${#} -lt 1 ]; then
    echo "Usage: ${0} <docker image> <cmd (optional)>"
    echo "Example: ${0} andrejorsula/drl_grasping:latest ros2 run drl_grasping ex_enjoy_pretrained_agent.bash"
    exit 1
fi

IMG=${1}
CMD=${@:2}

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f ${XAUTH} ]; then
    xauth_list=$(xauth nlist ${DISPLAY})
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f ${XAUTH} nmerge -
    else
        touch ${XAUTH}
    fi
    chmod a+r ${XAUTH}
fi

# Get the current version of docker-ce
# Strip leading stuff before the version number so it can be compared
DOCKER_OPTS=""
DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
if dpkg --compare-versions 19.03 gt "${DOCKER_VER}"; then
    echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
    if ! dpkg --list | grep nvidia-docker2; then
        echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
        exit 1
    fi
    DOCKER_OPTS="${DOCKER_OPTS} --runtime=nvidia"
else
    DOCKER_OPTS="${DOCKER_OPTS} --gpus all"
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f ${XAUTH} ]; then
    echo "[${XAUTH}] was not properly created. Exiting..."
    exit 1
fi

ADDITIONAL_VOLUMES=""
if [[ ! -z ${DRL_GRASPING_PBR_TEXTURES_DIR} ]]; then
    ADDITIONAL_VOLUMES="${ADDITIONAL_VOLUMES} -v ${DRL_GRASPING_PBR_TEXTURES_DIR}:/root/drl_grasping/pbr_textures"
fi

docker run -it \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${XAUTH} \
    -v "${XAUTH}:${XAUTH}" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev/input:/dev/input" \
    -v "${PWD}/training:/root/drl_grasping/training" \
    ${ADDITIONAL_VOLUMES} \
    --network host \
    --ipc host \
    --rm \
    -it \
    --privileged \
    --security-opt seccomp=unconfined \
    ${DOCKER_OPTS} \
    ${IMG} \
    ${CMD}

# Use this volume for custom config and models that are stored locally on your machine
# -v "${HOME}/.ignition:/root/.ignition"
