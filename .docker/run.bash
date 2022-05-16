#!/usr/bin/env bash

if [ ${#} -lt 1 ]; then
    echo "Usage: ${0} <docker image> <cmd (optional)>"
    echo "Example: ${0} andrejorsula/drl_grasping:latest ros2 run drl_grasping ex_evaluate_pretrained_agent.bash"
    exit 1
fi

IMG=${1}
CMD=${*:2}

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f ${XAUTH} ]; then
    xauth_list=$(xauth nlist "${DISPLAY}")
    # shellcheck disable=SC2001
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
GUI_ENVS=(
    --env XAUTHORITY="${XAUTH}"
    --env QT_X11_NO_MITSHM=1
    --env DISPLAY="${DISPLAY}"
    # --env DISPLAY=:1.0
    # --env MESA_GL_VERSION_OVERRIDE=3.3
)
GUI_VOLUMES=(
    --volume "${XAUTH}:${XAUTH}"
    --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    --volume "/dev/input:/dev/input"
)

VOLUMES="--volume ${PWD}/drl_grasping_training_docker:/root/drl_grasping_training"
if [[ -n ${DRL_GRASPING_PBR_TEXTURES_DIR} ]]; then
    VOLUMES="${VOLUMES} --volume ${DRL_GRASPING_PBR_TEXTURES_DIR}:/root/pbr_textures"
fi

ENVS=""
if [ -n "${RMW_IMPLEMENTATION}" ]; then
    ENVS="${ENVS} --env RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
fi
if [ -n "${CYCLONEDDS_URI}" ]; then
    ENVS="${ENVS} --env CYCLONEDDS_URI=${CYCLONEDDS_URI}"
    VOLUMES="${VOLUMES} --volume ${CYCLONEDDS_URI//file:\/\//}:${CYCLONEDDS_URI//file:\/\//}:ro"
fi
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]; then
    ENVS="${ENVS} --env FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"
    VOLUMES="${VOLUMES} --volume ${FASTRTPS_DEFAULT_PROFILES_FILE}:${FASTRTPS_DEFAULT_PROFILES_FILE}:ro"
fi
if [ -n "${ROS_DOMAIN_ID}" ]; then
    ENVS="${ENVS} --env ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
fi
if [ -n "${ROS_LOCALHOST_ONLY}" ]; then
    ENVS="${ENVS} --env ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
fi
if [ -n "${IGN_RELAY}" ]; then
    ENVS="${ENVS} --env IGN_RELAY=${IGN_RELAY}"
fi
if [ -n "${IGN_IP}" ]; then
    ENVS="${ENVS} --env IGN_IP=${IGN_IP}"
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
    "${DOCKER_OPTS}"
    --volume "/etc/localtime:/etc/localtime:ro"
    "${GUI_VOLUMES[@]}"
    "${GUI_ENVS[@]}"
    "${VOLUMES}"
    "${ENVS}"
    "${IMG}"
    "${CMD}"
)

echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
