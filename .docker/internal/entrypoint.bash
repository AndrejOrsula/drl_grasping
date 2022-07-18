#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"

## Source ROS 2 installation and workspace
source "/opt/ros/${ROS_DISTRO}/setup.bash" --
source "${WS_INSTALL_DIR}/local_setup.bash" --

## Configure ROS 2 RMW
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://${SCRIPT_DIR}/cyclonedds.xml

## Default ROS_DOMAIN_ID to 0 if not specified (ROS 2 default)
if [ -z "${ROS_DOMAIN_ID}" ]; then
    export ROS_DOMAIN_ID="0"
fi
## Default IGN_PARTITION to ROS_DOMAIN_ID if not specified
if [ -z "${IGN_PARTITION}" ]; then
    export IGN_PARTITION="${ROS_DOMAIN_ID}"
fi
## Configure behaviour of ROS 2 and Gazebo Transport based on selected ROS_DOMAIN_ID
if [ "${ROS_DOMAIN_ID}" == "69" ]; then
    ## ROS_DOMAIN_ID="69" - Default network interface and multicast configuration
    unset ROS_LOCALHOST_ONLY
    ## Gazebo Transport - Make sure the communication is not restricted
    unset IGN_IP
    unset IGN_RELAY
else
    ## ROS_DOMAIN_ID!=69 - Restrict to localhost
    export ROS_LOCALHOST_ONLY=1
    ## Gazebo Transport - Restrict to localhost
    export IGN_IP=127.0.0.1
    export IGN_RELAY=127.0.0.1
    if [ "${ROS_DOMAIN_ID}" == "42" ]; then
        ## ROS_DOMAIN_ID==42 - Enable multicast
        ifconfig lo multicast
        route add -net 224.0.0.0 netmask 240.0.0.0 dev lo 2>/dev/null
    else
        ## ROS_DOMAIN_ID!=42 - Disable Gazebo Transport broadcasting and user commands
        export DRL_GRASPING_BROADCAST_INTERACTIVE_GUI=false
    fi
fi

## Export paths to Gazebo plugins (ign_ros2_control)
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=${WS_INSTALL_DIR}/lib${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH}}

## Export paths for O-CNN
export PATH="${WS_SRC_DIR}/O-CNN/octree/build${PATH:+:${PATH}}"
export PYTHONPATH="${WS_SRC_DIR}/O-CNN/octree/build/python${PYTHONPATH:+:${PYTHONPATH}}"

## Source textures and SDF models
if [ -d "${ASSETS_DIR}/textures" ]; then
    source "${ASSETS_DIR}/textures/scripts/source.bash"
fi
if [ -d "${ASSETS_DIR}/sdf_models" ]; then
    source "${ASSETS_DIR}/sdf_models/scripts/source.bash"
fi

exec "$@"
