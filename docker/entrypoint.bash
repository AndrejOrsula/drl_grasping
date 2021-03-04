#!/usr/bin/env bash
set -e

## ROS 2
source /opt/ros/${ROS2_DISTRO}/setup.bash

## MoveIt2
source ${DRL_GRASPING_DEPS_DIR}/moveit2/install/local_setup.bash

## O-CNN Octree
export PATH=${DRL_GRASPING_DEPS_DIR}/O-CNN/octree/build:${PATH}
export PYTHONPATH=${DRL_GRASPING_DEPS_DIR}/O-CNN/octree/build/python:${PYTHONPATH}

## Custom packages
source ${DRL_GRASPING_REPOS_DIR}/drl_grasping/install/local_setup.bash

# PBR textures (optional for random ground plane)
export DRL_GRASPING_PBR_TEXTURES_DIR="${DRL_GRASPING_WS_DIR}/pbr_textures"

# If a CMD is passed, execute it
exec "$@"
