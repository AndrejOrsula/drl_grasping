#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping random_agent.launch.py` and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping random_agent.launch.py --show-args`.

### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="42"

## Robot to use during training
# ROBOT_MODEL="panda"
ROBOT_MODEL="lunalab_summit_xl_gen"

## ID of the environment
## Reach
# ENV="Reach-Gazebo-v0"
# ENV="Reach-ColorImage-Gazebo-v0"
# ENV="Reach-DepthImage-Gazebo-v0"
# ENV="Reach-Octree-Gazebo-v0"
# ENV="Reach-OctreeWithIntensity-Gazebo-v0"
# ENV="Reach-OctreeWithColor-Gazebo-v0"
## Grasp
# ENV="Grasp-Gazebo-v0"
# ENV="Grasp-Octree-Gazebo-v0"
# ENV="Grasp-OctreeWithIntensity-Gazebo-v0"
# ENV="Grasp-OctreeWithColor-Gazebo-v0"
## GraspPlanetary
# ENV="GraspPlanetary-Gazebo-v0"
# ENV="GraspPlanetary-MonoImage-Gazebo-v0"
# ENV="GraspPlanetary-ColorImage-Gazebo-v0"
# ENV="GraspPlanetary-DepthImage-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithColor-Gazebo-v0"
# ENV="GraspPlanetary-Octree-Gazebo-v0"
ENV="GraspPlanetary-OctreeWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-OctreeWithColor-Gazebo-v0"

### Arguments
LAUNCH_ARGS=(
    "seed:=${SEED}"
    "robot_model:=${ROBOT_MODEL}"
    "env:=${ENV}"
    "check_env:=false"
    "render:=true"
    "enable_rviz:=true"
    "log_level:=warn"
)

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping random_agent.launch.py
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]}
