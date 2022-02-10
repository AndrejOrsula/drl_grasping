#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping evaluate.launch.py` and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping evaluate.launch.py --show-args`.

### Global configuration
## OMP
export OMP_DYNAMIC=TRUE
export OMP_NUM_THREADS=4


### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="77"

## Robot to use during evaluation
# ROBOT_MODEL="panda"
ROBOT_MODEL="lunalab_summit_xl_gen"

## ID of the environment
# ENV="Reach-Gazebo-v0"
# ENV="Reach-ColorImage-Gazebo-v0"
# ENV="Reach-DepthImage-Gazebo-v0"
# ENV="Reach-Octree-Gazebo-v0"
# ENV="Reach-OctreeWithIntensity-Gazebo-v0"
# ENV="Reach-OctreeWithColor-Gazebo-v0"
# ENV="Grasp-Octree-Gazebo-v0"
# ENV="Grasp-OctreeWithIntensity-Gazebo-v0"
# ENV="Grasp-OctreeWithColor-Gazebo-v0"
# ENV="GraspPlanetary-DepthImage-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithColor-Gazebo-v0"
# ENV="GraspPlanetary-Octree-Gazebo-v0"
ENV="GraspPlanetary-OctreeWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-OctreeWithColor-Gazebo-v0"

## Selection of RL algorithm
# ALGO="td3"
# ALGO="sac"
ALGO="tqc"

## Path to logs directory
LOG_FOLDER="${PWD}/drl_grasping_training/train/${ENV}/logs"

## Path to reward log directory
REWARD_LOG="${PWD}/drl_grasping_training/evaluate/${ENV}"

## Load checkpoint instead of last model (# steps)
# LOAD_CHECKPOINT="0"

### Arguments
LAUNCH_ARGS=(
    "seed:=${SEED}"
    "robot_model:=${ROBOT_MODEL}"
    "env:=${ENV}"
    "algo:=${ALGO}"
    "log_folder:=${LOG_FOLDER}"
    "reward_log:=${REWARD_LOG}"
    "stochastic:=false"
    "n_episodes:=200"
    "load_best:=false"
    "enable_rviz:=true"
    "log_level:=warn"
)
if [[ -n ${LOAD_CHECKPOINT} ]]; then
    LAUNCH_ARGS+=("load_checkpoint:=${LOAD_CHECKPOINT}")
fi

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping evaluate.launch.py
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]}
