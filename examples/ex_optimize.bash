#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping optimize.launch.py` and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping optimize.launch.py --show-args`.

## Enable training dataset
ros2 run drl_grasping dataset_set_train.bash 2> /dev/null

### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="69"

## Robot to use during training
ROBOT_MODEL="panda"
# ROBOT_MODEL="lunalab_summit_xl_gen"

## ID of the environment
## Reach
# ENV="Reach-Gazebo-v0"
# ENV="Reach-ColorImage-Gazebo-v0"
# ENV="Reach-DepthImage-Gazebo-v0"
# ENV="Reach-Octree-Gazebo-v0"
# ENV="Reach-OctreeWithIntensity-Gazebo-v0"
# ENV="Reach-OctreeWithColor-Gazebo-v0"
## Grasp
# ENV="Grasp-Octree-Gazebo-v0"
# ENV="Grasp-OctreeWithIntensity-Gazebo-v0"
ENV="Grasp-OctreeWithColor-Gazebo-v0"
## GraspPlanetary
# ENV="GraspPlanetary-MonoImage-Gazebo-v0"
# ENV="GraspPlanetary-ColorImage-Gazebo-v0"
# ENV="GraspPlanetary-DepthImage-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-DepthImageWithColor-Gazebo-v0"
# ENV="GraspPlanetary-Octree-Gazebo-v0"
# ENV="GraspPlanetary-OctreeWithIntensity-Gazebo-v0"
# ENV="GraspPlanetary-OctreeWithColor-Gazebo-v0"

## Selection of RL algorithm
# ALGO="td3"
# ALGO="sac"
ALGO="tqc"

## Path to logs directory
LOG_FOLDER="${PWD}/drl_grasping_training/optimize/${ENV}/logs"

## Path to tensorboard logs directory
TENSORBOARD_LOG="${PWD}/drl_grasping_training/optimize/${ENV}/tensorboard_logs"

## Path to a replay buffer that should be loaded before each trial begins (`**/*.pkl`)
# PRELOAD_REPLAY_BUFFER=""

### Arguments
LAUNCH_ARGS=(
    "seed:=${SEED}"
    "robot_model:=${ROBOT_MODEL}"
    "env:=${ENV}"
    "algo:=${ALGO}"
    "log_folder:=${LOG_FOLDER}"
    "tensorboard_log:=${TENSORBOARD_LOG}"
    "n_timesteps:=1000000"
    "sampler:=tpe"
    "pruner:=median"
    "n_trials:=20"
    "n_startup_trials:=5"
    "n_evaluations:=4"
    "eval_episodes:=20"
    "log_interval:=-1"
    "enable_rviz:=true"
    "log_level:=fatal"
)
if [[ -n ${PRELOAD_REPLAY_BUFFER} ]]; then
    LAUNCH_ARGS+=("preload_replay_buffer:=${PRELOAD_REPLAY_BUFFER}")
fi

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping optimize.launch.py
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]}
