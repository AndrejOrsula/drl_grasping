#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping train.launch.py` and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping train.launch.py --show-args`.

### Global configuration
## OMP
export OMP_DYNAMIC=TRUE
export OMP_NUM_THREADS=3


### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="42"

## Robot to use during training
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

## Path to tensorboard logs directory
TENSORBOARD_LOG="${PWD}/drl_grasping_training/train/${ENV}/tensorboard_logs"

## Path to a trained agent to continue training (`**.zip`)
# TRAINED_AGENT_SESSION="1"
# TRAINED_AGENT_STEPS="0"
# TRAINED_AGENT="${LOG_FOLDER}/${ALGO}/${ENV}_${TRAINED_AGENT_SESSION}/rl_model_${TRAINED_AGENT_STEPS}_steps.zip"

## Path to a replay buffer that should be loaded before the training begins (`**.pkl`)
# PRELOAD_REPLAY_BUFFER=""

### Arguments
LAUNCH_ARGS=(
    "seed:=${SEED}"
    "robot_model:=${ROBOT_MODEL}"
    "env:=${ENV}"
    "algo:=${ALGO}"
    "log_folder:=${LOG_FOLDER}"
    "tensorboard_log:=${TENSORBOARD_LOG}"
    "save_freq:=10000"
    "save_replay_buffer:=true"
    "log_interval:=-1"
    "eval_freq:=-1"
    "eval_episodes:=5"
    "enable_rviz:=false"
    "log_level:=fatal"
)
if [[ -n ${TRAINED_AGENT} ]]; then
    LAUNCH_ARGS+=("trained_agent:=${TRAINED_AGENT}")
fi
if [[ -n ${PRELOAD_REPLAY_BUFFER} ]]; then
    LAUNCH_ARGS+=("preload_replay_buffer:=${PRELOAD_REPLAY_BUFFER}")
fi

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping train.launch.py
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]}
