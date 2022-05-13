#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping train_dreamerv2.launch.py` and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping train_dreamerv2.launch.py --show-args`.

### Global configuration
## OMP
export OMP_DYNAMIC=TRUE
export OMP_NUM_THREADS=4


### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="42"

## Robot to use during training
# ROBOT_MODEL="panda"
ROBOT_MODEL="lunalab_summit_xl_gen"

## ID of the environment
ENV="Reach-ColorImage-Gazebo-v0"
# ENV="GraspPlanetary-ColorImage-Gazebo-v0"
# ENV="GraspPlanetary-MonoImage-Gazebo-v0"

## Path to logs directory
LOG_FOLDER="${PWD}/drl_grasping_training/train/${ENV}/logs"

### Arguments
LAUNCH_ARGS=(
    "seed:=${SEED}"
    "robot_model:=${ROBOT_MODEL}"
    "env:=${ENV}"
    "log_folder:=${LOG_FOLDER}"
    "eval_freq:=10000"
    "enable_rviz:=false"
    "log_level:=fatal"
)

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping train_dreamerv2.launch.py
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]}
