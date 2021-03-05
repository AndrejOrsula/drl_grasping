#!/usr/bin/env bash

## Random seed to use for both the environment and agent
SEED="-1"

## Action that htis script should peform
# ACTION="optimize"
ACTION="train"
# ACTION="enjoy"

## ID of the environment
## Reach
ENV_ID="Reach-Gazebo-v0"
# ENV_ID="Reach-ColorImage-Gazebo-v0"
# ENV_ID="Reach-DepthImage-Gazebo-v0"
# ENV_ID="Reach-Octree-Gazebo-v0"
# ENV_ID="Reach-OctreeWithColor-Gazebo-v0"
## Grasp
# ENV_ID="Grasp-Gazebo-v0"
# ENV_ID="Grasp-Octree-Gazebo-v0"
# ENV_ID="Grasp-OctreeWithColor-Gazebo-v0"

## Algorithm to use
ALGO="sac"

## Path to trained agent (to continue training)
# TRAINED_AGENT=""${ENV_ID}"_1/rl_model_100000_steps.zip"

## Last checkpoint (to enjoy)
# CHECKPOINT=100000

## Args for optimization
OPTIMIZE_N_TIMESTAMPS=25000
OPTIMIZE_N_STARTUP_TRIALS=5
OPTIMIZE_N_TRIALS=10
OPTIMIZE_EVAL_EPISODES=10

## Path the parent training directory
TRAINING_DIR="training"
## Path to logs
LOG_DIR=""${TRAINING_DIR}"/"${ENV_ID}"/logs"
## Path to tensorboard logs (train)
TENSORBOARD_LOG_DIR=""${TRAINING_DIR}"/"${ENV_ID}"/tensorboard_logs"
## Path to reward logs (enjoy)
REWARD_LOG_DIR=""${TRAINING_DIR}"/"${ENV_ID}"/reward_logs"

## Spawn ign_moveit2 subprocess in background, while making sure to forward termination signals
IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_headless.launch.py"
echo "Launching ign_moveit2 in background:"
echo "${IGN_MOVEIT2_CMD}"
echo ""
${IGN_MOVEIT2_CMD} &
## Kill all subprocesses when SIGINT SIGTERM EXIT are received
subprocess_pid_ign_moveit2="${!}"
terminate_subprocesses() {
    echo "INFO: Caught signal, killing all subprocesses..."
    pkill -P "${subprocess_pid_ign_moveit2}"
}
trap 'terminate_subprocesses' SIGINT SIGTERM EXIT

## Locate scripts directory
if [ -f ""$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)")"/scripts" ]; then
    # If run from source code
    SCRIPT_DIR=""$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)")"/scripts"
else
    # If run from installed dir or via `ros2 run`
    SCRIPT_DIR=""$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)""
fi

# Arguments used with all actions
COMMON_ARGS="--env "${ENV_ID}" --seed "${SEED}" --algo "${ALGO}""

if [ "${ACTION}" = "train" ]; then
    ## Arguments
    TRAIN_ARGS=""${COMMON_ARGS}" --log-folder "${LOG_DIR}" --tensorboard-log "${TENSORBOARD_LOG_DIR}""
    ## Add trained agent to args in order to continue training
    if [ ! -z "${TRAINED_AGENT}" ]; then
        TRAIN_ARGS=""${TRAIN_ARGS}" --trained-agent "${LOG_DIR}"/"${ALGO}"/"${TRAINED_AGENT}""
    fi

    ## Execute train script
    TRAIN_CMD=""${SCRIPT_DIR}"/train.py "${TRAIN_ARGS}""
    echo "Executing train command:"
    echo "${TRAIN_CMD}"
    echo ""
    ${TRAIN_CMD}
elif [ "${ACTION}" = "enjoy" ]; then
    ## Arguments
    ENJOY_ARGS=""${COMMON_ARGS}" --folder "${LOG_DIR}" --reward-log "${REWARD_LOG_DIR}""
    ## Add trained agent to args in order to continue training
    if [ ! -z "${CHECKPOINT}" ]; then
        ENJOY_ARGS=""${ENJOY_ARGS}" --load-checkpoint "${CHECKPOINT}""
    fi

    ## Execute enjoy script
    ENJOY_CMD=""${SCRIPT_DIR}"/enjoy.py "${ENJOY_ARGS}""
    echo "Executing enjoy command:"
    echo "${ENJOY_CMD}"
    echo ""
    ${ENJOY_CMD}
elif [ "${ACTION}" = "optimize" ]; then
    ## Arguments
    OPTIMIZE_ARGS=""${COMMON_ARGS}" --log-folder "${LOG_DIR}" --optimize-hyperparameters --n-timesteps "${OPTIMIZE_N_TIMESTAMPS}" --n-startup-trials "${OPTIMIZE_N_STARTUP_TRIALS}" --n-trials "${OPTIMIZE_N_TRIALS}" --eval-episodes "${OPTIMIZE_EVAL_EPISODES}""

    ## Execute optimize script
    OPTIMIZE_CMD=""${SCRIPT_DIR}"/train.py "${OPTIMIZE_ARGS}""
    echo "Executing train command for optimization:"
    echo "${OPTIMIZE_CMD}"
    echo ""
    ${OPTIMIZE_CMD}
fi
