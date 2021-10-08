#!/usr/bin/env bash


## Random seed to use for both the environment and agent (-1 for random)
SEED="123"

## ID of the environment
## Note: `preload_replay_buffer` must be enabled in the environment manually
## Reach
# ENV_ID="Reach-Gazebo-v0"
# ENV_ID="Reach-ColorImage-Gazebo-v0"
# ENV_ID="Reach-Octree-Gazebo-v0"
# ENV_ID="Reach-OctreeWithColor-Gazebo-v0"
## Grasp
# ENV_ID="Grasp-Octree-Gazebo-v0"
ENV_ID="Grasp-OctreeWithColor-Gazebo-v0"

## Robot model
ROBOT_MODEL="panda"
# ROBOT_MODEL="ur5_rg2"
# ROBOT_MODEL="kinova_j2s7s300"

## Algorithm to use (might not matter too much as long as it is off-policy)
# ALGO="sac"
# ALGO="td3"
ALGO="tqc"

## Path to logs, where to save the replay buffer
LOG_DIR="training/preloaded_buffers"

## Arguments for the environment
ENV_ARGS="robot_model:\"${ROBOT_MODEL}\""

## Extra arguments to be passed into the script
EXTRA_ARGS=""

########################################################################################################################
########################################################################################################################

## Spawn ign_moveit2 subprocess in background, while making sure to forward termination signals
IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_headless.launch.py"
if [ "$ROBOT_MODEL" = "ur5_rg2" ]; then
    IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_headless_ur5_rg2.launch.py"
fi
if [ "$ROBOT_MODEL" = "kinova_j2s7s300" ]; then
    IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_headless_kinova_j2s7s300.launch.py"
fi
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
trap 'terminate_subprocesses' SIGINT SIGTERM EXIT ERR

## Arguments
PRELOAD_BUFFER_ARGS="--env "${ENV_ID}" --algo "${ALGO}" --seed "${SEED}" --log-folder "${LOG_DIR}" --env-kwargs "${ENV_ARGS}" "${EXTRA_ARGS}""

## Execute train script
PRELOAD_BUFFER_CMD="ros2 run drl_grasping preload_replay_buffer.py "${PRELOAD_BUFFER_ARGS}""
echo "Executing command that preloads replay buffer:"
echo "${PRELOAD_BUFFER_CMD}"
echo ""
${PRELOAD_BUFFER_CMD}
