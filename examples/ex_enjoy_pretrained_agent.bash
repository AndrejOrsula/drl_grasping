#!/usr/bin/env bash

## Random seed to use for both the environment and agent (-1 for random)
SEED="69"

## ID of the environment
# ENV_ID="Reach-Gazebo-v0"
# ENV_ID="Reach-ColorImage-Gazebo-v0"
# ENV_ID="Reach-DepthImage-Gazebo-v0"
# ENV_ID="Reach-Octree-Gazebo-v0"
# ENV_ID="Reach-OctreeWithColor-Gazebo-v0"

# ENV_ID="Grasp-Octree-Gazebo-v0"
# ENV_ID="Grasp-OctreeWithColor-Gazebo-v0"

# ENV_ID="GraspPlanetary-Octree-Gazebo-v0"
ENV_ID="GraspPlanetary-OctreeWithColor-Gazebo-v0"

## Robot model
# ROBOT_MODEL="panda"
# ROBOT_MODEL="ur5_rg2"
# ROBOT_MODEL="kinova_j2s7s300"
ROBOT_MODEL="lunalab_summit_xl_gen"

## Algorithm to use
# ALGO="sac"
# ALGO="td3"
ALGO="tqc"

## Arguments for the environment
ENV_ARGS="robot_model:\"${ROBOT_MODEL}\""

## Extra arguments to be passed into the script
EXTRA_ARGS=""

########################################################################################################################
########################################################################################################################

## Spawn ign_moveit2 subprocess in background, while making sure to forward termination signals
IGN_MOVEIT2_CMD="ros2 launch drl_grasping sim.launch.py robot_model:=${ROBOT_MODEL} enable_rviz:=true"
if [ "$ROBOT_MODEL" = "kinova_j2s7s300" ]; then
    # Robot name for `kinova_j2s7s300` is different
    IGN_MOVEIT2_CMD="${IGN_MOVEIT2_CMD} robot_name:=j2s7s300"
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

## Locate directory of pretrained agents
PRETRAINED_AGENTS_DIR=""$(ros2 pkg prefix drl_grasping)"/share/drl_grasping/pretrained_agents"
LOG_DIR=""${PRETRAINED_AGENTS_DIR}"/"${ENV_ID}"/"${ROBOT_MODEL}""

## Arguments
ENJOY_ARGS="--env "${ENV_ID}" --algo "${ALGO}" --seed "${SEED}" --folder "${LOG_DIR}" --env-kwargs "${ENV_ARGS}" "${EXTRA_ARGS}""

## Execute enjoy script
ENJOY_CMD="ros2 run drl_grasping enjoy.py "${ENJOY_ARGS}""
echo "Executing enjoy command:"
echo "${ENJOY_CMD}"
echo ""
${ENJOY_CMD}
