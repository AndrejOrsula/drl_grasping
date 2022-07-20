#!/usr/bin/env bash
#### This script serves as an example of utilising `ros2 launch drl_grasping evaluate.launch.py` on a real robot and configuring some of its most common arguments.
#### When this script is called, the corresponding launch string is printed to STDOUT. Therefore, feel free to modify and use such command directly.
#### To view all arguments, run `ros2 launch drl_grasping evaluate.launch.py --show-args`.

## Use the correct runtime
export DRL_GRASPING_REAL_EVALUATION=True

### Arguments
## Random seed to use for both the environment and agent (-1 for random)
SEED="77"

## Robot to use during training
ROBOT_MODEL="panda"
# ROBOT_MODEL="lunalab_summit_xl_gen"

## ID of the environment
## Reach
# ENV="Reach-v0"
# ENV="Reach-ColorImage-v0"
# ENV="Reach-DepthImage-v0"
# ENV="Reach-Octree-v0"
# ENV="Reach-OctreeWithIntensity-v0"
# ENV="Reach-OctreeWithColor-v0"
## Grasp
# ENV="Grasp-Octree-v0"
# ENV="Grasp-OctreeWithIntensity-v0"
ENV="Grasp-OctreeWithColor-v0"
## GraspPlanetary
# ENV="GraspPlanetary-MonoImage-v0"
# ENV="GraspPlanetary-ColorImage-v0"
# ENV="GraspPlanetary-DepthImage-v0"
# ENV="GraspPlanetary-DepthImageWithIntensity-v0"
# ENV="GraspPlanetary-DepthImageWithColor-v0"
# ENV="GraspPlanetary-Octree-v0"
# ENV="GraspPlanetary-OctreeWithIntensity-v0"
# ENV="GraspPlanetary-OctreeWithColor-v0"

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

#### Launch script ####
### Arguments
LAUNCH_ARGS=(
    "enable_rviz:=true"
    "log_level:=warn"
)

### Launch script
LAUNCH_CMD=(
    ros2 launch -a
    drl_grasping "real_${ROBOT_MODEL}.launch.py"
    "${LAUNCH_ARGS[*]}"
)

echo -e "\033[1;30m${LAUNCH_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${LAUNCH_CMD[*]} &

terminate_child_processes() {
    echo "Signal received. Terminating all child processes..."
    for job in $(jobs -p); do
        kill -TERM "$job" 2>/dev/null || echo -e "\033[31m$job could not be terminated...\033[0m" >&2
    done
}
trap terminate_child_processes SIGINT SIGTERM SIGQUIT

#### Evaluation node ####
# Note: Evaluation node is started separately in order to enable user input
### Arguments
NODE_ARGS=(
    "--env" "${ENV}"
    "--env-kwargs" "robot_model:\"${ROBOT_MODEL}\""
    "--algo" "${ALGO}"
    "--seed" "${SEED}"
    "--num-threads" "-1"
    "--n-episodes" "200"
    "--stochastic" "false"
    "--log-folder" "${LOG_FOLDER}"
    "--reward-log" "${REWARD_LOG}"
    "--exp-id" "0"
    "--load-best" "false"
    "--norm-reward" "false"
    "--no-render" "true"
    "--verbose" "1"
)
if [[ -n ${LOAD_CHECKPOINT} ]]; then
    NODE_ARGS+=("--load-checkpoint" "${LOAD_CHECKPOINT}")
fi

### ROS arguments
NODE_ARGS+=(
    "--ros-args"
    "--log-level" "warn"
    "--param" "use_sim_time:=false"
    "--remap" "/rgbd_camera/points:=/lunalab_summit_xl_gen_d435/depth/color/points"
    "--remap" "/rgbd_camera/image:=/lunalab_summit_xl_gen_d435/color/image_raw"
    "--remap" "/rgbd_camera/depth_image:=/lunalab_summit_xl_gen_d435/depth/image_rect_raw"
)

### Run the node
NODE_CMD=(
    ros2 run
    drl_grasping evaluate.py
    "${NODE_ARGS[*]}"
)

echo -e "\033[1;30m${NODE_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${NODE_CMD[*]}
