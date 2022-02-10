## ROS 2
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

## Source workspace overlay
source "${WS_INSTALL_DIR}/local_setup.bash"

## Export paths for O-CNN
export PATH="${WS_SRC_DIR}/O-CNN/octree/build${PATH:+:${PATH}}"
export PYTHONPATH="${WS_SRC_DIR}/O-CNN/octree/build/python${PYTHONPATH:+:${PYTHONPATH}}"

## Robot models
export IGN_GAZEBO_RESOURCE_PATH="${WS_SRC_DIR}/panda_ign_moveit2/panda_description${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"
export IGN_GAZEBO_RESOURCE_PATH="${WS_SRC_DIR}/lunalab_summit_xl_gen/lunalab_summit_xl_gen_description${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"

## Path to PBR textures
if [ -d "${WS_DIR}/pbr_textures" ]; then
    # Use external textures (if mounted as volume)
    export DRL_GRASPING_PBR_TEXTURES_DIR="${WS_DIR}/pbr_textures"
else
    # Use the default textures
    export DRL_GRASPING_PBR_TEXTURES_DIR="${ASSETS_DIR}/pbr_textures"
fi

## Appending source command to ~/.bashrc enables autocompletion (ENTRYPOINT alone does not support that)
grep -qxF ". ${WS_DIR}/entrypoint.bash" "${HOME}/.bashrc" || echo ". ${WS_DIR}/entrypoint.bash" >>"${HOME}/.bashrc"

## Aliases
alias _nano_envs='nano ${WS_SRC_DIR}/drl_grasping/drl_grasping/envs/__init__.py'
alias _nano_td3='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/td3.yml'
alias _nano_sac='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/sac.yml'
alias _nano_tqc='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/tqc.yml'
alias _nano_ex_train='nano ${WS_SRC_DIR}/drl_grasping/examples/ex_train.bash'
alias __train='ros2 run drl_grasping ex_train.bash'
