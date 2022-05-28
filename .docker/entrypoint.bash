## ROS 2
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

## Source workspace overlay
source "${WS_INSTALL_DIR}/local_setup.bash"

## Export paths for O-CNN
export PATH="${WS_SRC_DIR}/O-CNN/octree/build${PATH:+:${PATH}}"
export PYTHONPATH="${WS_SRC_DIR}/O-CNN/octree/build/python${PYTHONPATH:+:${PYTHONPATH}}"

## Ignition plugins (ign_ros2_control)
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=${WS_INSTALL_DIR}/lib${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH}}

## Ignition models (robots)
export IGN_GAZEBO_RESOURCE_PATH="${WS_SRC_DIR}/panda_ign_moveit2/panda_description${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"
export IGN_GAZEBO_RESOURCE_PATH="${WS_SRC_DIR}/lunalab_summit_xl_gen/lunalab_summit_xl_gen_description${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"

## Source textures
if [ -d "${ASSETS_DIR}/textures" ]; then
    source "${ASSETS_DIR}/textures/scripts/source.bash"
fi

## Source SDF models
if [ -d "${ASSETS_DIR}/sdf_models" ]; then
    source "${ASSETS_DIR}/sdf_models/scripts/source.bash"
fi

## Path to PBR textures if volume is mounted
if [ -d "${WS_DIR}/pbr_textures" ]; then
    # Use external textures (if mounted as volume)
    export TEXTURE_DIRS="${WS_DIR}/pbr_textures"
fi

## Appending source command to ~/.bashrc enables autocompletion (ENTRYPOINT alone does not support that)
grep -qxF ". ${WS_DIR}/entrypoint.bash" "${HOME}/.bashrc" || echo ". ${WS_DIR}/entrypoint.bash" >>"${HOME}/.bashrc"

## Aliases
alias _nano_envs='nano ${WS_SRC_DIR}/drl_grasping/drl_grasping/envs/__init__.py'
alias _nano_td3='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/td3.yml'
alias _nano_sac='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/sac.yml'
alias _nano_tqc='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/tqc.yml'
alias _nano_ex_train='nano ${WS_SRC_DIR}/drl_grasping/examples/ex_train.bash'
alias _nano_ex_train_dreamerv2='nano ${WS_SRC_DIR}/drl_grasping/examples/ex_train_dreamerv2.bash'
alias __train='ros2 run drl_grasping ex_train.bash'
alias __train_dreamerv2='ros2 run drl_grasping ex_train_dreamerv2.bash'
