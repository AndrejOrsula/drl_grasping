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
