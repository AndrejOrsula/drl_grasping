# Note: The `open3d` and `stable_baselines3` modules must be imported prior to `gym_ignition`
import open3d  # isort:skip
import stable_baselines3  # isort:skip

# Note: If installed, `tensorflow` module must be imported before `gym_ignition`/`scenario`
# Otherwise, protobuf version incompatibility will cause an error
try:
    from importlib.util import find_spec

    if find_spec("tensorflow") is not None:
        import tensorflow
except:
    pass

from os import environ, path
from typing import Dict, Tuple

import numpy as np
from ament_index_python.packages import get_package_share_directory
from gym.envs.registration import register

from drl_grasping.utils.utils import str2bool

from . import tasks

######################
# Runtime Entrypoint #
######################
# Entrypoint for tasks (can be simulated or real)
if str2bool(environ.get("DRL_GRASPING_REAL_EVALUATION", default=False)):
    DRL_GRASPING_TASK_ENTRYPOINT: str = (
        "drl_grasping.envs.runtimes:RealEvaluationRuntime"
    )
else:
    DRL_GRASPING_TASK_ENTRYPOINT: str = (
        "gym_ignition.runtimes.gazebo_runtime:GazeboRuntime"
    )


###################
# Robot Specifics #
###################
## Fully supported robots: "panda", "lunalab_summit_xl_gen" (mobile)
# Default robot model to use in the tasks where robot can be static
DRL_GRASPING_ROBOT_MODEL: str = "panda"
# Default robot model to use in the tasks where robot needs to be mobile
DRL_GRASPING_ROBOT_MODEL_MOBILE: str = "lunalab_summit_xl_gen"

######################
# Datasets and paths #
######################
# Path to directory containing base SDF worlds
DRL_GRASPING_WORLDS_DIR: str = path.join(
    get_package_share_directory("drl_grasping"), "worlds"
)


###########
# Presets #
###########
# Gravity preset for Earth
GRAVITY_EARTH: Tuple[float, float, float] = (0.0, 0.0, -9.80665)
GRAVITY_EARTH_STD: Tuple[float, float, float] = (0.0, 0.0, 0.0232)
# Gravity preset for Moon
GRAVITY_MOON: Tuple[float, float, float] = (0.0, 0.0, -1.625)
GRAVITY_MOON_STD: Tuple[float, float, float] = (0.0, 0.0, 0.0084)
# Gravity preset for Mars
GRAVITY_MARS: Tuple[float, float, float] = (0.0, 0.0, -3.72076)
GRAVITY_MARS_STD: Tuple[float, float, float] = (0.0, 0.0, 0.0191)


############################
# Additional Configuration #
############################
BROADCAST_GUI: bool = str2bool(
    environ.get("DRL_GRASPING_BROADCAST_INTERACTIVE_GUI", default=True)
)


#########
# Reach #
#########
REACH_MAX_EPISODE_STEPS: int = 100
REACH_KWARGS: Dict[str, any] = {
    "agent_rate": 4.0,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "world",
    "workspace_centre": (0.45, 0.0, 0.25),
    "workspace_volume": (0.5, 0.5, 0.5),
    "ignore_new_actions_while_executing": False,
    "use_servo": True,
    "scaling_factor_translation": 0.4,
    "scaling_factor_rotation": np.pi / 4.0,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "sparse_reward": False,
    "act_quick_reward": -0.01,
    "required_accuracy": 0.05,
    "num_threads": 3,
}
REACH_KWARGS_OCTREE: Dict[str, any] = {
    "octree_reference_frame_id": "world",
    "octree_min_bound": (0.45 - 0.25, 0.0 - 0.25, 0.25 - 0.25),
    "octree_max_bound": (0.45 + 0.25, 0.0 + 0.25, 0.25 + 0.25),
    "octree_depth": 3,
    "octree_full_depth": 2,
    "octree_n_stacked": 2,
}
REACH_KWARGS_SIM: Dict[str, any] = {
    "physics_rate": 200.0,
    "real_time_factor": float(np.finfo(np.float32).max),
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
REACH_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
REACH_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": BROADCAST_GUI,
    "plugin_user_commands": BROADCAST_GUI,
    "plugin_sensors_render_engine": "ogre2",
    "robot_random_pose": False,
    "robot_random_joint_positions": False,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.0,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.2,
    "terrain_enable": True,
    "light_type": "sun",
    "light_direction": (0.5, 0.4, -0.2),
    "light_random_minmax_elevation": (-0.15, -0.5),
    "light_distance": 1000.0,
    "light_visual": False,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "sphere",
    "objects_relative_to": "base_link",
    "object_static": True,
    "object_collision": False,
    "object_visual": True,
    "object_color": (0.0, 0.0, 1.0, 1.0),
    "object_dimensions": [0.025, 0.025, 0.025],
    "object_count": 1,
    "object_spawn_position": (0.55, 0, 0.4),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [],
    "object_random_spawn_volume": (0.3, 0.3, 0.3),
    "object_models_rollouts_num": 0,
    "underworld_collision_plane": False,
}
REACH_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 64,
    "camera_height": 64,
    "camera_update_rate": 1.2 * REACH_KWARGS["agent_rate"],
    "camera_horizontal_fov": np.pi / 3.0,
    "camera_vertical_fov": np.pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "base_link",
    "camera_spawn_position": (0.85, -0.4, 0.45),
    "camera_spawn_quat_xyzw": (-0.0402991, -0.0166924, 0.9230002, 0.3823192),
    "camera_random_pose_rollouts_num": 0,
    "camera_random_pose_mode": "orbit",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": np.pi / 8,
    "camera_random_pose_select_position_options": [],
    "camera_random_pose_focal_point_z_offset": 0.0,
}

# Task
register(
    id="Reach-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.Reach,
        **REACH_KWARGS,
    },
)
register(
    id="Reach-ColorImage-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.ReachColorImage,
        **REACH_KWARGS,
    },
)
register(
    id="Reach-DepthImage-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.ReachDepthImage,
        **REACH_KWARGS,
    },
)
register(
    id="Reach-Octree-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.ReachOctree,
        **REACH_KWARGS,
        **REACH_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": False,
        "octree_max_size": 20000,
    },
)
register(
    id="Reach-OctreeWithIntensity-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.ReachOctree,
        **REACH_KWARGS,
        **REACH_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": True,
        "octree_max_size": 25000,
    },
)
register(
    id="Reach-OctreeWithColor-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.ReachOctree,
        **REACH_KWARGS,
        **REACH_KWARGS_OCTREE,
        "octree_include_color": True,
        "octree_include_intensity": False,
        "octree_max_size": 35000,
    },
)
# Gazebo wrapper
register(
    id="Reach-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        "camera_enable": False,
    },
)
register(
    id="Reach-ColorImage-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-ColorImage-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        **REACH_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_color": True,
    },
)
register(
    id="Reach-DepthImage-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-DepthImage-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        **REACH_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "depth_camera",
        "camera_publish_depth": True,
    },
)
register(
    id="Reach-Octree-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-Octree-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        **REACH_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "depth_camera",
        "camera_publish_points": True,
    },
)
register(
    id="Reach-OctreeWithIntensity-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-OctreeWithIntensity-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        **REACH_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        # "camera_image_format": "L8",
        "camera_publish_points": True,
    },
)
register(
    id="Reach-OctreeWithColor-Gazebo-v0",
    entry_point=REACH_RANDOMIZER,
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Reach-OctreeWithColor-v0",
        **REACH_KWARGS_SIM,
        **REACH_KWARGS_RANDOMIZER,
        **REACH_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_points": True,
    },
)


#########
# Grasp #
#########
GRASP_MAX_EPISODE_STEPS: int = 100
GRASP_KWARGS: Dict[str, any] = {
    "agent_rate": 2.5,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "arm_base_link",
    "workspace_centre": (0.5, 0.0, 0.15),
    "workspace_volume": (0.24, 0.24, 0.3),
    "ignore_new_actions_while_executing": False,
    # "use_servo": True,
    # "scaling_factor_translation": 0.5,
    # "scaling_factor_rotation": np.pi / 8,
    # "full_3d_orientation": True,
    "use_servo": False,
    "scaling_factor_translation": 0.1,
    "scaling_factor_rotation": np.pi / 4,
    "full_3d_orientation": False,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "num_threads": 3,
}
GRASP_KWARGS_OCTREE: Dict[str, any] = {
    "octree_reference_frame_id": "arm_base_link",
    "octree_min_bound": (
        0.5 - 0.12,
        0.0 - 0.12,
        0.11 - 0.12,
    ),
    "octree_max_bound": (
        0.5 + 0.12,
        0.0 + 0.12,
        0.11 + 0.12,
    ),
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 3,
    "proprioceptive_observations": True,
}
GRASP_KWARGS_SIM: Dict[str, any] = {
    "physics_rate": 400.0,
    "real_time_factor": float(np.finfo(np.float32).max),
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
GRASP_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
GRASP_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": BROADCAST_GUI,
    "plugin_user_commands": BROADCAST_GUI,
    "plugin_sensors_render_engine": "ogre2",
    "robot_spawn_position": (0, 0, 0),
    "robot_spawn_quat_xyzw": (0, 0, 0, 1),
    "robot_random_pose": False,
    "robot_random_spawn_volume": (0, 0, 0),
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": np.pi / 20,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.15,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.1,
    "terrain_enable": True,
    "terrain_spawn_position": (0.25, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (1.5, 1.5),
    "terrain_model_rollouts_num": 1,
    "light_type": "random_sun",
    "light_direction": (-0.5, -0.4, -0.2),
    "light_random_minmax_elevation": (-0.15, -0.5),
    "light_distance": 1000.0,
    "light_visual": True,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "random_mesh",
    "objects_relative_to": "arm_base_link",
    "object_count": 4,
    "object_spawn_position": (0.5, 0.0, 0.125),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [],
    "object_random_spawn_position_update_workspace_centre": False,
    "object_random_spawn_volume": (0.22, 0.22, 0.1),
    "object_models_rollouts_num": 2,
    "underworld_collision_plane": True,
    "boundary_collision_walls": True,
}
GRASP_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 256,
    "camera_height": 256,
    "camera_update_rate": 4.0 * GRASP_KWARGS["agent_rate"],
    "camera_horizontal_fov": np.pi / 4.0,
    "camera_vertical_fov": np.pi / 4.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "base_link",
    "camera_spawn_position": (
        1.0054652820235743,
        -0.80636443067215891,
        0.72881734178675539,
    ),
    "camera_spawn_quat_xyzw": (
        -0.28270992102080017,
        0.19612786858776488,
        0.7714710414897703,
        0.5352021971762847,
    ),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_mode": "orbit",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": np.pi / 6,
    "camera_random_pose_select_position_options": [],
    "camera_random_pose_focal_point_z_offset": 0.0,
}
GRASP_KWARGS_CURRICULUM: Dict[str, any] = {
    "stages_base_reward": 1.0,
    "reach_required_distance": 0.1,
    "lift_required_height": 0.1,
    "lift_required_height_max": 0.2,
    "lift_required_height_max_threshold": 0.4,
    "persistent_reward_each_step": -0.005,
    "persistent_reward_terrain_collision": -1.0,
    "persistent_reward_all_objects_outside_workspace": 0.0,
    "persistent_reward_arm_stuck": -0.001,
    "enable_stage_reward_curriculum": True,
    "enable_workspace_scale_curriculum": False,
    "enable_object_spawn_volume_scale_curriculum": False,
    "enable_object_count_curriculum": False,
    "stage_reward_multiplier": 8.0,
    "dense_reward": False,
    "initial_success_rate": 0.0,
    "rolling_average_n": 100,
    "min_workspace_scale": 0.5,
    "max_workspace_volume": GRASP_KWARGS["workspace_volume"],
    "max_workspace_scale_success_rate_threshold": 0.6,
    "min_object_spawn_volume_scale": 0.5,
    "max_object_spawn_volume": GRASP_KWARGS_RANDOMIZER["object_random_spawn_volume"],
    "max_object_spawn_volume_scale_success_rate_threshold": 0.6,
    "object_count_min": 1,
    "object_count_max": GRASP_KWARGS_RANDOMIZER["object_count"],
    "max_object_count_success_rate_threshold": 0.6,
    "arm_stuck_n_steps": 20,
    "arm_stuck_min_joint_difference_norm": np.pi / 32,
}

# Task
register(
    id="Grasp-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.Grasp,
        **GRASP_KWARGS,
        **GRASP_KWARGS_CURRICULUM,
    },
)
register(
    id="Grasp-Octree-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspOctree,
        **GRASP_KWARGS,
        **GRASP_KWARGS_CURRICULUM,
        **GRASP_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": False,
        "octree_max_size": 50000,
    },
)
register(
    id="Grasp-OctreeWithIntensity-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspOctree,
        **GRASP_KWARGS,
        **GRASP_KWARGS_CURRICULUM,
        **GRASP_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": True,
        "octree_max_size": 60000,
    },
)
register(
    id="Grasp-OctreeWithColor-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspOctree,
        **GRASP_KWARGS,
        **GRASP_KWARGS_CURRICULUM,
        **GRASP_KWARGS_OCTREE,
        "octree_include_color": True,
        "octree_include_intensity": False,
        "octree_max_size": 75000,
    },
)
# Gazebo wrapper
register(
    id="Grasp-Gazebo-v0",
    entry_point=GRASP_RANDOMIZER,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Grasp-v0",
        **GRASP_KWARGS_SIM,
        **GRASP_KWARGS_RANDOMIZER,
        "camera_enable": False,
    },
)
register(
    id="Grasp-Octree-Gazebo-v0",
    entry_point=GRASP_RANDOMIZER,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Grasp-Octree-v0",
        **GRASP_KWARGS_SIM,
        **GRASP_KWARGS_RANDOMIZER,
        **GRASP_KWARGS_RANDOMIZER_CAMERA,
        "terrain_type": "flat",
        "camera_type": "depth_camera",
        "camera_publish_points": True,
    },
)
register(
    id="Grasp-OctreeWithIntensity-Gazebo-v0",
    entry_point=GRASP_RANDOMIZER,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Grasp-OctreeWithIntensity-v0",
        **GRASP_KWARGS_SIM,
        **GRASP_KWARGS_RANDOMIZER,
        **GRASP_KWARGS_RANDOMIZER_CAMERA,
        "terrain_type": "random_flat",
        "camera_type": "rgbd_camera",
        # "camera_image_format": "L8",
        "camera_publish_points": True,
    },
)
register(
    id="Grasp-OctreeWithColor-Gazebo-v0",
    entry_point=GRASP_RANDOMIZER,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Grasp-OctreeWithColor-v0",
        **GRASP_KWARGS_SIM,
        **GRASP_KWARGS_RANDOMIZER,
        **GRASP_KWARGS_RANDOMIZER_CAMERA,
        "terrain_type": "random_flat",
        "camera_type": "rgbd_camera",
        "camera_publish_points": True,
    },
)


##################
# GraspPlanetary #
##################
GRASP_PLANETARY_MAX_EPISODE_STEPS: int = 100
GRASP_PLANETARY_KWARGS: Dict[str, any] = {
    "agent_rate": 2.5,
    "robot_model": DRL_GRASPING_ROBOT_MODEL_MOBILE,
    "workspace_frame_id": "arm_base_link",
    "workspace_centre": (0.45, 0.0, 0.25),
    "workspace_volume": (0.35, 0.35, 0.6),
    "ignore_new_actions_while_executing": False,
    # "use_servo": True,
    # "scaling_factor_translation": 0.25,
    # "scaling_factor_rotation": np.pi / 3,
    # "full_3d_orientation": False,
    "use_servo": False,
    "scaling_factor_translation": 0.1,
    "scaling_factor_rotation": np.pi / 4,
    "full_3d_orientation": False,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "num_threads": 3,
}
GRASP_PLANETARY_KWARGS_DEPTH_IMAGE: Dict[str, any] = {
    "depth_max_distance": 1.0,
    "image_n_stacked": 2,
    "proprioceptive_observations": True,
}
GRASP_PLANETARY_KWARGS_OCTREE: Dict[str, any] = {
    "octree_reference_frame_id": "arm_base_link",
    # ## Large volume around the robot
    # "octree_min_bound": (0.1 - 0.6, 0.0 - 0.6, 0.0 - 0.6),
    # "octree_max_bound": (0.1 + 0.6, 0.0 + 0.6, 0.0 + 0.6),
    ## Front of robot
    "octree_min_bound": (
        0.5 - 0.2,
        0.0 - 0.2,
        0.14 - 0.2,
    ),
    "octree_max_bound": (
        0.5 + 0.2,
        0.0 + 0.2,
        0.14 + 0.2,
    ),
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 2,
    "proprioceptive_observations": True,
}
GRASP_PLANETARY_KWARGS_SIM: Dict[str, any] = {
    "physics_rate": 500.0,
    "real_time_factor": float(np.finfo(np.float32).max),
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "lunar.sdf"),
}
GRASP_PLANETARY_RANDOMIZER: str = (
    "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
)
GRASP_PLANETARY_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_MOON,
    "gravity_std": GRAVITY_MOON_STD,
    "plugin_scene_broadcaster": BROADCAST_GUI,
    "plugin_user_commands": BROADCAST_GUI,
    "plugin_sensors_render_engine": "ogre2",
    "robot_spawn_position": (0, 0, 0.1),
    "robot_spawn_quat_xyzw": (0, 0, 0, 1),
    "robot_random_pose": True,
    "robot_random_spawn_volume": (1.0, 1.0, 0),
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": np.pi / 20,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.15,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.1,
    "terrain_enable": True,
    "terrain_type": "random_lunar_surface",
    "terrain_spawn_position": (0, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (3.0, 3.0),
    "terrain_model_rollouts_num": 1,
    "light_type": "random_sun",
    "light_direction": (0.6, -0.4, -0.2),
    "light_random_minmax_elevation": (-0.1, -0.5),
    "light_distance": 1000.0,
    "light_visual": True,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "random_lunar_rock",
    "objects_relative_to": "arm_base_link",
    "object_count": 4,
    "object_spawn_position": (0.5, 0.0, 0.1),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [
        # (0.5, -0.01, 0.1),
        # (0.5, 0.01, 0.1),
        # (0.1, -0.6, 0.1),
        # (0.2, -0.5, 0.1),
        # (-0.1, -0.45, 0.1),
        # (0.4, -0.45, 0.1),
        # (0.4, 0.45, 0.1),
        # (-0.1, 0.45, 0.1),
    ],
    "object_random_spawn_position_update_workspace_centre": False,
    "object_random_spawn_volume": (0.25, 0.25, 0.1),
    "object_models_rollouts_num": 1,
    "underworld_collision_plane": True,
    "boundary_collision_walls": True,
}
GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 128,
    "camera_height": 128,
    "camera_update_rate": 4.0 * GRASP_PLANETARY_KWARGS["agent_rate"],
    "camera_horizontal_fov": np.pi / 2.0,
    "camera_vertical_fov": np.pi / 2.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "arm_base_link",
    # # Pole-mount
    # "camera_relative_to": "arm_base_link",
    # "camera_spawn_position": (-0.4, 0, 0.65),
    # "camera_spawn_quat_xyzw": (0, 0.258819, 0, 0.9659258),
    # # Bumper-mount
    # "camera_relative_to": "arm_base_link",
    # "camera_spawn_position": (0.05, 0, 0.15),
    # "camera_spawn_quat_xyzw": (0, 0.2164396, 0, 0.976296),
    # # End-effector
    # "camera_relative_to": "end_effector",
    # "camera_spawn_position": (0, 0.07, -0.05),
    # "camera_spawn_quat_xyzw": (0, -0.707107, 0, 0.707107),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_mode": "select_random",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": np.pi / 6,
    "camera_random_pose_select_position_options": [
        # (-0.4, 0.05, 0.65),
        # (-0.4, -0.05, 0.65),
        # (-0.45, 0, 0.65),
        # (-0.35, 0, 0.65),
        # (-0.45, 0.05, 0.65),
        # (-0.45, -0.05, 0.65),
        # (-0.35, 0.05, 0.65),
        # (-0.35, -0.05, 0.65),
        # (-0.1, 0.25, 0.2),
        # (-0.1, -0.25, 0.2),
        (0.13, 0, 0.11 - 0.04),
        (0.13, 0.05, 0.11 - 0.04),
        (0.13, -0.05, 0.11 - 0.04),
        (0.13, 0, 0.12 - 0.04),
        (0.13, 0.05, 0.12 - 0.04),
        (0.13, -0.05, 0.12 - 0.04),
        (0.13, 0, 0.1 - 0.04),
        (0.13, 0.05, 0.1 - 0.04),
        (0.13, -0.05, 0.1 - 0.04),
    ],
    "camera_random_pose_focal_point_z_offset": 0.0,
}
GRASP_PLANETARY_KWARGS_CURRICULUM: Dict[str, any] = {
    "stages_base_reward": 1.0,
    "reach_required_distance": 0.1,
    "lift_required_height": 0.075,
    "lift_required_height_max": 0.15,
    "lift_required_height_max_threshold": 0.33,
    "persistent_reward_each_step": -0.1,
    "persistent_reward_terrain_collision": 0.0,
    "persistent_reward_all_objects_outside_workspace": 0.0,
    "persistent_reward_arm_stuck": -0.001,
    "enable_stage_reward_curriculum": True,
    "enable_workspace_scale_curriculum": False,
    "enable_object_spawn_volume_scale_curriculum": False,
    "enable_object_count_curriculum": False,
    "stage_reward_multiplier": 8.0,
    "dense_reward": False,
    "initial_success_rate": 0.0,
    "rolling_average_n": 100,
    "min_workspace_scale": 0.5,
    "max_workspace_volume": GRASP_PLANETARY_KWARGS["workspace_volume"],
    "max_workspace_scale_success_rate_threshold": 0.33,
    "min_object_spawn_volume_scale": 0.5,
    "max_object_spawn_volume": GRASP_PLANETARY_KWARGS_RANDOMIZER[
        "object_random_spawn_volume"
    ],
    "max_object_spawn_volume_scale_success_rate_threshold": 0.33,
    "object_count_min": 1,
    "object_count_max": GRASP_PLANETARY_KWARGS_RANDOMIZER["object_count"],
    "max_object_count_success_rate_threshold": 0.33,
    "arm_stuck_n_steps": 20,
    "arm_stuck_min_joint_difference_norm": np.pi / 32,
}

# Task
register(
    id="GraspPlanetary-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetary,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
    },
)
register(
    id="GraspPlanetary-ColorImage-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryColorImage,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        "monochromatic": False,
    },
)
register(
    id="GraspPlanetary-MonoImage-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryColorImage,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        "monochromatic": True,
    },
)
register(
    id="GraspPlanetary-DepthImage-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryDepthImage,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_DEPTH_IMAGE,
        "image_include_color": False,
        "image_include_intensity": False,
    },
)
register(
    id="GraspPlanetary-DepthImageWithIntensity-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryDepthImage,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_DEPTH_IMAGE,
        "image_include_color": False,
        "image_include_intensity": True,
    },
)
register(
    id="GraspPlanetary-DepthImageWithColor-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryDepthImage,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_DEPTH_IMAGE,
        "image_include_color": True,
        "image_include_intensity": False,
    },
)
register(
    id="GraspPlanetary-Octree-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryOctree,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": False,
        "octree_max_size": 35000,
    },
)
register(
    id="GraspPlanetary-OctreeWithIntensity-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryOctree,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_OCTREE,
        "octree_include_color": False,
        "octree_include_intensity": True,
        "octree_max_size": 45000,
    },
)
register(
    id="GraspPlanetary-OctreeWithColor-v0",
    entry_point=DRL_GRASPING_TASK_ENTRYPOINT,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "task_cls": tasks.GraspPlanetaryOctree,
        **GRASP_PLANETARY_KWARGS,
        **GRASP_PLANETARY_KWARGS_CURRICULUM,
        **GRASP_PLANETARY_KWARGS_OCTREE,
        "octree_include_color": True,
        "octree_include_intensity": False,
        "octree_max_size": 60000,
    },
)
# Gazebo wrapper
register(
    id="GraspPlanetary-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        "camera_enable": False,
    },
)
register(
    id="GraspPlanetary-ColorImage-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-ColorImage-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_color": True,
    },
)
register(
    id="GraspPlanetary-MonoImage-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-MonoImage-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_color": True,
    },
)
register(
    id="GraspPlanetary-DepthImage-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-DepthImage-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "depth_camera",
        "camera_publish_depth": True,
    },
)
register(
    id="GraspPlanetary-DepthImageWithIntensity-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-DepthImageWithIntensity-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_depth": True,
        "camera_publish_color": True,
    },
)
register(
    id="GraspPlanetary-DepthImageWithColor-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-DepthImageWithColor-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_depth": True,
        "camera_publish_color": True,
    },
)
register(
    id="GraspPlanetary-Octree-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-Octree-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "depth_camera",
        "camera_publish_points": True,
    },
)
register(
    id="GraspPlanetary-OctreeWithIntensity-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-OctreeWithIntensity-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        # "camera_image_format": "L8",
        "camera_publish_points": True,
    },
)
register(
    id="GraspPlanetary-OctreeWithColor-Gazebo-v0",
    entry_point=GRASP_PLANETARY_RANDOMIZER,
    max_episode_steps=GRASP_PLANETARY_MAX_EPISODE_STEPS,
    kwargs={
        "env": "GraspPlanetary-OctreeWithColor-v0",
        **GRASP_PLANETARY_KWARGS_SIM,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER,
        **GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA,
        "camera_type": "rgbd_camera",
        "camera_publish_points": True,
    },
)
