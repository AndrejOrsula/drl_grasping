# Note: These two modules must be imported prior to gym_ignition (open3d and stable_baselines3)
import open3d  # isort:skip
import stable_baselines3  # isort:skip

from os import path
from typing import Dict, Tuple

from ament_index_python.packages import get_package_share_directory
from gym.envs.registration import register
from numpy import pi

from . import randomizers, tasks

######################
# Generic padameters #
######################
# Entrypoint for tasks (can be simulated or real)
DRL_GRASPING_TASK_ENTRYPOINT: str = "gym_ignition.runtimes.gazebo_runtime:GazeboRuntime"
# Robot model to use in the tasks where robot can be static
DRL_GRASPING_ROBOT_MODEL: str = "panda"
# Robot model to use in the tasks where robot needs to be mobile
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

# Offset of "lunalab_summit_xl_gen" from `base_arm_link` to it base footprint
LUNALAB_SUMMIT_XL_GEN_Z_OFFSET: float = -0.22

#########
# Reach #
#########
REACH_MAX_EPISODE_STEPS: int = 50
REACH_AGENT_RATE: float = 4.0
REACH_KWARGS: Dict[str, any] = {
    "agent_rate": REACH_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "world",
    "workspace_centre": (0.45, 0.0, 0.25),
    "workspace_volume": (0.5, 0.5, 0.5),
    "ignore_new_actions_while_executing": True,
    "use_servo": True,
    "scaling_factor_translation": 0.2,
    "scaling_factor_rotation": pi / 4.0,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": False,
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
    "physics_rate": 100.0,
    "real_time_factor": 20.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
REACH_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
REACH_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_random_pose": False,
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.0,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.2,
    "terrain_enable": True,
    "object_enable": True,
    "object_type": "sphere",
    "objects_relative_to": "world",
    "object_static": True,
    "object_collision": False,
    "object_visual": True,
    "object_color": (0.0, 0.0, 1.0, 1.0),
    "object_dimensions": [0.025, 0.025, 0.025],
    "object_count": 1,
    "object_spawn_position": (0.45, 0, 0.25),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [],
    "object_random_spawn_volume": (0.4, 0.4, 0.4),
    "object_models_rollouts_num": 0,
    "underworld_collision_plane": False,
}
REACH_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 128,
    "camera_height": 128,
    "camera_update_rate": 1.2 * REACH_AGENT_RATE,
    "camera_horizontal_fov": pi / 3.0,
    "camera_vertical_fov": pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "world",
    "camera_spawn_position": (1.1, -0.75, 0.45),
    "camera_spawn_quat_xyzw": (-0.0402991, -0.0166924, 0.9230002, 0.3823192),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_mode": "orbit",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": pi / 8,
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
        "octree_max_size": 20000,
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
        "camera_type": "camera",
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
GRASP_AGENT_RATE: float = 4.0
GRASP_KWARGS: Dict[str, any] = {
    "agent_rate": GRASP_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "arm_base_link",
    "workspace_centre": (0.5, 0.0, 0.11),
    "workspace_volume": (0.24, 0.24, 0.24),
    "ignore_new_actions_while_executing": True,
    "use_servo": True,
    "scaling_factor_translation": 0.5,
    "scaling_factor_rotation": pi / 3,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "full_3d_orientation": False,
    "num_threads": 4,
}
GRASP_KWARGS_OCTREE: Dict[str, any] = {
    "octree_reference_frame_id": "arm_base_link",
    "octree_min_bound": (
        0.5 - 0.12,
        0.0 - 0.12,
        LUNALAB_SUMMIT_XL_GEN_Z_OFFSET + 0.11 - 0.12,
    ),
    "octree_max_bound": (
        0.5 + 0.12,
        0.0 + 0.12,
        LUNALAB_SUMMIT_XL_GEN_Z_OFFSET + 0.11 + 0.12,
    ),
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 3,
    "proprieceptive_observations": True,
}
GRASP_KWARGS_SIM: Dict[str, any] = {
    "physics_rate": 200.0,
    "real_time_factor": 15.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
GRASP_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
GRASP_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_spawn_position": (0, 0, 0),
    "robot_spawn_quat_xyzw": (0, 0, 0, 1),
    "robot_random_pose": False,
    "robot_random_spawn_volume": (0, 0, 0),
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": True,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.1,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.2,
    "terrain_enable": True,
    "terrain_type": "flat",
    "terrain_spawn_position": (0.25, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (1.5, 1.5),
    "terrain_model_rollouts_num": 1,
    "light_type": "sun",
    "light_direction": (0.6, -0.4, -0.2),
    "light_random_minmax_elevation": (-0.1, -0.5),
    "light_distance": 1000.0,
    "light_visual": True,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "random_mesh",
    "objects_relative_to": "arm_base_link",
    "object_count": 4,
    "object_spawn_position": (0.5, 0.0, 0.1),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [],
    "object_random_spawn_volume": (0.18, 0.18, 0.075),
    "object_models_rollouts_num": 1,
    "underworld_collision_plane": True,
}
GRASP_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 256,
    "camera_height": 256,
    "camera_update_rate": 1.2 * GRASP_AGENT_RATE,
    "camera_horizontal_fov": pi / 3.0,
    "camera_vertical_fov": pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "arm_base_link",
    "camera_spawn_position": (0.95, -0.55, 0.25),
    "camera_spawn_quat_xyzw": (-0.0402991, -0.0166924, 0.9230002, 0.3823192),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_mode": "orbit",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": pi / 6,
    "camera_random_pose_select_position_options": [],
    "camera_random_pose_focal_point_z_offset": LUNALAB_SUMMIT_XL_GEN_Z_OFFSET,
}
GRASP_KWARGS_CURRICULUM: Dict[str, any] = {
    "stages_base_reward": 1.0,
    "reach_required_distance": 0.1,
    "lift_required_height": 0.125,
    "persistent_reward_each_step": -0.005,
    "persistent_reward_terrain_collision": -1.0,
    "persistent_reward_all_objects_outside_workspace": 0.0,
    "enable_workspace_scale_curriculum": False,
    "enable_stage_reward_curriculum": True,
    "enable_object_spawn_volume_scale_curriculum": False,
    "enable_object_count_curriculum": True,
    "stage_reward_multiplier": 7.0,
    "dense_reward": False,
    "initial_success_rate": 0.0,
    "rolling_average_n": 100,
    "min_workspace_scale": 0.1,
    "max_workspace_volume": GRASP_KWARGS["workspace_volume"],
    "max_workspace_scale_success_rate_threshold": 0.6,
    "min_object_spawn_volume_scale": 0.1,
    "max_object_spawn_volume": GRASP_KWARGS_RANDOMIZER["object_random_spawn_volume"],
    "max_object_spawn_volume_scale_success_rate_threshold": 0.6,
    "object_count_min": 1,
    "object_count_max": GRASP_KWARGS_RANDOMIZER["object_count"],
    "max_object_count_success_rate_threshold": 0.6,
    "enable_logger_sb3": True,
}

# Task
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
        "octree_max_size": 50000,
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
        "octree_max_size": 75000,
    },
)
# Gazebo wrapper
register(
    id="Grasp-Octree-Gazebo-v0",
    entry_point=GRASP_RANDOMIZER,
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={
        "env": "Grasp-Octree-v0",
        **GRASP_KWARGS_SIM,
        **GRASP_KWARGS_RANDOMIZER,
        **GRASP_KWARGS_RANDOMIZER_CAMERA,
        # "terrain_type": "flat",
        "camera_type": "depth_camera",
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
        # "terrain_type": "random_flat",
        "camera_type": "rgbd_camera",
        "camera_publish_points": True,
    },
)


##################
# GraspPlanetary #
##################
GRASP_PLANETARY_MAX_EPISODE_STEPS: int = 100
GRASP_PLANETARY_AGENT_RATE: float = 4.0
GRASP_PLANETARY_KWARGS: Dict[str, any] = {
    "agent_rate": GRASP_PLANETARY_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL_MOBILE,
    "workspace_frame_id": "arm_base_link",
    "workspace_centre": (0.0, 0.0, 0.0),
    "workspace_volume": (2.0, 2.0, 2.0),
    "ignore_new_actions_while_executing": True,
    "use_servo": True,
    "scaling_factor_translation": 0.5,
    "scaling_factor_rotation": pi / 3,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "full_3d_orientation": False,
    "num_threads": 4,
}
GRASP_PLANETARY_KWARGS_OCTREE: Dict[str, any] = {
    "octree_reference_frame_id": "arm_base_link",
    "octree_min_bound": (0.1 - 0.6, 0.0 - 0.6, 0.0 - 0.6),
    "octree_max_bound": (0.1 + 0.6, 0.0 + 0.6, 0.0 + 0.6),
    # "octree_min_bound": (0.1 - 0.6, 0.0 - 0.6, -0.1 - 0.3),
    # "octree_max_bound": (0.1 + 0.6, 0.0 + 0.6, -0.1 + 0.3),
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 3,
    "proprieceptive_observations": True,
}
GRASP_PLANETARY_KWARGS_SIM: Dict[str, any] = {
    "physics_rate": 200.0,
    "real_time_factor": 10.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "lunar.sdf"),
}
GRASP_PLANETARY_RANDOMIZER: str = (
    "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
)
GRASP_PLANETARY_KWARGS_RANDOMIZER: Dict[str, any] = {
    "gravity": GRAVITY_MOON,
    "gravity_std": GRAVITY_MOON_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_spawn_position": (0, 0, 1),
    "robot_spawn_quat_xyzw": (0, 0, 0, 1),
    "robot_random_pose": True,
    "robot_random_spawn_volume": (20.0, 20.0, 0),
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": True,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.1,
    "robot_random_joint_positions_above_object_spawn_xy_randomness": 0.2,
    "terrain_enable": True,
    "terrain_type": "lunar_surface",
    "terrain_spawn_position": (0, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (20, 20),
    "terrain_model_rollouts_num": 1,
    "light_type": "random_sun",
    "light_direction": (0.6, -0.4, -0.2),
    "light_random_minmax_elevation": (-0.1, -0.5),
    "light_distance": 1000.0,
    "light_visual": True,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "rock",
    "objects_relative_to": "arm_base_link",
    "object_count": 1,
    "object_spawn_position": (1.2, 0.0, 0.2),
    "object_random_pose": True,
    "object_random_spawn_position_segments": [
        (-0.2, -0.45, -0.05),
        (0.4, -0.45, -0.05),
        (0.4, 0.45, -0.05),
        (-0.2, 0.45, -0.05),
    ],
    "object_random_spawn_volume": (0.25, 0.25, 0.05),
    "object_models_rollouts_num": 1,
    "underworld_collision_plane": True,
}
GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA: Dict[str, any] = {
    "camera_enable": True,
    "camera_width": 256,
    "camera_height": 256,
    "camera_update_rate": 1.2 * GRASP_PLANETARY_AGENT_RATE,
    "camera_horizontal_fov": pi / 3.0,
    "camera_vertical_fov": pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "base_link",
    # # Above robot
    # "camera_relative_to": "base_link",
    # "camera_spawn_position": (0, 0, 1),
    # "camera_spawn_quat_xyzw": (0, 0.707107, 0, 0.707107),
    # # Pole-mount
    # "camera_relative_to": "base_link",
    # "camera_spawn_position": (-0.2, 0, 0.75),
    # "camera_spawn_quat_xyzw": (0, 0.258819, 0, 0.9659258),
    # # Bumper-mount
    # "camera_relative_to": "base_link",
    # "camera_spawn_position": (0.37, 0, 0.25),
    # "camera_spawn_quat_xyzw": (0, 0.2164396, 0, 0.976296),
    # # End-effector
    # "camera_relative_to": "end_effector",
    # "camera_spawn_position": (0, 0.07, -0.05),
    # "camera_spawn_quat_xyzw": (0, -0.707107, 0, 0.707107),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_mode": "select_nearest",
    "camera_random_pose_orbit_distance": 1.0,
    "camera_random_pose_orbit_height_range": (0.1, 0.7),
    "camera_random_pose_orbit_ignore_arc_behind_robot": pi / 6,
    "camera_random_pose_select_position_options": [
        (-0.2, 0, 0.75),
        (0.37, 0, 0.25),
        (0.1, 0.25, 0.3),
        (0.1, -0.25, 0.3),
    ],
    "camera_random_pose_focal_point_z_offset": LUNALAB_SUMMIT_XL_GEN_Z_OFFSET,
}
GRASP_PLANETARY_KWARGS_CURRICULUM: Dict[str, any] = {
    "stages_base_reward": 1.0,
    "reach_required_distance": 0.1,
    "lift_required_height": 0.125,
    "persistent_reward_each_step": -0.005,
    "persistent_reward_terrain_collision": -1.0,
    "persistent_reward_all_objects_outside_workspace": 0.0,
    "enable_workspace_scale_curriculum": False,
    "enable_stage_reward_curriculum": True,
    "enable_object_spawn_volume_scale_curriculum": False,
    "enable_object_count_curriculum": True,
    "stage_reward_multiplier": 7.0,
    "dense_reward": False,
    "initial_success_rate": 0.0,
    "rolling_average_n": 100,
    "min_workspace_scale": 0.1,
    "max_workspace_volume": GRASP_PLANETARY_KWARGS["workspace_volume"],
    "max_workspace_scale_success_rate_threshold": 0.6,
    "min_object_spawn_volume_scale": 0.1,
    "max_object_spawn_volume": GRASP_PLANETARY_KWARGS_RANDOMIZER[
        "object_random_spawn_volume"
    ],
    "max_object_spawn_volume_scale_success_rate_threshold": 0.6,
    "object_count_min": 1,
    "object_count_max": GRASP_PLANETARY_KWARGS_RANDOMIZER["object_count"],
    "max_object_count_success_rate_threshold": 0.6,
    "enable_logger_sb3": True,
}

# Task
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
        "octree_max_size": 100000,
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
        "octree_max_size": 150000,
    },
)
# Gazebo wrapper
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
