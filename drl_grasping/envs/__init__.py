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


#########
# Reach #
#########
REACH_MAX_EPISODE_STEPS: int = 100
REACH_AGENT_RATE: float = 5.0
REACH_KWARGS: Dict = {
    "agent_rate": REACH_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "world",
    "workspace_centre": (0.45, 0, 0.25),
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
REACH_KWARGS_OCTREE: Dict = {
    "octree_reference_frame_id": "world",
    "octree_dimension": 0.5,
    "octree_depth": 3,
    "octree_full_depth": 2,
    "octree_n_stacked": 2,
}
REACH_KWARGS_SIM: Dict = {
    "physics_rate": 100.0,
    "real_time_factor": 20.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
REACH_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
REACH_KWARGS_RANDOMIZER: Dict = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_random_pose": False,
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.2,
    "terrain_enable": True,
    "object_enable": True,
    "object_type": "sphere",
    "objects_relative_to": "world",
    "object_static": True,
    "object_collision": False,
    "object_visual": True,
    "object_color": (0.0, 0.0, 1.0, 1.0),
    "object_dimensions": [0.025, 0.025, 0.025],
    "object_model_count": 1,
    "object_random_pose": True,
    "object_spawn_position": (0.45, 0, 0.25),
    "object_random_spawn_volume": (0.4, 0.4, 0.4),
    "object_models_rollouts_num": 0,
    "underworld_collision_plane": False,
}
REACH_KWARGS_RANDOMIZER_CAMERA: Dict = {
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
    "camera_random_pose_distance": 1.0,
    "camera_random_pose_height_range": (0.1, 0.7),
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
GRASP_AGENT_RATE: float = 5.0
GRASP_KWARGS: Dict = {
    "agent_rate": GRASP_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL,
    "workspace_frame_id": "world",
    "workspace_centre": (0.5, 0.0, 0.11),
    "workspace_volume": (0.24, 0.24, 0.24),
    "ignore_new_actions_while_executing": True,
    "use_servo": True,
    "scaling_factor_translation": 0.2,
    "scaling_factor_rotation": pi / 4.0,
    "restrict_position_goal_to_workspace": True,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "full_3d_orientation": False,
    "num_threads": 4,
}
GRASP_KWARGS_CURRICULUM: Dict = {
    "sparse_reward": True,
    "normalize_reward": False,
    "required_reach_distance": 0.1,
    "required_lift_height": 0.125,
    "reach_dense_reward_multiplier": 5.0,
    "lift_dense_reward_multiplier": 10.0,
    "act_quick_reward": -0.005,
    "outside_workspace_reward": 0.0,
    "ground_collision_reward": -1.0,
    "n_ground_collisions_till_termination": GRASP_MAX_EPISODE_STEPS,
    "curriculum_enable_workspace_scale": False,
    "curriculum_min_workspace_scale": 0.1,
    "curriculum_enable_object_count_increase": False,
    "curriculum_max_object_count": 4,
    "curriculum_enable_stages": False,
    "curriculum_stage_reward_multiplier": 7.0,
    "curriculum_stage_increase_rewards": True,
    "curriculum_success_rate_threshold": 0.6,
    "curriculum_success_rate_rolling_average_n": 100,
    "curriculum_restart_every_n_steps": 0,
    "curriculum_skip_reach_stage": False,
    "curriculum_skip_grasp_stage": True,
    "curriculum_restart_exploration_at_start": False,
    "max_episode_length": GRASP_MAX_EPISODE_STEPS,
}
GRASP_KWARGS_OCTREE: Dict = {
    "octree_reference_frame_id": "world",
    "octree_dimension": 0.24,
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 3,
    "proprieceptive_observations": True,
}
GRASP_KWARGS_SIM: Dict = {
    "physics_rate": 250.0,
    "real_time_factor": 15.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "default.sdf"),
}
GRASP_RANDOMIZER: str = "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
GRASP_KWARGS_RANDOMIZER: Dict = {
    "gravity": GRAVITY_EARTH,
    "gravity_std": GRAVITY_EARTH_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_random_pose": False,
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.2,
    "terrain_enable": True,
    "terrain_spawn_position": (0.25, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (1.25, 1.25),
    "object_enable": True,
    "object_type": "random_mesh",
    "objects_relative_to": "world",
    "object_model_count": 4,
    "object_random_pose": True,
    "object_spawn_position": (0.5, 0.0, 0.15),
    "object_random_spawn_volume": (0.18, 0.18, 0.075),
    "object_models_rollouts_num": 1,
    "underworld_collision_plane": True,
}

GRASP_KWARGS_RANDOMIZER_CAMERA: Dict = {
    "camera_enable": True,
    "camera_width": 256,
    "camera_height": 256,
    "camera_update_rate": 1.2 * GRASP_AGENT_RATE,
    "camera_horizontal_fov": pi / 3.0,
    "camera_vertical_fov": pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    "camera_relative_to": "world",
    "camera_spawn_position": (0.95, -0.55, 0.25),
    "camera_spawn_quat_xyzw": (-0.0402991, -0.0166924, 0.9230002, 0.3823192),
    "camera_random_pose_rollouts_num": 1,
    "camera_random_pose_distance": 1.0,
    "camera_random_pose_height_range": (0.1, 0.7),
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
        "terrain_type": "flat",
        "terrain_model_rollouts_num": 0,
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
        "terrain_type": "random_flat",
        "terrain_model_rollouts_num": 1,
        "camera_type": "rgbd_camera",
        "camera_publish_points": True,
    },
)


##################
# GraspPlanetary #
##################
GRASP_PLANETARY_MAX_EPISODE_STEPS: int = 50
GRASP_PLANETARY_AGENT_RATE: float = 4.0
GRASP_PLANETARY_KWARGS: Dict = {
    "agent_rate": GRASP_PLANETARY_AGENT_RATE,
    "robot_model": DRL_GRASPING_ROBOT_MODEL_MOBILE,
    "workspace_frame_id": "base_link",
    "workspace_centre": (0.0, 0.0, 0.0),
    "workspace_volume": (200.0, 200.0, 200.0),
    "ignore_new_actions_while_executing": True,
    "use_servo": True,
    "scaling_factor_translation": 0.5,
    "scaling_factor_rotation": pi / 3,
    "restrict_position_goal_to_workspace": False,
    "enable_gripper": True,
    "gripper_dead_zone": 0.0,
    "full_3d_orientation": False,
    "num_threads": 4,
}
GRASP_PLANETARY_KWARGS_CURRICULUM: Dict = {
    "sparse_reward": True,
    "normalize_reward": False,
    "required_reach_distance": 0.1,
    "required_lift_height": 0.125,
    "reach_dense_reward_multiplier": 5.0,
    "lift_dense_reward_multiplier": 10.0,
    "act_quick_reward": -0.005,
    "outside_workspace_reward": 0.0,
    "ground_collision_reward": -1.0,
    "n_ground_collisions_till_termination": GRASP_PLANETARY_MAX_EPISODE_STEPS,
    "curriculum_enable_workspace_scale": False,
    "curriculum_min_workspace_scale": 0.1,
    "curriculum_enable_object_count_increase": False,
    "curriculum_max_object_count": 4,
    "curriculum_enable_stages": False,
    "curriculum_stage_reward_multiplier": 7.0,
    "curriculum_stage_increase_rewards": True,
    "curriculum_success_rate_threshold": 0.6,
    "curriculum_success_rate_rolling_average_n": 100,
    "curriculum_restart_every_n_steps": 0,
    "curriculum_skip_reach_stage": False,
    "curriculum_skip_grasp_stage": True,
    "curriculum_restart_exploration_at_start": False,
    "max_episode_length": GRASP_PLANETARY_MAX_EPISODE_STEPS,
}
GRASP_PLANETARY_KWARGS_OCTREE: Dict = {
    "octree_reference_frame_id": "base_link",
    "octree_dimension": 2.0,
    "octree_depth": 4,
    "octree_full_depth": 2,
    "octree_n_stacked": 2,
    "proprieceptive_observations": True,
}
GRASP_PLANETARY_KWARGS_SIM: Dict = {
    "physics_rate": 250.0,
    "real_time_factor": 10.0,
    "world": path.join(DRL_GRASPING_WORLDS_DIR, "lunar.sdf"),
}
GRASP_PLANETARY_RANDOMIZER: str = (
    "drl_grasping.envs.randomizers:ManipulationGazeboEnvRandomizer"
)
GRASP_PLANETARY_KWARGS_RANDOMIZER: Dict = {
    "gravity": GRAVITY_MOON,
    "gravity_std": GRAVITY_MOON_STD,
    "plugin_scene_broadcaster": True,
    "plugin_user_commands": True,
    "plugin_sensors_render_engine": "ogre2",
    "robot_spawn_position": (0, 0, 1.0),
    "robot_spawn_quat_xyzw": (0, 0, 0, 1),
    "robot_random_pose": True,
    "robot_random_spawn_volume": (30.0, 30.0, 0.0),
    "robot_random_joint_positions": True,
    "robot_random_joint_positions_std": 0.1,
    "robot_random_joint_positions_above_object_spawn": False,
    "robot_random_joint_positions_above_object_spawn_elevation": 0.1,
    "terrain_enable": True,
    "terrain_type": "lunar_surface",
    # "terrain_type": "lunar_heightmap",
    "terrain_spawn_position": (0.25, 0, 0),
    "terrain_spawn_quat_xyzw": (0, 0, 0, 1),
    "terrain_size": (1.25, 1.25),
    "terrain_model_rollouts_num": 1,
    "light_type": "random_sun",
    "light_random_minmax_elevation": (-0.1, -0.5),
    "light_distance": 1000.0,
    "light_visual": True,
    "light_radius": 25.0,
    "light_model_rollouts_num": 1,
    "object_enable": True,
    "object_type": "rock",
    "objects_relative_to": "base_link",
    "object_model_count": 4,
    "object_random_pose": True,
    "object_spawn_position": (1.2, 0.0, 0.2),
    "object_random_spawn_volume": (0.5, 0.5, 0.1),
    "object_models_rollouts_num": 1,
    "underworld_collision_plane": True,
}

GRASP_PLANETARY_KWARGS_RANDOMIZER_CAMERA: Dict = {
    "camera_enable": True,
    "camera_width": 256,
    "camera_height": 256,
    "camera_update_rate": 1.2 * GRASP_PLANETARY_AGENT_RATE,
    "camera_horizontal_fov": pi / 3.0,
    "camera_vertical_fov": pi / 3.0,
    "camera_noise_mean": 0.0,
    "camera_noise_stddev": 0.001,
    # # Above robot
    # "camera_relative_to": "base_link",
    # "camera_spawn_position": (0, 0, 1),
    # "camera_spawn_quat_xyzw": (0, 0.707107, 0, 0.707107),
    # # Pole-mount
    # "camera_relative_to": "base_link",
    # "camera_spawn_position": (-0.2, 0, 0.75),
    # "camera_spawn_quat_xyzw": (0, 0.258819, 0, 0.9659258),
    # Bumper-mount
    "camera_relative_to": "base_link",
    "camera_spawn_position": (0.37, 0, 0.25),
    "camera_spawn_quat_xyzw": (0, 0.2164396, 0, 0.976296),
    # # End-effector
    # "camera_relative_to": "end_effector",
    # "camera_spawn_position": (0, 0.07, -0.05),
    # "camera_spawn_quat_xyzw": (0, -0.707107, 0, 0.707107),
    "camera_random_pose_rollouts_num": 1,
    # "camera_random_pose_distance": 1.0,
    # "camera_random_pose_height_range": (0.1, 0.7),
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
