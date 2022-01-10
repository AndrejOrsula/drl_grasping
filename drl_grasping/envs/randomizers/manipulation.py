import abc
from os import environ
from typing import List, Tuple, Union

import numpy as np
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from scenario import gazebo as scenario
from scipy.spatial import distance
from scipy.spatial.transform import Rotation

from drl_grasping.envs import models, tasks
from drl_grasping.envs.utils.conversions import quat_to_wxyz
from drl_grasping.envs.utils.gazebo import *
from drl_grasping.envs.utils.logging import set_log_level

# Tasks that are supported by this randomizer (used primarily for type hinting)
SupportedTasks = Union[
    tasks.Reach,
    tasks.ReachColorImage,
    tasks.ReachDepthImage,
    tasks.ReachOctree,
    tasks.Grasp,
    tasks.GraspOctree,
    tasks.GraspPlanetary,
    tasks.GraspPlanetaryOctree,
]

# TODO: Gazebo run sometimes causes crash (e.g. some cases with `random_flat` terrain) - Investigate why


class ManipulationGazeboEnvRandomizer(
    gazebo_env_randomizer.GazeboEnvRandomizer,
    randomizers.abc.PhysicsRandomizer,
    randomizers.abc.TaskRandomizer,
    abc.ABC,
):
    """
    Basic randomizer of environments for robotic manipulation inside Ignition Gazebo. This randomizer
    also populates the simulated world with robot, terrain, lighting and other entities.
    """

    POST_RANDOMIZATION_MAX_STEPS = 50

    def __init__(
        self,
        env: MakeEnvCallable,
        # Physics
        physics_rollouts_num: int = 0,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.80665),
        gravity_std: Tuple[float, float, float] = (0.0, 0.0, 0.0232),
        # World plugins
        plugin_scene_broadcaster: bool = False,
        plugin_user_commands: bool = False,
        plugin_sensors_render_engine: str = "ogre2",
        # Robot
        robot_spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        robot_spawn_quat_xyzw: Tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        robot_random_pose: bool = False,
        robot_random_spawn_volume: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        robot_random_joint_positions: bool = False,
        robot_random_joint_positions_std: float = 0.1,
        robot_random_joint_positions_above_object_spawn: bool = False,
        robot_random_joint_positions_above_object_spawn_elevation: float = 0.2,
        robot_random_joint_positions_above_object_spawn_xy_randomness: float = 0.2,
        # Camera #
        camera_enable: bool = True,
        camera_type: str = "rgbd_camera",
        camera_relative_to: str = "base_link",
        camera_width: int = 128,
        camera_height: int = 128,
        camera_update_rate: int = 10,
        camera_horizontal_fov: float = np.pi / 3.0,
        camera_vertical_fov: float = np.pi / 3.0,
        camera_clip_color: Tuple[float, float] = (0.01, 1000.0),
        camera_clip_depth: Tuple[float, float] = (0.05, 10.0),
        camera_noise_mean: float = None,
        camera_noise_stddev: float = None,
        camera_publish_color: bool = False,
        camera_publish_depth: bool = False,
        camera_publish_points: bool = False,
        # Note: Camera pose is with respect to the pose of `camera_relative_to` link (or world)
        camera_spawn_position: Tuple[float, float, float] = (0, 0, 1),
        camera_spawn_quat_xyzw: Tuple[float, float, float, float] = (
            0,
            0.70710678118,
            0,
            0.70710678118,
        ),
        camera_random_pose_rollouts_num: int = 1,
        camera_random_pose_mode: str = "orbit",
        camera_random_pose_orbit_distance: float = 1.0,
        camera_random_pose_orbit_height_range: Tuple[float, float] = (0.1, 0.7),
        camera_random_pose_orbit_ignore_arc_behind_robot: float = np.pi / 8,
        camera_random_pose_select_position_options: List[
            Tuple[float, float, float]
        ] = [],
        camera_random_pose_focal_point_z_offset: float = -0.22,
        # Terrain
        terrain_enable: bool = True,
        terrain_type: str = "flat",
        terrain_spawn_position: Tuple[float, float, float] = (0, 0, 0),
        terrain_spawn_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1),
        terrain_size: Tuple[float, float] = (1.0, 1.0),
        terrain_model_rollouts_num: int = 1,
        # Light
        light_enable: bool = True,
        light_type: str = "sun",
        light_direction: Tuple[float, float, float] = (0.5, -0.25, -0.75),
        light_random_minmax_elevation: Tuple[float, float] = (-0.15, -0.65),
        light_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        light_distance: float = 1000.0,
        light_visual: bool = True,
        light_radius: float = 25.0,
        light_model_rollouts_num: int = 1,
        # Objects
        object_enable: bool = True,
        object_type: str = "box",
        objects_relative_to: str = "base_link",
        object_static: bool = False,
        object_collision: bool = True,
        object_visual: bool = True,
        object_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        object_dimensions: List[float] = [0.05, 0.05, 0.05],
        object_mass: float = 0.1,
        object_count: int = 1,
        object_spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        object_random_pose: bool = True,
        object_random_spawn_position_segments: List[Tuple[float, float, float]] = [],
        object_random_spawn_volume: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        object_models_rollouts_num: int = 1,
        # Collision plane below terrain
        underworld_collision_plane: bool = True,
        underworld_collision_plane_depth: float = -1.0,
        # Visual debugging
        visualise_workspace: bool = False,
        visualise_spawn_volume: bool = False,
        **kwargs,
    ):

        # TODO (a lot of work): Implement proper physics randomization.
        if physics_rollouts_num != 0:
            raise TypeError(
                "Proper physics randomization at each reset is not yet implemented. Please set `physics_rollouts_num=0`."
            )

        # Update kwargs before passing them to the task constructor (some tasks might need them)
        kwargs.update(
            {
                "camera_type": camera_type,
                "camera_width": camera_width,
                "camera_height": camera_height,
            }
        )

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=physics_rollouts_num
        )
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(
            self, env=env, physics_randomizer=self, **kwargs
        )

        # Store parameters for later use #
        # Physics
        self._gravity = gravity
        self._gravity_std = gravity_std

        # World plugins
        self._plugin_scene_broadcaster = plugin_scene_broadcaster
        self._plugin_user_commands = plugin_user_commands
        self._plugin_sensors_render_engine = plugin_sensors_render_engine

        # Robot
        self._robot_spawn_position = robot_spawn_position
        self._robot_spawn_quat_xyzw = robot_spawn_quat_xyzw
        self._robot_random_pose = robot_random_pose
        self._robot_random_spawn_volume = robot_random_spawn_volume
        self._robot_random_joint_positions = robot_random_joint_positions
        self._robot_random_joint_positions_std = robot_random_joint_positions_std
        self._robot_random_joint_positions_above_object_spawn = (
            robot_random_joint_positions_above_object_spawn
        )
        self._robot_random_joint_positions_above_object_spawn_elevation = (
            robot_random_joint_positions_above_object_spawn_elevation
        )
        self._robot_random_joint_positions_above_object_spawn_xy_randomness = (
            robot_random_joint_positions_above_object_spawn_xy_randomness
        )

        # Camera
        self._camera_enable = camera_enable
        self._camera_type = camera_type
        self._camera_relative_to = camera_relative_to
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_update_rate = camera_update_rate
        self._camera_horizontal_fov = camera_horizontal_fov
        self._camera_vertical_fov = camera_vertical_fov
        self._camera_clip_color = camera_clip_color
        self._camera_clip_depth = camera_clip_depth
        self._camera_noise_mean = camera_noise_mean
        self._camera_noise_stddev = camera_noise_stddev
        self._camera_publish_color = camera_publish_color
        self._camera_publish_depth = camera_publish_depth
        self._camera_publish_points = camera_publish_points
        self._camera_spawn_position = camera_spawn_position
        self._camera_spawn_quat_xyzw = camera_spawn_quat_xyzw
        self._camera_random_pose_rollouts_num = camera_random_pose_rollouts_num
        self._camera_random_pose_mode = camera_random_pose_mode
        self._camera_random_pose_orbit_distance = camera_random_pose_orbit_distance
        self._camera_random_pose_orbit_height_range = (
            camera_random_pose_orbit_height_range
        )
        self._camera_random_pose_orbit_ignore_arc_behind_robot = (
            camera_random_pose_orbit_ignore_arc_behind_robot
        )
        self._camera_random_pose_select_position_options = (
            camera_random_pose_select_position_options
        )
        self._camera_random_pose_focal_point_z_offset = (
            camera_random_pose_focal_point_z_offset
        )

        # Terrain
        self._terrain_enable = terrain_enable
        self._terrain_spawn_position = terrain_spawn_position
        self._terrain_spawn_quat_xyzw = terrain_spawn_quat_xyzw
        self._terrain_size = terrain_size
        self._terrain_model_rollouts_num = terrain_model_rollouts_num

        # Light
        self._light_enable = light_enable
        self._light_direction = light_direction
        self._light_random_minmax_elevation = light_random_minmax_elevation
        self._light_color = light_color
        self._light_distance = light_distance
        self._light_visual = light_visual
        self._light_radius = light_radius
        self._light_model_rollouts_num = light_model_rollouts_num

        # Objects
        self._object_enable = object_enable
        self._objects_relative_to = objects_relative_to
        self._object_static = object_static
        self._object_collision = object_collision
        self._object_visual = object_visual
        self._object_color = object_color
        self._object_dimensions = object_dimensions
        self._object_mass = object_mass
        self._object_count = object_count
        self._object_spawn_position = object_spawn_position
        self._object_random_pose = object_random_pose
        self._object_random_spawn_position_segments = (
            object_random_spawn_position_segments
        )
        self._object_random_spawn_volume = object_random_spawn_volume
        self._object_models_rollouts_num = object_models_rollouts_num

        # Collision plane beneath the terrain (prevent objects from falling forever)
        self._underworld_collision_plane = underworld_collision_plane
        self._underworld_collision_plane_depth = underworld_collision_plane_depth

        # Visual debugging
        self._visualise_workspace = visualise_workspace
        self._visualise_spawn_volume = visualise_spawn_volume

        # Derived variables #
        # Model classes and whether these are randomizable
        self.__terrain_model_class = models.get_terrain_model_class(terrain_type)
        self.__is_terrain_type_randomizable = models.is_terrain_type_randomizable(
            terrain_type
        )
        self.__light_model_class = models.get_light_model_class(light_type)
        self.__is_light_type_randomizable = models.is_light_type_randomizable(
            light_type
        )
        self.__object_model_class = models.get_object_model_class(object_type)
        self.__is_object_type_randomizable = models.is_object_type_randomizable(
            object_type
        )

        # Variable initialisation #
        # Rollout counters
        self.__camera_pose_rollout_counter = camera_random_pose_rollouts_num
        self.__terrain_model_rollout_counter = terrain_model_rollouts_num
        self.__light_model_rollout_counter = light_model_rollouts_num
        self.__object_models_rollout_counter = object_models_rollouts_num

        # Flag that determines whether the camera is attached to the robot via detachable joint
        self.__is_camera_attached = False

        # Flag that determines whether environment has already been initialised
        self.__env_initialised = False

        # Dict to keep track of set object positions - without stepping (faster than lookup through gazebo)
        # It is used to make sure that objects are not spawned inside each other
        self.__object_positions = {}

    ##########################
    # PhysicsRandomizer impl #
    ##########################

    def init_physics_preset(self, task: SupportedTasks):

        self.set_gravity(task=task)

    def randomize_physics(self, task: SupportedTasks, **kwargs):

        self.set_gravity(task=task)

    def set_gravity(self, task: SupportedTasks):

        if not task.world.to_gazebo().set_gravity(
            (
                task.np_random.normal(loc=self._gravity[0], scale=self._gravity_std[0]),
                task.np_random.normal(loc=self._gravity[1], scale=self._gravity_std[1]),
                task.np_random.normal(loc=self._gravity[2], scale=self._gravity_std[2]),
            )
        ):
            raise RuntimeError("Failed to set the gravity")

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    #######################
    # TaskRandomizer impl #
    #######################

    def randomize_task(self, task: SupportedTasks, **kwargs):
        """
        Randomization of the task, which is called on each reset of the environment
        """

        # Get gazebo instance associated with the task
        if "gazebo" not in kwargs:
            raise ValueError("Randomizer does not have access to the gazebo interface")
        gazebo = kwargs["gazebo"]

        # Perform external overrides (e.g. from curriculum)
        self.external_overrides(task=task)

        # Initialise the environment on the first iteration
        if not self.__env_initialised:
            self.init_env(task=task, gazebo=gazebo)
            self.__env_initialised = True

        # Perform pre-randomization steps
        self.pre_randomization(task=task)

        # Randomize models if needed
        self.randomize_models(task=task, gazebo=gazebo)

        # Perform post-randomization steps
        self.post_randomization(task, gazebo)

    ###################
    # Randomizer impl #
    ###################

    # Initialisation #
    def init_env(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Initialise an instance of the environment before the very first iteration
        """

        # Set log level for (Gym) Ignition
        set_log_level(log_level=task.get_logger().get_effective_level().name)

        # Execute a paused run for the first time before initialising everything
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Substitute frame names if needed
        self._camera_relative_to = task.substitute_special_frame(
            self._camera_relative_to
        )
        self._objects_relative_to = task.substitute_special_frame(
            self._objects_relative_to
        )

        # Initialise custom physics preset
        self.init_physics_preset(task=task)

        # Insert world plugins needed by the task or selected by user
        self.init_world_plugins(task=task, gazebo=gazebo)

        # Initialise all models that are persustent throughout the entire training
        self.init_models(task=task, gazebo=gazebo)

    def init_world_plugins(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        # SceneBroadcaster
        if self._plugin_scene_broadcaster:
            if not gazebo.scene_broadcaster_active(
                task.substitute_special_frame("world")
            ):
                task.get_logger().info(
                    "Inserting world plugins for broadcasting GUI with enabled user commands..."
                )
                task.world.to_gazebo().insert_world_plugin(
                    "ignition-gazebo-user-commands-system",
                    "ignition::gazebo::systems::UserCommands",
                )

                # Execute a paused run to process world plugin insertion
                if not gazebo.run(paused=True):
                    raise RuntimeError("Failed to execute a paused Gazebo run")

        # UserCommands
        if self._plugin_user_commands:
            task.get_logger().info(
                "Inserting world plugins for broadcasting GUI with enabled user commands..."
            )
            task.world.to_gazebo().insert_world_plugin(
                "ignition-gazebo-scene-broadcaster-system",
                "ignition::gazebo::systems::SceneBroadcaster",
            )

            # Execute a paused run to process world plugin insertion
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Sensors
        if self._camera_enable:
            task.get_logger().info(
                f"Inserting world plugins for sensors with {self._plugin_sensors_render_engine} rendering engine..."
            )
            task.world.to_gazebo().insert_world_plugin(
                "libignition-gazebo-sensors-system.so",
                "ignition::gazebo::systems::Sensors",
                "<sdf version='1.9'>"
                f"<render_engine>{self._plugin_sensors_render_engine}</render_engine>"
                "</sdf>",
            )

            # Execute a paused run to process world plugin insertion
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

    def init_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Initialise all models that are persistent throughout the entire training (they do not need to be re-spawned).
        All other models that need to be re-spawned on each reset are ignored here
        """

        model_names = task.world.to_gazebo().model_names()
        if len(model_names) > 0:
            task.get_logger().warn(
                "Before initialisation, the world already contains the following models:"
                f"\n\t{model_names}"
            )

        # Insert default light if enabled and light randomization is disabled
        if self._light_enable and not self.__light_model_randomizer_enabled():
            task.get_logger().info("Inserting default light into the environment...")
            self.add_default_light(task=task, gazebo=gazebo)

        # Insert default terrain if enabled and terrain randomization is disabled
        if self._terrain_enable and not self.__terrain_model_randomizer_enabled():
            task.get_logger().info("Inserting default terrain into the environment...")
            self.add_default_terrain(task=task, gazebo=gazebo)

        # Insert robot
        task.get_logger().info("Inserting robot into the environment...")
        self.add_robot(task=task, gazebo=gazebo)

        # Insert camera
        if self._camera_enable:
            task.get_logger().info("Inserting camera into the environment...")
            self.add_camera(task=task, gazebo=gazebo)

        # Insert default object if enabled and object randomization is disabled
        if self._object_enable and not self.__object_models_randomizer_enabled():
            task.get_logger().info("Inserting default objects into the environment...")
            self.add_default_objects(task=task, gazebo=gazebo)

        # Insert invisible plane below the terrain to prevent objects from falling into the abyss and causing physics errors
        # TODO (medium): Consider replacing invisible plane with removal of all objects that are too low along z axis
        if self._underworld_collision_plane:
            task.get_logger().info(
                "Inserting invisible plane below the terrain into the environment..."
            )
            self.add_underworld_collision_plane(task=task, gazebo=gazebo)

        # Visualise volumes in GUI if desired
        # TODO: Visualization must follow the robot - consider using RViZ geometry markers instead of this appraoch
        if self._visualise_workspace:
            self.visualise_workspace(task=task, gazebo=gazebo)
        if self._visualise_spawn_volume:
            self.visualise_spawn_volume(task=task, gazebo=gazebo)

    def add_robot(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert robot into the simulation
        """

        # Instantiate robot class based on the selected model
        self.robot = task.robot_model_class(
            world=task.world,
            name=task.robot_name,
            prefix=task.robot_prefix,
            position=self._robot_spawn_position,
            orientation=quat_to_wxyz(self._robot_spawn_quat_xyzw),
            initial_arm_joint_positions=task.initial_arm_joint_positions,
            initial_gripper_joint_positions=task.initial_gripper_joint_positions,
            # TODO (medium): Pass xacro mappings to the function
            # xacro_mappings={},
            # kwargs={},
        )

        # The desired name is passed as arg on creation, however, a different name might be selected to be unique
        # Therefore, return the name back to the task
        task.robot_name = self.robot.name()

        # Enable contact detection for all gripper links (fingers)
        robot_gazebo = self.robot.to_gazebo()
        for gripper_link_name in task.robot_gripper_link_names:
            finger = robot_gazebo.get_link(link_name=gripper_link_name)
            finger.enable_contact_detection(True)

        # Execute a paused run to process robot model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Reset robot joints to the defaults
        self.reset_robot_joint_positions(
            task=task, gazebo=gazebo, above_object_spawn=False, randomize=False
        )

    def add_camera(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
    ):
        """
        Configure and insert camera into the simulation. Camera is places with respect to the robot
        """

        if task.world.to_gazebo().name() == self._camera_relative_to:
            camera_position = self._camera_spawn_position
            camera_quat_wxyz = quat_to_wxyz(self._camera_spawn_quat_xyzw)
        else:
            # Transform the pose of camera to be with respect to robot - but still represented in world reference frame for insertion into the world
            camera_position, camera_quat_wxyz = transform_move_to_model_pose(
                world=task.world,
                position=self._camera_spawn_position,
                quat=quat_to_wxyz(self._camera_spawn_quat_xyzw),
                target_model=self.robot,
                target_link=self._camera_relative_to,
                xyzw=False,
            )

        # Create camera
        self.camera = models.Camera(
            world=task.world,
            position=camera_position,
            orientation=camera_quat_wxyz,
            camera_type=self._camera_type,
            width=self._camera_width,
            height=self._camera_height,
            update_rate=self._camera_update_rate,
            horizontal_fov=self._camera_horizontal_fov,
            vertical_fov=self._camera_vertical_fov,
            clip_color=self._camera_clip_color,
            clip_depth=self._camera_clip_depth,
            noise_mean=self._camera_noise_mean,
            noise_stddev=self._camera_noise_stddev,
            ros2_bridge_color=self._camera_publish_color,
            ros2_bridge_depth=self._camera_publish_depth,
            ros2_bridge_points=self._camera_publish_points,
        )

        # Execute a paused run to process camera model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Attach to robot if needed
        if task.world.to_gazebo().name() != self._camera_relative_to:
            if not self.robot.to_gazebo().attach_link(
                self._camera_relative_to, self.camera.name(), self.camera.link_name
            ):
                raise Exception("Cannot attach camera link to robot")
            self.__is_camera_attached = True

            # Execute a paused run to process camera link attachment
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Broadcast tf
        task.tf2_broadcaster.broadcast_tf(
            parent_frame_id=self._camera_relative_to,
            child_frame_id=self.camera.frame_id,
            translation=self._camera_spawn_position,
            rotation=self._camera_spawn_quat_xyzw,
            xyzw=True,
        )

    def add_default_terrain(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Configure and insert default terrain into the simulation
        """

        # Create terrain
        self.terrain = self.__terrain_model_class(
            world=task.world,
            name=task.terrain_name,
            position=self._terrain_spawn_position,
            orientation=quat_to_wxyz(self._terrain_spawn_quat_xyzw),
            size=self._terrain_size,
            np_random=task.np_random,
            texture_dir=environ.get("DRL_GRASPING_PBR_TEXTURES_DIR", default=""),
        )

        # The desired name is passed as arg on creation, however, a different name might be selected to be unique
        # Therefore, return the name back to the task
        task.terrain_name = self.terrain.name()

        # Enable contact detection
        for link_name in self.terrain.link_names():
            link = self.terrain.to_gazebo().get_link(link_name=link_name)
            link.enable_contact_detection(True)

        # Execute a paused run to process terrain model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_default_light(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert default light into the simulation
        """

        # Create light
        self.light = self.__light_model_class(
            world=task.world,
            name="sun",
            direction=self._light_direction,
            minmax_elevation=self._light_random_minmax_elevation,
            color=self._light_color,
            distance=self._light_distance,
            visual=self._light_visual,
            radius=self._light_radius,
            np_random=task.np_random,
        )

        # Execute a paused run to process light model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_default_objects(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Configure and insert default object into the simulation
        """

        # Insert new models with random pose
        while len(self.task.object_names) < self._object_count:
            if self._object_count > 1:
                object_position, object_quat_wxyz = self.get_random_object_pose(
                    task=task,
                    centre=self._object_spawn_position,
                    volume=self._object_random_spawn_volume,
                )
            else:
                object_position = self._object_spawn_position
                object_quat_wxyz = (1.0, 0.0, 0.0, 0.0)

                if task.world.to_gazebo().name() != self._objects_relative_to:
                    # Transform the pose of camera to be with respect to robot - but represented in world reference frame for insertion into the world
                    object_position, object_quat_wxyz = transform_move_to_model_pose(
                        world=task.world,
                        position=object_position,
                        quat=object_quat_wxyz,
                        target_model=self.robot,
                        target_link=self._objects_relative_to,
                        xyzw=False,
                    )

            try:

                # Create object
                object_model = self.__object_model_class(
                    world=task.world,
                    position=object_position,
                    orientation=object_quat_wxyz,
                    size=self._object_dimensions,
                    radius=self._object_dimensions[0],
                    length=self._object_dimensions[1],
                    mass=self._object_mass,
                    collision=self._object_collision,
                    visual=self._object_visual,
                    static=self._object_static,
                    color=self._object_color,
                )

                model_name = object_model.name()

                # Expose name of the object for task (append in case of more)
                task.object_names.append(model_name)

                # Enable contact detection
                for link_name in object_model.link_names():
                    link = object_model.to_gazebo().get_link(link_name=link_name)
                    link.enable_contact_detection(True)

            except Exception as ex:
                task.get_logger().warn(
                    "Model "
                    + model_name
                    + " could not be insterted. Reason: "
                    + str(ex)
                )

        # Execute a paused run to process insertion of object model
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_underworld_collision_plane(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Add an infinitely large collision plane below the terrain in order to prevent object from falling into the abyss forever
        """

        models.Plane(
            world=task.world,
            position=(0.0, 0.0, self._underworld_collision_plane_depth),
            orientation=(1.0, 0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            visual=False,
            collision=True,
            friction=1000.0,
        )

        # Execute a paused run to process model insertion of underworld collision plane
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    # Randomization #
    def randomize_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Randomize models if needed
        """

        # Randomize light plane if needed
        if self._light_enable and self._light_model_expired():
            self.randomize_light(task=task, gazebo=gazebo)

        # Randomize terrain plane if needed
        if self._terrain_enable and self._terrain_model_expired():
            self.randomize_terrain(task=task, gazebo=gazebo)

        # Randomize robot model pose if needed
        if self.robot.is_mobile:
            self.reset_robot_pose(
                task=task, gazebo=gazebo, randomize=self._robot_random_pose
            )

        # Reset/randomize robot joint positions
        self.reset_robot_joint_positions(
            task=task,
            gazebo=gazebo,
            above_object_spawn=self._robot_random_joint_positions_above_object_spawn,
            randomize=self._robot_random_joint_positions,
        )

        # Randomize camera if needed
        if self._camera_enable and self._camera_pose_expired():
            self.randomize_camera_pose(
                task=task, gazebo=gazebo, mode=self._camera_random_pose_mode
            )

        # Randomize objects if needed
        # Note: No need to randomize pose of new models because they are already spawned randomly
        self.__object_positions.clear()
        if self._object_enable:
            if self._object_models_expired():
                self.randomize_object_models(task=task, gazebo=gazebo)
            elif self._object_random_pose:
                self.object_random_pose(task=task, gazebo=gazebo)
            else:
                self.reset_default_object_pose(task=task, gazebo=gazebo)

    def reset_robot_pose(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
        randomize: bool = False,
    ):

        if randomize:
            position = [
                self._robot_spawn_position[0]
                + task.np_random.uniform(
                    -self._robot_random_spawn_volume[0] / 2,
                    self._robot_random_spawn_volume[0] / 2,
                ),
                self._robot_spawn_position[1]
                + task.np_random.uniform(
                    -self._robot_random_spawn_volume[1] / 2,
                    self._robot_random_spawn_volume[1] / 2,
                ),
                self._robot_spawn_position[2]
                + task.np_random.uniform(
                    -self._robot_random_spawn_volume[2] / 2,
                    self._robot_random_spawn_volume[2] / 2,
                ),
            ]
            quat_xyzw = Rotation.from_euler(
                "xyz", (0, 0, task.np_random.uniform(-np.pi, np.pi))
            ).as_quat()
        else:
            position = self._robot_spawn_position
            quat_xyzw = self._robot_spawn_quat_xyzw

        gazebo_robot = self.robot.to_gazebo()
        gazebo_robot.reset_base_pose(position, quat_to_wxyz(quat_xyzw))
        gazebo_robot.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Execute a paused run to process model modification
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def reset_robot_joint_positions(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
        above_object_spawn: bool = False,
        randomize: bool = False,
    ):

        # Stop servoing
        if task._use_servo:
            task.servo()
            if task.servo.is_enabled:
                task.servo.disable(sync=True)

        gazebo_robot = self.robot.to_gazebo()

        if above_object_spawn:
            # If desired, compute IK above object spawn
            if randomize:
                rnd_displacement = (
                    self._robot_random_joint_positions_above_object_spawn_xy_randomness
                    * task.np_random.uniform(
                        (
                            -self._object_random_spawn_volume[0],
                            -self._object_random_spawn_volume[1],
                        ),
                        self._object_random_spawn_volume[:2],
                    )
                )
                position = (
                    self._object_spawn_position[0] + rnd_displacement[0],
                    self._object_spawn_position[1] + rnd_displacement[1],
                    self._object_spawn_position[2]
                    + self._robot_random_joint_positions_above_object_spawn_elevation,
                )
                quat_xyzw = Rotation.from_euler(
                    "xyz", (0, np.pi, task.np_random.uniform(-np.pi, np.pi))
                ).as_quat()
            else:
                position = (
                    self._object_spawn_position[0],
                    self._object_spawn_position[1],
                    self._object_spawn_position[2]
                    + self._robot_random_joint_positions_above_object_spawn_elevation,
                )
                quat_xyzw = (1.0, 0.0, 0.0, 0.0)

            joint_configuration = task.moveit2.compute_ik(
                position=position,
                quat_xyzw=quat_xyzw,
                start_joint_state=task.initial_arm_joint_positions,
            )
            if joint_configuration is not None:
                arm_joint_positions = joint_configuration.position[
                    : len(task.initial_arm_joint_positions)
                ]
            else:
                task.get_logger().warn(
                    "Robot configuration could not be reset above the object spawn. Using initial arm joint positions instead."
                )
                arm_joint_positions = task.initial_arm_joint_positions
        else:
            # Otherwise get initial arm joint positions from the task (each task might need something different)
            arm_joint_positions = task.initial_arm_joint_positions

        # Add normal noise if desired
        if randomize:
            for joint_position in arm_joint_positions:
                joint_position += task.np_random.normal(
                    loc=0.0, scale=self._robot_random_joint_positions_std
                )

        # Arm joints - apply joint positions zero out velocities
        if not gazebo_robot.reset_joint_positions(
            arm_joint_positions, self.robot.arm_joint_names
        ):
            raise RuntimeError("Failed to reset robot joint positions")
        if not gazebo_robot.reset_joint_velocities(
            [0.0] * len(self.robot.arm_joint_names),
            self.robot.arm_joint_names,
        ):
            raise RuntimeError("Failed to reset robot joint velocities")

        # Gripper joints - apply joint positions zero out velocities
        if task._enable_gripper and self.robot.gripper_joint_names:
            if not gazebo_robot.reset_joint_positions(
                task.initial_gripper_joint_positions, self.robot.gripper_joint_names
            ):
                raise RuntimeError("Failed to reset gripper joint positions")
            if not gazebo_robot.reset_joint_velocities(
                [0.0] * len(self.robot.gripper_joint_names),
                self.robot.gripper_joint_names,
            ):
                raise RuntimeError("Failed to reset gripper joint velocities")

        # Passive joints - zero out velocities
        if self.robot.passive_joint_names:
            if not gazebo_robot.reset_joint_velocities(
                [0.0] * len(self.robot.passive_joint_names),
                self.robot.passive_joint_names,
            ):
                raise RuntimeError("Failed to reset passive joint velocities")

        # Execute a paused run to process model modification
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Start servoing again
        if task._use_servo:
            task.servo.enable(sync=False)

        # Execute an unpaused run to process model modification and get new JointStates
        if not gazebo.step():
            raise RuntimeError("Failed to execute an unpaused Gazebo step")

        # Reset also the controllers
        task.moveit2.reset_controller(joint_state=arm_joint_positions)
        # TODO (low): Reset of gripper causes the motion of robot with servo to become jittery. No idea why. Not really necessary to reset gripper controller either
        # if task._enable_gripper and self.robot.gripper_joint_names:
        #     task.gripper.reset_controller(task.initial_gripper_joint_positions)
        #     if (
        #         self.robot.CLOSED_GRIPPER_JOINT_POSITIONS
        #         == task.initial_gripper_joint_positions
        #     ):
        #         task.gripper.close()
        #     else:
        #         task.gripper.open()

    def randomize_camera_pose(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, mode: str
    ):

        # Get random camera pose, centred at object position (or centre of object spawn box)
        if "orbit" == mode:
            camera_position, camera_quat_xyzw = self.get_random_camera_pose_orbit(
                task=task,
                centre=self._object_spawn_position,
                distance=self._camera_random_pose_orbit_distance,
                height=self._camera_random_pose_orbit_height_range,
                ignore_arc_behind_robot=self._camera_random_pose_orbit_ignore_arc_behind_robot,
                focal_point_z_offset=self._camera_random_pose_focal_point_z_offset,
            )
        elif "select_random" == mode:
            (
                camera_position,
                camera_quat_xyzw,
            ) = self.get_random_camera_pose_sample_random(
                task=task,
                centre=self._object_spawn_position,
                options=self._camera_random_pose_select_position_options,
            )
        elif "select_nearest" == mode:
            (
                camera_position,
                camera_quat_xyzw,
            ) = self.get_random_camera_pose_sample_nearest(
                centre=self._object_spawn_position,
                options=self._camera_random_pose_select_position_options,
            )
        else:
            raise TypeError("Invalid mode for camera pose randomization.")

        if task.world.to_gazebo().name() == self._camera_relative_to:
            transformed_camera_position = camera_position
            transformed_camera_quat_wxyz = quat_to_wxyz(camera_quat_xyzw)
        else:
            # Transform the pose of camera to be with respect to robot - but represented in world reference frame for insertion into the world
            (
                transformed_camera_position,
                transformed_camera_quat_wxyz,
            ) = transform_move_to_model_pose(
                world=task.world,
                position=camera_position,
                quat=quat_to_wxyz(camera_quat_xyzw),
                target_model=self.robot,
                target_link=self._camera_relative_to,
                xyzw=False,
            )

        # Detach camera if needed
        if self.__is_camera_attached:
            if not self.robot.to_gazebo().detach_link(
                self._camera_relative_to, self.camera.name(), self.camera.link_name
            ):
                raise Exception("Cannot detach camera link from robot")

            # Execute a paused run to process camera detachment
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Move pose of the camera
        camera_gazebo = self.camera.to_gazebo()
        camera_gazebo.reset_base_pose(
            transformed_camera_position, transformed_camera_quat_wxyz
        )

        # Execute a paused run to process change of camera pose
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Attach to robot again if needed
        if self.__is_camera_attached:
            if not self.robot.to_gazebo().attach_link(
                self._camera_relative_to, self.camera.name(), self.camera.link_name
            ):
                raise Exception("Cannot attach camera link to robot")

            # Execute a paused run to process link attachment
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Broadcast tf
        task.tf2_broadcaster.broadcast_tf(
            parent_frame_id=self._camera_relative_to,
            child_frame_id=self.camera.frame_id,
            translation=camera_position,
            rotation=camera_quat_xyzw,
            xyzw=True,
        )

    def get_random_camera_pose_orbit(
        self,
        task: SupportedTasks,
        centre: Tuple[float, float, float],
        distance: float,
        height: Tuple[float, float],
        ignore_arc_behind_robot: float,
        focal_point_z_offset: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

        # Select a random 3D position (with restricted min height)
        while True:
            position = task.np_random.uniform(
                low=(-1.0, -1.0, height[0]), high=(1.0, 1.0, height[1])
            )
            # Normalize
            position /= np.linalg.norm(position)

            # Make sure it does not get spawned directly behind the robot
            if (
                abs(np.arctan2(position[0], position[1]) + np.pi / 2)
                > ignore_arc_behind_robot
            ):
                break

        # Determine orientation such that camera faces the origin
        rpy = [
            0.0,
            np.arctan2(
                position[2] - focal_point_z_offset, np.linalg.norm(position[:2], 2)
            ),
            np.arctan2(position[1], position[0]) + np.pi,
        ]
        quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()

        # Scale normal vector by distance and translate camera to point at the workspace centre
        position *= distance
        position[:2] += centre[:2]

        return position, quat_xyzw

    def get_random_camera_pose_sample_random(
        self,
        task: SupportedTasks,
        centre: Tuple[float, float, float],
        options: List[Tuple[float, float, float]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

        # Select a random entry from the options
        selection = options[task.np_random.randint(len(options))]

        # Process it and return
        return self.get_random_camera_pose_sample_process(
            centre=centre,
            position=selection,
            focal_point_z_offset=self._camera_random_pose_focal_point_z_offset,
        )

    def get_random_camera_pose_sample_nearest(
        self,
        centre: Tuple[float, float, float],
        options: List[Tuple[float, float, float]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

        # Select the nearest entry
        dist_sqr = np.sum((np.array(options) - np.array(centre)) ** 2, axis=1)
        nearest = options[np.argmin(dist_sqr)]

        # Process it and return
        return self.get_random_camera_pose_sample_process(
            centre=centre,
            position=nearest,
            focal_point_z_offset=self._camera_random_pose_focal_point_z_offset,
        )

    def get_random_camera_pose_sample_process(
        self,
        centre: Tuple[float, float, float],
        position: Tuple[float, float, float],
        focal_point_z_offset: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

        # Determine orientation such that camera faces the centre
        rpy = [
            0.0,
            np.arctan2(
                position[2] - focal_point_z_offset,
                np.linalg.norm((position[0] - centre[0], position[1] - centre[1]), 2),
            ),
            np.arctan2(position[1] - centre[1], position[0] - centre[0]) + np.pi,
        ]
        quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()

        return position, quat_xyzw

    def randomize_terrain(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):

        # Remove existing terrain
        if hasattr(self, "terrain"):
            if not task.world.to_gazebo().remove_model(self.terrain.name()):
                raise RuntimeError(f"Failed to remove {self.terrain.name()}")

            # Execute a paused run to process model removal
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Choose one of the random orientations for the texture (4 directions)
        orientation = [
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0.70710678118, 0, 0, 0.70710678118),
            (0.70710678118, 0, 0, -0.70710678118),
        ][task.np_random.randint(4)]

        # Create terrain
        self.terrain = self.__terrain_model_class(
            world=task.world,
            name=task.terrain_name,
            position=self._terrain_spawn_position,
            orientation=orientation,
            size=self._terrain_size,
            np_random=task.np_random,
            texture_dir=environ.get("DRL_GRASPING_PBR_TEXTURES_DIR", default=""),
        )

        # Expose name of the terrain for task
        task.terrain_name = self.terrain.name()

        # Enable contact detection
        for link_name in self.terrain.link_names():
            link = self.terrain.to_gazebo().get_link(link_name=link_name)
            link.enable_contact_detection(True)

        # Execute a paused run to process model removal and insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def randomize_light(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):

        # Remove existing light
        if hasattr(self, "light"):
            if not task.world.to_gazebo().remove_model(self.light.name()):
                raise RuntimeError(f"Failed to remove {self.light.name()}")

            # Execute a paused run to process model removal
            if not gazebo.run(paused=True):
                raise RuntimeError("Failed to execute a paused Gazebo run")

        # Create light
        self.light = self.__light_model_class(
            world=task.world,
            name="sun",
            direction=self._light_direction,
            minmax_elevation=self._light_random_minmax_elevation,
            color=self._light_color,
            distance=self._light_distance,
            visual=self._light_visual,
            radius=self._light_radius,
            np_random=task.np_random,
        )

        # Execute a paused run to process model removal and insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def reset_default_object_pose(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):

        assert len(task.object_names) == 1

        obj = task.world.to_gazebo().get_model(task.object_names[0]).to_gazebo()
        obj.reset_base_pose(self._object_spawn_position, (1.0, 0.0, 0.0, 0.0))
        obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Execute a paused run to process model modification
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def randomize_object_models(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):

        # Remove all existing models
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f"Failed to remove {object_name}")
            self.task.object_names.clear()

        # Insert new models with random pose
        while len(self.task.object_names) < self._object_count:
            position, quat_random = self.get_random_object_pose(
                task=task,
                centre=self._object_spawn_position,
                volume=self._object_random_spawn_volume,
            )
            try:
                model = self.__object_model_class(
                    world=task.world,
                    position=position,
                    orientation=quat_random,
                    np_random=task.np_random,
                )
                model_name = model.name()
                self.task.object_names.append(model_name)
                self.__object_positions[model_name] = position
                # Enable contact detection
                for link_name in model.link_names():
                    link = model.to_gazebo().get_link(link_name=link_name)
                    link.enable_contact_detection(True)

            except Exception as ex:
                task.get_logger().warn(
                    "Model "
                    + model_name
                    + " could not be insterted. Reason: "
                    + str(ex)
                )

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def object_random_pose(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):

        for object_name in self.task.object_names:
            position, quat_random = self.get_random_object_pose(
                task=task,
                centre=self._object_spawn_position,
                volume=self._object_random_spawn_volume,
            )
            obj = task.world.to_gazebo().get_model(object_name).to_gazebo()
            obj.reset_base_pose(position, quat_random)
            obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.__object_positions[object_name] = position

        # Execute a paused run to process model modification
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def get_random_object_pose(
        self,
        task: SupportedTasks,
        centre: Tuple[float, float, float],
        volume: Tuple[float, float, float],
        name: str = "",
        min_distance_to_other_objects: float = 0.25,
        min_distance_decay_factor: float = 0.9,
    ):

        is_too_close = True
        while is_too_close:
            object_position = [
                centre[0] + task.np_random.uniform(-volume[0] / 2, volume[0] / 2),
                centre[1] + task.np_random.uniform(-volume[1] / 2, volume[1] / 2),
                centre[2] + task.np_random.uniform(-volume[2] / 2, volume[2] / 2),
            ]

            if task.world.to_gazebo().name() != self._objects_relative_to:
                # Transform the pose of camera to be with respect to robot - but represented in world reference frame for insertion into the world
                object_position = transform_move_to_model_position(
                    world=task.world,
                    position=object_position,
                    target_model=self.robot,
                    target_link=self._objects_relative_to,
                )

            # Check if position is far enough from other
            is_too_close = False
            for (
                existing_object_name,
                existing_object_position,
            ) in self.__object_positions.items():
                if existing_object_name == name:
                    # Do not compare to itself
                    continue
                if (
                    distance.euclidean(object_position, existing_object_position)
                    < min_distance_to_other_objects
                ):
                    min_distance_to_other_objects *= min_distance_decay_factor
                    is_too_close = True
                    break

        quat = task.np_random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)

        return object_position, quat

    # External overrides #
    def external_overrides(self, task: SupportedTasks):
        """
        Perform external overrides from either task level or environment before initialising/randomising the task
        """

        self.__consume_parameter_overrides(task=task)

    # Pre-randomization #
    def pre_randomization(self, task: SupportedTasks):
        """
        Perform steps that are required before randomization is performed.
        """

        # If desired, select random spawn position from the segments
        # It is performed here because object spawn position might be of interest also for robot and camera randomization
        segments_len = len(self._object_random_spawn_position_segments)
        if segments_len > 1:
            # Randomly select a segment between two points
            start_index = task.np_random.randint(segments_len - 1)
            segment = (
                self._object_random_spawn_position_segments[start_index],
                self._object_random_spawn_position_segments[start_index + 1],
            )

            # Randomly select a point on the segment and use it as the new object spawn position
            intersect = task.np_random.random()
            direction = (
                segment[1][0] - segment[0][0],
                segment[1][1] - segment[0][1],
                segment[1][2] - segment[0][2],
            )
            self._object_spawn_position = (
                segment[0][0] + intersect * direction[0],
                segment[0][1] + intersect * direction[1],
                segment[0][2] + intersect * direction[2],
            )

    # Post-randomization #
    def post_randomization(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Perform steps that are required once randomization is complete and the simulation can be stepped a few times unpaused.
        """

        attempts = 0
        object_overlapping_ok = False

        # Execute steps until new observations are available
        observations_ready = False
        task.moveit2.reset_new_joint_state_checker()
        if task._enable_gripper:
            task.gripper.reset_new_joint_state_checker()
        if hasattr(task, "camera_sub"):
            task.camera_sub.reset_new_observation_checker()
        while observations_ready:
            attempts += 1
            if 0 == attempts % self.POST_RANDOMIZATION_MAX_STEPS:
                task.get_logger().warn(
                    f"Waiting for new joint state after reset. Iteration #{attempts}..."
                )
            else:
                task.get_logger().debug("Waiting for new joint state after reset.")
            if not gazebo.step():
                raise RuntimeError("Failed to execute an unpaused Gazebo step")
            object_overlapping_ok = self.check_object_overlapping(task=task)

            # Break once all observaions are available
            if not task.moveit2.new_joint_state_available:
                continue
            if task._enable_gripper:
                if not task.gripper.new_joint_state_available:
                    continue
            if hasattr(task, "camera_sub"):
                if not task.camera_sub.new_observation_available:
                    continue
            observations_ready = True

        # Make sure no objects are overlapping (intersections between collision geometry)
        while (
            not object_overlapping_ok and attempts < self.POST_RANDOMIZATION_MAX_STEPS
        ):
            attempts += 1
            task.get_logger().info("Objects overlapping, trying new positions")
            if not gazebo.step():
                raise RuntimeError("Failed to execute an unpaused Gazebo step")
            object_overlapping_ok = self.check_object_overlapping(task=task)
        if self.POST_RANDOMIZATION_MAX_STEPS == attempts:
            task.get_logger().warn(
                "Objects could not be spawned overlapping. The workspace might be too crowded!"
            )

    def check_object_overlapping(
        self,
        task: SupportedTasks,
        allowed_penetration_depth: float = 0.001,
        terrain_allowed_penetration_depth: float = 0.01,
    ) -> bool:
        """
        Go through all objects and make sure that none of them are overlapping.
        If an object is overlapping, reset its position.
        Positions are reset also if object is in collision with robot right after reset.
        Collisions/overlaps with terrain are ignored.
        Returns True if all objects are okay, false if they had to be reset
        """

        # Update object positions
        for object_name in self.task.object_names:
            model = task.world.get_model(object_name).to_gazebo()
            self.__object_positions[object_name] = model.get_link(
                link_name=model.link_names()[0]
            ).position()

        for object_name in self.task.object_names:
            obj = task.world.get_model(object_name).to_gazebo()
            for contact in obj.contacts():
                depth = np.mean([point.depth for point in contact.points])
                if (
                    self.terrain.name() in contact.body_b
                    and depth < terrain_allowed_penetration_depth
                ):
                    continue
                if (
                    task.robot_name in contact.body_b
                    or depth > allowed_penetration_depth
                ):
                    position, quat_random = self.get_random_object_pose(
                        task=task,
                        centre=self._object_spawn_position,
                        volume=self._object_random_spawn_volume,
                        name=object_name,
                    )
                    obj.reset_base_pose(position, quat_random)
                    obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                    return False

        return True

    # ============================
    # Randomizer rollouts checking
    # ============================

    def __camera_pose_randomizer_enabled(self) -> bool:
        """
        Checks if camera pose randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self._camera_random_pose_rollouts_num == 0:
            return False
        else:
            return True

    def _camera_pose_expired(self) -> bool:
        """
        Checks if camera pose needs to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.__camera_pose_randomizer_enabled():
            return False

        self.__camera_pose_rollout_counter += 1

        if self.__camera_pose_rollout_counter >= self._camera_random_pose_rollouts_num:
            self.__camera_pose_rollout_counter = 0
            return True

        return False

    def __terrain_model_randomizer_enabled(self) -> bool:
        """
        Checks if terrain randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self._terrain_model_rollouts_num == 0:
            return False
        else:
            return self.__is_terrain_type_randomizable

    def _terrain_model_expired(self) -> bool:
        """
        Checks if terrain model needs to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.__terrain_model_randomizer_enabled():
            return False

        self.__terrain_model_rollout_counter += 1

        if self.__terrain_model_rollout_counter >= self._terrain_model_rollouts_num:
            self.__terrain_model_rollout_counter = 0
            return True

        return False

    def __light_model_randomizer_enabled(self) -> bool:
        """
        Checks if light model randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self._light_model_rollouts_num == 0:
            return False
        else:
            return self.__is_light_type_randomizable

    def _light_model_expired(self) -> bool:
        """
        Checks if light models need to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.__light_model_randomizer_enabled():
            return False

        self.__light_model_rollout_counter += 1

        if self.__light_model_rollout_counter >= self._light_model_rollouts_num:
            self.__light_model_rollout_counter = 0
            return True

        return False

    def __object_models_randomizer_enabled(self) -> bool:
        """
        Checks if object model randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self._object_models_rollouts_num == 0:
            return False
        else:
            return self.__is_object_type_randomizable

    def _object_models_expired(self) -> bool:
        """
        Checks if object models need to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.__object_models_randomizer_enabled():
            return False

        self.__object_models_rollout_counter += 1

        if self.__object_models_rollout_counter >= self._object_models_rollouts_num:
            self.__object_models_rollout_counter = 0
            return True

        return False

    def __consume_parameter_overrides(self, task: SupportedTasks):

        for key, value in task._randomizer_parameter_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

        task._randomizer_parameter_overrides.clear()

    # =============================
    # Additional features and debug
    # =============================

    def visualise_workspace(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
        color: Tuple[float, float, float, float] = (0, 1, 0, 0.8),
    ):

        # Insert a translucent box visible only in simulation with no physical interactions
        models.Box(
            world=task.world,
            name="workspace_volume",
            position=self._object_spawn_position,
            orientation=(0, 0, 0, 1),
            size=task.workspace_volume,
            collision=False,
            visual=True,
            gui_only=True,
            static=True,
            color=color,
        )
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def visualise_spawn_volume(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
        color: Tuple[float, float, float, float] = (0, 0, 1, 0.8),
        color_with_height: Tuple[float, float, float, float] = (1, 0, 1, 0.7),
    ):

        # Insert translucent boxes visible only in simulation with no physical interactions
        models.Box(
            world=task.world,
            name="object_random_spawn_volume",
            position=self._object_spawn_position,
            orientation=(0, 0, 0, 1),
            size=self._object_random_spawn_volume,
            collision=False,
            visual=True,
            gui_only=True,
            static=True,
            color=color,
        )
        models.Box(
            world=task.world,
            name="object_random_spawn_volume_with_height",
            position=self._object_spawn_position,
            orientation=(0, 0, 0, 1),
            size=self._object_random_spawn_volume,
            collision=False,
            visual=True,
            gui_only=True,
            static=True,
            color=color_with_height,
        )
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")
