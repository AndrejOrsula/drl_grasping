from drl_grasping.envs import tasks, models
from drl_grasping.envs.utils.conversions import quat_to_xyzw, quat_to_wxyz
from drl_grasping.envs.utils.gazebo import get_model_pose
from drl_grasping.envs.utils.math import quat_mul
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from os import environ
from scenario import gazebo as scenario
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from typing import Union, Tuple, List
import abc
import numpy as np
import operator

# Tasks that are supported by this randomizer (used primarily for type hinting)
SupportedTasks = Union[tasks.Reach, tasks.ReachOctree, tasks.Grasp, tasks.GraspOctree]


class ManipulationGazeboEnvRandomizer(
    gazebo_env_randomizer.GazeboEnvRandomizer,
    randomizers.abc.PhysicsRandomizer,
    randomizers.abc.TaskRandomizer,
    abc.ABC,
):
    """
    Basic randomizer for robotic manipulation environments that also populates the simulated world.
    """

    def __init__(
        self,
        env: MakeEnvCallable,
        physics_rollouts_num: int = 0,
        # Robot
        robot_spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        robot_spawn_quat_xyzw: Tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        robot_random_joint_positions: bool = False,
        robot_random_joint_positions_std: float = 0.1,
        # Camera #
        camera_enable: bool = True,
        camera_type: str = "rgbd_camera",
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
        # Note: Camera pose is with respect to the pose of robot base link
        camera_spawn_position: Tuple[float, float, float] = (0, 0, 1),
        camera_spawn_quat_xyzw: Tuple[float, float, float, float] = (
            0,
            0.70710678118,
            0,
            0.70710678118,
        ),
        camera_random_pose_rollouts_num: int = 0,
        camera_random_pose_distance: float = 1.0,
        camera_random_pose_height_range: Tuple[float, float] = (0.1, 0.7),
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
        light_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        light_distance: float = 1000.0,
        light_visual: bool = True,
        light_radius: float = 25.0,
        light_model_rollouts_num: int = 1,
        # Objects
        object_enable: bool = True,
        object_type: str = "box",
        object_static: bool = False,
        object_collision: bool = True,
        object_visual: bool = True,
        object_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        object_dimensions: List[float] = [0.05, 0.05, 0.05],
        object_mass: float = 0.1,
        object_model_count: int = 1,
        object_spawn_position: List[float] = [0.0, 0.0, 0.0],
        object_random_pose: bool = True,
        object_random_spawn_volume: List[float] = [0.5, 0.5, 0.5],
        object_models_rollouts_num: int = 1,
        # Collision plane below terrain
        underworld_collision_plane: bool = True,
        underworld_collision_plane_depth: float = -1.0,
        # Visual debugging
        visualise_workspace: bool = False,
        visualise_spawn_volume: bool = False,
        **kwargs,
    ):

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
        # Robot
        self._robot_spawn_position = robot_spawn_position
        self._robot_spawn_quat_xyzw = robot_spawn_quat_xyzw
        self._robot_random_joint_positions = robot_random_joint_positions
        self._robot_random_joint_positions_std = robot_random_joint_positions_std

        # Camera
        self._camera_enable = camera_enable
        self._camera_type = camera_type
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
        self._camera_random_pose_distance = camera_random_pose_distance
        self._camera_random_pose_height_range = camera_random_pose_height_range

        # Terrain
        self._terrain_enable = terrain_enable
        self._terrain_spawn_position = terrain_spawn_position
        self._terrain_spawn_quat_xyzw = terrain_spawn_quat_xyzw
        self._terrain_size = terrain_size
        self._terrain_model_rollouts_num = terrain_model_rollouts_num

        # Light
        self._light_enable = light_enable
        self._light_direction = light_direction
        self._light_color = light_color
        self._light_distance = light_distance
        self._light_visual = light_visual
        self._light_radius = light_radius
        self._light_model_rollouts_num = light_model_rollouts_num

        # Objects
        self._object_enable = object_enable
        self._object_static = object_static
        self._object_collision = object_collision
        self._object_visual = object_visual
        self._object_color = object_color
        self._object_dimensions = object_dimensions
        self._object_mass = object_mass
        self._object_model_count = object_model_count
        self._object_spawn_position = object_spawn_position
        self._object_random_pose = object_random_pose
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

        # Flag that determines whether environment has already been initialised
        self.__env_initialised = False

        # Dict to keep track of set object positions - without stepping (faster than lookup through gazebo)
        # It is used to make sure that objects are not spawned inside each other
        self.__object_positions = {}

    ##########################
    # PhysicsRandomizer impl #
    ##########################

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs):

        # TODO: Add gravity preset for Moon (and other bodies)
        gravity_z = task.np_random.normal(loc=-9.80665, scale=0.02)
        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

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
        self.external_overrides(task)

        # Initialise the environment on the first iteration
        if not self.__env_initialised:
            self.init_env(task=task, gazebo=gazebo)
            self.__env_initialised = True

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

        # Insert world plugins needed by the task or selected by user
        self.init_world_plugins(task=task, gazebo=gazebo)

        # Initialise all models that are persustent throughout the entire training
        self.init_models(task=task, gazebo=gazebo)

        # Visualise volumes in GUI if desired
        if self._visualise_workspace:
            self.visualise_workspace(task=task, gazebo=gazebo)
        if self._visualise_spawn_volume:
            self.visualise_spawn_volume(task=task, gazebo=gazebo)

    def init_world_plugins(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        # SceneBroadcaster and UcerCommands
        if environ.get(
            "DRL_GRASPING_BROADCAST_INTERACTIVE_GUI", default="false"
        ).lower() in ("true", "1"):
            # TODO: Do not open GUI client when DRL_GRASPING_BROADCAST_INTERACTIVE_GUI is set, only enable the broadcaster plugin
            if task._verbose:
                print(
                    "Inserting world plugins for broadcasting GUI with enabled user commands..."
                )
            gazebo.gui()
            task.world.to_gazebo().insert_world_plugin(
                "ignition-gazebo-user-commands-system",
                "ignition::gazebo::systems::UserCommands",
            )

        # Sensors
        if self._camera_enable:
            camera_render_engine = environ.get(
                "DRL_GRASPING_SENSORS_RENDER_ENGINE", default="ogre2"
            )
            if task._verbose:
                print(
                    f"Inserting world plugins for sensors with {camera_render_engine} rendering engine..."
                )
            task.world.to_gazebo().insert_world_plugin(
                "libignition-gazebo-sensors-system.so",
                "ignition::gazebo::systems::Sensors",
                "<sdf version='1.9'>"
                f"<render_engine>{camera_render_engine}</render_engine>"
                "</sdf>",
            )

    def init_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Initialise all models that are persistent throughout the entire training (they do not need to be re-spawned).
        All other models that need to be re-spawned on each reset are ignored here
        """

        model_names = task.world.to_gazebo().model_names()
        if task._verbose and len(model_names) > 0:
            print(
                "Before initialisation, the world already contains the following models:"
                f"\n\t{model_names}"
            )

        # Insert robot
        if task._verbose:
            print("Inserting robot into the environment...")
        self.add_robot(task=task, gazebo=gazebo)

        # Insert camera
        if self._camera_enable:
            if task._verbose:
                print("Inserting camera into the environment...")
            self.add_camera(
                task=task, gazebo=gazebo, attach_to_robot=self.robot.is_mobile
            )

        # Insert default terrain if enabled and terrain randomization is disabled
        if self._terrain_enable and not self.__terrain_model_randomizer_enabled():
            if task._verbose:
                print("Inserting default terrain into the environment...")
            self.add_default_terrain(task=task, gazebo=gazebo)

        # Insert default light if enabled and light randomization is disabled
        if self._light_enable and not self.__light_model_randomizer_enabled():
            if task._verbose:
                print("Inserting default light into the environment...")
            self.add_default_light(task=task, gazebo=gazebo)

        # Insert default object if enabled and object randomization is disabled
        if self._object_enable and not self.__object_models_randomizer_enabled():
            if task._verbose:
                print("Inserting default objects into the environment...")
            self.add_defaults_object(task=task, gazebo=gazebo)

        # Insert invisible plane below the terrain to prevent objects from falling into the abyss and causing physics errors
        # TODO: Consider replacing invisiable plane with removal of all objects that are too low along z axis
        if self._underworld_collision_plane:
            if task._verbose:
                print(
                    "Inserting invisible plane below the terrain into the environment..."
                )
            self.add_underworld_collision_plane(task=task, gazebo=gazebo)

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
            # TODO: Pass xacro mappings to the function
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

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_camera(
        self,
        task: SupportedTasks,
        gazebo: scenario.GazeboSimulator,
        attach_to_robot: bool = True,
    ):
        """
        Configure and insert camera into the simulation. Camera is places with respect to the robot
        """

        # Transform the pose of camera to be with respect to robot - but represented in world reference frame for insertion into the world
        robot_base_link_position, robot_base_link_quat_xyzw = get_model_pose(
            task.world,
            model=self.robot,
            link=self.robot.robot_base_link_name,
            xyzw=True,
        )
        camera_position_wrt_robot_in_world_ref = (
            Rotation.from_quat(robot_base_link_quat_xyzw).apply(
                self._camera_spawn_position
            )
            + robot_base_link_position
        )
        camera_orientation_wrt_robot_in_world_ref = quat_mul(
            self._camera_spawn_quat_xyzw, robot_base_link_quat_xyzw, xyzw=True
        )

        # Create camera
        self.camera = models.Camera(
            world=task.world,
            position=camera_position_wrt_robot_in_world_ref,
            orientation=quat_to_wxyz(camera_orientation_wrt_robot_in_world_ref),
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

        # Attach to robot
        if attach_to_robot:
            detach_camera_topic = f"{self.robot.name()}/detach_{self.camera.name()}"
            self.robot.to_gazebo().insert_model_plugin(
                "libignition-gazebo-detachable-joint-system.so",
                "ignition::gazebo::systems::DetachableJoint",
                "<sdf version='1.9'>"
                f"<parent_link>{self.robot.robot_base_link_name}</parent_link>"
                f"<child_model>{self.camera.name()}</child_model>"
                f"<child_link>{self.camera.link_name}</child_link>"
                f"<topic>/{detach_camera_topic}</topic>"
                "</sdf>",
            )

        # Broadcast tf
        task.tf2_broadcaster.broadcast_tf(
            translation=self._camera_spawn_position,
            rotation=self._camera_spawn_quat_xyzw,
            xyzw=True,
            child_frame_id=self.camera.frame_id,
            parent_frame_id=self.robot.robot_base_link_name,
        )

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

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
        link = self.terrain.to_gazebo().get_link(link_name=self.terrain.link_names()[0])
        link.enable_contact_detection(True)

        # Execute a paused run to process model insertion
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
            color=self._light_color,
            distance=self._light_distance,
            visual=self._light_visual,
            radius=self._light_radius,
            np_random=task.np_random,
        )

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_defaults_object(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Configure and insert default object into the simulation
        """

        # Insert new models with random pose
        while len(self.task.object_names) < self._object_model_count:
            if self._object_model_count > 1:
                position, quat_wxyz = self.get_random_object_pose(
                    centre=self._object_spawn_position,
                    volume=self._object_random_spawn_volume,
                    np_random=task.np_random,
                )
            else:
                position = self._object_spawn_position
                quat_wxyz = (1.0, 0.0, 0.0, 0.0)

            # Create object
            object_model = self.__object_model_class(
                world=task.world,
                position=position,
                orientation=quat_wxyz,
                size=self._object_dimensions,
                radius=self._object_dimensions[0],
                length=self._object_dimensions[1],
                mass=self._object_mass,
                collision=self._object_collision,
                visual=self._object_visual,
                static=self._object_static,
                color=self._object_color,
            )

            # Expose name of the object for task (append in case of more)
            task.object_names.append(object_model.name())

            # Enable contact detection
            link = object_model.to_gazebo().get_link(
                link_name=object_model.link_names()[0]
            )
            link.enable_contact_detection(True)

        # Execute a paused run to process model insertion
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

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    # Randomization #
    def randomize_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Randomize models if needed
        """

        # Randomize robot joint positions if needed, else reset to
        self.reset_robot_joint_positions(
            task=task, randomize=self._robot_random_joint_positions
        )

        # Randomize terrain plane if needed
        if self._terrain_enable and self._terrain_model_expired():
            self.randomize_terrain(task=task)

        # Randomize light plane if needed
        if self._light_enable and self._light_model_expired():
            self.randomize_light(task=task)

        # Randomize camera if needed
        if self._camera_enable and self._camera_pose_expired():
            self.randomize_camera_pose(task=task)

        # Randomize objects if needed
        # Note: No need to randomize pose of new models because they are already spawned randomly
        self.__object_positions.clear()
        if self._object_enable:
            if self._object_models_expired():
                self.randomize_object_models(task=task)
            elif self._object_random_pose:
                self.object_random_pose(task=task)
            else:
                self.reset_default_object_pose(task=task)

        # Execute a paused run to process these randomization operations
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def reset_robot_joint_positions(
        self, task: SupportedTasks, randomize: bool = False
    ):

        gazebo_robot = self.robot.to_gazebo()

        # Get initial arm joint positions from the task (each task might need something different)
        arm_joint_positions = task.initial_arm_joint_positions
        # Add normal noise if desired
        if randomize:
            for joint_position in arm_joint_positions:
                joint_position += task.np_random.normal(
                    loc=0.0, scale=self._robot_random_joint_positions_std
                )
        # Arm joints - apply positions and 0 velocities to
        if not gazebo_robot.reset_joint_positions(
            arm_joint_positions, self.robot.arm_joint_names
        ):
            raise RuntimeError("Failed to reset robot joint positions")
        if not gazebo_robot.reset_joint_velocities(
            [0.0] * len(self.robot.arm_joint_names),
            self.robot.arm_joint_names,
        ):
            raise RuntimeError("Failed to reset robot joint velocities")
        # Send new positions also to the controller
        # Note: 'move_to_joint_positions' requires that joints need to be sorted alphabetically
        arm_joint_names = self.robot.arm_joint_names
        task.moveit2.move_to_joint_positions(
            list(
                operator.itemgetter(
                    *sorted(
                        range(len(arm_joint_names)), key=arm_joint_names.__getitem__
                    )
                )(arm_joint_positions)
            )
        )

        # Gripper joints - apply positions and 0 velocities
        if self.robot.gripper_joint_names:
            if not gazebo_robot.reset_joint_positions(
                task.initial_gripper_joint_positions, self.robot.gripper_joint_names
            ):
                raise RuntimeError("Failed to reset gripper joint positions")
            if not gazebo_robot.reset_joint_velocities(
                [0.0] * len(self.robot.gripper_joint_names),
                self.robot.gripper_joint_names,
            ):
                raise RuntimeError("Failed to reset gripper joint velocities")
            # TODO: Send gripper commands also to the controller (JointTrajectoryController). Test if needed first

        # Passive joints - apply 0 velocities to all
        if self.robot.passive_joint_names:
            if not gazebo_robot.reset_joint_velocities(
                [0.0] * len(self.robot.passive_joint_names),
                self.robot.passive_joint_names,
            ):
                raise RuntimeError("Failed to reset passive joint velocities")

    def randomize_camera_pose(self, task: SupportedTasks):

        if self.robot.is_mobile:
            # TODO: Implement camera pose randomization for mobile robots (currently not working due to detachable joint)
            raise NotImplementedError

        # Get random camera pose, centred at object position (or centre of object spawn box)
        position, quat_xyzw = self.get_random_camera_pose(
            task,
            centre=self.object_spawn_position,
            distance=self._camera_random_pose_distance,
            height=self._camera_random_pose_height_range,
        )

        # Move pose of the camera
        camera_gazebo = self.camera.to_gazebo()
        camera_gazebo.reset_base_pose(position, quat_to_wxyz(quat_xyzw))

        # Broadcast tf
        task.tf2_broadcaster.broadcast_tf(
            translation=position,
            rotation=quat_xyzw,
            xyzw=True,
            child_frame_id=self.camera.frame_id,
        )

    def get_random_camera_pose(
        self,
        task: SupportedTasks,
        centre: Tuple[float, float, float],
        distance: float,
        height: Tuple[float, float],
    ):

        # Select a random 3D position (with restricted min height)
        while True:
            position = np.array(
                [
                    task.np_random.uniform(low=-1.0, high=1.0),
                    task.np_random.uniform(low=-1.0, high=1.0),
                    task.np_random.uniform(low=height[0], high=height[1]),
                ]
            )
            # Normalize
            position /= np.linalg.norm(position)

            # Make sure it does not get spawned directly behind the robot (checking for +-22.5 deg)
            if abs(np.arctan2(position[0], position[1]) + np.pi / 2) > np.pi / 8:
                break

        # Determine orientation such that camera faces the origin
        rpy = [
            0.0,
            np.arctan2(position[2], np.linalg.norm(position[:2], 2)),
            np.arctan2(position[1], position[0]) + np.pi,
        ]
        quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()

        # Scale normal vector by distance and translate camera to point at the workspace centre
        position *= distance
        position += centre

        return position, quat_xyzw

    def randomize_terrain(self, task: SupportedTasks):

        # Remove existing terrain
        if hasattr(self, "terrain"):
            if not task.world.to_gazebo().remove_model(self.terrain.name()):
                raise RuntimeError(f"Failed to remove {self.terrain.name()}")

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
        link = self.terrain.to_gazebo().get_link(link_name=self.terrain.link_names()[0])
        link.enable_contact_detection(True)

    def randomize_light(self, task: SupportedTasks):

        # Remove existing light
        if hasattr(self, "light"):
            if not task.world.to_gazebo().remove_model(self.light.name()):
                raise RuntimeError(f"Failed to remove {self.light.name()}")

        # Create light
        self.light = self.__light_model_class(
            world=task.world,
            name="sun",
            direction=self._light_direction,
            color=self._light_color,
            distance=self._light_distance,
            visual=self._light_visual,
            radius=self._light_radius,
            np_random=task.np_random,
        )

    def reset_default_object_pose(self, task: SupportedTasks):

        assert len(task.object_names) == 1

        obj = task.world.to_gazebo().get_model(task.object_names[0]).to_gazebo()
        obj.reset_base_pose(self._object_spawn_position, (1.0, 0.0, 0.0, 0.0))
        obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def randomize_object_models(self, task: SupportedTasks):

        # Remove all existing models
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f"Failed to remove {object_name}")
            self.task.object_names.clear()

        # Insert new models with random pose
        while len(self.task.object_names) < self._object_model_count:
            position, quat_random = self.get_random_object_pose(
                centre=self._object_spawn_position,
                volume=self._object_random_spawn_volume,
                np_random=task.np_random,
            )
            try:
                model_name = ""
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
                link = model.to_gazebo().get_link(link_name=model.link_names()[0])
                link.enable_contact_detection(True)
            except Exception as ex:
                print(
                    "Model "
                    + model_name
                    + " could not be insterted. Reason: "
                    + str(ex)
                )

    def object_random_pose(self, task: SupportedTasks):

        for object_name in self.task.object_names:
            position, quat_random = self.get_random_object_pose(
                centre=self._object_spawn_position,
                volume=self._object_random_spawn_volume,
                np_random=task.np_random,
            )
            obj = task.world.to_gazebo().get_model(object_name).to_gazebo()
            obj.reset_base_pose(position, quat_random)
            obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.__object_positions[object_name] = position

    def get_random_object_pose(
        self,
        centre,
        volume,
        np_random,
        name: str = "",
        min_distance_to_other_objects: float = 0.25,
        min_distance_decay_factor: float = 0.9,
    ):

        is_too_close = True
        while is_too_close:
            position = [
                centre[0] + np_random.uniform(-volume[0] / 2, volume[0] / 2),
                centre[1] + np_random.uniform(-volume[1] / 2, volume[1] / 2),
                centre[2] + np_random.uniform(-volume[2] / 2, volume[2] / 2),
            ]

            # Check if position is far enough from other
            is_too_close = False
            for obj_name, obj_position in self.__object_positions.items():
                if obj_name == name:
                    # Do not compare to itself
                    continue
                if (
                    distance.euclidean(position, obj_position)
                    < min_distance_to_other_objects
                ):
                    min_distance_to_other_objects *= min_distance_decay_factor
                    is_too_close = True
                    break

        quat = np_random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)

        return position, quat

    # External overrides #
    def external_overrides(self, task: SupportedTasks):
        """
        Perform external overrides from either task level or environment before initialising/randomising the task
        """

        # Override number of objects in the scene if task requires it (e.g. if curriculum has this functionality)
        if hasattr(task, "object_count_override"):
            self._object_model_count = task.object_count_override

    # Post-randomization #
    def post_randomization(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """ """

        object_overlapping_ok = False
        if hasattr(task, "camera_sub"):
            # Execute steps until observation after reset is available
            task.camera_sub.reset_new_observation_checker()
            while not task.camera_sub.new_observation_available():
                if task._verbose:
                    print("Waiting for new observation after reset")
                if not gazebo.run(paused=False):
                    raise RuntimeError("Failed to execute a running Gazebo run")
                object_overlapping_ok = self.check_object_overlapping(task=task)
        else:
            # Otherwise execute exactly one unpaused steps to update JointStatePublisher (and potentially others)
            if not gazebo.run(paused=False):
                raise RuntimeError("Failed to execute a running Gazebo run")
            object_overlapping_ok = self.check_object_overlapping(task=task)

        # Make sure no objects are overlapping (intersections between collision geometry)
        attemps = 0
        while not object_overlapping_ok and attemps < 5:
            attemps += 1
            if task._verbose:
                print("Objects overlapping, trying new positions")
            if not gazebo.run(paused=False):
                raise RuntimeError("Failed to execute a running Gazebo run")
            object_overlapping_ok = self.check_object_overlapping(task=task)

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
                        centre=self._object_spawn_position,
                        volume=self._object_random_spawn_volume,
                        np_random=task.np_random,
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
