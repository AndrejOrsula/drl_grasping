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
from typing import Union, Tuple
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
        robot_random_joint_positions: bool = False,
        robot_random_joint_positions_std: float = 0.1,
        camera_pose_rollouts_num: int = 0,
        camera_random_pose_distance: float = 1.0,
        camera_random_pose_height_range: Tuple[float, float] = (0.1, 0.7),
        camera_noise_mean: float = None,
        camera_noise_stddev: float = None,
        terrain_type: str = "random_flat",
        terrain_model_rollouts_num: int = 1,
        object_type: str = "random_mesh",
        object_random_pose: bool = True,
        object_random_use_mesh_models: bool = True,
        object_models_rollouts_num: int = 1,
        object_random_model_count: int = 1,
        invisible_world_bottom_collision_plane: bool = True,
        visualise_workspace: bool = False,
        visualise_spawn_volume: bool = False,
        **kwargs,
    ):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=physics_rollouts_num
        )
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(
            self, env=env, physics_randomizer=self, **kwargs
        )

        # Randomizers, their frequency and counters for different randomizers
        self.robot_random_joint_positions = robot_random_joint_positions
        self.camera_pose_rollouts_num = camera_pose_rollouts_num
        self.camera_pose_rollout_counter = camera_pose_rollouts_num
        self.terrain_type = terrain_type
        self.terrain_model_rollouts_num = terrain_model_rollouts_num
        self.terrain_model_rollout_counter = terrain_model_rollouts_num
        self.object_type = object_type
        self._object_random_pose = object_random_pose
        self._object_random_use_mesh_models = object_random_use_mesh_models
        self._object_models_rollouts_num = object_models_rollouts_num
        self._object_models_rollout_counter = object_models_rollouts_num

        # Additional parameters
        self.robot_random_joint_positions_std = robot_random_joint_positions_std
        self.camera_random_pose_distance = camera_random_pose_distance
        self.camera_random_pose_height_range = camera_random_pose_height_range
        self.camera_noise_mean = camera_noise_mean
        self.camera_noise_stddev = camera_noise_stddev
        self._object_random_model_count = object_random_model_count
        self._invisible_world_bottom_collision_plane = (
            invisible_world_bottom_collision_plane
        )
        self._visualise_workspace = visualise_workspace
        self._visualise_spawn_volume = visualise_spawn_volume

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
        if task.camera_enable:
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
        if task.camera_enable:
            if task._verbose:
                print("Inserting camera into the environment...")
            self.add_camera(
                task=task, gazebo=gazebo, attach_to_robot=self.robot.is_mobile
            )

        # Insert default terrain if enabled and terrain randomization is disabled
        if task.terrain_enable and not self.terrain_model_randomizer_enabled():
            if task._verbose:
                print("Inserting default terrain into the environment...")
            self.add_default_terrain(task=task, gazebo=gazebo)

        # Insert default object if enabled and object randomization is disabled
        if task.object_enable and not self.object_models_randomizer_enabled():
            if task._verbose:
                print("Inserting default object into the environment...")
            self.add_default_object(task=task, gazebo=gazebo)

        # Insert invisible plane below the terrain to prevent objects from falling into the abyss and causing physics errors
        # TODO: Consider replacing invisiable plane with removal of all objects that are too low along z axis
        if self._invisible_world_bottom_collision_plane:
            if task._verbose:
                print(
                    "Inserting invisible plane below the terrain into the environment..."
                )
            self.add_invisible_world_bottom_collision_plane(task=task, gazebo=gazebo)

    def add_robot(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert robot into the simulation
        """

        # Instantiate robot class based on the selected model
        self.robot = task.robot_model_class(
            world=task.world,
            name=task.robot_name,
            prefix=task.robot_prefix,
            position=task.initial_robot_position,
            orientation=quat_to_wxyz(task.initial_robot_quat_xyzw),
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
            Rotation.from_quat(robot_base_link_quat_xyzw).apply(task.camera_position)
            + robot_base_link_position
        )
        camera_orientation_wrt_robot_in_world_ref = quat_mul(
            task.camera_quat_xyzw, robot_base_link_quat_xyzw, xyzw=True
        )

        # Create camera
        self.camera = models.Camera(
            world=task.world,
            position=camera_position_wrt_robot_in_world_ref,
            orientation=quat_to_wxyz(camera_orientation_wrt_robot_in_world_ref),
            camera_type=task.camera_type,
            width=task.camera_width,
            height=task.camera_height,
            update_rate=task.camera_update_rate,
            horizontal_fov=task.camera_horizontal_fov,
            vertical_fov=task.camera_vertical_fov,
            clip_color=task.camera_clip_color,
            clip_depth=task.camera_clip_depth,
            noise_mean=self.camera_noise_mean,
            noise_stddev=self.camera_noise_stddev,
            ros2_bridge_color=task.camera_publish_color,
            ros2_bridge_depth=task.camera_publish_depth,
            ros2_bridge_points=task.camera_publish_points,
        )

        # Expose name of the camera for task
        task.camera_name = self.camera.name()

        # Attach to robot
        if attach_to_robot:
            detach_camera_topic = f"{self.robot.name()}/detach_{self.camera.name()}"
            self.robot.to_gazebo().insert_model_plugin(
                "libignition-gazebo-detachable-joint-system.so",
                "ignition::gazebo::systems::DetachableJoint",
                "<sdf version='1.9'>"
                f"<parent_link>{self.robot.robot_base_link_name}</parent_link>"
                f"<child_model>{self.camera.name()}</child_model>"
                f"<child_link>{self.camera.link_name()}</child_link>"
                f"<topic>/{detach_camera_topic}</topic>"
                "</sdf>",
            )

        # Broadcast tf
        task.tf2_broadcaster.broadcast_tf(
            translation=task.camera_position,
            rotation=task.camera_quat_xyzw,
            xyzw=True,
            child_frame_id=self.camera.frame_id(),
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

        # Get model class based on the selected terrain type
        terrain_model_class = models.get_terrain_model_class(self.terrain_type)

        # Create terrain
        self.terrain = terrain_model_class(
            world=task.world,
            name=task.terrain_name,
            position=task.terrain_position,
            orientation=quat_to_wxyz(task.terrain_quat_xyzw),
            size=task.terrain_size,
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

    def add_default_object(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Configure and insert default object into the simulation
        """

        # Get model class based on the selected object type
        object_model_class = models.get_object_model_class(self.object_type)

        # Create object
        object_model = object_model_class(
            world=task.world,
            position=task._object_spawn_centre,
            orientation=quat_to_wxyz(task.object_quat_xyzw),
            size=task.object_dimensions,
            radius=task.object_dimensions[0],
            length=task.object_dimensions[1],
            mass=task.object_mass,
            collision=task.object_collision,
            visual=task.object_visual,
            static=task.object_static,
            color=task.object_color,
        )

        # Expose name of the object for task (append in case of more)
        task.object_names.append(object_model.name())

        # Enable contact detection
        link = object_model.to_gazebo().get_link(link_name=object_model.link_names()[0])
        link.enable_contact_detection(True)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_invisible_world_bottom_collision_plane(
        self, task: SupportedTasks, gazebo: scenario.GazeboSimulator
    ):
        """
        Add an infinitely large collision plane below the terrain in order to prevent object from falling into the abyss forever
        """

        models.Plane(
            world=task.world,
            position=(0.0, 0.0, -10.0),
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
            task=task, randomize=self.robot_joint_position_randomizer_enabled()
        )

        # Randomize terrain plane if needed
        if (
            task.terrain_enable
            and self.terrain_model_expired()
            and models.is_terrain_type_randomizable(self.terrain_type)
        ):
            self.randomize_terrain(task=task)

        # Randomize camera if needed
        if task.camera_enable and self.camera_pose_expired():
            self.randomize_camera_pose(task=task)

        # Randomize objects if needed
        # Note: No need to randomize pose of new models because they are already spawned randomly
        self.__object_positions.clear()
        if task.object_enable:
            if self.object_models_expired() and models.is_object_type_randomizable(
                self.object_type
            ):
                if self._object_random_use_mesh_models:
                    self.randomize_object_models(task=task)
                else:
                    self.randomize_object_primitives(task=task)
            elif self.object_poses_randomizer_enabled():
                self.object_random_pose(task=task)
            elif not self.object_models_randomizer_enabled():
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
                    loc=0.0, scale=self.robot_random_joint_positions_std
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
            centre=task.object_spawn_centre,
            distance=self.camera_random_pose_distance,
            height=self.camera_random_pose_height_range,
        )

        # Move pose of the camera
        camera = task.world.to_gazebo().get_model(task.camera_name)
        camera.to_gazebo().reset_base_pose(position, quat_to_wxyz(quat_xyzw))

        # Broadcast tf
        camera_base_frame_id = models.Camera.frame_id_name(task.camera_name)
        task.tf2_broadcaster.broadcast_tf(
            translation=position,
            rotation=quat_xyzw,
            xyzw=True,
            child_frame_id=camera_base_frame_id,
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

        # Get model class based on the selected terrain type
        terrain_model_class = models.get_terrain_model_class(self.terrain_type)

        # Create terrain
        self.terrain = terrain_model_class(
            world=task.world,
            position=task.terrain_position,
            orientation=orientation,
            size=task.terrain_size,
            np_random=task.np_random,
            texture_dir=environ.get("DRL_GRASPING_PBR_TEXTURES_DIR", default=""),
        )

        # Expose name of the terrain for task
        task.terrain_name = self.terrain.name()

        # Enable contact detection
        link = self.terrain.to_gazebo().get_link(link_name=self.terrain.link_names()[0])
        link.enable_contact_detection(True)

    def reset_default_object_pose(self, task: SupportedTasks):

        assert len(task.object_names) == 1

        obj = task.world.to_gazebo().get_model(task.object_names[0]).to_gazebo()
        obj.reset_base_pose(
            task._object_spawn_centre, quat_to_wxyz(task.object_quat_xyzw)
        )
        obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def randomize_object_models(self, task: SupportedTasks):

        # Remove all existing models
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f"Failed to remove {object_name}")
            self.task.object_names.clear()

        # Insert new models with random pose
        while len(self.task.object_names) < self._object_random_model_count:
            position, quat_random = self.get_random_object_pose(
                centre=task._object_spawn_centre,
                volume=task.object_spawn_volume,
                np_random=task.np_random,
            )
            try:
                model_name = ""
                model = models.RandomObject(
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

    def randomize_object_primitives(self, task: SupportedTasks):

        # Remove all existing models
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f"Failed to remove {object_name}")
            self.task.object_names.clear()

        # Insert new primitives with random pose
        while len(self.task.object_names) < self._object_random_model_count:
            position, quat_random = self.get_random_object_pose(
                centre=task._object_spawn_centre,
                volume=task.object_spawn_volume,
                np_random=task.np_random,
            )
            try:
                model = models.RandomPrimitive(
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
            except:
                pass

    def object_random_pose(self, task: SupportedTasks):

        for object_name in self.task.object_names:
            position, quat_random = self.get_random_object_pose(
                centre=task._object_spawn_centre,
                volume=task.object_spawn_volume,
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
            self._object_random_model_count = task.object_count_override

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
                    task.terrain_name in contact.body_b
                    and depth < terrain_allowed_penetration_depth
                ):
                    continue
                if (
                    task.robot_name in contact.body_b
                    or depth > allowed_penetration_depth
                ):
                    position, quat_random = self.get_random_object_pose(
                        centre=task._object_spawn_centre,
                        volume=task.object_spawn_volume,
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

    def object_models_randomizer_enabled(self) -> bool:
        """
        Checks if object model randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self._object_models_rollouts_num == 0:
            return False
        else:
            return True

    def object_models_expired(self) -> bool:
        """
        Checks if object models need to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.object_models_randomizer_enabled():
            return False

        self._object_models_rollout_counter += 1

        if self._object_models_rollout_counter >= self._object_models_rollouts_num:
            self._object_models_rollout_counter = 0
            return True

        return False

    def object_poses_randomizer_enabled(self) -> bool:
        """
        Checks if object poses randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        return self._object_random_pose

    def robot_joint_position_randomizer_enabled(self) -> bool:
        """
        Checks if robot joint position randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        return self.robot_random_joint_positions

    def terrain_model_randomizer_enabled(self) -> bool:
        """
        Checks if terrain randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self.terrain_model_rollouts_num == 0:
            return False
        else:
            return True

    def terrain_model_expired(self) -> bool:
        """
        Checks if terrain model needs to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.terrain_model_randomizer_enabled():
            return False

        self.terrain_model_rollout_counter += 1

        if self.terrain_model_rollout_counter >= self.terrain_model_rollouts_num:
            self.terrain_model_rollout_counter = 0
            return True

        return False

    def camera_pose_randomizer_enabled(self) -> bool:
        """
        Checks if camera pose randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """

        if self.camera_pose_rollouts_num == 0:
            return False
        else:
            return True

    def camera_pose_expired(self) -> bool:
        """
        Checks if camera pose needs to be randomized.

        Return:
            True if expired, false otherwise
        """

        if not self.camera_pose_randomizer_enabled():
            return False

        self.camera_pose_rollout_counter += 1

        if self.camera_pose_rollout_counter >= self.camera_pose_rollouts_num:
            self.camera_pose_rollout_counter = 0
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
            position=task.object_spawn_centre,
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
            name="object_spawn_volume",
            position=task._object_spawn_centre,
            orientation=(0, 0, 0, 1),
            size=task.object_spawn_volume,
            collision=False,
            visual=True,
            gui_only=True,
            static=True,
            color=color,
        )
        models.Box(
            world=task.world,
            name="object_spawn_volume_with_height",
            position=task._object_spawn_centre,
            orientation=(0, 0, 0, 1),
            size=task.object_spawn_volume,
            collision=False,
            visual=True,
            gui_only=True,
            static=True,
            color=color_with_height,
        )
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")
