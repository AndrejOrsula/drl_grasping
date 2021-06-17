from drl_grasping.envs import tasks, models
from drl_grasping.utils import Tf2Broadcaster
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.rbd import conversions
from scenario import gazebo as scenario
from scipy.spatial.transform import Rotation
from typing import Union, Tuple
import abc
import numpy as np


# Tasks that are supported by this randomizer (used primarily for type hinting)
SupportedTasks = Union[tasks.FidgetSpin, tasks.FidgetSpinOctree]


class FidgetSpinGazeboEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer,
                                    randomizers.abc.PhysicsRandomizer,
                                    randomizers.abc.TaskRandomizer,
                                    abc.ABC):
    """
    Randomizer for FidgetSpin task.
    """

    def __init__(self,
                 env: MakeEnvCallable,
                 physics_rollouts_num: int = 0,
                 hand_random_joint_positions: bool = True,
                 hand_random_joint_positions_std: float = 0.2,
                 spinner_random_pose: bool = True,
                 camera_pose_rollouts_num: int = 0,
                 camera_random_pose_distance: float = 1.0,
                 camera_random_pose_height_range: Tuple[float, float] = (
                     0.1, 0.7),
                 camera_noise_mean: float = None,
                 camera_noise_stddev: float = None,
                 verbose: bool = False,
                 **kwargs):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(self,
                                                   randomize_after_rollouts_num=physics_rollouts_num)
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self,
                                                           env=env,
                                                           physics_randomizer=self,
                                                           **kwargs)

        # Randomizers, their frequency and counters for different randomizers
        self._hand_random_joint_positions = hand_random_joint_positions
        self._camera_pose_rollouts_num = camera_pose_rollouts_num
        self._camera_pose_rollout_counter = camera_pose_rollouts_num

        # Additional parameters
        self._hand_random_joint_positions_std = hand_random_joint_positions_std
        self._spinner_random_pose = spinner_random_pose
        self._camera_random_pose_distance = camera_random_pose_distance
        self._camera_random_pose_height_range = camera_random_pose_height_range
        self._camera_noise_mean = camera_noise_mean
        self._camera_noise_stddev = camera_noise_stddev
        self._verbose = verbose

        # Flag that determines whether environment has already been initialised
        self.__env_initialised = False

    # ===========================
    # PhysicsRandomizer interface
    # ===========================

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        gravity_z = task.np_random.normal(loc=-9.80665, scale=0.02)
        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        # Get gazebo instance associated with the task
        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")
        gazebo = kwargs["gazebo"]

        if not self.__env_initialised:
            # Broadcaster of tf (world -> hand, world -> camera)
            self._tf2_broadcaster = Tf2Broadcaster(
                node_name=f'drl_grasping_camera_tf_broadcaster_{task.id}')

            # Initialise all models and world plugins
            self.init_models(task=task,
                             gazebo=gazebo)

            # Insert world plugins
            if task._insert_scene_broadcaster_plugin:
                task.world.to_gazebo().insert_world_plugin("ignition-gazebo-scene-broadcaster-system",
                                                           "ignition::gazebo::systems::SceneBroadcaster")
            if task._insert_user_commands_plugin:
                task.world.to_gazebo().insert_world_plugin("ignition-gazebo-user-commands-system",
                                                           "ignition::gazebo::systems::UserCommands")
            self.__env_initialised = True

        # Randomize models if needed
        self.randomize_models(task=task,
                              gazebo=gazebo)

        if hasattr(task, 'camera_sub'):
            # Execute steps until observation after reset is available
            task.camera_sub.reset_new_observation_checker()
            while not task.camera_sub.new_observation_available():
                if self._verbose:
                    print("Waiting for new observation after reset")
                if not gazebo.run(paused=False):
                    raise RuntimeError(
                        "Failed to execute a running Gazebo run")
        else:
            # Otherwise execute exactly one unpaused steps to update JointStatePublisher (and potentially others)
            if not gazebo.run(paused=False):
                raise RuntimeError("Failed to execute a running Gazebo run")

    def init_models(self,
                    task: SupportedTasks,
                    gazebo: scenario.GazeboSimulator):
        """
        Initialise all models at beginning.
        All models that are re-spawned with randomizers are ignored here.
        """

        model_names = task.world.to_gazebo().model_names()
        if len(model_names) > 0:
            print(f"World currently contains models: {model_names}")

        # Insert hand
        if task.hand_name is None:
            print("Inserting hand")
            self.add_hand(task=task,
                          gazebo=gazebo)

        # Insert spinner
        if task.spinner_name is None:
            print("Inserting spinner")
            self.add_spinner(task=task,
                             gazebo=gazebo)

        # Insert camera (if enabled)
        if task._camera_enable and task.camera_name is None:
            print("Inserting camera")
            self.add_camera(task=task,
                            gazebo=gazebo)

        # Invisible collision plane at the bottom of the world that avoids out of bounds errors
        self.add_invisible_world_bottom_collision_plane(task=task,
                                                        gazebo=gazebo)

    def add_hand(self,
                 task: SupportedTasks,
                 gazebo: scenario.GazeboSimulator):

        self._hand = models.ShadowHand(world=task.world,
                                       position=task._hand_position,
                                       orientation=conversions.Quaternion.to_wxyz(
                                           task._hand_quat_xyzw))
        task.hand_name = self._hand.name()
        task.hand_joint_names = self._hand.get_joint_names()

        hand_base_frame_id = self._hand.link_names()[0]
        self._tf2_broadcaster.broadcast_tf(translation=task._hand_position,
                                           rotation=task._hand_quat_xyzw,
                                           xyzw=True,
                                           child_frame_id=hand_base_frame_id)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_spinner(self,
                    task: SupportedTasks,
                    gazebo: scenario.GazeboSimulator):

        spinner = models.FidgetSpinner(world=task.world,
                                       position=task._spinner_position,
                                       orientation=conversions.Quaternion.to_wxyz(
                                           task._spinner_quat_xyzw))
        task.spinner_name = spinner.name()

        spinner_base_frame_id = spinner.link_names()[0]
        self._tf2_broadcaster.broadcast_tf(translation=task._spinner_position,
                                           rotation=task._spinner_quat_xyzw,
                                           xyzw=True,
                                           child_frame_id=spinner_base_frame_id)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_camera(self,
                   task: SupportedTasks,
                   gazebo: scenario.GazeboSimulator):

        # First add a sensor plugin to the world to enable
        task.world.to_gazebo().insert_world_plugin("libignition-gazebo-sensors-system.so",
                                                   "ignition::gazebo::systems::Sensors",
                                                   f"""
                                                    <sdf version="1.7">
                                                        <render_engine>{task._camera_render_engine}</render_engine>
                                                    </sdf>
                                                    """)

        # Create camera
        camera = models.Camera(world=task.world,
                               position=task._camera_position,
                               orientation=conversions.Quaternion.to_wxyz(
                                   task._camera_quat_xyzw),
                               camera_type=task._camera_type,
                               width=task._camera_width,
                               height=task._camera_height,
                               update_rate=task._camera_update_rate,
                               horizontal_fov=task._camera_horizontal_fov,
                               vertical_fov=task._camera_vertical_fov,
                               clip_color=task._camera_clip_color,
                               clip_depth=task._camera_clip_depth,
                               noise_mean=self._camera_noise_mean,
                               noise_stddev=self._camera_noise_stddev,
                               ros2_bridge_color=task._camera_ros2_bridge_color,
                               ros2_bridge_depth=task._camera_ros2_bridge_depth,
                               ros2_bridge_points=task._camera_ros2_bridge_points)
        task.camera_name = camera.name()

        camera_base_frame_id = camera.frame_id()
        self._tf2_broadcaster.broadcast_tf(translation=task._camera_position,
                                           rotation=task._camera_quat_xyzw,
                                           xyzw=True,
                                           child_frame_id=camera_base_frame_id)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_invisible_world_bottom_collision_plane(self,
                                                   task: SupportedTasks,
                                                   gazebo: scenario.GazeboSimulator):

        models.Plane(world=task.world,
                     position=(0.0, 0.0, -1.0),
                     orientation=(1.0, 0.0, 0.0, 0.0),
                     direction=(0.0, 0.0, 1.0),
                     collision=True,
                     friction=10.0,
                     visual=False)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def randomize_models(self,
                         task: SupportedTasks,
                         gazebo: scenario.GazeboSimulator):
        """
        Randomize models if needed.
        """

        # Randomize hand joint positions if needed, else reset
        if self.hand_joint_position_randomizer_enabled():
            self.hand_random_joint_positions(task=task)
        else:
            self.reset_hand_joint_positions(task=task)

        # Randomize camera if needed
        if task._camera_enable and self.camera_pose_expired():
            self.randomize_camera_pose(task=task)

        # Randomize fidget spinner position
        if self.spinner_poses_randomizer_enabled():
            self.spinner_random_pose(task=task)
        else:
            self.reset_spinner_pose(task=task)

        # Execute a paused run to process these randomization operations
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def hand_random_joint_positions(self,
                                    task: SupportedTasks):

        # Get random joint positions around the initial position
        joint_positions = [joint_position +
                           task.np_random.normal(loc=0.0,
                                                 scale=self._hand_random_joint_positions_std)
                           for joint_position in task._hand_initial_joint_positions]

        hand = task.world.to_gazebo().get_model(task.hand_name)
        if not hand.to_gazebo().reset_joint_positions(joint_positions):
            raise RuntimeError("Failed to reset hand joint positions")
        if not hand.to_gazebo().reset_joint_velocities([0.0] * len(joint_positions)):
            raise RuntimeError("Failed to reset hand joint velocities")

        # Reset joint effort
        hand.set_joint_generalized_force_targets([0.0] * len(task._hand_initial_joint_positions))

    def reset_hand_joint_positions(self,
                                   task: SupportedTasks):

        hand = task.world.to_gazebo().get_model(task.hand_name)
        if not hand.to_gazebo().reset_joint_positions(task._hand_initial_joint_positions):
            raise RuntimeError("Failed to reset hand joint positions")
        if not hand.to_gazebo().reset_joint_velocities([0.0] * len(task._hand_initial_joint_positions)):
            raise RuntimeError("Failed to reset hand joint velocities")

        # Reset joint effort
        hand.set_joint_generalized_force_targets([0.0] * len(task._hand_initial_joint_positions))

    def randomize_camera_pose(self,
                              task: SupportedTasks):

        # Get random camera pose, centred at spinner position (or centre of spinner spawn box)
        position, quat_xyzw = self.get_random_camera_pose(
            task,
            centre=task._workspace_centre,
            distance=self._camera_random_pose_distance,
            height=self._camera_random_pose_height_range)

        # Move pose of the camera
        camera = task.world.to_gazebo().get_model(task.camera_name)
        camera.to_gazebo().reset_base_pose(position,
                                           conversions.Quaternion.to_wxyz(quat_xyzw))

        camera_base_frame_id = models.Camera.frame_id_name(task.camera_name)
        self._tf2_broadcaster.broadcast_tf(translation=position,
                                           rotation=quat_xyzw,
                                           xyzw=True,
                                           child_frame_id=camera_base_frame_id)

    def get_random_camera_pose(self,
                               task: SupportedTasks,
                               centre: Tuple[float, float, float],
                               distance: float,
                               height: Tuple[float, float]):

        # Select a random 3D position (with restricted min height)
        while True:
            position = np.array([task.np_random.uniform(low=-1.0, high=1.0),
                                 task.np_random.uniform(low=-1.0, high=1.0),
                                 task.np_random.uniform(low=height[0], high=height[1])])
            # Normalize
            position /= np.linalg.norm(position)

            # Make sure it does not get spawned directly behind the hand (checking for +-22.5 deg)
            if abs(np.arctan2(position[0], position[1]) + np.pi/2) > np.pi/8:
                break

        # Determine orientation such that camera faces the origin
        rpy = [0.0,
               np.arctan2(position[2], np.linalg.norm(position[:2], 2)),
               np.arctan2(position[1], position[0]) + np.pi]
        quat_xyzw = Rotation.from_euler('xyz', rpy).as_quat()

        # Scale normal vector by distance and translate camera to point at the workspace centre
        position *= distance
        position += centre

        return position, quat_xyzw

    def reset_spinner_pose(self,
                           task: SupportedTasks):

        spinner = task.world.to_gazebo().get_model(
            task.spinner_name).to_gazebo()
        spinner.reset_base_pose(task._spinner_spawn_centre,
                                conversions.Quaternion.to_wxyz(task._spinner_quat_xyzw))
        spinner.reset_base_world_velocity([0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0])
        spinner.reset_joint_positions([0.0])
        spinner.reset_joint_velocities([0.0])

    def spinner_random_pose(self,
                            task: SupportedTasks):

        position, quat_random = self.get_random_spinner_pose(centre=task._spinner_spawn_centre,
                                                             volume=task._spinner_spawn_volume,
                                                             np_random=task.np_random)
        spinner = task.world.to_gazebo().get_model(task.spinner_name).to_gazebo()
        spinner.reset_base_pose(position, quat_random)
        spinner.reset_base_world_velocity([0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0])
        spinner.reset_joint_positions([0.0])
        spinner.reset_joint_velocities([0.0])

    def get_random_spinner_pose(self, centre, volume, np_random):

        position = [
            centre[0] + np_random.uniform(-volume[0]/2, volume[0]/2),
            centre[1] + np_random.uniform(-volume[1]/2, volume[1]/2),
            centre[2] + np_random.uniform(-volume[2]/2, volume[2]/2),
        ]
        quat = np_random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)

        return position, quat

    # ============================
    # Randomizer rollouts checking
    # ============================

    def spinner_poses_randomizer_enabled(self) -> bool:
        """
        Checks if spinner poses randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        return self._spinner_random_pose

    def hand_joint_position_randomizer_enabled(self) -> bool:
        """
        Checks if hand joint position randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        return self._hand_random_joint_positions

    def camera_pose_randomizer_enabled(self) -> bool:
        """
        Checks if camera pose randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        if self._camera_pose_rollouts_num == 0:
            return False
        else:
            return True

    def camera_pose_expired(self) -> bool:
        """
        Checks if camera pose needs to be randomized.

        Return:
            True if expired, false otherwise.
        """

        if not self.camera_pose_randomizer_enabled():
            return False

        self._camera_pose_rollout_counter += 1

        if self._camera_pose_rollout_counter >= self._camera_pose_rollouts_num:
            self._camera_pose_rollout_counter = 0
            return True

        return False

    # =============================
    # Additional features and debug
    # =============================
