from drl_grasping.envs import tasks, models
from drl_grasping.utils import Tf2Broadcaster
from gym_ignition import randomizers
from gym_ignition import utils
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.rbd import conversions
from gym_ignition.utils import misc
from scenario import gazebo as scenario
from scipy.spatial.transform import Rotation
from typing import Union, List, Tuple
import abc
import numpy as np
import os

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.Grasping]


class GraspingGazeboEnvRandomizerImpl(randomizers.abc.TaskRandomizer,
                                      randomizers.abc.PhysicsRandomizer,
                                      # randomizers.abc.ModelDescriptionRandomizer,
                                      abc.ABC):
    """
    Mixin that collects the implementation of task, model and physics randomizations for
    grasping environments.
    """

    def __init__(self,
                 randomize_robot_joint_positions: bool,
                 randomize_object_poses: bool,
                 object_models_rollouts_num: int,
                 ground_model_rollouts_num: int,
                 camera_pose_rollouts_num: int,
                 physics_rollouts_num: int,
                 random_object_count: int):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(self,
                                                   randomize_after_rollouts_num=physics_rollouts_num)
        # randomizers.abc.ModelDescriptionRandomizer.__init__(self)

        # TODO (low priority): TF2 - Move this to task
        # Broadcaster of tf (world -> robot, world -> camera)
        self._tf2_broadcaster = Tf2Broadcaster()

        # Randomizers, their frequency and counters for different randomizers
        self._randomize_object_poses = randomize_object_poses
        self._randomize_robot_joint_positions = randomize_robot_joint_positions
        self._object_models_rollouts_num = object_models_rollouts_num
        self._object_models_rollout_counter = object_models_rollouts_num
        self._ground_model_rollouts_num = ground_model_rollouts_num
        self._ground_model_rollout_counter = ground_model_rollouts_num
        self._camera_pose_rollouts_num = camera_pose_rollouts_num
        self._camera_pose_rollout_counter = camera_pose_rollouts_num

        # TODO (low priority): Expose these as args if needed (or move them to task)
        self._robot_position: List[float] = (0, 0, 0)
        self._robot_orientation: List[float] = (1, 0, 0, 0)
        self._camera_distance: float = 0.5
        self._camera_camera_height: List[float] = (0.5, 1.0)
        # self._default_camera_position: List[float] = (0.5, 0, 1)
        # self._default_camera_orientation: List[float] = (0, -0.707, 0, 0.707)
        self._default_camera_position: List[float] = (0.9, 0, 1)
        self._default_camera_orientation: List[float] = (0, -0.461749, 0, 0.887010)
        self._ground_position: List[float] = (0.5, 0, 0)
        self._ground_orientation: List[float] = (1, 0, 0, 0)
        self._ground_size: List[float] = (0.5, 0.5)
        self._random_object_count = random_object_count
        self._object_height: float = 0.25
        self._object_spawn_volume: List[float] = (0.5, 0.5, 0.05)
        self._default_object_position: List[float] = (0.5, 0, 0.03)
        self._default_object_orientation: List[float] = (1, 0, 0, 0)
        self._default_object_size: List[float] = (0.06, 0.06, 0.06)
        self._default_object_mass: float = 0.2

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        # Get gazebo instance associated with the task
        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")
        gazebo = kwargs["gazebo"]

        # Initialise all models
        self.init_models(task=task,
                         gazebo=gazebo,
                         robot_position=self._robot_position,
                         robot_orientation=self._robot_orientation,
                         robot_default_joint_positions=task._robot_default_joint_positions,
                         default_camera_position=self._default_camera_position,
                         default_camera_orientation=self._default_camera_orientation,
                         default_ground_position=self._ground_position,
                         default_ground_orientation=self._ground_orientation,
                         default_ground_size=self._ground_size,
                         default_object_position=self._default_object_position,
                         default_object_orientation=self._default_object_orientation,
                         default_object_size=self._default_object_size,
                         default_object_mass=self._default_object_mass)

        # Randomize models if needed
        self.randomize_models(task=task,
                              gazebo=gazebo,
                              robot_default_joint_positions=task._robot_default_joint_positions,
                              ground_position=self._ground_position,
                              ground_orientation=self._ground_orientation,
                              ground_size=self._ground_size,
                              camera_distance=self._camera_distance,
                              camera_camera_height=self._camera_camera_height,
                              random_object_count=self._random_object_count,
                              object_height=self._object_height,
                              object_spawn_volume=self._object_spawn_volume,
                              default_object_position=self._default_object_position,
                              default_object_orientation=self._default_object_orientation)

        # Execute few unpaused steps in order to:
        #   Camera gets the latest observation
        #       - This could be improved by waiting until camera publishes new topic
        #   Robot reaches its starting configuration
        #       - Using controller to reach pose because reset of joint positions causes segfault
        #       - I currently have no suggestion/solution to this. The segfault must be fixed
        # TODO (performance): Eliminate the need to run unpaused steps during reset
        for i in range(5):
            if not gazebo.run(paused=False):
                raise RuntimeError("Failed to execute a running Gazebo run")

    def init_models(self,
                    task: SupportedTasks,
                    gazebo: scenario.GazeboSimulator,
                    robot_position: List[float],
                    robot_orientation: List[float],
                    robot_default_joint_positions: List[float],
                    default_camera_position: List[float],
                    default_camera_orientation: List[float],
                    default_ground_position: List[float],
                    default_ground_orientation: List[float],
                    default_ground_size: List[float],
                    default_object_position: List[float],
                    default_object_orientation: List[float],
                    default_object_size: List[float],
                    default_object_mass: float):
        """
        Initialise all models at beginning.
        All models that are re-spawned with randomizers are ignored here.
        """

        # Insert robot (first time only)
        if task.robot_name is None:
            self.add_robot(task=task,
                           gazebo=gazebo,
                           position=robot_position,
                           orientation=robot_orientation,
                           initial_joint_positions=robot_default_joint_positions)

        # Insert camera (first time only)
        if task.camera_name is None:
            self.add_camera(task=task,
                            gazebo=gazebo,
                            position=default_camera_position,
                            orientation=default_camera_orientation)

        # Insert default ground plane if there is none and randomization is disabled
        if not self.ground_model_randomizer_enabled() and task.ground_name is None:
            self.add_default_ground(task=task,
                                    gazebo=gazebo,
                                    position=default_ground_position,
                                    orientation=default_ground_orientation,
                                    size=default_ground_size)

        # Insert default object if there is none and randomization is disabled
        if not self.object_models_randomizer_enabled() and len(task.object_names) == 0:
            self.add_default_object(task=task,
                                    gazebo=gazebo,
                                    position=default_object_position,
                                    orientation=default_object_orientation,
                                    size=default_object_size,
                                    mass=default_object_mass)

    def add_robot(self,
                  task: SupportedTasks,
                  gazebo: scenario.GazeboSimulator,
                  position: List[float],
                  orientation: List[float],
                  initial_joint_positions: List[float]):

        robot = models.Panda(world=task.world,
                             position=position,
                             orientation=orientation,
                             initial_joint_positions=initial_joint_positions)
        task.robot_name = robot.name()

        # TODO (low priority): TF2 - Move this to task
        robot_base_frame_id = robot.link_names()[0]
        self._tf2_broadcaster.broadcast_tf(translation=position,
                                           rotation=orientation,
                                           child_frame_id=robot_base_frame_id)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_camera(self,
                   task: SupportedTasks,
                   gazebo: scenario.GazeboSimulator,
                   position: List[float],
                   orientation: List[float]):

        camera = models.Camera(world=task.world,
                               position=position,
                               orientation=orientation,
                               camera_type = task._camera_type,
                               width=task._camera_width,
                               height=task._camera_height,
                               update_rate=task._camera_update_rate,
                               horizontal_fov=task._camera_horizontal_fov,
                               vertical_fov=task._camera_vertical_fov,
                               clip_color=task._camera_clip_color,
                               clip_depth=task._camera_clip_depth,
                               noise_mean=task._camera_noise_mean,
                               noise_stddev=task._camera_noise_stddev,
                               ros2_bridge_color=task._camera_ros2_bridge_color,
                               ros2_bridge_depth=task._camera_ros2_bridge_depth,
                               ros2_bridge_points=task._camera_ros2_bridge_points)
        task.camera_name = camera.name()

        # TODO (low priority): TF2 - Move this to task
        camera_base_frame_id = camera.frame_id()
        self._tf2_broadcaster.broadcast_tf(translation=position,
                                           rotation=orientation,
                                           child_frame_id=camera_base_frame_id)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_default_ground(self,
                           task: SupportedTasks,
                           gazebo: scenario.GazeboSimulator,
                           position: List[float],
                           orientation: List[float],
                           size: List[float]):

        ground = models.Ground(world=task.world,
                               position=position,
                               orientation=orientation,
                               size=size)
        task.ground_name = ground.name()

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def add_default_object(self,
                           task: SupportedTasks,
                           gazebo: scenario.GazeboSimulator,
                           position: List[float],
                           orientation: List[float],
                           size: List[float],
                           mass: float):

        box = models.Box(world=task.world,
                         position=position,
                         orientation=orientation,
                         size=size,
                         mass=mass)
        task.object_names.append(box.name())

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def randomize_models(self,
                         task: SupportedTasks,
                         gazebo: scenario.GazeboSimulator,
                         robot_default_joint_positions: List[float],
                         ground_position: List[float],
                         ground_orientation: List[float],
                         ground_size: List[float],
                         camera_distance: float,
                         camera_camera_height: List[float],
                         random_object_count: int,
                         object_height: float,
                         object_spawn_volume: List[float],
                         default_object_position: List[float],
                         default_object_orientation: List[float]):
        """
        Randomize models if needed.
        """

        # Randomize robot joint positions if needed, else reset
        if self.robot_joint_position_randomizer_enabled():
            self.randomize_robot_joint_positions(task=task)
        else:
            self.reset_robot_joint_positions(task=task,
                                             joint_positions=robot_default_joint_positions)

        # Randomize ground plane if needed
        if self.ground_model_expired():
            self.randomize_ground(task=task,
                                  position=ground_position,
                                  orientation=ground_orientation,
                                  size=ground_size)

        # Randomize camera if needed
        if self.camera_pose_expired():
            self.randomize_camera_pose(task=task,
                                       centre=ground_position,
                                       distance=camera_distance,
                                       camera_height=camera_camera_height)

        # Randomize objects if needed
        # Note: No need to randomize pose of new models because they are already spawned randomly
        spawn_centre = [ground_position[0],
                        ground_position[1],
                        ground_position[2] + object_height]
        if self.object_models_expired():
            self.randomize_object_models(task=task,
                                         object_count=random_object_count,
                                         centre=spawn_centre,
                                         volume=object_spawn_volume)
        elif self.object_poses_randomizer_enabled():
            self.randomize_object_poses(task=task,
                                        centre=spawn_centre,
                                        volume=object_spawn_volume)
        elif not self.object_models_randomizer_enabled():
            self.reset_default_object_pose(task=task,
                                           position=default_object_position,
                                           orientation=default_object_orientation)

        # Execute a paused run to process these randomization operations
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def randomize_robot_joint_positions(self,
                                        task: SupportedTasks,
                                        std_scale: float = 0.1):

        joint_positions = []
        for joint_limits in models.Panda.get_joint_limits():
            mean = (joint_limits[0] + joint_limits[1])/2
            std = std_scale*abs(joint_limits[1] - joint_limits[0])
            random_position = task.np_random.normal(loc=mean, scale=std)
            joint_positions.append(random_position)

        # # TODO: Reset of joint positions eventually causes segfault
        # robot = task.world.to_gazebo().get_model(task.robot_name)
        # if not robot.to_gazebo().reset_joint_positions(joint_positions):
        #     raise RuntimeError("Failed to reset the robot joint position")

        # Send new positions also to the controller
        finger_count = models.Panda.get_finger_count()
        task.moveit2.move_to_joint_positions(joint_positions[:-finger_count])

    def reset_robot_joint_positions(self,
                                    task: SupportedTasks,
                                    joint_positions: List[float]):

        # # TODO: Reset of joint positions eventually causes segfault
        # robot = task.world.to_gazebo().get_model(task.robot_name)
        # if not robot.to_gazebo().reset_joint_positions(joint_positions):
        #     raise RuntimeError("Failed to reset the robot joint position")

        # Send new positions also to the controller
        finger_count = models.Panda.get_finger_count()
        task.moveit2.move_to_joint_positions(joint_positions[:-finger_count])

    def randomize_camera_pose(self,
                              task: SupportedTasks,
                              centre: List[float],
                              distance: float,
                              camera_height: List[float]):

        # TODO: Take another look at this (random camera pose)
        # Get random camera pose
        position, orientation = self.get_random_camera_pose(
            task,
            centre=centre,
            distance=distance,
            camera_height=camera_height)

        # Move pose of the camera
        camera = task.world.to_gazebo().get_model(task.camera_name)
        camera.to_gazebo().reset_base_pose(position, orientation)

        # TODO (low priority): TF2 - Move this to task
        camera_base_frame_id = models.Camera.frame_id_name(task.camera_name)
        self._tf2_broadcaster.broadcast_tf(translation=position,
                                           rotation=orientation,
                                           child_frame_id=camera_base_frame_id)

    def get_random_camera_pose(self, task: SupportedTasks, centre, distance, camera_height):

        # Range [0;pi] [-pi;pi]
        theta = task.np_random.uniform(0.0, 1.0) * np.pi
        phi = task.np_random.uniform(-1.0, 1.0) * np.pi

        # Switch to cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = task.np_random.uniform(camera_height[0], camera_height[1])

        pitch = np.arctan2(z, np.sqrt(x**2+y**2))
        yaw = np.arctan2(y, x) + np.pi

        position = [x*distance + centre[0],
                    y*distance + centre[1],
                    z*distance + centre[2]]
        orientation = conversions.Quaternion.to_wxyz(
            Rotation.from_euler('xyz', [0.0, pitch, yaw],).as_quat())

        return position, orientation

    def randomize_ground(self,
                         task: SupportedTasks,
                         position: List[float],
                         orientation: List[float],
                         size: List[float]):

        # Remove existing ground
        if task.ground_name is not None:
            if not task.world.to_gazebo().remove_model(task.ground_name):
                raise RuntimeError(f"Failed to remove {task.ground_name}")

        # Add new random ground
        plane = models.RandomGround(world=task.world,
                                    position=position,
                                    orientation=orientation,
                                    size=size,
                                    np_random=task.np_random,
                                    texture_dir=os.environ.get('DRL_GRASPING_PBR_TEXTURES_DIR',
                                                               default=''))
        task.ground_name = plane.name()

    def reset_default_object_pose(self,
                                  task: SupportedTasks,
                                  position: List[float],
                                  orientation: List[float]):

        assert(len(task.object_names) == 1)

        obj = task.world.to_gazebo().get_model(task.object_names[0])
        obj.to_gazebo().reset_base_pose(position, orientation)

    def randomize_object_models(self,
                                task: SupportedTasks,
                                object_count: int,
                                centre: List[float],
                                volume: List[float]):

        # Remove all existing models
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f"Failed to remove {object_name}")
            self.task.object_names.clear()

        # Insert new models with random pose
        while len(self.task.object_names) < object_count:
            position, orientation = self.get_random_object_pose(centre=centre,
                                                                volume=volume,
                                                                np_random=task.np_random)
            try:
                model = models.RandomObject(world=task.world,
                                            position=position,
                                            orientation=orientation,
                                            np_random=task.np_random)
                self.task.object_names.append(model.name())
            except:
                # TODO (low priority): Automatically blacklist a model if Gazebo does not accept it
                pass

    def randomize_object_poses(self,
                               task: SupportedTasks,
                               centre: List[float],
                               volume: List[float]):

        for object_name in self.task.object_names:
            position, orientation = self.get_random_object_pose(centre=centre,
                                                                volume=volume,
                                                                np_random=task.np_random)
            obj = task.world.to_gazebo().get_model(object_name)
            obj.to_gazebo().reset_base_pose(position, orientation)

    def get_random_object_pose(self, centre, volume, np_random):

        position = [
            centre[0] + np_random.uniform(-volume[0]/2, volume[0]/2),
            centre[1] + np_random.uniform(-volume[1]/2, volume[1]/2),
            centre[2] + np_random.uniform(-volume[2]/2, volume[2]/2),
        ]
        orientation = conversions.Quaternion.to_wxyz(
            Rotation.from_quat([np_random.uniform(-1, 1),
                                np_random.uniform(-1, 1),
                                np_random.uniform(-1, 1),
                                np_random.uniform(-1, 1)]).as_quat())

        return position, orientation

    # ===========================
    # PhysicsRandomizer interface
    # ===========================

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        gravity_z = task.np_random.normal(loc=-9.80665, scale=0.02)

        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

    # ============================
    # Randomizer rollouts checking
    # ============================

    def object_models_randomizer_enabled(self) -> bool:
        """
        Checks if object model randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        if self._object_models_rollouts_num == 0:
            return False
        else:
            return True

    def object_models_expired(self) -> bool:
        """
        Checks if object models need to be randomized.

        Return:
            True if expired, false otherwise.
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
            True if enabled, false otherwise.
        """

        return self._randomize_object_poses

    def robot_joint_position_randomizer_enabled(self) -> bool:
        """
        Checks if robot joint position randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        return self._randomize_robot_joint_positions

    def ground_model_randomizer_enabled(self) -> bool:
        """
        Checks if ground randomizer is enabled.

        Return:
            True if enabled, false otherwise.
        """

        if self._ground_model_rollouts_num == 0:
            return False
        else:
            return True

    def ground_model_expired(self) -> bool:
        """
        Checks if ground model needs to be randomized.

        Return:
            True if expired, false otherwise.
        """

        if not self.ground_model_randomizer_enabled():
            return False

        self._ground_model_rollout_counter += 1

        if self._ground_model_rollout_counter >= self._ground_model_rollouts_num:
            self._ground_model_rollout_counter = 0
            return True

        return False

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


class GraspingGazeboEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer,
                                  GraspingGazeboEnvRandomizerImpl):
    """
    Concrete implementation of grasping environment randomization.
    """

    def __init__(self,
                 env: MakeEnvCallable,
                 randomize_robot_joint_positions: bool = False,
                 randomize_object_poses: bool = False,
                 object_models_rollouts_num: int = 0,
                 ground_model_rollouts_num: int = 0,
                 camera_pose_rollouts_num: int = 0,
                 physics_rollouts_num: int = 0,
                 random_object_count: int = 2):

        # Initialize the mixin
        GraspingGazeboEnvRandomizerImpl.__init__(
            self,
            randomize_robot_joint_positions=randomize_robot_joint_positions,
            randomize_object_poses=randomize_object_poses,
            object_models_rollouts_num=object_models_rollouts_num,
            ground_model_rollouts_num=ground_model_rollouts_num,
            camera_pose_rollouts_num=camera_pose_rollouts_num,
            physics_rollouts_num=physics_rollouts_num,
            random_object_count=random_object_count)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self,
                                                           env=env,
                                                           physics_randomizer=self)
