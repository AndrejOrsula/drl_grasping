import abc
import multiprocessing
import sys
from itertools import count
from threading import Thread
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rclpy
from gym_ignition.base.task import Task
from gym_ignition.utils.typing import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
    Reward,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from drl_grasping.envs.control import MoveIt2, MoveIt2Gripper, MoveIt2Servo
from drl_grasping.envs.models.robots import get_robot_model_class
from drl_grasping.envs.utils import Tf2Broadcaster, Tf2Listener
from drl_grasping.envs.utils.conversions import orientation_6d_to_quat
from drl_grasping.envs.utils.gazebo import *
from drl_grasping.envs.utils.math import quat_mul


class Manipulation(Task, Node, abc.ABC):
    _ids = count(0)

    def __init__(
        self,
        agent_rate: float,
        robot_model: str,
        workspace_frame_id: str,
        workspace_centre: Tuple[float, float, float],
        workspace_volume: Tuple[float, float, float],
        ignore_new_actions_while_executing: bool,
        use_servo: bool,
        scaling_factor_translation: float,
        scaling_factor_rotation: float,
        restrict_position_goal_to_workspace: bool,
        enable_gripper: bool,
        num_threads: int,
        **kwargs,
    ):

        # Get next ID for this task instance
        self.id = next(self._ids)

        # Initialize the Task base class
        Task.__init__(self, agent_rate=agent_rate)

        # Initialize ROS 2 context (if not done before)
        try:
            rclpy.init()
        except Exception as e:
            if not rclpy.ok():
                sys.exit(f"ROS 2 context could not be initialised: {e}")

        # Initialize ROS 2 Node base class
        Node.__init__(self, f"drl_grasping_{self.id}")

        # Create callback group that allows execution of callbacks in parallel without restrictions
        self._callback_group = ReentrantCallbackGroup()

        # Create executor
        if num_threads == 1:
            executor = SingleThreadedExecutor()
        elif num_threads > 1:
            executor = MultiThreadedExecutor(
                num_threads=num_threads,
            )
        else:
            executor = MultiThreadedExecutor(num_threads=multiprocessing.cpu_count())

        # Add this node to the executor
        executor.add_node(self)

        # Spin this node in background thread(s)
        self._executor_thread = Thread(target=executor.spin, daemon=True, args=())
        self._executor_thread.start()

        # Store passed arguments for later use
        self.workspace_centre = workspace_centre
        self.workspace_volume = workspace_volume
        self.__restrict_position_goal_to_workspace = restrict_position_goal_to_workspace
        self._use_servo = use_servo
        self.__scaling_factor_translation = scaling_factor_translation
        self.__scaling_factor_rotation = scaling_factor_rotation
        self._enable_gripper = enable_gripper

        # Get workspace bounds, useful is many computations
        workspace_volume_half = (
            workspace_volume[0] / 2,
            workspace_volume[1] / 2,
            workspace_volume[2] / 2,
        )
        self.workspace_min_bound = (
            self.workspace_centre[0] - workspace_volume_half[0],
            self.workspace_centre[1] - workspace_volume_half[1],
            self.workspace_centre[2] - workspace_volume_half[2],
        )
        self.workspace_max_bound = (
            self.workspace_centre[0] + workspace_volume_half[0],
            self.workspace_centre[1] + workspace_volume_half[1],
            self.workspace_centre[2] + workspace_volume_half[2],
        )

        # Get class of the robot model based on passed argument
        self.robot_model_class = get_robot_model_class(robot_model)

        # Determine robot name and prefix based on current ID of the task
        self.robot_prefix = self.robot_model_class.DEFAULT_PREFIX
        if 0 == self.id:
            self.robot_name = self.robot_model_class.ROBOT_MODEL_NAME
        else:
            self.robot_name = f"{self.robot_model_class.ROBOT_MODEL_NAME}{self.id}"
            if self.robot_prefix.endswith("_"):
                self.robot_prefix = f"{self.robot_prefix[:-1]}{self.id}_"
            elif self.robot_prefix.empty():
                self.robot_prefix = f"robot{self.id}_"

        # Names of specific robot links, useful all around the code
        self.robot_base_link_name = self.robot_model_class.get_robot_base_link_name(
            self.robot_prefix
        )
        self.robot_arm_base_link_name = self.robot_model_class.get_arm_base_link_name(
            self.robot_prefix
        )
        self.robot_ee_link_name = self.robot_model_class.get_ee_link_name(
            self.robot_prefix
        )
        self.robot_arm_link_names = self.robot_model_class.get_arm_link_names(
            self.robot_prefix
        )
        self.robot_gripper_link_names = self.robot_model_class.get_gripper_link_names(
            self.robot_prefix
        )
        self.robot_arm_joint_names = self.robot_model_class.get_arm_joint_names(
            self.robot_prefix
        )
        self.robot_gripper_joint_names = self.robot_model_class.get_gripper_joint_names(
            self.robot_prefix
        )

        # Get exact name substitution of the frame for workspace
        self.workspace_frame_id = self.substitute_special_frame(workspace_frame_id)

        # Specify initial positions (default configuration is used here)
        self.initial_arm_joint_positions = (
            self.robot_model_class.DEFAULT_ARM_JOINT_POSITIONS
        )
        self.initial_gripper_joint_positions = (
            self.robot_model_class.DEFAULT_GRIPPER_JOINT_POSITIONS
        )

        # Names of important models (in addition to robot model)
        self.terrain_name = "terrain"
        self.object_names = []

        # Setup listener and broadcaster of transforms via tf2
        self.tf2_listener = Tf2Listener(node=self)
        self.tf2_broadcaster = Tf2Broadcaster(node=self)

        # MoveIt 2 for motion planning with arm (always needed at least for joint position resets)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.robot_arm_joint_names,
            base_link_name=self.robot_arm_base_link_name,
            end_effector_name=self.robot_ee_link_name,
            execute_via_moveit=False,
            ignore_new_calls_while_executing=ignore_new_actions_while_executing,
            callback_group=self._callback_group,
        )
        # MoveIt2 real-time control (servo)
        if self._use_servo:
            self.servo = MoveIt2Servo(
                node=self,
                frame_id=self.robot_arm_base_link_name,
                linear_speed=scaling_factor_translation,
                angular_speed=scaling_factor_rotation,
                callback_group=self._callback_group,
            )
        # Gripper interface
        if self._enable_gripper:
            self.gripper = MoveIt2Gripper(
                node=self,
                gripper_joint_names=self.robot_gripper_joint_names,
                open_gripper_joint_positions=self.robot_model_class.OPEN_GRIPPER_JOINT_POSITIONS,
                closed_gripper_joint_positions=self.robot_model_class.CLOSED_GRIPPER_JOINT_POSITIONS,
                skip_planning=True,
                ignore_new_calls_while_executing=ignore_new_actions_while_executing,
                callback_group=self._callback_group,
            )

        # Initialize task and randomizer overrides (e.g. from curriculum)
        # Both of these are consumed at the beginning of reset
        self.__task_parameter_overrides: Dict[str, any] = {}
        self._randomizer_parameter_overrides: Dict[str, any] = {}

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        action_space = self.create_action_space()
        observation_space = self.create_observation_space()

        return action_space, observation_space

    def create_action_space(self) -> ActionSpace:

        raise NotImplementedError()

    def create_observation_space(self) -> ObservationSpace:

        raise NotImplementedError()

    def set_action(self, action: Action):

        raise NotImplementedError()

    def get_observation(self) -> Observation:

        raise NotImplementedError()

    def get_reward(self) -> Reward:

        raise NotImplementedError()

    def is_done(self) -> bool:

        raise NotImplementedError()

    def reset_task(self):

        self.__consume_parameter_overrides()

    # Helper functions #
    def get_relative_ee_position(
        self, translation: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:

        # Scale relative action to metric units
        translation = self.scale_relative_translation(translation)
        # Get current position
        current_position = self.get_ee_position()
        # Compute target position
        target_position = (
            current_position[0] + translation[0],
            current_position[1] + translation[1],
            current_position[2] + translation[2],
        )

        # Restrict target position to a limited workspace, if desired
        if self.__restrict_position_goal_to_workspace:
            target_position = self.restrict_position_goal_to_workspace(target_position)

        return target_position

    def get_relative_ee_orientation(
        self,
        rotation: Union[
            float,
            Tuple[float, float, float, float],
            Tuple[float, float, float, float, float, float],
        ],
        representation: str = "quat",
    ) -> Tuple[float, float, float, float]:

        # Get current orientation
        current_quat_xyzw = self.get_ee_orientation()

        # For 'z' representation, result should always point down
        # Therefore, create a new quatertnion that contains only yaw component
        if "z" == representation:
            current_yaw = Rotation.from_quat(current_quat_xyzw).as_euler("xyz")[2]
            current_quat_xyzw = Rotation.from_euler(
                "xyz", [np.pi, 0, current_yaw]
            ).as_quat()

        # Convert relative orientation representation to quaternion
        relative_quat_xyzw = None
        if "quat" == representation:
            relative_quat_xyzw = rotation
        elif "6d" == representation:
            vectors = tuple(
                rotation[x : x + 3] for x, _ in enumerate(rotation) if x % 3 == 0
            )
            relative_quat_xyzw = orientation_6d_to_quat(vectors[0], vectors[1])
        elif "z" == representation:
            rotation = self.scale_relative_rotation(rotation)
            relative_quat_xyzw = Rotation.from_euler("xyz", [0, 0, rotation]).as_quat()

        # Compute target position (combine quaternions)
        target_quat_xyzw = quat_mul(current_quat_xyzw, relative_quat_xyzw)

        # Normalise quaternion (should not be needed, but just to be safe)
        target_quat_xyzw /= np.linalg.norm(target_quat_xyzw)

        return target_quat_xyzw

    def scale_relative_translation(
        self, translation: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:

        return (
            self.__scaling_factor_translation * translation[0],
            self.__scaling_factor_translation * translation[1],
            self.__scaling_factor_translation * translation[2],
        )

    def scale_relative_rotation(
        self,
        rotation: Union[float, Tuple[float, float, float], np.floating, np.ndarray],
    ) -> float:

        if not hasattr(rotation, "__len__"):
            return self.__scaling_factor_rotation * rotation
        else:
            return (
                self.__scaling_factor_rotation * rotation[0],
                self.__scaling_factor_rotation * rotation[1],
                self.__scaling_factor_rotation * rotation[2],
            )

    def restrict_position_goal_to_workspace(
        self, position: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:

        return (
            min(
                self.workspace_max_bound[0],
                max(
                    self.workspace_min_bound[0],
                    position[0],
                ),
            ),
            min(
                self.workspace_max_bound[1],
                max(
                    self.workspace_min_bound[1],
                    position[1],
                ),
            ),
            min(
                self.workspace_max_bound[2],
                max(
                    self.workspace_min_bound[2],
                    position[2],
                ),
            ),
        )

    def get_ee_pose(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """
        Return the current pose of the end effector with respect to arm base link.
        """

        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_position, ee_quat_xyzw = get_model_pose(
                world=self.world,
                model=robot_model,
                link=self.robot_ee_link_name,
                xyzw=True,
            )
            return transform_change_reference_frame_pose(
                world=self.world,
                position=ee_position,
                quat=ee_quat_xyzw,
                target_model=robot_model,
                target_link=self.robot_arm_base_link_name,
                xyzw=True,
            )
        except Exception as e:
            self.get_logger().warn(
                f"Cannot get end effector pose from Gazebo ({e}), using tf2..."
            )
            transform = self.tf2_listener.lookup_transform_sync(
                source_frame=self.robot_ee_link_name,
                target_frame=self.robot_arm_base_link_name,
                retry=False,
            )
            if transform is not None:
                return (
                    (
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ),
                    (
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                        transform.rotation.w,
                    ),
                )
            else:
                self.get_logger().error(
                    "Cannot get pose of the end effector (default values are returned)"
                )
                return (
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )

    def get_ee_position(self) -> Tuple[float, float, float]:
        """
        Return the current position of the end effector with respect to arm base link.
        """

        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_position = get_model_position(
                world=self.world,
                model=robot_model,
                link=self.robot_ee_link_name,
            )
            return transform_change_reference_frame_position(
                world=self.world,
                position=ee_position,
                target_model=robot_model,
                target_link=self.robot_arm_base_link_name,
            )
        except Exception as e:
            self.get_logger().warn(
                f"Cannot get end effector position from Gazebo ({e}), using tf2..."
            )
            transform = self.tf2_listener.lookup_transform_sync(
                source_frame=self.robot_ee_link_name,
                target_frame=self.robot_arm_base_link_name,
                retry=False,
            )
            if transform is not None:
                return (
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                )
            else:
                self.get_logger().error(
                    "Cannot get position of the end effector (default values are returned)"
                )
                return (0.0, 0.0, 0.0)

    def get_ee_orientation(self) -> Tuple[float, float, float, float]:
        """
        Return the current xyzw quaternion of the end effector with respect to arm base link.
        """

        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_quat_xyzw = get_model_orientation(
                world=self.world,
                model=robot_model,
                link=self.robot_ee_link_name,
                xyzw=True,
            )
            return transform_change_reference_frame_orientation(
                world=self.world,
                quat=ee_quat_xyzw,
                target_model=robot_model,
                target_link=self.robot_arm_base_link_name,
                xyzw=True,
            )
        except Exception as e:
            self.get_logger().warn(
                f"Cannot get end effector orientation from Gazebo ({e}), using tf2..."
            )
            transform = self.tf2_listener.lookup_transform_sync(
                source_frame=self.robot_ee_link_name,
                target_frame=self.robot_arm_base_link_name,
                retry=False,
            )
            if transform is not None:
                return (
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                )
            else:
                self.get_logger().error(
                    "Cannot get orientation of the end effector (default values are returned)"
                )
                return (0.0, 0.0, 0.0, 1.0)

    def get_object_position(
        self, object_model: Union[ModelWrapper, str]
    ) -> Tuple[float, float, float]:
        """
        Return the current position of an object with respect to arm base link.
        Note: Only simulated objects are currently supported.
        """

        try:
            object_position = get_model_position(
                world=self.world,
                model=object_model,
            )
            return transform_change_reference_frame_position(
                world=self.world,
                position=object_position,
                target_model=self.robot_name,
                target_link=self.robot_arm_base_link_name,
            )
        except Exception as e:
            self.get_logger().error(
                f"Cannot get position of {object_model} object (default values are returned): {e}"
            )
            return (0.0, 0.0, 0.0)

    def get_object_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Return the current position of all objects with respect to arm base link.
        Note: Only simulated objects are currently supported.
        """

        object_positions = {}

        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            robot_arm_base_link = robot_model.get_link(
                link_name=self.robot_arm_base_link_name
            )
            for object_name in self.object_names:
                object_position = get_model_position(
                    world=self.world,
                    model=object_name,
                )
                object_positions[
                    object_name
                ] = transform_change_reference_frame_position(
                    world=self.world,
                    position=object_position,
                    target_model=robot_model,
                    target_link=robot_arm_base_link,
                )
        except Exception as e:
            self.get_logger().error(
                f"Cannot get positions of all objects (empty Dict is returned): {e}"
            )

        return object_positions

    def substitute_special_frame(self, frame_id: str) -> str:

        if "arm_base_link" == frame_id:
            return self.robot_arm_base_link_name
        elif "base_link" == frame_id:
            return self.robot_base_link_name
        elif "end_effector" == frame_id:
            return self.robot_ee_link_name
        elif "world" == frame_id:
            try:
                # In Gazebo, where multiple worlds are allowed
                return self.world.to_gazebo().name()
            except Exception as e:
                self.get_logger().warn(f"")
                # Otherwise (e.g. real world)
                return "drl_grasping_world"
        else:
            return frame_id

    def add_parameter_overrides(self, parameter_overrides: Dict[str, any]):

        self.add_task_parameter_overrides(parameter_overrides)
        self.add_randomizer_parameter_overrides(parameter_overrides)

    def add_task_parameter_overrides(self, parameter_overrides: Dict[str, any]):

        self.__task_parameter_overrides.update(parameter_overrides)

    def add_randomizer_parameter_overrides(self, parameter_overrides: Dict[str, any]):

        self._randomizer_parameter_overrides.update(parameter_overrides)

    def __consume_parameter_overrides(self):

        for key, value in self.__task_parameter_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            elif hasattr(self, f"__{key}"):
                setattr(self, f"__{key}", value)
            else:
                self.get_logger().error(
                    f"Override '{key}' is not supperted by the task."
                )

        self.__task_parameter_overrides.clear()
