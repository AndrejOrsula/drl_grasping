from drl_grasping.envs.control import MoveIt2
from drl_grasping.envs.models.robots import get_robot_model_class
from drl_grasping.envs.utils import Tf2Broadcaster
from drl_grasping.envs.utils.conversions import orientation_6d_to_quat, quat_to_xyzw
from drl_grasping.envs.utils.math import quat_mul
from gym_ignition.base import task
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from itertools import count
from scipy.spatial.transform import Rotation
from typing import Tuple, Union
import abc
import numpy as np


class Manipulation(task.Task, abc.ABC):
    _ids = count(0)

    def __init__(
        self,
        agent_rate: float,
        robot_model: str,
        workspace_frame_id: str,
        workspace_centre: Tuple[float, float, float],
        workspace_volume: Tuple[float, float, float],
        restrict_position_goal_to_workspace: bool,
        relative_position_scaling_factor: float,
        z_relative_orientation_scaling_factor: float,
        use_sim_time: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)

        # Get next ID for this task instance
        self.id = next(self._ids)

        # Store passed arguments for later use
        self.workspace_centre = workspace_centre
        self.workspace_volume = workspace_volume
        self.__restrict_position_goal_to_workspace = restrict_position_goal_to_workspace
        self.__relative_position_scaling_factor = relative_position_scaling_factor
        self.__z_relative_orientation_scaling_factor = (
            z_relative_orientation_scaling_factor
        )
        self._use_sim_time = use_sim_time
        self._verbose = verbose

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
        self.robot_gripper_link_names = self.robot_model_class.get_gripper_link_names(
            self.robot_prefix
        )

        # Get exact name substitution of the frame for workspace
        self.workspace_frame_id = self.substitute_special_frames(workspace_frame_id)

        # Specify initial positions (default configuration is used here)
        self.initial_arm_joint_positions = (
            self.robot_model_class.DEFAULT_ARM_JOINT_POSITIONS
        )
        self.initial_gripper_joint_positions = (
            self.robot_model_class.DEFAULT_GRIPPER_JOINT_POSITIONS
        )

        # Setup broadcaster of transforms via tf2
        self.tf2_broadcaster = Tf2Broadcaster(
            node_name=f"drl_grasping_tf_broadcaster_{self.id}",
            use_sim_time=self._use_sim_time,
        )

        # Setup control of the manipulator with MoveIt 2
        self.moveit2 = MoveIt2(
            robot_model=robot_model,
            node_name=f"drl_grasping_moveit2_py_{self.id}",
            use_sim_time=self._use_sim_time,
        )

        # Names of important models (in addition to robot model)
        self.terrain_name = "terrain"
        self.object_names = []

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

        raise NotImplementedError()

    # Helper functions #
    def set_position_goal(
        self,
        absolute: Union[Tuple[float, float, float], None] = None,
        relative: Union[Tuple[float, float, float], None] = None,
    ):

        target_pos = None

        if absolute is not None:
            # If absolute position is selected, directly use the action as target
            target_pos = absolute
        elif relative is not None:
            # Scale relative action to metric units
            relative_pos = self.__relative_position_scaling_factor * relative
            # Get current position
            current_pos = self.get_ee_position()

            # Compute target position
            target_pos = [
                current_pos[0] + relative_pos[0],
                current_pos[1] + relative_pos[1],
                current_pos[2] + relative_pos[2],
            ]

        if target_pos is not None:
            # Restrict target position to a limited workspace
            if self.__restrict_position_goal_to_workspace:
                centre = self.workspace_centre
                volume = self.workspace_volume
                for i in range(3):
                    target_pos[i] = min(
                        centre[i] + volume[i] / 2,
                        max(centre[i] - volume[i] / 2, target_pos[i]),
                    )
            # Set position goal
            # TODO: This needs to be fixed (get_ee_pose must return pose w.r.t arm base)
            self.moveit2.set_position_goal(target_pos, frame="drl_grasping_world")
        else:
            print("error: Neither absolute or relative position is set")

    def set_orientation_goal(
        self,
        absolute: Union[Tuple[float, ...], None] = None,
        relative: Union[Tuple[float, ...], None] = None,
        representation: str = "quat",
        xyzw: bool = True,
    ):

        target_quat_xyzw = None

        if absolute is not None:
            # Convert absolute orientation representation to quaternion
            if "quat" == representation:
                if xyzw:
                    target_quat_xyzw = absolute
                else:
                    target_quat_xyzw = quat_to_xyzw(absolute)
            elif "6d" == representation:
                vectors = tuple(
                    absolute[x : x + 3] for x, _ in enumerate(absolute) if x % 3 == 0
                )
                target_quat_xyzw = orientation_6d_to_quat(vectors[0], vectors[1])
            elif "z" == representation:
                target_quat_xyzw = Rotation.from_euler(
                    "xyz", [np.pi, 0, absolute]
                ).as_quat()

        elif relative is not None:
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
                if xyzw:
                    relative_quat_xyzw = relative
                else:
                    relative_quat_xyzw = quat_to_xyzw(relative)
            elif "6d" == representation:
                vectors = tuple(
                    relative[x : x + 3] for x, _ in enumerate(relative) if x % 3 == 0
                )
                relative_quat_xyzw = orientation_6d_to_quat(vectors[0], vectors[1])
            elif "z" == representation:
                relative *= self.__z_relative_orientation_scaling_factor
                relative_quat_xyzw = Rotation.from_euler(
                    "xyz", [0, 0, relative]
                ).as_quat()

            # Compute target position (combine quaternions)
            target_quat_xyzw = quat_mul(current_quat_xyzw, relative_quat_xyzw)

        if target_quat_xyzw is not None:
            # Normalise quaternion (should not be needed, but just to be safe)
            target_quat_xyzw /= np.linalg.norm(target_quat_xyzw)
            # Set orientation goal
            self.moveit2.set_orientation_goal(target_quat_xyzw)
        else:
            print("error: Neither absolute or relative orientation is set")

    def get_ee_position(self) -> Tuple[float, float, float]:
        """
        Return the current position of the end effector
        """

        robot = self.world.get_model(self.robot_name)
        return robot.get_link(self.robot_ee_link_name).position()

    def get_ee_orientation(self) -> Tuple[float, float, float, float]:
        """
        Return the current xyzw quaternion of the end effector
        """

        robot = self.world.get_model(self.robot_name)
        quat_wxyz = robot.get_link(self.robot_ee_link_name).orientation()
        return quat_to_xyzw(quat_wxyz)

    def substitute_special_frames(self, frame_id: str) -> str:

        if "world" == frame_id:
            try:
                return self.world.to_gazebo().name()
            except:
                return "drl_grasping_world"
        elif "base_link" == frame_id:
            return self.robot_base_link_name
        elif "arm_base_link" == frame_id:
            return self.arm_base_link_name
        elif "end_effector" == frame_id:
            return self.ee_link_name
        else:
            return frame_id
