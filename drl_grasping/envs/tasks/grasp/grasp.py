import abc
import itertools
from typing import Dict, List, Tuple

import gym
import numpy as np
from gym_ignition.utils.typing import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
    Reward,
)
from scipy.spatial.transform import Rotation

from drl_grasping.envs.tasks.curriculums import GraspCurriculum
from drl_grasping.envs.tasks.manipulation import Manipulation
from drl_grasping.envs.utils.conversions import quat_to_xyzw


class Grasp(Manipulation, abc.ABC):
    def __init__(
        self,
        gripper_dead_zone: float,
        full_3d_orientation: bool,
        preload_replay_buffer: bool = False,
        **kwargs,
    ):

        # Initialize the Task base class
        Manipulation.__init__(
            self,
            **kwargs,
        )

        self.curriculum = GraspCurriculum(
            task=self,
            **kwargs,
        )

        # Additional parameters
        self.__gripper_dead_zone = gripper_dead_zone
        self.__full_3d_orientation = full_3d_orientation
        self.__preload_replay_buffer = preload_replay_buffer

    def create_action_space(self) -> ActionSpace:

        if self.__full_3d_orientation:
            if self._use_servo:
                # 0   - (gripper) Gripper action
                #       - Open if positive (i.e. increase width)
                #       - Close if negative (i.e. decrease width)
                # 1:4 - (x, y, z) displacement
                #       - rescaled to metric units before use
                # 4:7 - (angular_x, angular_y, angular_z) relative rotation for moveit_servo
                return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            else:
                # 0   - (gripper) Gripper action
                #       - Open if positive (i.e. increase width)
                #       - Close if negative (i.e. decrease width)
                # 1:4 - (x, y, z) displacement
                #       - rescaled to metric units before use
                # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) relative 3D orientation in "6D representation"
                return gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        else:
            # 0   - (gripper) Gripper action
            #       - Open if positive (i.e. increase width)
            #       - Close if negative (i.e. decrease width)
            # 1:4 - (x, y, z) displacement
            #       - rescaled to metric units before use
            # 4   - (yaw) relative orientation around Z
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:

        pass

    def set_action(self, action: Action):

        if self.__preload_replay_buffer:
            action = self._demonstrate_action()

        self.get_logger().debug(f"action: {action}")

        # Gripper action
        gripper_action = action[0]
        if gripper_action < -self.__gripper_dead_zone:
            self.gripper.close()
        elif gripper_action > self.__gripper_dead_zone:
            self.gripper.open()
        else:
            # No-op for the gripper if action is within the dead zone
            pass

        # Motion
        if self._use_servo:
            if self.__full_3d_orientation:
                angular = action[4:7]
            else:
                angular = [0.0, 0.0, action[4]]
            self.servo(linear=action[1:4], angular=angular)
        else:
            position = self.get_relative_ee_position(action[1:4])
            if self.__full_3d_orientation:
                quat_xyzw = self.get_relative_ee_orientation(
                    rotation=action[4:10], representation="6d"
                )
            else:
                quat_xyzw = self.get_relative_ee_orientation(
                    rotation=action[4], representation="z"
                )
            self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)

    def get_observation(self) -> Observation:

        raise NotImplementedError()

    def get_reward(self) -> Reward:

        return self.curriculum.get_reward()

    def is_done(self) -> bool:

        return self.curriculum.is_done()

    def get_info(self) -> Dict:

        info = self.curriculum.get_info()

        if self.__preload_replay_buffer:
            info.update({"actual_actions": self.__actual_actions})

        return info

    def reset_task(self):

        Manipulation.reset_task(self)
        self.curriculum.reset_task()

    # Helper functions #
    def get_touched_objects(self) -> List[str]:
        """
        Returns list of all objects that are in contact with any finger.
        """

        robot = self.world.get_model(self.robot_name).to_gazebo()
        touched_objects = []

        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()

            # Add model to list of touched objects if in contact
            for contact in finger_contacts:
                # Keep only the model name (disregard the link of object that is in collision)
                model_name = contact.body_b.split("::", 1)[0]
                if model_name not in touched_objects and any(
                    object_name in model_name for object_name in self.object_names
                ):
                    touched_objects.append(model_name)

        return touched_objects

    def get_grasped_objects(self) -> List[str]:
        """
        Returns list of all currently grasped objects.
        Grasped object must be in contact with all gripper links (fingers) and their contant normals must be dissimilar.
        """

        if self.gripper.is_open:
            # Return empty if the gripper is opened
            return []

        robot = self.world.get_model(self.robot_name)
        grasp_candidates = {}

        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.to_gazebo().get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()

            if 0 == len(finger_contacts):
                # If any of the fingers has no contact, immediately return None
                return []
            else:
                # Otherwise, add all contacted objects as grasp candidates (together with contact points - used later)
                for contact in finger_contacts:
                    # Keep only the model name (disregard the link of object that is in collision)
                    model_name = contact.body_b.split("::", 1)[0]
                    if any(
                        object_name in model_name for object_name in self.object_names
                    ):
                        if model_name not in grasp_candidates:
                            grasp_candidates[model_name] = []
                        grasp_candidates[model_name].append(contact.points)

        # Determine what grasp candidates are indeed grasped objects
        # First make sure it has contact with more than half of the fingers
        # Then make sure that their normals are dissimilar
        grasped_objects = []
        for model_name, contact_points_list in grasp_candidates.items():
            if len(contact_points_list) < len(self.robot_gripper_link_names) // 2 + 1:
                continue

            # Compute average normal of all finger-object collisions
            average_normals = []
            for contact_points in contact_points_list:
                average_normal = np.array([0.0, 0.0, 0.0])
                for point in contact_points:
                    average_normal += point.normal
                average_normal /= np.linalg.norm(average_normal)
                average_normals.append(average_normal)
            # Compare normals (via their angle) and reject candidates that are too similar
            normal_angles = []
            for n1, n2 in itertools.combinations(average_normals, 2):
                normal_angles.append(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

            # Angle between at least two normals must be larger than 0.25*pi
            sufficient_angle = 0.25 * np.pi
            for angle in normal_angles:
                if angle > sufficient_angle:
                    # If sufficient, add to list and process other candidates
                    grasped_objects.append(model_name)
                    break

        return grasped_objects

    def check_terrain_collision(self) -> bool:
        """
        Returns true if robot links are in collision with the ground.
        """

        robot_name_len = len(self.robot_name)
        for contact in self.world.get_model(self.terrain_name).contacts():
            if len(contact.body_b) > robot_name_len:
                if contact.body_b[:robot_name_len] == self.robot_name:
                    link = contact.body_b[len(self.robot_name) + 2 :]

                    if not self.robot_base_link_name == link and (
                        link in self.robot_arm_link_names
                        or link in self.robot_gripper_link_names
                    ):
                        return True
        return False

    def check_all_objects_outside_workspace(
        self,
        object_positions: Dict[str, Tuple[float, float, float]],
    ) -> bool:
        """
        Returns true if all objects are outside the workspace
        """

        return all(
            [
                object_position[0] < self.workspace_min_bound[0]
                or object_position[1] < self.workspace_min_bound[1]
                or object_position[2] < self.workspace_min_bound[2]
                or object_position[0] > self.workspace_max_bound[0]
                or object_position[1] > self.workspace_max_bound[1]
                or object_position[2] > self.workspace_max_bound[2]
                for object_position in object_positions.values()
            ]
        )

    def _demonstrate_action(self) -> np.ndarray:

        self.__actual_actions = np.zeros(self.action_space.shape)

        ee_position, ee_orientation = self.get_ee_pose()
        ee_position = np.array(ee_position)
        ee_orientation = np.array(ee_orientation)

        object_position = np.array(self.get_object_position(self.object_names[0]))

        distance = object_position - ee_position
        distance_mag = np.linalg.norm(distance)

        if distance_mag < 0.02:
            # Object is approached
            if self.gripper.is_open:
                # Gripper is currently opened and should be closed
                self.__actual_actions[0] = -1.0
                # Don't move this step
                self.__actual_actions[1:4] = np.zeros((3,))
            else:
                # Gripper is already closed, keep it that way gripper close
                self.__actual_actions[0] = -1.0
                # Move upwards
                self.__actual_actions[1:4] = np.array((0.0, 0.0, 1.0))

            # Do not change orientation in either case
            if self.__full_3d_orientation:
                # self.__actual_actions[4:10] = orientation_6d
                pass
            else:
                self.__actual_actions[4] = 0.0
        else:
            # Object is not yet approached
            # Keep the gripper open
            self.__actual_actions[0] = 1.0

            # Move towards the object
            if distance_mag > self._relative_position_scaling_factor:
                relative_position = distance / distance_mag
            else:
                relative_position = distance / self._relative_position_scaling_factor
            self.__actual_actions[1:4] = relative_position

            # Stay above object until xy position is correct
            distance_mag_xy = np.linalg.norm(distance[:2])
            if distance_mag_xy > 0.01 and ee_position[2] < 0.1:
                self.__actual_actions[3] = max(0.0, self.__actual_actions[3])

            # Orient gripper appropriately
            object_orientation = quat_to_xyzw(
                np.array(self.get_object_orientation(self.object_names[0]))
            )
            if self.__full_3d_orientation:
                # self.__actual_actions[4:10] = orientation_6d
                pass
            else:
                current_ee_yaw = Rotation.from_quat(ee_orientation).as_euler("xyz")[2]
                current_object_yaw = Rotation.from_quat(object_orientation).as_euler(
                    "xyz"
                )[2]
                yaw_diff = current_object_yaw - current_ee_yaw
                if yaw_diff > np.pi:
                    yaw_diff -= np.pi / 2
                elif yaw_diff < -np.pi:
                    yaw_diff += np.pi / 2
                yaw_diff = min(
                    1.0, 1.0 / (self._z_relative_orientation_scaling_factor / yaw_diff)
                )
                self.__actual_actions[4] = yaw_diff

        # Make sure robot does not collide with the table
        if ee_position[2] < 0.025:
            self.__actual_actions[3] = max(0.0, self.__actual_actions[3])

        return self.__actual_actions
