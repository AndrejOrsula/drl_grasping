from drl_grasping.envs.tasks.manipulation import Manipulation
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import Tuple, List
from typing import Tuple, Union, Dict
import abc
import gym
import itertools
import numpy as np


class Grasp(Manipulation, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _robot_arm_collision: bool = False
    _robot_hand_collision: bool = True
    _ground_enable: bool = True
    _ground_position: Tuple[float, float, float] = (0.5, 0, 0)
    _ground_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1)
    _ground_size: Tuple[float, float] = (1.2, 1.2)

    _object_enable: bool = True
    _workspace_centre: Tuple[float, float, float] = (0.5, 0, 0.25)
    _workspace_volume: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    _object_spawn_volume: Tuple[float, float, float] = (0.3, 0.3, 0.01)
    _object_spawn_height: float = 0.05
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __init__(self,
                 agent_rate: float,
                 restrict_position_goal_to_workspace: bool,
                 gripper_dead_zone: float,
                 full_3d_orientation: bool,
                 shaped_reward: bool,
                 object_distance_reward_scale: float,
                 object_height_reward_scale: float,
                 grasping_object_reward: float,
                 act_quick_reward: float,
                 ground_collision_reward: float,
                 required_object_height: float,
                 verbose: bool,
                 **kwargs):

        # Initialize the Task base class
        Manipulation.__init__(self,
                              agent_rate=agent_rate,
                              restrict_position_goal_to_workspace=restrict_position_goal_to_workspace,
                              verbose=verbose,
                              **kwargs)

        # Additional parameters
        self._gripper_dead_zone: float = gripper_dead_zone
        self._full_3d_orientation: bool = full_3d_orientation
        self._shaped_reward: bool = shaped_reward
        self._object_distance_reward_scale: float = object_distance_reward_scale
        self._object_height_reward_scale: float = object_height_reward_scale
        self._grasping_object_reward: float = grasping_object_reward
        self._act_quick_reward: float = act_quick_reward
        self._ground_collision_reward: float = ground_collision_reward
        self._required_object_height: float = required_object_height

        # Flag indicating if the task is done (performance - get_reward + is_done)
        self._is_done: bool = False

        # Distance to closest object in the previous step (or after reset)
        self._previous_min_distance: float = None
        # Heights of objects in the scene in the previous step (or after reset)
        self._previous_object_heights: Dict[str, float] = {}

    def create_action_space(self) -> ActionSpace:

        if self._full_3d_orientation:
            # 0   - (gripper) Gripper action
            #       - Open if positive (i.e. increase width)
            #       - Close if negative (i.e. decrease width)
            # 1:4 - (x, y, z) displacement
            #       - rescaled to metric units before use
            # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) relative 3D orientation in "6D representation"
            return gym.spaces.Box(low=-1.0,
                                  high=1.0,
                                  shape=(10,),
                                  dtype=np.float32)
        else:
            # 0   - (gripper) Gripper action
            #       - Open if positive (i.e. increase width)
            #       - Close if negative (i.e. decrease width)
            # 1:4 - (x, y, z) displacement
            #       - rescaled to metric units before use
            # 4   - (yaw) relative orientation around Z
            return gym.spaces.Box(low=-1.0,
                                  high=1.0,
                                  shape=(5,),
                                  dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:

        pass

    def set_action(self, action: Action):

        if self._verbose:
            print(f"action: {action}")

        # Gripper action
        # gripper_action = action['gripper_action']
        gripper_action = action[0]
        if gripper_action < -self._gripper_dead_zone:
            self.moveit2.gripper_close(manual_plan=True)
        elif gripper_action > self._gripper_dead_zone:
            self.moveit2.gripper_open(manual_plan=True)
        else:
            # No-op for the gripper as it is in the dead zone
            pass

        # Set position goal
        relative_position = action[1:4]
        self.set_position_goal(relative=relative_position)

        # Set orientation goal
        if self._full_3d_orientation:
            orientation_6d = action[4:10]
            self.set_orientation_goal(
                relative=orientation_6d, representation='6d')
        else:
            orientation_z = action[4]
            self.set_orientation_goal(
                relative=orientation_z, representation='z')

        # Plan and execute motion to target pose
        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

    def get_observation(self) -> Observation:

        pass

    def get_reward(self) -> Reward:

        reward = 0.0

        # Get positions of all objects in the scene
        object_positions = self.get_object_positions()
        # Get list of grasped objects (in contact with gripper links)
        grasped_objects = self.get_grasped_objects()

        # Return reward of 1.0 if any object was elevated above certain height and mark the episode done
        for grasped_object in grasped_objects:
            if object_positions[grasped_object][2] > self._required_object_height:
                reward += 1.0
                self._is_done = True

        # If the episode is not done, give a shaped/act-quick reward (if enabled)
        if not self._is_done:
            if self._shaped_reward:
                # Give reward based on how much closer robot got relative to the closest object
                if self._object_distance_reward_scale != 0.0:
                    current_min_distance = self.get_closest_object_distance(
                        object_positions)
                    # TODO: Consider giving only positive reward (just like in object height below)
                    reward += self._object_distance_reward_scale * \
                        (self._previous_min_distance - current_min_distance)
                    self._previous_min_distance = current_min_distance

                # Give reward based on increase in object's height above the ground (only positive)
                if self._object_height_reward_scale != 0.0:
                    for object_name, object_position in object_positions.items():
                        reward += self._object_height_reward_scale * max(0.0, object_position[2] -
                                                                         self._previous_object_heights[object_name])
                        self._previous_object_heights[object_name] = object_position[2]

                # Give a small positive reward if an object is grasped
                if self._grasping_object_reward != 0.0:
                    if len(grasped_objects) > 0:
                        if self._verbose:
                            print(f"Object(s) grasped: {grasped_objects}")
                        reward += self._grasping_object_reward

                # Give negative reward for collisions with ground
                if self._ground_collision_reward != 0.0:
                    if self.check_ground_collision():
                        reward += self._ground_collision_reward

            # Subtract a small reward each step to provide incentive to act quickly (if enabled)
            if self._act_quick_reward != 0.0:
                reward += self._act_quick_reward

        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        done = self._is_done

        if self._verbose:
            print(f"done: {done}")

        return done

    def reset_task(self):

        if self._verbose:
            print(f"\nreset task")

        self._is_done = False

        if self._shaped_reward:
            # Get current positions of all objects in the scene
            object_positions = self.get_object_positions()
            # Get distance to the closest object after the reset
            if self._object_distance_reward_scale != 0.0:
                self._previous_min_distance = self.get_closest_object_distance(
                    object_positions)
            # Get height of all objects in the scene after the reset
            if self._object_height_reward_scale != 0.0:
                self._previous_object_heights.clear()
                for object_name, object_position in object_positions.items():
                    self._previous_object_heights[object_name] = object_position[2]

    def get_closest_object_distance(self, object_positions:  Dict[str, Tuple[float, float, float]]) -> float:

        min_distance = 1.0

        ee_position = self.get_ee_position()
        for object_position in object_positions.values():
            distance = np.linalg.norm([ee_position[0] - object_position[0],
                                       ee_position[1] - object_position[1],
                                       ee_position[2] - object_position[2]])
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_object_positions(self) -> Dict[str, Tuple[float, float, float]]:

        object_positions = {}
        for object_name in self.object_names:
            object_positions[object_name] = tuple(self.get_object_position(
                object_name=object_name))

        return object_positions

    def get_object_position(self, object_name: str, link_name: Union[str, None] = None) -> Tuple[float, float, float]:

        # Get reference to the model of the object
        object_model = self.world.get_model(object_name).to_gazebo()

        # Use the first link if not specified
        if link_name is None:
            link_name = object_model.link_names()[0]

        # Return position of the object's link
        return object_model.get_link(link_name=link_name).position()

    def get_grasped_objects(self) -> List[str]:
        """
        Returns list of all currently grasped objects.
        Grasped object must be in contact with all gripper links (fingers).
        """

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
                    model_name = contact.body_b.split('::', 1)[0]
                    if any(object_name in model_name for object_name in self.object_names):
                        if model_name not in grasp_candidates:
                            grasp_candidates[model_name] = []
                        grasp_candidates[model_name].append(contact.points)

        # Determine what grasp candidates are indeed grasped objects
        # First make sure it has concact with all fingers
        # Then make sure that their normals are dissimilar
        grasped_objects = []
        for model_name, contact_points_list in grasp_candidates.items():
            if len(contact_points_list) < len(self.robot_gripper_link_names):
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
                normal_angles.append(
                    np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

            # Angle between at least two normals must be larger than 0.5*pi/(number_of_fingers)
            sufficient_angle = 0.5*np.pi/len(self.robot_gripper_link_names)
            for angle in normal_angles:
                if angle > sufficient_angle:
                    # If sufficient, add to list and process other candidates
                    grasped_objects.append(model_name)
                    continue

        return grasped_objects

    def check_ground_collision(self) -> bool:
        """
        Returns true if robot links are in collision with the ground.
        """

        ground = self.world.get_model(self.ground_name)
        for contact in ground.contacts():
            if self.robot_name in contact.body_b and not self.robot_base_link_name in contact.body_b:
                return True
        return False
