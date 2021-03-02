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
    _ground_enable: bool = True
    _ground_position: Tuple[float, float, float] = (0.5, 0, 0)
    _ground_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1)
    _ground_size: Tuple[float, float] = (1.2, 1.2)

    _object_enable: bool = True
    _workspace_centre: Tuple[float, float, float] = (0.5, 0, 0.25)
    _workspace_volume: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    _object_spawn_volume: Tuple[float, float, float] = (0.3, 0.3, 0.1)
    _object_spawn_height: float = 0.1
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __init__(self,
                 agent_rate: float,
                 restrict_position_goal_to_workspace: bool = True,
                 gripper_dead_zone: float = 0.25,
                 full_3d_orientation: bool = False,
                 shaped_reward: bool = True,
                 grasping_object_reward: float = 0.01,
                 act_quick_reward: float = -0.001,
                 required_object_height: float = 0.25,
                 verbose: bool = False,
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
        self._grasping_object_reward: float = grasping_object_reward
        self._act_quick_reward: float = act_quick_reward
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
            self.moveit2.gripper_close(speed=0.1, force=20)
        elif gripper_action > self._gripper_dead_zone:
            self.moveit2.gripper_open(speed=0.1)
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
                current_min_distance = self.get_closest_object_distance(
                    object_positions)
                reward += self._previous_min_distance - current_min_distance
                self._previous_min_distance = current_min_distance

                # Give reward based on increase in object's height above the ground (only positive)
                for object_name, object_position in self.get_object_positions().items():
                    reward += max(0.0, object_position[2] -
                                  self._previous_object_heights[object_name])
                    self._previous_object_heights[object_name] = object_position[2]

                # Give a small positive reward if an object is grasped
                if len(grasped_objects) > 0:
                    reward += self._grasping_object_reward

            # Subtract a small reward each step to provide incentive to act quickly (if enabled)
            if self._act_quick_reward < 0.0:
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
            self._previous_min_distance = self.get_closest_object_distance(
                object_positions)
            # Get height of all objects in the scene after the reset
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
        grasped_objects = []

        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.to_gazebo().get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()

            if 0 == len(finger_contacts):
                # If any of the fingers has no contact, immediately return None
                return []
            elif 0 == len(grasped_objects):
                # If there are no candidates (first link being checked), add all current contact bodies
                for contact in finger_contacts:
                    for object_name in self.object_names:
                        if object_name in contact.body_b:
                            grasped_objects.append(contact.body_b)
            else:
                # Otherwise, keep only candidates that also have contact with the other finger(s)
                contact_bodies = [contact.body_b
                                  for contact in finger_contacts]
                grasped_objects = list(itertools.filterfalse(lambda x: x not in grasped_objects,
                                                             contact_bodies))

        # Keep only the model name (disregard the link of object that is in collision)
        grasped_objects = [model_name.split('::', 1)[0]
                           for model_name in grasped_objects]

        return grasped_objects
