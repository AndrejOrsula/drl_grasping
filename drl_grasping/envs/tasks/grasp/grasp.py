from drl_grasping.envs.tasks.manipulation import Manipulation
from drl_grasping.envs.tasks.grasp.curriculum import GraspCurriculum
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import Tuple, List, Union, Dict
import abc
import gym
import itertools
import numpy as np
import sys


class Grasp(Manipulation, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _robot_arm_collision: bool = False
    _robot_hand_collision: bool = True
    _robot_initial_joint_positions: Tuple[float, ...] = (0.0,
                                                         0.0,
                                                         0.0,
                                                         -1.57,
                                                         0.0,
                                                         1.57,
                                                         0.79,
                                                         0.04,
                                                         0.04)

    _workspace_centre: Tuple[float, float, float] = (0.45, 0, 0.2)
    _workspace_volume: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    _ground_enable: bool = True
    _ground_position: Tuple[float, float, float] = (0, 0, 0)
    _ground_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1)
    _ground_size: Tuple[float, float] = (2.0, 2.0)

    _object_enable: bool = True
    # 'box' [x, y, z], 'sphere' [radius], 'cylinder' [radius, height]
    _object_type: str = 'box'
    _object_dimensions: List[float] = [0.05, 0.05, 0.05]
    _object_mass: float = 0.1
    _object_collision: bool = True
    _object_visual: bool = True
    _object_static: bool = False
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    _object_spawn_centre: Tuple[float, float, float] = \
        (_workspace_centre[0],
         _workspace_centre[1],
         0.05)
    _object_spawn_volume_proportion: float = 0.75
    _object_spawn_volume: Tuple[float, float, float] = \
        (_object_spawn_volume_proportion*_workspace_volume[0],
         _object_spawn_volume_proportion*_workspace_volume[1],
         0.0)

    def __init__(self,
                 agent_rate: float,
                 restrict_position_goal_to_workspace: bool,
                 gripper_dead_zone: float,
                 full_3d_orientation: bool,
                 sparse_reward: bool,
                 required_reach_distance: float,
                 required_lift_height: float,
                 act_quick_reward: float,
                 curriculum_enabled: bool,
                 curriculum_success_rate_threshold: float,
                 curriculum_success_rate_rolling_average_n: int,
                 curriculum_stage_reward_multiplier: float,
                 curriculum_restart_every_n_steps: int,
                 curriculum_min_workspace_scale: float,
                 curriculum_scale_negative_reward: bool,
                 verbose: bool,
                 **kwargs):

        # Initialize the Task base class
        Manipulation.__init__(self,
                              agent_rate=agent_rate,
                              restrict_position_goal_to_workspace=restrict_position_goal_to_workspace,
                              verbose=verbose,
                              **kwargs)

        self.curriculum = GraspCurriculum(task=self,
                                          enabled=curriculum_enabled,
                                          sparse_reward=sparse_reward,
                                          required_reach_distance=required_reach_distance,
                                          required_lift_height=required_lift_height,
                                          act_quick_reward=act_quick_reward,
                                          success_rate_threshold=curriculum_success_rate_threshold,
                                          success_rate_rolling_average_n=curriculum_success_rate_rolling_average_n,
                                          stage_reward_multiplier=curriculum_stage_reward_multiplier,
                                          restart_every_n_steps=curriculum_restart_every_n_steps,
                                          min_workspace_scale=curriculum_min_workspace_scale,
                                          scale_negative_rewards=curriculum_scale_negative_reward,
                                          verbose=verbose)

        # Additional parameters
        self._gripper_dead_zone: float = gripper_dead_zone
        self._full_3d_orientation: bool = full_3d_orientation


        self._original_workspace_volume = self._workspace_volume

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

        reward = self.curriculum.get_reward()

        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        done = self.curriculum.is_done()

        if self._verbose:
            print(f"done: {done}")

        return done

    def get_info(self) -> Dict:

        info = {}

        info.update(self.curriculum.get_info())

        return info

    def reset_task(self):

        self.curriculum.reset_task()

        if self._verbose:
            print(f"\ntask reset")

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

    def get_closest_object_distance(self, object_positions:  Dict[str, Tuple[float, float, float]]) -> float:

        min_distance = sys.float_info.max

        ee_position = self.get_ee_position()
        for object_position in object_positions.values():
            distance = np.linalg.norm([ee_position[0] - object_position[0],
                                       ee_position[1] - object_position[1],
                                       ee_position[2] - object_position[2]])
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_touched_objects(self) -> List[str]:
        """
        Returns list of all objects that are in contact with any finger.
        """

        robot = self.world.get_model(self.robot_name)
        touched_objects = []

        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.to_gazebo().get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()

            # Add model to list of touched objects if in contact
            for contact in finger_contacts:
                # Keep only the model name (disregard the link of object that is in collision)
                model_name = contact.body_b.split('::', 1)[0]
                if model_name not in touched_objects and \
                        any(object_name in model_name for object_name in self.object_names):
                    touched_objects.append(model_name)

        return touched_objects

    def get_grasped_objects(self) -> List[str]:
        """
        Returns list of all currently grasped objects.
        Grasped object must be in contact with all gripper links (fingers) and their contant normals must be dissimilar.
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
        # First make sure it has contact with all fingers
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

    def check_all_objects_outside_workspace(self, object_positions:  Dict[str, Tuple[float, float, float]]) -> bool:
        """
        Returns true if all objects are outside the workspace
        """

        ws_min_bound = \
            (self._workspace_centre[0] - self._workspace_volume[0]/2,
             self._workspace_centre[1] - self._workspace_volume[1]/2,
             self._workspace_centre[2] - self._workspace_volume[2]/2)
        ws_max_bound = \
            (self._workspace_centre[0] + self._workspace_volume[0]/2,
             self._workspace_centre[1] + self._workspace_volume[1]/2,
             self._workspace_centre[2] + self._workspace_volume[2]/2)

        return all([object_position[0] < ws_min_bound[0] or
                    object_position[1] < ws_min_bound[1] or
                    object_position[2] < ws_min_bound[2] or
                    object_position[0] > ws_max_bound[0] or
                    object_position[1] > ws_max_bound[1] or
                    object_position[2] > ws_max_bound[2]
                    for object_position in object_positions.values()])

    def update_workspace_size(self, scale: float):

        self._workspace_volume = (scale*self._original_workspace_volume[0],
                                  scale*self._original_workspace_volume[1],
                                  self._original_workspace_volume[2])
        self._object_spawn_volume = (self._object_spawn_volume_proportion*self._workspace_volume[0],
                                     self._object_spawn_volume_proportion*self._workspace_volume[1],
                                     self._object_spawn_volume[2])
