from drl_grasping.control import MoveIt2
from drl_grasping.perception import PointCloudSub, ImageSub
from gym_ignition.base import task
from gym_ignition.runtimes.gazebo_runtime import GazeboRuntime
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from scenario import core as scenario
from scipy.spatial.transform import Rotation
from typing import Tuple, List
import abc
import gym
import numpy as np


class Grasping(task.Task, abc.ABC):

    _camera_type: str = 'camera'
    _camera_width: int = 128
    _camera_height: int = 128
    _camera_update_rate: int = 10
    _camera_horizontal_fov: float = 1.0
    _camera_vertical_fov: float = 1.0
    _camera_clip_color: List[float] = (0.01, 1000.0)
    _camera_clip_depth: List[float] = (0.01, 10.0)
    _camera_noise_mean: float = None
    _camera_noise_stddev: float = None
    _camera_ros2_bridge_color: bool = True
    _camera_ros2_bridge_depth: bool = False
    _camera_ros2_bridge_points: bool = False

    _robot_default_joint_positions: List[float] = (0.0,
                                                   0.0,
                                                   0.0,
                                                   -1.57,
                                                   0.0,
                                                   1.57,
                                                   0.79,
                                                   0.04,
                                                   0.04)

    def __init__(self,
                 agent_rate: float,
                 **kwargs):

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)

        # Perception (RGB-D camera)
        # self.point_cloud_sub = PointCloudSub()
        self.color_image_sub = ImageSub(topic=f'/{self._camera_type}')

        # Control (MoveIt2)
        self.moveit2 = MoveIt2()

        # Names of important models
        self.robot_name = None
        self.camera_name = None
        self.ground_name = None
        self.object_names = []
        self.target_object_name = None

        # Flag indicating if the task is done (performance - get_reward + is_done)
        self.__is_done = False

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # print(f"Creating spaces")
        # TODO: Make use of dictionaries to simplify work with spaces - needs to be custom, sb3 does not support (besides HER)

        # Action space
        # action_space_dict = self._get_action_space_dict()
        action_space = gym.spaces.Box(low=np.array((-0.2, -0.2, -0.2, -1.0)),
                                      high=np.array((0.2, 0.2, 0.2, 1.0)),
                                      shape=(4,),
                                      dtype=np.float32)

        # Observation space
        observation_space_dict = self._get_observation_space_dict()
        observation_space = observation_space_dict['color_image']

        return action_space, observation_space

    def set_action(self, action: Action) -> None:

        # print(f"Applying action: {action}")

        # Position goal (relative)
        # pos_xyz = action['pos_xyz']
        pos_xyz = action[0:3]
        current_pos = self.moveit2.compute_fk().pose_stamped[0].pose.position
        # Restrict target position (no crashing into table or going in the opposite direction)
        target_pos = [max(current_pos.x + pos_xyz[0], 0.1),
                      current_pos.y + pos_xyz[1],
                      max(current_pos.z + pos_xyz[2], 0.01)]
        self.moveit2.set_position_goal(target_pos)

        # Orientation goal
        # quat_xyzw = Rotation.from_quat(action['quat_xyzw']).as_quat()
        quat_xyzw = [1.0, 0.0, 0.0, 0.0]
        self.moveit2.set_orientation_goal(quat_xyzw)

        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

        # Gripper action
        # gripper_action = action['gripper_action']
        gripper_action = action[3]
        # gripper_width = action['gripper_width']
        gripper_width_close = 0.0
        gripper_width_open = 1.0
        # gripper_speed = action['gripper_speed']
        gripper_speed = 0.5
        # gripper_force = action['gripper_force']
        gripper_force = 1.0
        if gripper_action < 0:
            # TODO: Put limits somewhere else
            self.moveit2.gripper_close(width=0.08*gripper_width_close,
                                       speed=0.2*gripper_speed,
                                       force=20*gripper_force)
        else:
            self.moveit2.gripper_open(width=0.08*gripper_width_open,
                                      speed=0.2*gripper_speed)

    def get_observation(self) -> Observation:

        # point_cloud = self.point_cloud_sub.get_point_cloud()
        image = self.color_image_sub.get_image()

        if image.height != self._camera_height or image.width != self._camera_width:
            print("Received image with incorrect resolution",
                  image.height, image.width)
            image.data = [0] * (self._camera_height * self._camera_width * 3)

        if len(image.data) == 0:
            print("Empty image received")
            image.data = [0] * (self._camera_height * self._camera_width * 3)

        color_image = np.array(image.data, dtype=np.uint8).reshape(self._camera_height,
                                                                   self._camera_width, 3)

        # joint_state = self.moveit2.get_joint_state()
        # joint_positions = joint_state.position
        # joint_velocities = joint_state.velocity
        # ee_pose = self.moveit2.compute_fk(
        #     joint_state=joint_state).pose_stamped[0].pose
        # ee_pos = [ee_pose.position.x,
        #           ee_pose.position.y,
        #           ee_pose.position.z]
        # ee_quat = [ee_pose.orientation.x,
        #            ee_pose.orientation.y,
        #            ee_pose.orientation.z,
        #            ee_pose.orientation.w]

        # Create the observation
        observation = Observation(np.array([
            color_image,
            # point_cloud.data,
            # ee_pos,
            # ee_quat,
        ]))

        # print(f"observation: {observation}")

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        reward = 0.0

        # Give reward if object is grasped
        if self.is_target_object_grasped():
            reward += 1.0

            # Give extra reward if the object is grasped and moved 25 cm above ground
            target_object = self.world.get_model(
                self.target_object_name).to_gazebo()
            target_position = target_object.get_link(
                link_name=target_object.link_names()[0]).position()
            if target_position[2] > 0.25:
                reward += 10.0
                self.__is_done = True

        # print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        done = self.__is_done

        # print(f"done: {done}")

        return done

    def reset_task(self) -> None:

        # print(f"reset task")

        # TODO: Reset (determine what to put here and what to put inside randomizer)
        # The less is here, the easier Sim2Real transfer will be

        self.__is_done = False

        pass

    def _get_action_space_dict(self) -> dict:

        actions: dict = {}

        # Grasp end-effector position (x, y, z), relative
        actions['pos_xyz'] = gym.spaces.Box(low=np.array((-0.1, -0.1, -0.1)),
                                            high=np.array((0.1, 0.1, 0.1)),
                                            shape=(3,),
                                            dtype=np.float32)

        # # Grasp end-effector orientation (x, y, z, w)
        # actions['quat_xyzw'] = gym.spaces.Box(low=-1.0,
        #                                       high=1.0,
        #                                       shape=(4,),
        #                                       dtype=np.float32)

        # Gripper action
        # CLOSE < 0
        # OPEN >= 0
        actions['gripper_action'] = gym.spaces.Box(low=-1.0,
                                                   high=1.0,
                                                   shape=(1,),
                                                   dtype=np.float16)
        # actions['gripper_width'] = gym.spaces.Box(low=0.0,
        #                                           high=1.0,
        #                                           shape=(1,),
        #                                           dtype=np.float16)
        # actions['gripper_speed'] = gym.spaces.Box(low=0.0,
        #                                           high=1.0,
        #                                           shape=(1,),
        #                                           dtype=np.float16)
        # actions['gripper_force'] = gym.spaces.Box(low=0.0,
        #                                           high=1.0,
        #                                           shape=(1,),
        #                                           dtype=np.float16)

        return actions

    def _get_observation_space_dict(self) -> dict:

        observations: dict = {}
        inf = np.finfo(np.float32).max

        # Color image
        observations['color_image'] = gym.spaces.Box(low=0,
                                                     high=255,
                                                     shape=(self._camera_height,
                                                            self._camera_width, 3),
                                                     dtype=np.uint8)

        # # Depth image
        # observations['depth_image'] = gym.spaces.Box(low=0,
        #                                              high=255,
        #                                              shape=(self._camera_height,
        #                                                     self._camera_width, 1),
        #                                              dtype=np.float32)

        # # TODO: convert data array from bytes to XYZRGB floats
        # observations['xyzrgb_point_cloud'] = gym.spaces.Box(low=-inf,
        #                                                     high=inf,
        #                                                     shape=(
        #                                                         self._camera_width*self._camera_height*3*4
        #                                                         + self._camera_width*self._camera_height*3*4,),
        #                                                     dtype=np.float32)

        # TODO: Split perception for object to be grasped and obstables (avoidance)

        # Joint positions
        # observations['joint_pos'] = gym.spaces.Box(low=-inf,
        #                                            high=inf,
        #                                            shape=(
        #                                                self._number_of_joints,),
        #                                            dtype=np.float32)

        # # Joint velocities
        # observations['joint_vel'] = gym.spaces.Box(low=-inf,
        #                                            high=inf,
        #                                            shape=(
        #                                                self._number_of_joints,),
        #                                            dtype=np.float32)

        # # End-effector position (x, y, z)
        # observations['pos_xyz'] = gym.spaces.Box(low=-1.0,
        #                                          high=1.0,
        #                                          shape=(3,),
        #                                          dtype=np.float32)

        # # End-effector orientation (x, y, z, w)
        # observations['quat_xyzw'] = gym.spaces.Box(low=-1.0,
        #                                            high=1.0,
        #                                            shape=(4,),
        #                                            dtype=np.float32)

        return observations

    def is_target_object_grasped(self) -> bool:

        # TODO: Select target object (the first is used now)
        self.target_object_name = self.object_names[0]

        robot = self.world.get_model(self.robot_name)

        # TODO: Make robot gripper (fingers) more general inside task
        finger_left = robot.to_gazebo().get_link(link_name="panda_leftfinger")
        finger_right = robot.to_gazebo().get_link(link_name="panda_rightfinger")

        contacts_left = finger_left.contacts()
        contacts_right = finger_right.contacts()

        if len(contacts_left) > 0 and len(contacts_right) > 0:
            left_target_object_contact = False
            right_target_object_contact = False

            for contact in contacts_left:
                if self.target_object_name in contact.body_a \
                        or self.target_object_name in contact.body_b:
                    left_target_object_contact = True

            for contact in contacts_right:
                if self.target_object_name in contact.body_a \
                        or self.target_object_name in contact.body_b:
                    right_target_object_contact = True

            if left_target_object_contact and right_target_object_contact:
                return True
