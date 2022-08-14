import abc
from collections import deque
from typing import Tuple

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber, OctreeCreator
from drl_grasping.envs.tasks.grasp import Grasp
from drl_grasping.envs.utils.conversions import orientation_quat_to_6d


class GraspOctree(Grasp, abc.ABC):
    def __init__(
        self,
        octree_reference_frame_id: str,
        octree_min_bound: Tuple[float, float, float],
        octree_max_bound: Tuple[float, float, float],
        octree_depth: int,
        octree_full_depth: int,
        octree_include_color: bool,
        octree_include_intensity: bool,
        octree_n_stacked: int,
        octree_max_size: int,
        proprioceptive_observations: bool,
        camera_type: str = "rgbd_camera",
        **kwargs,
    ):

        # Initialize the Task base class
        Grasp.__init__(
            self,
            **kwargs,
        )

        # Perception (depth/RGB-D camera - point cloud)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_points_topic(camera_type),
            is_point_cloud=True,
            callback_group=self._callback_group,
        )

        # Offset octree bounds by the robot base offset
        octree_min_bound = (
            octree_min_bound[0],
            octree_min_bound[1],
            octree_min_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET,
        )
        octree_max_bound = (
            octree_max_bound[0],
            octree_max_bound[1],
            octree_max_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET,
        )

        # Octree creator
        self.octree_creator = OctreeCreator(
            node=self,
            tf2_listener=self.tf2_listener,
            reference_frame_id=self.substitute_special_frame(octree_reference_frame_id),
            min_bound=octree_min_bound,
            max_bound=octree_max_bound,
            include_color=octree_include_color,
            include_intensity=octree_include_intensity,
            depth=octree_depth,
            full_depth=octree_full_depth,
        )

        # Additional parameters
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size
        self._proprioceptive_observations = proprioceptive_observations

        # List of all octrees
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

    def create_observation_space(self) -> ObservationSpace:

        # 0:n - octree
        # Note: octree is currently padded with zeros to have constant size
        # TODO: Customize replay buffer to support variable sized observations
        # If enabled, proprieceptive observations will be embedded inside octree in a hacky way
        # (replace with Dict once https://github.com/DLR-RM/stable-baselines3/pull/243 is merged)
        # 0   - (gripper) Gripper state
        #       - 1.0: opened
        #       - -1.0: closed
        # 1:4 - (x, y, z) displacement
        #       - metric units, unbound
        # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) 3D orientation in "6D representation"
        #       - normalised
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._octree_n_stacked, self._octree_max_size),
            dtype=np.uint8,
        )

    def get_observation(self) -> Observation:

        # Get the latest point cloud
        point_cloud = self.camera_sub.get_observation()

        # Contrust octree from this point cloud
        octree = self.octree_creator(point_cloud).numpy()

        # Pad octree with zeros to have a consistent length
        # TODO: Customize replay buffer to support variable sized observations
        octree_size = octree.shape[0]
        if octree_size > self._octree_max_size:
            self.get_logger().error(
                f"Octree is larger than the maximum allowed size of {self._octree_max_size} (exceeded with {octree_size})"
            )
        octree = np.pad(
            octree,
            (0, self._octree_max_size - octree_size),
            "constant",
            constant_values=0,
        )

        # Write the original length into the padded octree for reference
        octree[-4:] = np.ndarray(
            buffer=np.array([octree_size], dtype=np.uint32).tobytes(),
            shape=(4,),
            dtype=np.uint8,
        )
        # To get it back:
        # octree_size = np.frombuffer(buffer=octree[-4:], dtype=np.uint32, count=1)

        if self._proprioceptive_observations:
            # Add number of auxiliary observations to octree structure
            octree[-8:-4] = np.ndarray(
                buffer=np.array([10], dtype=np.uint32).tobytes(),
                shape=(4,),
                dtype=np.uint8,
            )

            # Gather proprioceptive observations
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = (
                (1.0 if self.gripper.is_open else -1.0,)
                + ee_position
                + ee_orientation[0]
                + ee_orientation[1]
            )

            # Add auxiliary observations into the octree structure
            octree[-48:-8] = np.ndarray(
                buffer=np.array(aux_obs, dtype=np.float32).tobytes(),
                shape=(40,),
                dtype=np.uint8,
            )

        self.__stacked_octrees.append(octree)
        # For the first buffer after reset, fill with identical observations until deque is full
        while not self._octree_n_stacked == len(self.__stacked_octrees):
            self.__stacked_octrees.append(octree)

        # Create the observation
        observation = Observation(np.array(self.__stacked_octrees, dtype=np.uint8))

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def reset_task(self):

        self.__stacked_octrees.clear()
        Grasp.reset_task(self)
