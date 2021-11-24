from collections import deque
from drl_grasping.envs.tasks.reach import Reach
from drl_grasping.envs.perception import CameraSubscriber, OctreeCreator
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
from drl_grasping.envs.models.sensors import Camera
import abc
import gym
import numpy as np

from tf2_ros import transform_listener


class ReachOctree(Reach, abc.ABC):

    _octree_min_bound: Tuple[float, float, float] = (0.15, -0.3, 0.0)
    _octree_max_bound: Tuple[float, float, float] = (0.75, 0.3, 0.6)

    def __init__(
        self,
        camera_type: str,
        octree_reference_frame_id: str,
        octree_dimension: float,
        octree_depth: int,
        octree_full_depth: int,
        octree_include_color: bool,
        octree_n_stacked: int,
        octree_max_size: int,
        **kwargs,
    ):

        # Initialize the Task base class
        Reach.__init__(
            self,
            **kwargs,
        )

        # Store parameters for later use
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size

        # Perception (RGB-D camera - point cloud)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_points_topic(camera_type),
            is_point_cloud=True,
        )

        # Get exact name substitution of the frame for octree
        octree_reference_frame_id = self.substitute_special_frames(
            octree_reference_frame_id
        )

        octree_min_bound: Tuple[float, float, float] = (
            self.workspace_centre[0] - octree_dimension / 2,
            self.workspace_centre[1] - octree_dimension / 2,
            self.workspace_centre[2] - octree_dimension / 2,
        )
        octree_max_bound: Tuple[float, float, float] = (
            self.workspace_centre[0] + octree_dimension / 2,
            self.workspace_centre[1] + octree_dimension / 2,
            self.workspace_centre[2] + octree_dimension / 2,
        )
        self.octree_creator = OctreeCreator(
            tf2_listener=self.tf2_listener,
            reference_frame_id=octree_reference_frame_id,
            min_bound=octree_min_bound,
            max_bound=octree_max_bound,
            include_color=octree_include_color,
            depth=octree_depth,
            full_depth=octree_full_depth,
        )

        # Variable initialisation
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

    def create_observation_space(self) -> ObservationSpace:

        # 0:n - octree
        # Note: octree is currently padded with zeros to have constant size
        # TODO: Customize replay buffer to support variable sized observations
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
            print(
                f"ERROR: Octree is larger than the maximum "
                f"allowed size (exceeded with {octree_size})"
            )
        octree = np.pad(
            octree,
            (0, self._octree_max_size - octree_size),
            "constant",
            constant_values=0,
        )

        # Write the original length into the padded octree for reference
        octree[-4:] = np.ndarray(
            buffer=np.array([octree_size], dtype="uint32").tobytes(),
            shape=(4,),
            dtype="uint8",
        )
        # To get it back:
        # octree_size = np.frombuffer(buffer=octree[-4:],
        #                             dtype='uint32',
        #                             count=1)

        self.__stacked_octrees.append(octree)
        # For the first buffer after reset, fill with identical observations until deque is full
        while not self._octree_n_stacked == len(self.__stacked_octrees):
            self.__stacked_octrees.append(octree)

        # Create the observation
        observation = Observation(np.array(self.__stacked_octrees))

        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def reset_task(self):

        self.__stacked_octrees.clear()
        Reach.reset_task(self)
