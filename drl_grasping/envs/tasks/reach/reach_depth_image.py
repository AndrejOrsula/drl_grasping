import abc

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber
from drl_grasping.envs.tasks.reach import Reach

# TODO: ReachDepthImage environment currently does not currently have a working CnnPolicy


class ReachDepthImage(Reach, abc.ABC):
    def __init__(
        self,
        camera_type: str,
        camera_width: int,
        camera_height: int,
        **kwargs,
    ):

        # Initialize the Task base class
        Reach.__init__(
            self,
            **kwargs,
        )

        # Store parameters for later use
        self._camera_width = camera_width
        self._camera_height = camera_height

        # Perception (depth camera)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_depth_topic(camera_type),
            is_point_cloud=False,
            callback_group=self._callback_group,
        )

    def create_observation_space(self) -> ObservationSpace:

        # 0:height*width - depth image
        return gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self._camera_height, self._camera_width, 1),
            dtype=np.float32,
        )

    def get_observation(self) -> Observation:

        # Get the latest image
        image = self.camera_sub.get_observation()

        # Construct from buffer and reshape
        depth_image = np.frombuffer(image.data, dtype=np.float32).reshape(
            self._camera_height, self._camera_width, 1
        )
        # Replace all instances of infinity with 0
        depth_image[depth_image == np.inf] = 0.0

        # Create the observation
        observation = Observation(depth_image)

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation
