from drl_grasping.envs.tasks.reach import Reach
from drl_grasping.envs.perception import CameraSubscriber
from drl_grasping.envs.models.sensors import Camera
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class ReachColorImage(Reach, abc.ABC):
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

        # Perception (RGB camera)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_color_topic(camera_type),
            is_point_cloud=False,
        )

    def create_observation_space(self) -> ObservationSpace:

        # 0:3*height*width - rgb image
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._camera_height, self._camera_width, 3),
            dtype=np.uint8,
        )

    def get_observation(self) -> Observation:

        # Get the latest image
        image = self.camera_sub.get_observation()

        # Reshape and create the observation
        color_image = np.array(image.data, dtype=np.uint8).reshape(
            self._camera_height, self._camera_width, 3
        )

        observation = Observation(color_image)

        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation
