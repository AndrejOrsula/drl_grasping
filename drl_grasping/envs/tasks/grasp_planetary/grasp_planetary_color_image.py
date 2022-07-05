import abc

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber
from drl_grasping.envs.tasks.grasp_planetary import GraspPlanetary


class GraspPlanetaryColorImage(GraspPlanetary, abc.ABC):
    def __init__(
        self,
        camera_width: int,
        camera_height: int,
        camera_type: str = "camera",
        monochromatic: bool = False,
        **kwargs,
    ):

        # Initialize the Task base class
        GraspPlanetary.__init__(
            self,
            **kwargs,
        )

        # Store parameters for later use
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._monochromatic = monochromatic

        # Perception (RGB camera)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_color_topic(camera_type),
            is_point_cloud=False,
            callback_group=self._callback_group,
        )

    def create_observation_space(self) -> ObservationSpace:

        # 0:3*height*width - rgb image
        # 0:1*height*width - monochromatic (intensity) image
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self._camera_height,
                self._camera_width,
                1 if self._monochromatic else 3,
            ),
            dtype=np.uint8,
        )

    def get_observation(self) -> Observation:

        # Get the latest image
        image = self.camera_sub.get_observation()

        assert (
            image.width == self._camera_width and image.height == self._camera_height
        ), f"Error: Resolution of the input image does not match the configured observation space. ({image.width}x{image.height} instead of {self._camera_width}x{self._camera_height})"

        # Reshape and create the observation
        color_image = np.array(image.data, dtype=np.uint8).reshape(
            self._camera_height, self._camera_width, 3
        )

        # # Debug save images
        # from PIL import Image
        # img_color = Image.fromarray(color_image)
        # img_color.save("img_color.png")

        if self._monochromatic:
            observation = Observation(color_image[:, :, 0])
        else:
            observation = Observation(color_image)

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation
