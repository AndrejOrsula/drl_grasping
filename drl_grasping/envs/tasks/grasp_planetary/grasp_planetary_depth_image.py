import abc
from collections import deque

import gym
import numpy as np
from gym_ignition.utils.typing import Observation, ObservationSpace

from drl_grasping.envs.models.sensors import Camera
from drl_grasping.envs.perception import CameraSubscriber
from drl_grasping.envs.tasks.grasp_planetary import GraspPlanetary
from drl_grasping.envs.utils.conversions import orientation_quat_to_6d


class GraspPlanetaryDepthImage(GraspPlanetary, abc.ABC):
    def __init__(
        self,
        depth_max_distance: float,
        image_include_color: bool,
        image_include_intensity: bool,
        image_n_stacked: int,
        proprioceptive_observations: bool,
        camera_type: str = "rgbd_camera",
        camera_width: int = 128,
        camera_height: int = 128,
        **kwargs,
    ):

        # Initialize the Task base class
        GraspPlanetary.__init__(
            self,
            **kwargs,
        )

        # Perception (depth map)
        self.camera_sub = CameraSubscriber(
            node=self,
            topic=Camera.get_depth_topic(camera_type),
            is_point_cloud=False,
            callback_group=self._callback_group,
        )
        # Perception (RGB image)
        if image_include_color or image_include_intensity:
            assert camera_type == "rgbd_camera"
            self.camera_sub_color = CameraSubscriber(
                node=self,
                topic=Camera.get_color_topic(camera_type),
                is_point_cloud=False,
                callback_group=self._callback_group,
            )

        # Additional parameters
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._depth_max_distance = depth_max_distance
        self._image_n_stacked = image_n_stacked
        self._image_include_color = image_include_color
        self._image_include_intensity = image_include_intensity
        self._proprioceptive_observations = proprioceptive_observations

        self._num_pixels = camera_height * camera_width
        # Queue of images
        self.__stacked_images = deque([], maxlen=self._image_n_stacked)

    def create_observation_space(self) -> ObservationSpace:

        # Size of depth channel
        # Note: Size is expressed in float32
        size = self._num_pixels

        if self._image_include_color:
            # Add 3 channels (RGB)
            size += 3 * self._num_pixels
        elif self._image_include_intensity:
            # Add 1 channel (intensity)
            size += self._num_pixels

        if self._proprioceptive_observations:
            size += 11

        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._image_n_stacked, size),
            dtype=np.float32,
        )

    def create_proprioceptive_observation_space(self) -> ObservationSpace:

        # 0   - (gripper) Gripper state
        #       - 1.0: opened
        #       - -1.0: closed
        # 1:4 - (x, y, z) displacement
        #       - metric units, unbound
        # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) 3D orientation in "6D representation"
        #       - normalised
        return gym.spaces.Box(
            low=np.array(
                (-1.0, -np.inf, -np.inf, -np.inf, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            ),
            high=np.array((1.0, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            shape=(10,),
            dtype=np.float32,
        )

    def get_observation(self) -> Observation:

        # Get the latest depth map
        depth_image_msg = self.camera_sub.get_observation()

        img_res = depth_image_msg.height * depth_image_msg.width
        if 2 * img_res == len(depth_image_msg.data):
            depth_data_type = np.float16
        else:
            depth_data_type = np.float32

        if (
            depth_image_msg.height != self._camera_width
            or depth_image_msg.width != self._camera_height
        ):
            # TODO: Add cv2 to requirements or find a better alternative for quick resizing of images
            import cv2

            # Convert to ndarray
            depth_image = np.ndarray(
                buffer=depth_image_msg.data,
                dtype=depth_data_type,
                shape=(depth_image_msg.height, depth_image_msg.width),
            ).astype(dtype=np.float32)

            # Crop and resize to the desired resolution
            if depth_image_msg.height > depth_image_msg.width:
                diff = depth_image_msg.height - depth_image_msg.width
                diff_2 = diff // 2
                depth_image = depth_image[diff_2:-diff_2, :]
            elif depth_image_msg.height < depth_image_msg.width:
                diff = depth_image_msg.width - depth_image_msg.height
                diff_2 = diff // 2
                depth_image = depth_image[:, diff_2:-diff_2]
            depth_image = cv2.resize(
                depth_image,
                dsize=(self._camera_height, self._camera_width),
                interpolation=cv2.INTER_CUBIC,
            ).reshape(self._num_pixels)

        else:
            # Convert to ndarray
            depth_image = np.ndarray(
                buffer=depth_image_msg.data,
                dtype=depth_data_type,
                shape=(self._num_pixels,),
            ).astype(dtype=np.float32)

        # Replace nan and inf with zero
        np.nan_to_num(depth_image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize depth based on pre-defined max distance
        depth_image[depth_image > self._depth_max_distance] = self._depth_max_distance
        depth_image = depth_image / self._depth_max_distance

        if self._image_include_color or self._image_include_intensity:
            # Get the latest color image
            color_image_msg = self.camera_sub_color.get_observation()

            if (
                color_image_msg.height != self._camera_width
                or color_image_msg.width != self._camera_height
            ):
                import cv2

                # Convert to ndarray
                color_image = np.ndarray(
                    buffer=color_image_msg.data,
                    dtype=np.uint8,
                    shape=(color_image_msg.height, color_image_msg.width, 3),
                )

                # Crop and resize to the desired resolution
                if color_image_msg.height > color_image_msg.width:
                    diff = color_image_msg.height - color_image_msg.width
                    diff_2 = diff // 2
                    color_image = color_image[diff_2:-diff_2, :, :]
                elif color_image_msg.height < color_image_msg.width:
                    diff = color_image_msg.width - color_image_msg.height
                    diff_2 = diff // 2
                    color_image = color_image[:, diff_2:-diff_2, :]
                color_image = cv2.resize(
                    color_image,
                    dsize=(self._camera_width, self._camera_height),
                    interpolation=cv2.INTER_CUBIC,
                ).reshape(3 * self._num_pixels)

            else:
                # Convert to ndarray
                color_image = np.ndarray(
                    buffer=color_image_msg.data,
                    dtype=np.uint8,
                    shape=(3 * self._num_pixels,),
                )

            if self._image_include_intensity:
                # Use only the first channel as the intensity observation
                color_image = color_image.reshape(
                    self._camera_width, self._camera_height, 3
                )[:, :, 0].reshape(-1)

            # # Debug save images
            # from PIL import Image
            # img_intensity = Image.fromarray(
            #     color_image.reshape(self._camera_width, self._camera_height), "L"
            # )
            # img_intensity.save("img_intensity.png")
            # img_depth = Image.fromarray(
            #     (depth_image * 255)
            #     .astype(np.uint8)
            #     .reshape(self._camera_width, self._camera_height),
            #     "L",
            # )
            # img_depth.save("img_depth.png")

            # Normalize color
            color_image.astype(dtype=np.float32)
            color_image = color_image / 255.0

            depth_image = np.concatenate((depth_image, color_image))

        if self._proprioceptive_observations:
            # Pad image with zeros to have a place for proprioceptive observations
            depth_image = np.pad(depth_image, (0, 11), "constant", constant_values=0)

            # Add number of auxiliary observations to image structure
            depth_image[-1] = np.array(10, dtype=np.float32)

            # Gather proprioceptive observations
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = (
                (1 if self.gripper.is_open else -1,)
                + ee_position
                + ee_orientation[0]
                + ee_orientation[1]
            )

            # Add auxiliary observations into the image structure
            depth_image[-11:-1] = np.array(aux_obs, dtype=np.float32)

        self.__stacked_images.append(depth_image)
        # For the first buffer after reset, fill with identical observations until deque is full
        while not self._image_n_stacked == len(self.__stacked_images):
            self.__stacked_images.append(depth_image)

        # Create the observation
        observation = Observation(np.array(self.__stacked_images))

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def reset_task(self):
        self.__stacked_images.clear()
        GraspPlanetary.reset_task(self)
