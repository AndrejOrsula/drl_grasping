from collections import deque
from drl_grasping.envs.perception import CameraSubscriber, OctreeCreator
from drl_grasping.envs.tasks.grasp import Grasp
from drl_grasping.envs.utils.conversions import orientation_quat_to_6d
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class GraspOctree(Grasp, abc.ABC):
    def __init__(
        self,
        octree_reference_frame_id: str,
        octree_dimension: float,
        octree_depth: int,
        octree_full_depth: int,
        octree_include_color: bool,
        octree_n_stacked: int,
        octree_max_size: int,
        proprieceptive_observations: bool,
        **kwargs,
    ):

        # Initialize the Task base class
        Grasp.__init__(
            self,
            **kwargs,
        )

        if octree_include_color:
            self.camera_type = "rgbd_camera"
        else:
            self.camera_type = "depth_camera"

        # Perception (RGB-D camera - point cloud)
        self.camera_sub = CameraSubscriber(
            topic=f"/{self.camera_type}/points",
            is_point_cloud=True,
            node_name=f"drl_grasping_camera_sub_{self.id}",
            use_sim_time=self._use_sim_time,
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
            min_bound=octree_min_bound,
            max_bound=octree_max_bound,
            depth=octree_depth,
            full_depth=octree_full_depth,
            include_color=octree_include_color,
            use_sim_time=self._use_sim_time,
            debug_draw=False,
            debug_write_octree=False,
            reference_frame_id=octree_reference_frame_id,
            node_name=f"drl_grasping_octree_creator_{self.id}",
        )

        # Additional parameters
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size
        self._proprieceptive_observations = proprieceptive_observations

        # List of all octrees
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

    def create_observation_space(self) -> ObservationSpace:

        # 0:n - octree
        # Note: octree is currently padded with zeros to have constant size
        # TODO: Customize replay buffer to support variable sized observations
        # If enabled, proprieceptive observations will be embedded inside octree in a hacky way
        # (replace with Dict once https://github.com/DLR-RM/stable-baselines3/pull/243 is merged)
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._octree_n_stacked, self._octree_max_size),
            dtype=np.uint8,
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

        if self._proprieceptive_observations:
            # Add number of auxiliary observations to octree structure
            octree[-8:-4] = np.ndarray(
                buffer=np.array([10], dtype="uint32").tobytes(),
                shape=(4,),
                dtype="uint8",
            )

            # Gather proprioceptive observations
            ee_position = self.get_ee_position()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=self.get_ee_orientation())
            aux_obs = (
                (self._gripper_state,)
                + ee_position
                + ee_orientation[0]
                + ee_orientation[1]
            )

            # Add auxiliary observations into the octree structure
            octree[-48:-8] = np.ndarray(
                buffer=np.array(aux_obs, dtype="float32").tobytes(),
                shape=(40,),
                dtype="uint8",
            )

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
        Grasp.reset_task(self)
