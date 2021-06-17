from collections import deque
from drl_grasping.envs.tasks.fidget_spin import FidgetSpin
from drl_grasping.perception import CameraSubscriber, OctreeCreator
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class FidgetSpinOctree(FidgetSpin, abc.ABC):

    _camera_enable: bool = True
    _camera_type: str = 'auto'
    _camera_render_engine: str = 'ogre2'
    _camera_position: Tuple[float, float, float] = (0.3, 0, 0.4)
    _camera_quat_xyzw: Tuple[float, float,
                             float, float] = (-0.707, 0, 0.707, 0)
    _camera_width: int = 256
    _camera_height: int = 256
    _camera_update_rate: int = 30
    _camera_horizontal_fov: float = 1.0
    _camera_vertical_fov: float = 1.0
    _camera_clip_color: Tuple[float, float] = (0.01, 1000.0)
    _camera_clip_depth: Tuple[float, float] = (0.01, 10.0)
    _camera_ros2_bridge_color: bool = False
    _camera_ros2_bridge_depth: bool = False
    _camera_ros2_bridge_points: bool = True

    _workspace_centre: Tuple[float, float, float] = (0.3, 0.0, 0.0)
    _octree_size: float = 0.24
    _octree_min_bound: Tuple[float, float, float] = (_workspace_centre[0]-_octree_size/2,
                                                     _workspace_centre[1]-_octree_size/2,
                                                     _workspace_centre[2]-_octree_size/2,)
    _octree_max_bound: Tuple[float, float, float] = (_workspace_centre[0]+_octree_size/2,
                                                     _workspace_centre[1]+_octree_size/2,
                                                     _workspace_centre[2]+_octree_size/2,)

    def __init__(self,
                 agent_rate: float,
                 octree_depth: int,
                 octree_full_depth: int,
                 octree_include_color: bool,
                 octree_n_stacked: int,
                 octree_max_size: int,
                 verbose: bool,
                 **kwargs):

        # Initialize the Task base class
        FidgetSpin.__init__(self,
                            agent_rate=agent_rate,
                            verbose=verbose,
                            **kwargs)

        if octree_include_color:
            self._camera_type = 'rgbd_camera'
        else:
            self._camera_type = 'depth_camera'

        # Perception (RGB-D camera - point cloud)
        self.camera_sub = CameraSubscriber(topic=f'/{self._camera_type}/points',
                                           is_point_cloud=True,
                                           node_name=f'drl_grasping_point_cloud_sub_{self.id}')

        self.octree_creator = OctreeCreator(min_bound=self._octree_min_bound,
                                            max_bound=self._octree_max_bound,
                                            depth=octree_depth,
                                            full_depth=octree_full_depth,
                                            include_color=octree_include_color,
                                            use_sim_time=True,
                                            debug_draw=False,
                                            debug_write_octree=False,
                                            robot_frame_id='world',
                                            node_name=f'drl_grasping_octree_creator_{self.id}')

        # Additional parameters
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size

        # List of all octrees
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

        # Name of camera model
        self.camera_name = None

    def create_observation_space(self) -> ObservationSpace:

        # 0:n - octree
        # Note: octree is currently padded with zeros to have constant size
        # TODO: Customize replay buffer to support variable sized observations
        # If enabled, proprieceptive observations will be embedded inside octree in a hacky way
        # (replace with Dict once https://github.com/DLR-RM/stable-baselines3/pull/243 is merged)
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._octree_n_stacked,
                                     self._octree_max_size),
                              dtype=np.uint8)

    def get_observation(self) -> Observation:

        # Get the latest point cloud
        point_cloud = self.camera_sub.get_observation()

        # Contrust octree from this point cloud
        octree = self.octree_creator(point_cloud).numpy()

        # Pad octree with zeros to have a consistent length
        # TODO: Customize replay buffer to support variable sized observations
        octree_size = octree.shape[0]
        if octree_size > self._octree_max_size:
            print(f"ERROR: Octree is larger than the maximum "
                  f"allowed size (exceeded with {octree_size})")
        octree = np.pad(octree,
                        (0, self._octree_max_size - octree_size),
                        'constant',
                        constant_values=0)

        # Write the original length into the padded octree for reference
        octree[-4:] = np.ndarray(buffer=np.array([octree_size],
                                                 dtype='uint32').tobytes(),
                                 shape=(4,),
                                 dtype='uint8')
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
        FidgetSpin.reset_task(self)
