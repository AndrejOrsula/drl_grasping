from drl_grasping.envs.tasks.grasp import Grasp
from drl_grasping.perception import CameraSubscriber
from drl_grasping.perception import OctreeCreator
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class GraspOctree(Grasp, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _camera_enable: bool = True
    _camera_type: str = 'rgbd_camera'
    _camera_width: int = 256
    _camera_height: int = 256
    _camera_update_rate: int = 10
    _camera_horizontal_fov: float = 0.9
    _camera_vertical_fov: float = 0.9
    _camera_position: Tuple[float, float, float] = (1.1, -0.75, 0.3)
    _camera_quat_xyzw: Tuple[float, float,
                             float, float] = (-0.0402991, -0.0166924, 0.9230002, 0.3823192)
    _camera_ros2_bridge_points: bool = True

    _workspace_centre: Tuple[float, float, float] = (0.45, 0, 0.2)
    _workspace_volume: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    _octree_ground_offset: float = 0.01
    _octree_min_bound: Tuple[float, float, float] = (0.15,
                                                     -0.3,
                                                     0.0 - _octree_ground_offset)
    _octree_max_bound: Tuple[float, float, float] = (0.75,
                                                     0.3,
                                                     0.6 - _octree_ground_offset)

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
                 octree_depth: int,
                 octree_full_depth: int,
                 octree_include_color: bool,
                 octree_max_size: int,
                 verbose: bool,
                 **kwargs):

        # Initialize the Task base class
        Grasp.__init__(self,
                       agent_rate=agent_rate,
                       restrict_position_goal_to_workspace=restrict_position_goal_to_workspace,
                       gripper_dead_zone=gripper_dead_zone,
                       full_3d_orientation=full_3d_orientation,
                       sparse_reward=sparse_reward,
                       required_reach_distance=required_reach_distance,
                       required_lift_height=required_lift_height,
                       act_quick_reward=act_quick_reward,
                       curriculum_enabled=curriculum_enabled,
                       curriculum_success_rate_threshold=curriculum_success_rate_threshold,
                       curriculum_success_rate_rolling_average_n=curriculum_success_rate_rolling_average_n,
                       curriculum_stage_reward_multiplier=curriculum_stage_reward_multiplier,
                       curriculum_restart_every_n_steps=curriculum_restart_every_n_steps,
                       curriculum_min_workspace_scale=curriculum_min_workspace_scale,
                       curriculum_scale_negative_reward=curriculum_scale_negative_reward,
                       verbose=verbose,
                       **kwargs)

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
                                            node_name=f'drl_grasping_octree_creator_{self.id}')

        # Additional parameters
        self._octree_max_size = octree_max_size

    def create_observation_space(self) -> ObservationSpace:

        # 0:n - octree
        # Note: octree is currently padded with zeros to have constant size
        # TODO: Customize replay buffer to support variable sized observations
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._octree_max_size,),
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
                  "allowed size (exceeded with {octree_size})")
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

        # Create the observation
        observation = Observation(octree)

        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation
