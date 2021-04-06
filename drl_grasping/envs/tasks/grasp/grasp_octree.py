from collections import deque
from drl_grasping.envs.tasks.grasp import Grasp
from drl_grasping.perception import CameraSubscriber, OctreeCreator
from drl_grasping.utils.conversions import orientation_quat_to_6d
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class GraspOctree(Grasp, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _camera_enable: bool = True
    _camera_type: str = 'auto'
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

    # A small offset to include ground inside the observations
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
                 curriculum_enable_stages: bool,
                 curriculum_enable_workspace_scale: bool,
                 curriculum_success_rate_threshold: float,
                 curriculum_success_rate_rolling_average_n: int,
                 curriculum_restart_every_n_steps: int,
                 curriculum_min_workspace_scale: float,
                 curriculum_stage_reward_multiplier: float,
                 curriculum_stage_increase_rewards: bool,
                 curriculum_scale_negative_reward: bool,
                 curriculum_n_ground_collisions_till_termination: bool,
                 curriculum_reward_multiplier: float,
                 curriculum_reach_dense_reward_multiplier: float,
                 curriculum_lift_dense_reward_multiplier: float,
                 curriculum_skip_grasp_stage: bool,
                 curriculum_restart_exploration_at_start: bool,
                 octree_depth: int,
                 octree_full_depth: int,
                 octree_include_color: bool,
                 octree_n_stacked: int,
                 octree_max_size: int,
                 proprieceptive_observations: bool,
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
                       curriculum_enable_stages=curriculum_enable_stages,
                       curriculum_enable_workspace_scale=curriculum_enable_workspace_scale,
                       curriculum_success_rate_threshold=curriculum_success_rate_threshold,
                       curriculum_success_rate_rolling_average_n=curriculum_success_rate_rolling_average_n,
                       curriculum_restart_every_n_steps=curriculum_restart_every_n_steps,
                       curriculum_min_workspace_scale=curriculum_min_workspace_scale,
                       curriculum_stage_reward_multiplier=curriculum_stage_reward_multiplier,
                       curriculum_stage_increase_rewards=curriculum_stage_increase_rewards,
                       curriculum_scale_negative_reward=curriculum_scale_negative_reward,
                       curriculum_n_ground_collisions_till_termination=curriculum_n_ground_collisions_till_termination,
                       curriculum_reward_multiplier=curriculum_reward_multiplier,
                       curriculum_reach_dense_reward_multiplier=curriculum_reach_dense_reward_multiplier,
                       curriculum_lift_dense_reward_multiplier=curriculum_lift_dense_reward_multiplier,
                       curriculum_skip_grasp_stage=curriculum_skip_grasp_stage,
                       curriculum_restart_exploration_at_start=curriculum_restart_exploration_at_start,
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
                                            node_name=f'drl_grasping_octree_creator_{self.id}')

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
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._octree_n_stacked,
                                     self._octree_max_size),
                              dtype=np.uint8)

    def create_proprioceptive_observation_space(self) -> ObservationSpace:

        # 0   - (gripper) Gripper state
        #       - 1.0: opened
        #       - -1.0: closed
        # 1:4 - (x, y, z) displacement
        #       - metric units, unbound
        # 4:10 - (v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) 3D orientation in "6D representation"
        #       - normalised
        return gym.spaces.Box(low=np.array((-1.0,
                                            -np.inf, -np.inf, -np.inf,
                                            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)),
                              high=np.array((1.0,
                                             np.inf, np.inf, np.inf,
                                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
                              shape=(10,),
                              dtype=np.float32)

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

        if self._proprieceptive_observations:
            # Add number of auxiliary observations to octree structure
            octree[-8:-4] = np.ndarray(buffer=np.array([10],
                                                       dtype='uint32').tobytes(),
                                       shape=(4,),
                                       dtype='uint8')

            # Gather proprioceptive observations
            ee_position = self.get_ee_position()
            # TODO: get_ee_orientation depends on realtime runtime PR (xyzw order)
            ee_orientation = orientation_quat_to_6d(
                quat_xyzw=self.get_ee_orientation())
            aux_obs = (self._gripper_state,) + ee_position + \
                ee_orientation[0] + ee_orientation[1]
            
            # Add auxiliary observations into the octree structure
            octree[-48:-8] = np.ndarray(buffer=np.array(aux_obs, dtype='float32').tobytes(),
                       shape=(40,),
                       dtype='uint8')
            

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
