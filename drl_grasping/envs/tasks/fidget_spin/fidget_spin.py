from gym_ignition.base import task
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from itertools import count
from typing import Tuple, List, Union, Dict
import abc
import gym
import numpy as np

class FidgetSpin(task.Task, abc.ABC):
    _ids = count(0)

    _hand_position: Tuple[float, float, float] = (0, 0, 0)
    _hand_quat_xyzw: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)
    _hand_initial_joint_positions: List[float] = [0.0] * 22

    _camera_enable: bool = False
    _camera_type: str = 'rgbd_camera'
    _camera_render_engine: str = 'ogre2'
    _camera_position: Tuple[float, float, float] = (0, 0, 0)
    _camera_quat_xyzw: Tuple[float, float,
                             float, float] = (0, 0, 0, 1)
    _camera_width: int = 256
    _camera_height: int = 256
    _camera_update_rate: int = 10
    _camera_horizontal_fov: float = 1.0
    _camera_vertical_fov: float = 1.0
    _camera_clip_color: Tuple[float, float] = (0.01, 1000.0)
    _camera_clip_depth: Tuple[float, float] = (0.01, 10.0)
    _camera_ros2_bridge_color: bool = False
    _camera_ros2_bridge_depth: bool = False
    _camera_ros2_bridge_points: bool = False

    _spinner_position: Tuple[float, float, float] = (0.35, 0.0, 0.15)
    _spinner_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1)

    _spinner_spawn_centre: Tuple[float, float, float] = _spinner_position
    _spinner_spawn_volume: Tuple[float, float, float] = (0.1,
                                                         0.05,
                                                         0.0)

    _reset_spinner_height: float = -0.5

    _insert_scene_broadcaster_plugin: bool = True
    _insert_user_commands_plugin: bool = True

    def __init__(self,
                 agent_rate: float,
                 verbose: bool,
                 **kwargs):
        # Add to ids
        self.id = next(self._ids)

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)

        # Additional parameters
        self._verbose = verbose

        # Names of hand and spinner
        self.hand_name = None
        self.hand_joint_names = None
        self.spinner_name = None

        # Tracker of joint position
        self._previous_joint_position = 0.0

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Action space
        action_space = self.create_action_space()
        # Observation space
        observation_space = self.create_observation_space()

        return action_space, observation_space

    def create_action_space(self) -> ActionSpace:

        # Effort for all actuated joints
        # 0  - 'thumb_joint1'           +-100
        # 1  - 'thumb_joint2'           +-100
        # 2  - 'thumb_joint3'           +-20
        # 3  - 'thumb_joint4'           +-20
        # 4  - 'thumb_joint5'           +-20
        # 5  - 'index_finger_joint1'    +-20
        # 6  - 'index_finger_joint2'    +-20
        # 7  - 'index_finger_joint3'    +-20
        # 8  - 'index_finger_joint4'    +-20
        # 9  - 'middle_finger_joint1'   +-20
        # 10 - 'middle_finger_joint2'   +-20
        # 11 - 'middle_finger_joint3'   +-20
        # 12 - 'middle_finger_joint4'   +-20
        # 13 - 'ring_finger_joint1'     +-20
        # 14 - 'ring_finger_joint2'     +-20
        # 15 - 'ring_finger_joint3'     +-20
        # 16 - 'ring_finger_joint4'     +-20
        # 17 - 'little_finger_joint1'   +-50
        # 18 - 'little_finger_joint2'   +-20
        # 19 - 'little_finger_joint3'   +-20
        # 20 - 'little_finger_joint4'   +-20
        # 21 - 'little_finger_joint5'   +-20
        return gym.spaces.Box(low=-1.0,
                              high=1.0,
                              shape=(22,),
                              dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:

        pass

    def set_action(self, action: Action):

        if self._verbose:
            print(f"action: {action}")

        # Rescale to joint limits
        action *= 20.0
        action[0] *= 5.0
        action[1] *= 5.0
        action[17] *= 2.5

        # Execute
        hand_model = self.world.get_model(self.hand_name).to_gazebo()
        hand_model.set_joint_generalized_force_targets(action.tolist())

    def get_observation(self) -> Observation:

        pass

    def get_reward(self) -> Reward:

        # Set reward based how much further the spinner spun since last time
        new_joint_position = self.get_spinner_joint_position()
        if new_joint_position < 0:
            reward = self._previous_joint_position-new_joint_position
        else:
            reward = new_joint_position-self._previous_joint_position
        self._previous_joint_position = new_joint_position

        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        # Reset if spinner fell out of the hand
        if self.get_spinner_position()[2] < self._reset_spinner_height:
            return True
        else:
            return False

    def get_info(self) -> Dict:

        return {}

    def reset_task(self):

        # Task is reset by the randomizer

        if self._verbose:
            print(f"\ntask reset")

    def get_spinner_position(self, link_name: Union[str, None] = None) -> Tuple[float, float, float]:

        # Get reference to the model of the object
        spinner_model = self.world.get_model(self.spinner_name).to_gazebo()

        # Use the first link if not specified
        if link_name is None:
            link_name = spinner_model.link_names()[0]

        # Return position of the object's link
        return spinner_model.get_link(link_name=link_name).position()

    def get_spinner_joint_position(self, joint_name: Union[str, None] = None) -> float:

        # Get reference to the model of the object
        spinner_model = self.world.get_model(self.spinner_name).to_gazebo()

        # Use the first link if not specified
        if joint_name is None:
            joint_name = spinner_model.joint_names()[0]

        # Return position of the joint
        return spinner_model.get_joint(joint_name=joint_name).position(0)
