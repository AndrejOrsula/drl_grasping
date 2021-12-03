from drl_grasping.envs.tasks.manipulation import Manipulation
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class Reach(Manipulation, abc.ABC):
    def __init__(
        self,
        sparse_reward: bool,
        act_quick_reward: float,
        required_accuracy: float,
        **kwargs,
    ):

        # Initialize the Task base class
        Manipulation.__init__(
            self,
            **kwargs,
        )

        # Additional parameters
        self._sparse_reward: bool = sparse_reward
        self._act_quick_reward = (
            act_quick_reward if act_quick_reward >= 0.0 else -act_quick_reward
        )
        self._required_accuracy: float = required_accuracy

        # Flag indicating if the task is done (performance - get_reward + is_done)
        self._is_done: bool = False

        # Distance to target in the previous step (or after reset)
        self._previous_distance: float = None

    def create_action_space(self) -> ActionSpace:

        # 0:3 - (x, y, z) displacement
        #     - rescaled to metric units before use
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:

        # 0:3 - (x, y, z) end effector position
        # 3:6 - (x, y, z) target position
        # Note: These could theoretically be restricted to the workspace and object spawn area instead of inf
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def set_action(self, action: Action):

        self.get_logger().debug(f"action: {action}")

        if self._use_servo:
            self.servo(linear=action[0:3])
        else:
            position = self.get_relative_position(action[0:3])
            quat_xyzw = (1.0, 0.0, 0.0, 0.0)
            self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)

    def get_observation(self) -> Observation:

        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()

        # Create the observation
        observation = Observation(np.concatenate([ee_position, target_position]))

        self.get_logger().debug(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        reward = 0.0

        # Compute the current distance to the target
        current_distance = self.get_distance_to_target()

        # Mark the episode done if target is reached
        if current_distance < self._required_accuracy:
            self._is_done = True
            if self._sparse_reward:
                reward += 1.0

        # Give reward based on how much closer robot got relative to the target for dense reward
        if not self._sparse_reward:
            reward += self._previous_distance - current_distance
            self._previous_distance = current_distance

        # Subtract a small reward each step to provide incentive to act quickly (if enabled)
        reward -= self._act_quick_reward

        self.get_logger().debug(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        done = self._is_done

        self.get_logger().debug(f"done: {done}")

        return done

    def reset_task(self):

        self._is_done = False

        # Compute and store the distance after reset if using dense reward
        if not self._sparse_reward:
            self._previous_distance = self.get_distance_to_target()

        self.get_logger().debug(f"\ntask reset")

    def get_distance_to_target(self) -> Tuple[float, float, float]:

        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()

        # Compute the current distance to the target
        return np.linalg.norm(
            [
                ee_position[0] - target_position[0],
                ee_position[1] - target_position[1],
                ee_position[2] - target_position[2],
            ]
        )

    def get_target_position(self) -> Tuple[float, float, float]:

        target_object = self.world.get_model(self.object_names[0]).to_gazebo()
        return target_object.get_link(
            link_name=target_object.link_names()[0]
        ).position()
