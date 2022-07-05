# This module adds a monkey patch to ReplayBuffer, such that octrees are
# directly supported and there is no extra RAM -> VRAM -> RAM overhead

# Note: needs to be included before `from stable_baselines3.common.buffers import ReplayBuffer` in the module that uses this

from typing import Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from .utils import preprocess_stacked_depth_image_batch, preprocess_stacked_octree_batch

__old__init__ = ReplayBuffer.__init__
__old_get_samples__ = ReplayBuffer._get_samples


def __init___with_checking_for_stacked_images_and_octrees(
    self,
    buffer_size: int,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    device: Union[th.device, str] = "cpu",
    n_envs: int = 1,
    optimize_memory_usage: bool = False,
    separate_networks_for_stacks: bool = True,
):
    __old__init__(
        self,
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        n_envs=n_envs,
        optimize_memory_usage=optimize_memory_usage,
    )

    # Determine if octrees are used
    # Note: This is not 100% reliable as there could be other observations that do the same (outside of this repo)
    self.contains_octree_obs = False
    self.contains_stacked_image_obs = False
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 2:
        if (
            np.uint8 == observation_space.dtype
            and np.all(0 == observation_space.low)
            and np.all(255 == observation_space.high)
        ):
            self.contains_octree_obs = True
            self._separate_networks_for_stacks = separate_networks_for_stacks
        elif (
            np.float32 == observation_space.dtype
            and np.all(-1.0 == observation_space.low)
            and np.all(1.0 == observation_space.high)
        ):
            self.contains_stacked_image_obs = True
            self._separate_networks_for_stacks = separate_networks_for_stacks


def _get_samples_with_support_for_octree(
    self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
) -> ReplayBufferSamples:

    if self.contains_octree_obs:
        # Current observations
        obs = self.observations[batch_inds, 0, :]
        obs = preprocess_stacked_octree_batch(
            obs, self.device, separate_batches=self._separate_networks_for_stacks
        )

        # Next observations
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, 0, :]
        else:
            next_obs = self.next_observations[batch_inds, 0, :]
        next_obs = preprocess_stacked_octree_batch(
            next_obs, self.device, separate_batches=self._separate_networks_for_stacks
        )

        return ReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds, 0, :]),
            next_observations=next_obs,
            dones=self.to_torch(self.dones[batch_inds]),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds], env)
            ),
        )
    elif self.contains_stacked_image_obs:
        # Current observations
        obs = self.observations[batch_inds, 0, :]
        obs = preprocess_stacked_depth_image_batch(
            obs, self.device, separate_batches=self._separate_networks_for_stacks
        )

        # Next observations
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, 0, :]
        else:
            next_obs = self.next_observations[batch_inds, 0, :]
        next_obs = preprocess_stacked_depth_image_batch(
            next_obs, self.device, separate_batches=self._separate_networks_for_stacks
        )

        return ReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds, 0, :]),
            next_observations=next_obs,
            dones=self.to_torch(self.dones[batch_inds]),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds], env)
            ),
        )
    else:
        return __old_get_samples__(self, batch_inds=batch_inds, env=env)


ReplayBuffer.__init__ = __init___with_checking_for_stacked_images_and_octrees
ReplayBuffer._get_samples = _get_samples_with_support_for_octree
