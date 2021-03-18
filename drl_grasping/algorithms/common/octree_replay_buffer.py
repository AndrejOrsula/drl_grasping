# This module adds a monkey patch to ReplayBuffer, such that octrees are
# directly supported and there is no extra RAM -> VRAM -> RAM overhead

# Note: needs to be included before `from stable_baselines3.common.buffers import ReplayBuffer` in the module that uses this

from stable_baselines3.common.buffers import ReplayBuffer

import numpy as np
import torch as th
from typing import Optional, Union
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gym import spaces

import ocnn


old_init = ReplayBuffer.__init__
old_get_samples = ReplayBuffer._get_samples


def __init___with_checking_for_octree(self,
                                      buffer_size: int,
                                      observation_space: spaces.Space,
                                      action_space: spaces.Space,
                                      device: Union[th.device, str] = "cpu",
                                      n_envs: int = 1,
                                      optimize_memory_usage: bool = False):
    old_init(self,
             buffer_size=buffer_size,
             observation_space=observation_space,
             action_space=action_space,
             device=device,
             n_envs=n_envs,
             optimize_memory_usage=optimize_memory_usage)

    # Determine if octrees are used
    # Note: This is not 100% reliable as there could be other observations that do the same (outside of this repo)
    self.contains_octree_obs = False
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1:
        if np.uint8 == observation_space.dtype and \
            np.all(0 == observation_space.low) and \
                np.all(255 == observation_space.high):
            self.contains_octree_obs = True


def _get_samples_with_support_for_octree(self,
                                         batch_inds: np.ndarray,
                                         env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

    if not self.contains_octree_obs:
        return old_get_samples(self, batch_inds=batch_inds, env=env)

    obs_octree_list = []
    obs_batch = self.observations[batch_inds, 0, :]
    for obs_i in obs_batch:
        octree_size = np.frombuffer(buffer=obs_i[-4:],
                                    dtype='uint32',
                                    count=1)
        octree_tensor = th.from_numpy(obs_i[:octree_size[0]])
        obs_octree_list.append(octree_tensor)
    obs = ocnn.octree_batch(obs_octree_list)

    next_obs_octree_list = []
    if self.optimize_memory_usage:
        next_obs_batch = self.observations[(
            batch_inds + 1) % self.buffer_size, 0, :]
    else:
        next_obs_batch = self.next_observations[batch_inds, 0, :]
    for next_obs_i in next_obs_batch:
        octree_size = np.frombuffer(buffer=next_obs_i[-4:],
                                    dtype='uint32',
                                    count=1)
        octree_tensor = th.from_numpy(next_obs_i[:octree_size[0]])
        next_obs_octree_list.append(octree_tensor)
    next_obs = ocnn.octree_batch(next_obs_octree_list)

    return ReplayBufferSamples(
        observations=obs.to(self.device),
        actions=self.to_torch(self.actions[batch_inds, 0, :]),
        next_observations=next_obs.to(self.device),
        dones=self.to_torch(self.dones[batch_inds]),
        rewards=self.to_torch(self._normalize_reward(
            self.rewards[batch_inds], env)),
    )


ReplayBuffer.__init__ = __init___with_checking_for_octree
ReplayBuffer._get_samples = _get_samples_with_support_for_octree
