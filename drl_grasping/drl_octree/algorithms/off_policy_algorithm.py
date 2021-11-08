# This module adds a monkey patch to OffPolicyAlgorithm `_setup_model` function such that separae octree batches for stacks are supported
# Note: needs to be included before `from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm` in the module that uses this

# Note: Import monkey patch of ReplayBuffer before stable_baselines3 ReplayBuffer
from drl_grasping.drl_octree.replay_buffer import octree_replay_buffer
from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


def _setup_model_with_separate_octree_batches_for_stacks(self) -> None:
    self._setup_lr_schedule()
    self.set_random_seed(self.seed)
    if 'separate_networks_for_stacks' in self.policy_kwargs:
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            separate_networks_for_stacks=self.policy_kwargs['separate_networks_for_stacks'],
        )
    else:
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
    self.policy = self.policy_class(  # pytype:disable=not-instantiable
        self.observation_space,
        self.action_space,
        self.lr_schedule,
        **self.policy_kwargs,  # pytype:disable=not-instantiable
    )
    self.policy = self.policy.to(self.device)

    # Convert train freq parameter to TrainFreq object
    self._convert_train_freq()


OffPolicyAlgorithm._setup_model = _setup_model_with_separate_octree_batches_for_stacks
