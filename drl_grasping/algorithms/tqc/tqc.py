# This module adds a monkey patch to TQC `_store_transition` function
# The only difference is that if info contains 'curriculum.restart_exploration' key set to True, the ent_coef and its optimizer will be reset.

# Note: needs to be included before `from sb3_contrib import TQC` in the module that uses this

from drl_grasping.algorithms.common import off_policy_algorithm
from sb3_contrib import TQC
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List
import numpy as np
import torch as th


__old_store_transition__ = TQC._store_transition


def _setup_model_store_init_and_reset_ent_coef(self) -> None:
    super(TQC, self)._setup_model()
    self._create_aliases()
    self.replay_buffer.actor = self.actor
    self.replay_buffer.ent_coef = 0.0

    # Target entropy is used when learning the entropy coefficient
    if self.target_entropy == "auto":
        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
    else:
        # Force conversion
        # this will also throw an error for unexpected string
        self.target_entropy = float(self.target_entropy)

    # The entropy coefficient or entropy can be learned automatically
    # see Automating Entropy Adjustment for Maximum Entropy RL section
    # of https://arxiv.org/abs/1812.05905
    if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
        # Default initial value of ent_coef when learned
        init_value = 1.0
        if "_" in self.ent_coef:
            init_value = float(self.ent_coef.split("_")[1])
            assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # NOTE: The monkey patch is on the following block of code
            if len(self.ent_coef.split("_")) == 3:
                restart_value = float(self.ent_coef.split("_")[2])
                assert restart_value > 0.0, "The retart value of ent_coef must be greater than 0"
                self.ent_coef_restart_value = restart_value
            else:
                self.ent_coef_restart_value = init_value

        # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
    else:
        # Force conversion to float
        # this will throw an error if a malformed string (different from 'auto')
        # is passed
        self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


def _store_transition_allow_exploration_reset(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]]):

    # Chain up original implementation
    __old_store_transition__(self,
                             replay_buffer=replay_buffer,
                             buffer_action=buffer_action,
                             new_obs=new_obs,
                             reward=reward,
                             done=done,
                             infos=infos)

    if infos[0].get("curriculum.restart_exploration", False):
        if self.ent_coef_optimizer is not None:
            restart_value = self.ent_coef_restart_value if hasattr(
                self, 'ent_coef_restart_value') else 1.0
            print("Curriculum:\n"
                  f"\tRestarting exploration (ent_coef = {restart_value})")
            del self.log_ent_coef
            self.log_ent_coef = th.log(th.ones(
                1, device=self.device) * restart_value).requires_grad_(True)

            del self.ent_coef_optimizer
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1))


TQC._setup_model = _setup_model_store_init_and_reset_ent_coef
TQC._store_transition = _store_transition_allow_exploration_reset
