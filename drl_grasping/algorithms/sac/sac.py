# This module adds a monkey patch to SAC `_store_transition` function
# The only difference is that if info contains 'curriculum.restart_exploration' key set to True, the ent_coef and its optimizer will be reset.

# Note: needs to be included before `from stable_baselines3 import SAC` in the module that uses this

from drl_grasping.algorithms.common import off_policy_algorithm
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List
import numpy as np
import torch as th


__old_store_transition__ = SAC._store_transition


def _setup_model_store_init_and_reset_ent_coef(self) -> None:
    super(SAC, self)._setup_model()
    self._create_aliases()
    # Target entropy is used when learning the entropy coefficient
    if self.target_entropy == "auto":
        # automatically set target entropy if needed
        self.target_entropy = - \
            np.prod(self.env.action_space.shape).astype(np.float32)
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
        self.log_ent_coef = th.log(
            th.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam(
            [self.log_ent_coef], lr=self.lr_schedule(1))
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


SAC._setup_model = _setup_model_store_init_and_reset_ent_coef
SAC._store_transition = _store_transition_allow_exploration_reset


# Note: The patch below was a test to compare stable-baselines3 implementation compared to the original paper.
# Conclusion: No significant difference was found, except a bit longer exploration when using the patch (might have also been a run-to-run noise).
# # This module adds a monkey patch to SAC `train` function
# # The only difference is in the optimization objective for temperature (entropy_coef / alpha),
# # where `alpha * log_prob` is optimised instead of `log_alpha * log_prob`, such that it matches original paper

# # Note: needs to be included before `from stable_baselines3 import SAC` in the module that uses this

# import numpy as np
# import torch as th
# from torch.nn import functional as F
# from stable_baselines3.common import logger
# from stable_baselines3.common.utils import polyak_update

# def _setup_model_optimize_alpha_instead_of_log_alpha(self) -> None:
#     super(SAC, self)._setup_model()
#     self._create_aliases()
#     # Target entropy is used when learning the entropy coefficient
#     if self.target_entropy == "auto":
#         # automatically set target entropy if needed
#         self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
#     else:
#         # Force conversion
#         # this will also throw an error for unexpected string
#         self.target_entropy = float(self.target_entropy)

#     # The entropy coefficient or entropy can be learned automatically
#     # see Automating Entropy Adjustment for Maximum Entropy RL section
#     # of https://arxiv.org/abs/1812.05905
#     if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
#         # Default initial value of ent_coef when learned
#         init_value = 1.0
#         if "_" in self.ent_coef:
#             init_value = float(self.ent_coef.split("_")[1])
#             assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

#         # Note: we optimize the log of the entropy coeff which is slightly different from the paper
#         # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
#         ### NOTE: The monkey patch is on the following line of code
#         self.log_ent_coef = (th.ones(1, device=self.device) * init_value).requires_grad_(True)
#         # self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
#         self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
#     else:
#         # Force conversion to float
#         # this will throw an error if a malformed string (different from 'auto')
#         # is passed
#         self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

# def train_optimize_alpha_instead_of_log_alpha(self, gradient_steps: int, batch_size: int = 64) -> None:
#     # Update optimizers learning rate
#     optimizers = [self.actor.optimizer, self.critic.optimizer]
#     if self.ent_coef_optimizer is not None:
#         optimizers += [self.ent_coef_optimizer]

#     # Update learning rate according to lr schedule
#     self._update_learning_rate(optimizers)

#     ent_coef_losses, ent_coefs = [], []
#     actor_losses, critic_losses = [], []

#     for gradient_step in range(gradient_steps):
#         # Sample replay buffer
#         replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

#         # We need to sample because `log_std` may have changed between two gradient steps
#         if self.use_sde:
#             self.actor.reset_noise()

#         # Action by the current actor for the sampled state
#         actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
#         log_prob = log_prob.reshape(-1, 1)

#         ent_coef_loss = None
#         if self.ent_coef_optimizer is not None:
#             # Important: detach the variable from the graph
#             # so we don't change it with other losses
#             # see https://github.com/rail-berkeley/softlearning/issues/60
#             ### NOTE: The monkey patch is on the following line of code
#             ent_coef = self.log_ent_coef.detach()
#             # ent_coef = th.exp(self.log_ent_coef.detach())
#             ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
#             ent_coef_losses.append(ent_coef_loss.item())
#         else:
#             ent_coef = self.ent_coef_tensor

#         ent_coefs.append(ent_coef.item())

#         # Optimize entropy coefficient, also called
#         # entropy temperature or alpha in the paper
#         if ent_coef_loss is not None:
#             self.ent_coef_optimizer.zero_grad()
#             ent_coef_loss.backward()
#             self.ent_coef_optimizer.step()

#         with th.no_grad():
#             # Select action according to policy
#             next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
#             # Compute the next Q values: min over all critics targets
#             next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
#             next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
#             # add entropy term
#             next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
#             # td error + entropy term
#             target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

#         # Get current Q-values estimates for each critic network
#         # using action from the replay buffer
#         current_q_values = self.critic(replay_data.observations, replay_data.actions)

#         # Compute critic loss
#         critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
#         critic_losses.append(critic_loss.item())

#         # Optimize the critic
#         self.critic.optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic.optimizer.step()

#         # Compute actor loss
#         # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
#         # Mean over all critic networks
#         q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
#         min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
#         actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
#         actor_losses.append(actor_loss.item())

#         # Optimize the actor
#         self.actor.optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor.optimizer.step()

#         # Update target networks
#         if gradient_step % self.target_update_interval == 0:
#             polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

#     self._n_updates += gradient_steps

#     logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#     logger.record("train/ent_coef", np.mean(ent_coefs))
#     logger.record("train/actor_loss", np.mean(actor_losses))
#     logger.record("train/critic_loss", np.mean(critic_losses))
#     if len(ent_coef_losses) > 0:
#         logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

# SAC._setup_model = _setup_model_optimize_alpha_instead_of_log_alpha
# SAC.train = train_optimize_alpha_instead_of_log_alpha
