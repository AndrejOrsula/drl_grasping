# This module adds a monkey patch to SAC `train` function
# The only difference is in the optimization objective for temperature (entropy_coef / alpha),
# where `alpha * log_prob` is optimised instead of `log_alpha * log_prob`, such that it matches original paper

# Note: needs to be included before `from stable_baselines3 import SAC` in the module that uses this

from stable_baselines3 import SAC

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.utils import polyak_update


def _setup_model_optimize_for_alpha_instead_of_log_alpha(self) -> None:
    super(SAC, self)._setup_model()
    self._create_aliases()
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

        # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        ### NOTE: The monkey patch is on the following line of code 
        self.log_ent_coef = (th.ones(1, device=self.device) * init_value).requires_grad_(True)
        # self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
    else:
        # Force conversion to float
        # this will throw an error if a malformed string (different from 'auto')
        # is passed
        self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


SAC._setup_model = _setup_model_optimize_for_alpha_instead_of_log_alpha
