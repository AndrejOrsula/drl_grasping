from stable_baselines3.common.noise import NormalActionNoise
from torch import nn as nn
from typing import Any, Dict
import numpy as np
import optuna


def sample_sac_params(trial: optuna.Trial,
                      octree_observations: bool = True,
                      octree_depth: int = 6,
                      octree_full_depth: int = 2,
                      octree_channels_in: int = 7,
                      octree_fast_conv: bool = True,
                      octree_batch_norm: bool = True) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparameters
    """

    buffer_size = 80000
    # learning_starts = trial.suggest_categorical(
    #     "learning_starts", [5000, 10000, 20000])
    learning_starts = 5000

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_loguniform(
        "learning_rate", low=0.000001, high=0.001)

    gamma = trial.suggest_loguniform("gamma", low=0.98, high=1.0)
    tau = trial.suggest_loguniform("tau", low=0.001, high=0.025)

    ent_coef = "auto_0.5_0.1"
    target_entropy = "auto"

    noise_std = trial.suggest_loguniform("noise_std", low=0.01, high=0.2)
    action_noise = NormalActionNoise(mean=np.zeros(trial.n_actions),
                                     sigma=np.ones(trial.n_actions)*noise_std)

    train_freq = 1
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2])

    policy_kwargs = dict()
    net_arch = trial.suggest_categorical("net_arch", ["small [256, 128]",
                                                      "medium [384, 256]",
                                                      "big [512, 384]"])
    policy_kwargs["net_arch"] = {"small [256, 128]": [256, 128],
                                 "medium [384, 256]": [384, 256],
                                 "big [512, 384]": [512, 384]}[net_arch]
    if octree_observations:
        features_extractor_kwargs = dict()

        features_extractor_kwargs["depth"] = octree_depth
        features_extractor_kwargs["full_depth"] = octree_full_depth
        features_extractor_kwargs["channels_in"] = octree_channels_in

        features_extractor_kwargs["channel_multiplier"] = \
            trial.suggest_categorical("channel_multiplier", [8, 16, 32])

        features_extractor_kwargs["features_dim"] = \
            trial.suggest_categorical("features_dim", [256, 512, 768])

        features_extractor_kwargs["fast_conv"] = octree_fast_conv
        features_extractor_kwargs["batch_normalization"] = octree_batch_norm

        policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs

    return {
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "tau": tau,
        "ent_coef": ent_coef,
        "target_entropy": target_entropy,
        "action_noise": action_noise,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": policy_kwargs,
    }


def sample_td3_params(trial: optuna.Trial,
                      octree_observations: bool = True,
                      octree_depth: int = 6,
                      octree_full_depth: int = 2,
                      octree_channels_in: int = 7,
                      octree_fast_conv: bool = True,
                      octree_batch_norm: bool = True) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparameters
    """

    buffer_size = 80000
    # learning_starts = trial.suggest_categorical(
    #     "learning_starts", [5000, 10000, 20000])
    learning_starts = 5000

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_loguniform(
        "learning_rate", low=0.000001, high=0.001)

    gamma = trial.suggest_loguniform("gamma", low=0.98, high=1.0)
    tau = trial.suggest_loguniform("tau", low=0.001, high=0.025)

    target_policy_noise = trial.suggest_loguniform(
        "target_policy_noise", low=0.0, high=0.5)
    target_noise_clip = 0.5

    noise_std = trial.suggest_loguniform("noise_std", low=0.025, high=0.5)
    action_noise = NormalActionNoise(mean=np.zeros(trial.n_actions),
                                     sigma=np.ones(trial.n_actions)*noise_std)

    train_freq = 1
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2])

    policy_kwargs = dict()
    net_arch = trial.suggest_categorical("net_arch", ["small [256, 128]",
                                                      "medium [384, 256]",
                                                      "big [512, 384]"])
    policy_kwargs["net_arch"] = {"small [256, 128]": [256, 128],
                                 "medium [384, 256]": [384, 256],
                                 "big [512, 384]": [512, 384]}[net_arch]
    if octree_observations:
        features_extractor_kwargs = dict()

        features_extractor_kwargs["depth"] = octree_depth
        features_extractor_kwargs["full_depth"] = octree_full_depth
        features_extractor_kwargs["channels_in"] = octree_channels_in

        features_extractor_kwargs["channel_multiplier"] = \
            trial.suggest_categorical("channel_multiplier", [8, 16, 32])

        features_extractor_kwargs["features_dim"] = \
            trial.suggest_categorical("features_dim", [256, 512, 768])

        features_extractor_kwargs["fast_conv"] = octree_fast_conv
        features_extractor_kwargs["batch_normalization"] = octree_batch_norm

        policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs

    return {
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "tau": tau,
        "target_policy_noise": target_policy_noise,
        "target_noise_clip": target_noise_clip,
        "action_noise": action_noise,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": policy_kwargs,
    }


HYPERPARAMS_SAMPLER = {
    "sac": sample_sac_params,
    "td3": sample_td3_params,
}
