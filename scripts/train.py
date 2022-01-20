#!/usr/bin/env -S python3 -O

import argparse
import difflib
import os
import uuid
from typing import Dict

import gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed

from drl_grasping.utils import import_envs
from drl_grasping.utils.exp_manager import ExperimentManager
from drl_grasping.utils.utils import ALGOS, StoreDict, empty_str2none, str2bool


def main(args: Dict):

    # Check if the selected environment is valid
    # If it could not be found, suggest the closest match
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{args.env} not found in gym registry, you maybe meant {closest_match}?"
        )

    # If no specific seed is selected, choose a random one
    if args.seed < 0:
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    # Set the random seed across platforms
    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    # Verify that pre-trained agent exists before continuing to train it
    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    # If enabled, ensure that the run has a unique ID
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""

    print("=" * 10, args.env, "=" * 10)
    print(f"Seed: {args.seed}")

    exp_manager = ExperimentManager(
        args,
        args.algo,
        args.env,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        preload_replay_buffer=args.preload_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
    )

    # Prepare experiment
    model = exp_manager.setup_experiment()

    exp_manager.learn(model)
    exp_manager.save_trained_model(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment and its parameters
    parser.add_argument(
        "--env", type=str, default="Reach-Gazebo-v0", help="Environment ID"
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=["dummy", "subproc"],
        default="dummy",
        help="Type of VecEnv to use",
    )

    # Algorithm and training
    parser.add_argument(
        "--algo",
        type=str,
        choices=list(ALGOS.keys()),
        required=False,
        default="sac",
        help="RL algorithm to use during the training",
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional RL hyperparameter overwrite (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-n",
        "--n-timesteps",
        type=int,
        default=-1,
        help="Overwrite the number of timesteps",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help="Number of threads for PyTorch (-1 to use default)",
    )

    # Continue training an already trained agent
    parser.add_argument(
        "-i",
        "--trained-agent",
        type=str,
        default="",
        help="Path to a pretrained agent to continue training",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=-1, help="Random generator seed")

    # Saving of model
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save the model every n steps (if negative, no checkpoint)",
    )
    parser.add_argument(
        "--save-replay-buffer",
        type=str2bool,
        default=False,
        help="Save the replay buffer too (when applicable)",
    )

    # Pre-load a replay buffer and start training on it
    parser.add_argument(
        "--preload-replay-buffer",
        type=empty_str2none,
        default="",
        help="Path to a replay buffer that should be preloaded before starting the training process",
    )

    # Logging
    parser.add_argument(
        "-f", "--log-folder", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument(
        "-tb",
        "--tensorboard-log",
        type=empty_str2none,
        default="tensorboard_logs",
        help="Tensorboard log dir",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=-1,
        help="Override log interval (default: -1, no change)",
    )
    parser.add_argument(
        "-uuid",
        "--uuid",
        type=str2bool,
        default=False,
        help="Ensure that the run has a unique ID",
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=-1,
        help="Evaluate the agent every n steps (if negative, no evaluation)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes to use for evaluation",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: INFO)"
    )

    # HER specifics
    parser.add_argument(
        "--truncate-last-trajectory",
        type=str2bool,
        default=True,
        help="When using HER with online sampling the last trajectory in the replay buffer will be truncated after reloading the replay buffer.",
    )

    args, unknown = parser.parse_known_args()

    main(args=args)
