#!/usr/bin/env -S python3 -O

import argparse
import difflib
import os
import uuid

import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common.utils import set_random_seed

from drl_grasping.utils.training.exp_manager import ExperimentManager
from drl_grasping.utils.training.utils import ALGOS, StoreDict

seaborn.set()


def main(args=None):

    # Check if the selected environment is valid
    # If it could not be found, suggest the closest match
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(
                args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{args.env} not found in gym registry, you maybe meant {closest_match}?")

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

    print("=" * 10, "Preloading buffer for ", args.env, "=" * 10)
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
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()
    # Collect transitions for demonstration
    exp_manager.collect_demonstration(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment and its parameters
    parser.add_argument("--env", type=str,
                        default="Reach-Gazebo-v0",
                        help="environment ID")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict,
                        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"],
                        default="dummy",
                        help="VecEnv type")

    # Algorithm
    parser.add_argument("--algo", type=str, choices=list(ALGOS.keys()), required=False,
                        default="sac", help="RL Algorithm")
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("--num-threads", type=int,
                        default=-1,
                        help="Number of threads for PyTorch (-1 to use default)")

    # Training duration
    parser.add_argument("-n", "--n-timesteps", type=int,
                        default=-1,
                        help="Overwrite the number of timesteps")

    # Continue training an already trained agent
    parser.add_argument("-i", "--trained-agent", type=str,
                        default="",
                        help="Path to a pretrained agent to continue training")

    # Random seed
    parser.add_argument("--seed", type=int,
                        default=-1,
                        help="Random generator seed")

    # Saving of model
    parser.add_argument("--save-freq", type=int,
                        default=10000,
                        help="Save the model every n steps (if negative, no checkpoint)")
    parser.add_argument("--save-replay-buffer", action="store_true",
                        default=True,
                        help="Save the replay buffer too (when applicable)")

    # Logging
    parser.add_argument("-f", "--log-folder", type=str,
                        default="logs",
                        help="Log folder")
    parser.add_argument("-tb", "--tensorboard-log", type=str,
                        default="tensorboard_logs",
                        help="Tensorboard log dir")
    parser.add_argument("--log-interval", type=int,
                        default=-1,
                        help="Override log interval (default: -1, no change)")
    parser.add_argument("-uuid", "--uuid", action="store_true",
                        default=False,
                        help="Ensure that the run has a unique ID")

    # Hyperparameter optimization
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true",
                        default=False,
                        help="Run hyperparameters search")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "skopt"],
                        default="tpe",
                        help="Sampler to use when optimizing hyperparameters")
    parser.add_argument("--pruner", type=str, choices=["halving", "median", "none"],
                        default="median",
                        help="Pruner to use when optimizing hyperparameters")
    parser.add_argument("--n-trials", type=int,
                        default=10,
                        help="Number of trials for optimizing hyperparameters")
    parser.add_argument("--n-startup-trials", type=int,
                        default=5,
                        help="Number of trials before using optuna sampler")
    parser.add_argument("--n-evaluations", type=int,
                        default=2,
                        help="Number of evaluations for hyperparameter optimization")
    parser.add_argument("--n-jobs", type=int,
                        default=1,
                        help="Number of parallel jobs when optimizing hyperparameters")
    parser.add_argument("--storage", type=str,
                        default=None,
                        help="Database storage path if distributed optimization should be used")
    parser.add_argument("--study-name", type=str,
                        default=None,
                        help="Study name for distributed optimization")

    # Evaluation
    parser.add_argument("--eval-freq", type=int,
                        default=-1,
                        help="Evaluate the agent every n steps (if negative, no evaluation)")
    parser.add_argument("--eval-episodes", type=int,
                        default=5,
                        help="Number of episodes to use for evaluation")

    # Verbosity
    parser.add_argument("--verbose", type=int,
                        default=1,
                        help="Verbose mode (0: no output, 1: INFO)")

    # HER specifics
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )

    args = parser.parse_args()

    main(args=args)
