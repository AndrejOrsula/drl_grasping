#!/usr/bin/env -S python3 -O

import argparse
import difflib
from typing import Dict

import dreamerv2.api as dv2
import gym
import numpy as np

from drl_grasping import envs as drl_grasping_envs
from drl_grasping.utils.utils import StoreDict


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
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    config, _ = dv2.defaults.update(
        {
            "logdir": args.log_folder,
            "eval_every": args.eval_freq,
            "prefill": 500,
            "pretrain": 100,
            "clip_rewards": "identity",
            "pred_discount": False,
            "replay": {
                "capacity": 1e6,
                "ongoing": False,
                "minlen": 10,
                "maxlen": 10,
                "prioritize_ends": True,
            },
            "dataset": {"batch": 16, "length": 10},
            "grad_heads": ["decoder", "reward"],
            "rssm": {"hidden": 200, "deter": 200},
            "model_opt": {"lr": 1e-4},
            "actor_opt": {"lr": 1e-5},
            "critic_opt": {"lr": 1e-5},
            "actor_ent": 1e-4,
            "render_size": [64, 64],
            "kl": {"free": 1.0},
        }
    ).parse_flags(known_only=True)

    # Set the random seed across platforms
    np.random.seed(args.seed)

    print("=" * 10, args.env, "=" * 10)
    print(f"Seed: {args.seed}")

    env = gym.make(args.env, **args.env_kwargs)
    env.seed(args.seed)

    dv2.train(env, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment and its parameters
    parser.add_argument(
        "--env",
        type=str,
        default="GraspPlanetary-ColorImage-Gazebo-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=-1, help="Random generator seed")

    # Logging
    parser.add_argument(
        "-f", "--log-folder", type=str, default="logs", help="Path to the log directory"
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate the agent every n steps (if negative, no evaluation)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: INFO)"
    )

    args, unknown = parser.parse_known_args()

    main(args=args)
