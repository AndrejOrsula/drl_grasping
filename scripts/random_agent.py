#!/usr/bin/env python3

import argparse
from typing import Dict

import gym
from stable_baselines3.common.env_checker import check_env

from drl_grasping.utils import import_envs
from drl_grasping.utils.utils import StoreDict, str2bool


def main(args: Dict):

    # Create the environment
    env = gym.make(args.env, **args.env_kwargs)

    # Initialize random seed
    env.seed(args.seed)

    # Check the environment
    if args.check_env:
        check_env(env, warn=True, skip_render_check=True)

    # Step environment for bunch of episodes
    for episode in range(args.n_timesteps):

        # Initialize returned values
        done = False
        total_reward = 0

        # Reset the environment
        observation = env.reset()

        # Step through the current episode until it is done
        while not done:

            # Sample random action
            action = env.action_space.sample()

            # Step the environment with the random action
            observation, reward, done, info = env.step(action)

            # Accumulate the reward
            total_reward += reward

        print(f"Episode #{episode}\n\treward: {total_reward}")

    # Cleanup once done
    env.close()


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

    # Number of timesteps to run
    parser.add_argument(
        "-n",
        "--n-timesteps",
        type=int,
        default=10000,
        help="Overwrite the number of timesteps",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=69, help="Random generator seed")

    # Flag to check environment
    parser.add_argument(
        "--check-env",
        type=str2bool,
        default=True,
        help="Flag to check the environment before running the random agent",
    )

    args, unknown = parser.parse_known_args()

    main(args=args)
