#!/usr/bin/env python3

import argparse
import os

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

from drl_grasping.utils.training import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from drl_grasping.utils.training.utils import StoreDict


def main(args=None):

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(
            os.path.join(args.folder, args.algo), args.env)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(args.folder, args.algo,
                                f"{args.env}_{args.exp_id}")
    else:
        log_path = os.path.join(args.folder, args.algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{args.env}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(
            log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(
            f"No model found for {args.algo} on {args.env}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if args.algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, args.env)
    hyperparams, stats_path = get_saved_hyperparams(
        stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, args.env, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            # pytype: disable=module-attr
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        args.env,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if args.algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    model = ALGOS[args.algo].load(model_path, env=env, **kwargs)

    obs = env.reset()

    # Deterministic by default
    stochastic = args.stochastic
    deterministic = not stochastic

    print(f"Evaluating for {args.n_episodes} episodes with a",
          "deterministic" if deterministic else "stochastic", "policy.")

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths, success_episode_lengths = [], [], []
    ep_len = 0
    episode = 0
    # For HER, monitor success rate
    successes = []
    while episode < args.n_episodes:
        action, state = model.predict(
            obs, state=state, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render("human")

        episode_reward += reward[0]
        ep_len += 1

        if done and args.verbose > 0:
            episode += 1
            print(f"--- Episode {episode}/{args.n_episodes}")
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print(f"Episode Reward: {episode_reward:.2f}")
            episode_rewards.append(episode_reward)
            print("Episode Length", ep_len)
            episode_lengths.append(ep_len)
            if infos[0].get("is_success") is not None:
                print("Success?:", infos[0].get("is_success", False))
                successes.append(infos[0].get("is_success", False))
                if infos[0].get("is_success"):
                    success_episode_lengths.append(ep_len)
                print(f"Current success rate: {100 * np.mean(successes):.2f}%")
            episode_reward = 0.0
            ep_len = 0
            state = None

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"Mean reward: {np.mean(episode_rewards):.2f} "
              f"+/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} "
              f"+/- {np.std(episode_lengths):.2f}")

    if args.verbose > 0 and len(success_episode_lengths) > 0:
        print(f"Mean episode length of successful episodes: {np.mean(success_episode_lengths):.2f} "
              f"+/- {np.std(success_episode_lengths):.2f}")

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and "Bullet" not in args.env and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            # SubprocVecEnv
            env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment and its parameters
    parser.add_argument("--env", type=str,
                        default="Reach-Gazebo-v0",
                        help="environment ID")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict,
                        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--n-envs", type=int,
                        default=1,
                        help="number of environments")

    # Algorithm
    parser.add_argument("--algo", type=str, choices=list(ALGOS.keys()), required=False,
                        default="sac", help="RL Algorithm")
    parser.add_argument("--num-threads", type=int,
                        default=-1,
                        help="Number of threads for PyTorch (-1 to use default)")

    # Test duration
    parser.add_argument("-n", "--n-episodes", type=int,
                        default=200,
                        help="Overwrite the number of episodes")

    # Random seed
    parser.add_argument("--seed", type=int,
                        default=0,
                        help="Random generator seed")

    # Model to test
    parser.add_argument("-f", "--folder", type=str,
                        default="logs",
                        help="Log folder")
    parser.add_argument("--exp-id", type=int,
                        default=0,
                        help="Experiment ID (default: 0: latest, -1: no exp folder)")
    parser.add_argument("--load-best", action="store_true",
                        default=False,
                        help="Load best model instead of last model if available")
    parser.add_argument("--load-checkpoint", type=int,
                        help="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it")

    # Deterministic/stochastic actions
    parser.add_argument("--deterministic", action="store_true",
                        default=True,
                        help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true",
                        default=False,
                        help="Use stochastic actions")

    # Logging
    parser.add_argument("--reward-log", type=str,
                        default="reward_logs",
                        help="Where to log reward")
    parser.add_argument("--norm-reward", action="store_true",
                        default=False,
                        help="Normalize reward if applicable (trained with VecNormalize)")

    # Disable render
    parser.add_argument("--no-render", action="store_true",
                        default=False,
                        help="Do not render the environment (useful for tests)")

    # Verbosity
    parser.add_argument("--verbose", type=int,
                        default=1,
                        help="Verbose mode (0: no output, 1: INFO)")

    args = parser.parse_args()

    main(args)
