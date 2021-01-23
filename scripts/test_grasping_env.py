#!/usr/bin/env python3

from drl_grasping.envs.randomizers import GraspingGazeboEnvRandomizer
from gym_ignition.utils import logger
import functools
import gym


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    return gym.make(env_id, **kwargs)


def main(args=None):

    # Set verbosity
    logger.set_level(gym.logger.ERROR)

    # Create a partial function passing the environment id
    make_env = functools.partial(make_env_from_id, env_id="Grasping-Gazebo-v0")

    # Wrap environment with randomizer
    env = GraspingGazeboEnvRandomizer(env=make_env)

    # Initialize random seed
    env.seed(42)

    # Enable rendering
    env.render('human')

    # Step environment for bunch of episodes
    for episode in range(10000):

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
            observation, reward, done, _ = env.step(action)

            # Accumulate the reward
            total_reward += reward

        print(f"Episode #{episode} reward: {total_reward}")

    # Cleanup once done
    env.close()


if __name__ == "__main__":
    main()
