#!/usr/bin/env python3

from stable_baselines3.common.env_checker import check_env
from drl_grasping.envs.randomizers import FidgetSpinGazeboEnvRandomizer
from gym_ignition.utils import logger
import functools
import gym

# Reach
# env_id="Reach-Gazebo-v0"
# env_id="Reach-ColorImage-Gazebo-v0"
# env_id="Reach-Octree-Gazebo-v0"
# env_id="Reach-OctreeWithColor-Gazebo-v0"
# Grasp
env_id = "FidgetSpin-OctreeWithColor-Gazebo-v0"
# env_id = "Grasp-OctreeWithColor-Gazebo-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    return gym.make(env_id, **kwargs)


def main(args=None):

    # Set verbosity
    logger.set_level(gym.logger.ERROR)

    # Create a partial function passing the environment id
    make_env = functools.partial(make_env_from_id, env_id=env_id)

    # Wrap environment with randomizer
    env = FidgetSpinGazeboEnvRandomizer(env=make_env)

    # Initialize random seed
    env.seed(42)

    # Enable rendering
    env.render('human')

    # Check it
    check_env(env, warn=True, skip_render_check=True)

    # Step environment for bunch of episodes
    for episode in range(100000):

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

        print(f"Episode #{episode} reward: {total_reward}")

    # Cleanup once done
    env.close()


if __name__ == "__main__":
    main()
