#!/usr/bin/env python3

from drl_grasping import envs as drl_grasping_envs
from stable_baselines3.common.env_checker import check_env
from gym_ignition.utils import logger
import gym
from gym import logger as gym_logger
from os import environ


# env_id = "Reach-Gazebo-v0"
# env_id = "Reach-ColorImage-Gazebo-v0"
# env_id = "Reach-DepthImage-Gazebo-v0"
# env_id = "Reach-Octree-Gazebo-v0"
# env_id = "Reach-OctreeWithColor-Gazebo-v0"

# env_id = "Grasp-Octree-Gazebo-v0"
# env_id = "Grasp-OctreeWithColor-Gazebo-v0"

# env_id = "GraspPlanetary-Octree-Gazebo-v0"
env_id = "GraspPlanetary-OctreeWithColor-Gazebo-v0"


def main(args=None):

    # Set verbosity
    debug_level = environ.get("DRL_GRASPING_DEBUG_LEVEL", default="ERROR").upper()
    logger.set_level(getattr(gym_logger, debug_level))

    # Make environment
    env = gym.make(env_id)

    # Initialize random seed
    env.seed(42)

    # Enable rendering
    # env.render("human")

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
