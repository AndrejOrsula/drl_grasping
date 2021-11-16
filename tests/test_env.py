#!/usr/bin/env python3

from stable_baselines3.common.env_checker import check_env
from drl_grasping.envs.randomizers import (
    ManipulationGazeboEnvRandomizer,
    ManipulationPlanetaryGazeboEnvRandomizer,
)
from gym_ignition.utils import logger
import functools
import gym

# Reach
# env_id="Reach-Gazebo-v0"
# env_id="Reach-ColorImage-Gazebo-v0"
# env_id="Reach-Octree-Gazebo-v0"
# env_id="Reach-OctreeWithColor-Gazebo-v0"
# Grasp
# env_id = "Grasp-Octree-Gazebo-v0"
# env_id = "Grasp-OctreeWithColor-Gazebo-v0"
# GraspPlanetary
# env_id = "GraspPlanetary-Gazebo-v0"
env_id = "GraspPlanetary-OctreeWithColor-Gazebo-v0"

from gym_ignition.utils import logger as gym_ign_logger
from gym import logger as gym_logger
from os import environ


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    return gym.make(env_id, **kwargs)


def main(args=None):

    # Set verbosity
    debug_level = environ.get("DRL_GRASPING_DEBUG_LEVEL", default="ERROR").upper()
    logger.set_level(getattr(gym_logger, debug_level))

    # Create a partial function passing the environment id
    make_env = functools.partial(make_env_from_id, env_id=env_id)

    # # Wrap environment with randomizer
    # env = ManipulationGazeboEnvRandomizer(
    #     env=make_env,
    #     object_random_pose=True,
    #     object_models_rollouts_num=1,
    #     object_random_use_mesh_models=True,
    #     object_random_model_count=3,
    #     ground_model_rollouts_num=1,
    # )

    # Wrap environment with randomizer
    env = ManipulationPlanetaryGazeboEnvRandomizer(env=make_env)

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
