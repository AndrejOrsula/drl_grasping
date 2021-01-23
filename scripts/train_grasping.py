#!/usr/bin/env python3

# Must be included before gym ignition (protobuf)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from drl_grasping.envs.randomizers import GraspingGazeboEnvRandomizer
from gym_ignition.utils import logger
from os import path
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

    # # Enable rendering
    # env.render('human')

    # Define training itself
    training_steps = 10000000
    training_dir = "/home/andrej/uni/training/grasping_image_cnn_sac/"
    model_name = "model0"
    model_path = path.join(training_dir, model_name)
    tensorboard_path = path.join(training_dir, "tensorboard")

    load_checkpoint = False
    previous_checkpoint = path.join(training_dir,
                                    "checkpoint0X", model_name + "_XX000_steps")
    checkpoint_save_path = path.join(training_dir, "checkpoint00")
    save_freq = 10000

    test_trained_model = False

    # Initialize or load the model
    model = None
    if not load_checkpoint:

        model = SAC(policy="CnnPolicy",
                    env=env,
                    buffer_size=65536,
                    batch_size=256,
                    learning_starts=2048,
                    learning_rate=0.001,
                    gradient_steps=1,
                    tensorboard_log=tensorboard_path,
                    verbose=1)
    else:

        model = SAC.load(path=previous_checkpoint,
                         env=env,
                         tensorboard_log=tensorboard_path,
                         verbose=1)

    # Train or test the model
    if not test_trained_model:

        # Create callback for checkpoints
        checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                 save_path=checkpoint_save_path,
                                                 name_prefix=model_name,
                                                 verbose=1)

        # Train
        model.learn(total_timesteps=training_steps,
                    callback=checkpoint_callback,
                    log_interval=5)

        # Save the model once finished
        model.save(model_path)
    else:

        for episode in range(10000):

            # Initialize returned values
            done = False
            total_reward = 0

            # Reset the environment
            observation = env.reset()

            # Step through the current episode until it is done
            while not done:

                # Take an action (greedy) based on last observation
                action, _states = model.predict(observation)

                # Step the environment with the given action
                observation, reward, done, _info = env.step(action)

                # Accumulate the reward
                total_reward += reward

            print(f"Episode #{episode} reward: {total_reward}")

    # Cleanup once done
    env.close()


if __name__ == "__main__":
    main()
