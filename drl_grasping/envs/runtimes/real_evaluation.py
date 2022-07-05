import sys
import time
from typing import Optional

import numpy as np
from gym_ignition.base.runtime import Runtime
from gym_ignition.base.task import Task
from gym_ignition.utils import logger
from gym_ignition.utils.typing import (
    Action,
    Done,
    Info,
    Observation,
    Reward,
    SeedList,
    State,
)
from pynput import keyboard

from drl_grasping.envs.tasks.manipulation import Manipulation


class RealEvaluationRuntime(Runtime):
    """
    Implementation of :py:class:`~gym_ignition.base.runtime.Runtime` for execution
    of trained agents on real robots (evaluation only).

    It is assumed that the task has an interface that is invariant to sim/real domain for both actions and observations (e.g. ROS 2 middleware).

    This runtime requires manual reset of the workspace as well as manual logging
    of success rate.

    Enable `manual_stepping` to manually step through the execution (safe mode).
    """

    def __init__(
        self, task_cls: type, agent_rate: float, manual_stepping: bool = True, **kwargs
    ):

        # Create the Task object
        task = task_cls(agent_rate=agent_rate, **kwargs)

        if not isinstance(task, Task):
            raise RuntimeError("The task is not compatible with the runtime")

        # Wrap the task with the runtime
        super().__init__(task=task, agent_rate=agent_rate)

        # Initialise spaces
        self.action_space, self.observation_space = self.task.create_spaces()
        # Store the spaces also in the task
        self.task.action_space = self.action_space
        self.task.observation_space = self.observation_space

        # Seed the environment
        self.seed()

        # Other parameters
        self._manual_stepping = manual_stepping
        if manual_stepping:
            print(
                "Safety feature for manual stepping is enabled. 'Enter' must be pressed to perform each step."
            )
        print("Press 'ESC' to terminate.")
        print(
            "Press 'd' once episode is done (either success of failure). Success rate must be logged manually."
        )

        # Initialise flags that are set manually
        self._manual_done = False
        self._manual_terminate = False

        # Register keyboard listener to manually trigger events (task is done, etc...)
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        # Initialise start time
        self.running_time = 0.0
        self.start_time = time.time()

    # =================
    # Runtime interface
    # =================

    def timestamp(self, running_time: bool = True) -> float:

        if running_time:
            return self.running_time
        else:
            return time.time() - self.start_time

    # =================
    # gym.Env interface
    # =================

    def step(self, action: Action) -> State:

        # Wait until Enter is pressed (if manual stepping is enabled)
        if self._manual_stepping:
            input("Press a key to continue...")

        # Time the execution of the step to determine running time
        pre_step_time = time.time()

        # Terminate if desired
        if self._manual_terminate:
            print("Terminating...")
            sys.exit()

        # Set the action
        if not self._manual_done:
            if not self.action_space.contains(action):
                logger.warn("The action does not belong to the action space")

            print("Performing action...")
            self.task.set_action(action)

            if isinstance(self.task, Manipulation):
                print("Waiting until the action is executed...")
                self.task.wait_until_action_executed()

        # Get the observation
        observation = self.task.get_observation()
        assert isinstance(observation, np.ndarray)
        if not self.observation_space.contains(observation):
            logger.warn("The observation does not belong to the observation space")

        # Get the reward
        # Note: Automatic reward function in real world is not yet implemented
        # reward = self.task.get_reward()
        # assert isinstance(reward, float), "Failed to get the reward"
        reward = 0.0

        # Check termination
        # Note: Automatic done checking is not yet implemented for real world
        # done = self.task.is_done()
        done = self._manual_done

        # Get info
        # Note: There is currently no info relevant for real world use
        # info = self.task.get_info()
        info = {}

        # Update running time
        self.running_time += time.time() - pre_step_time

        # Return state
        return State((Observation(observation), Reward(reward), Done(done), Info(info)))

    def reset(self) -> Observation:

        # Instruct the operator to reset the task
        input(
            "Episode done, please reset the workspace for a new episode. Once the workspace is reset, press any key."
        )

        print(
            "After 5 seconds, the robot will move to its initial joint configuration. Be ready..."
        )
        time.sleep(5.0)

        # Move to initial joint configuration
        if isinstance(self.task, Manipulation):
            print("Moving to the initial joint configuration...")
            self.task.move_to_initial_joint_configuration()

        input("Press any key to confirm that robot and workspace are reset...")

        # Reset internals
        self._manual_done = False

        # Reset the task
        self.task.reset_task()

        # Get the observation
        observation = self.task.get_observation()
        assert isinstance(observation, np.ndarray)
        if not self.observation_space.contains(observation):
            logger.warn("The observation does not belong to the observation space")

        return Observation(observation)

    def seed(self, seed: Optional[int] = None) -> SeedList:

        # Seed the task
        seed = self.task.seed_task(seed)
        return seed

    def render(self, mode: str = "human"):

        pass

    def close(self):

        pass

    def on_press(self, key: keyboard.KeyCode):

        print("")
        if keyboard.KeyCode.from_char("d") == key:
            print(
                "'d' pressed: This episode is now considered to be finished. Please log whether it was success or failure."
            )
            self._manual_done = True
        elif keyboard.Key.esc == key:
            print("'ESC' pressed: Termination signal received...")
            self._manual_terminate = True
