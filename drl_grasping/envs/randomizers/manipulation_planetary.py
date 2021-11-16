from drl_grasping.envs import tasks, models
from drl_grasping.envs.utils import Tf2Broadcaster
from drl_grasping.envs.utils.conversions import quat_to_xyzw, quat_to_wxyz
from drl_grasping.envs.utils.gazebo import get_model_pose
from drl_grasping.envs.utils.math import quat_mul
from drl_grasping.envs.randomizers import ManipulationGazeboEnvRandomizer
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from os import environ
from scenario import gazebo as scenario
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from typing import Union, Tuple
import abc
import numpy as np
import operator

# Tasks that are supported by this randomizer (used primarily for type hinting)
SupportedTasks = Union[tasks.GraspPlanetary, tasks.GraspPlanetaryOctree]


class ManipulationPlanetaryGazeboEnvRandomizer(
    ManipulationGazeboEnvRandomizer,
    abc.ABC,
):
    """
    Randomizer for robotic manipulation environments that focuses on planetary tasks.
    """

    def __init__(
        self,
        env: MakeEnvCallable,
        **kwargs,
    ):

        # Initialize base class
        ManipulationGazeboEnvRandomizer.__init__(self, env=env, **kwargs)
