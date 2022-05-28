import os
from typing import List, Optional

import numpy as np
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from numpy.random import RandomState
from scenario import core as scenario


class RandomLunarRock(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "rock",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        models_dir: Optional[str] = None,
        np_random: Optional[RandomState] = None,
        **kwargs,
    ):

        if np_random is None:
            np_random = np.random.default_rng()

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get path to all lunar rock models
        if not models_dir:
            models_dir = os.environ.get("SDF_PATH_LUNAR_ROCK", default="")

        # Make sure the path exists
        if not os.path.exists(models_dir):
            raise ValueError(
                f"Invalid path '{models_dir}' pointed by 'SDF_PATH_LUNAR_ROCK' environment variable."
            )

        # Select a single model at random
        model_dir = np_random.choice(os.listdir(models_dir))
        sdf_filepath = os.path.join(model_dir, "model.sdf")

        # Insert the model
        ok_model = world.to_gazebo().insert_model_from_file(
            sdf_filepath, initial_pose, model_name
        )
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
