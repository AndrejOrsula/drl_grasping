from drl_grasping.envs.utils.model_collection_randomizer import (
    ModelCollectionRandomizer,
)
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List


class RandomObject(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "object",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_paths: str = None,
        owner: str = "GoogleResearch",
        collection: str = "Google Scanned Objects",
        server: str = "https://fuel.ignitionrobotics.org",
        server_version: str = "1.0",
        unique_cache: bool = False,
        reset_collection: bool = False,
        np_random=None,
    ):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        model_collection_randomizer = ModelCollectionRandomizer(
            model_paths=model_paths,
            owner=owner,
            collection=collection,
            server=server,
            server_version=server_version,
            unique_cache=unique_cache,
            reset_collection=reset_collection,
            np_random=np_random,
        )

        # Note: using default arguments here
        modified_sdf_file = model_collection_randomizer.random_model()

        # Insert the model
        ok_model = world.to_gazebo().insert_model(
            modified_sdf_file, initial_pose, model_name
        )
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
