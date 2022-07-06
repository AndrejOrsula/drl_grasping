from typing import List

from gym_ignition.scenario import model_with_file, model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo


class LunarSurface(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):
    def __init__(
        self,
        world: scenario.World,
        name: str = "lunar_surface",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_file: str = None,
        use_fuel: bool = False,
        variant: str = "tycho",
        **kwargs,
    ):

        # Allow passing of custom model file as an argument
        if model_file is None:
            model_file = self.get_model_file(fuel=use_fuel, variant=variant)

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Setup initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Insert the model
        ok_model = world.to_gazebo().insert_model_from_file(
            model_file, initial_pose, model_name
        )
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel: bool = False, variant: str = "tycho") -> str:
        if fuel:
            raise NotImplementedError
            return scenario_gazebo.get_model_file_from_fuel(
                f"https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/lunar_surface_{variant}"
            )
        else:
            return f"lunar_surface_{variant}"
