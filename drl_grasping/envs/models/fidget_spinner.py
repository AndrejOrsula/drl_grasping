from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List


class FidgetSpinner(model_wrapper.ModelWrapper,
                    model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'fidget_spinner',
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_file: str = None,
                 use_fuel: bool = True):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(fuel=use_fuel)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=True) -> str:
        if fuel:
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/fidget_spinner")
        else:
            return "fidget_spinner"
