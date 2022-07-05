from gym_ignition.scenario.model_wrapper import ModelWrapper

from .lunalab_summit_xl_gen import LunalabSummitXlGen
from .panda import Panda

# TODO: When adding new a robot, create abstract classes to simplify such process


def get_robot_model_class(robot_model: str) -> ModelWrapper:
    # TODO: Refactor into enum

    if "panda" == robot_model:
        return Panda
    elif "lunalab_summit_xl_gen" == robot_model:
        return LunalabSummitXlGen
