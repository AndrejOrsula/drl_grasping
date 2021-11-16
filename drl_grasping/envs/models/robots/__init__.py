from gym_ignition.scenario.model_wrapper import ModelWrapper
from .kinova_j2s7s300 import KinovaJ2s7s300
from .lunalab_summit_xl_gen import LunalabSummitXlGen
from .panda import Panda
from .ur5_rg2 import UR5RG2

# TODO: Update all robots according to the new template used for LunalabSummitXlGen


def get_robot_model_class(robot_model: str) -> ModelWrapper:

    if "panda" == robot_model:
        return Panda
    elif "ur5_rg2" == robot_model:
        return UR5RG2
    elif "kinova_j2s7s300" == robot_model:
        return KinovaJ2s7s300
    elif "lunalab_summit_xl_gen" == robot_model:
        return LunalabSummitXlGen
