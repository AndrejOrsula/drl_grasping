from gym_ignition.scenario.model_wrapper import ModelWrapper
from .random_sun import RandomSun
from .sun import Sun


def get_light_model_class(light_type: str) -> ModelWrapper:

    if "sun" == light_type:
        return Sun
    elif "random_sun" == light_type:
        return RandomSun


def is_light_type_randomizable(light_type: str) -> bool:

    if "random_sun" == light_type:
        return True
    return False
