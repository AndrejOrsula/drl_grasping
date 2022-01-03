from gym_ignition.scenario.model_wrapper import ModelWrapper

from .primitives import Box, Cylinder, Plane, Sphere
from .random_object import RandomObject
from .random_primitive import RandomPrimitive
from .rock import Rock


def get_object_model_class(object_type: str) -> ModelWrapper:

    if "box" == object_type:
        return Box
    elif "sphere" == object_type:
        return Sphere
    elif "cylinder" == object_type:
        return Cylinder
    elif "random_primitive" == object_type:
        return RandomPrimitive
    elif "random_mesh" == object_type:
        return RandomObject
    elif "rock" == object_type:
        return Rock


def is_object_type_randomizable(object_type: str) -> bool:

    if "random_primitive" == object_type or "random_mesh" == object_type:
        return True
    return False
