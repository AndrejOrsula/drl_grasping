from gym_ignition.scenario.model_wrapper import ModelWrapper
from .primitives import Box, Cylinder, Sphere, Plane
from .random_object import RandomObject
from .random_primitive import RandomPrimitive


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


def is_object_type_randomizable(object_type: str) -> ModelWrapper:

    if object_type in ("random_primitive", "random_mesh"):
        return True
    return False
