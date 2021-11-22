from gym_ignition.scenario.model_wrapper import ModelWrapper
from .ground import Ground
from .lunar_surface import LunarSurface
from .random_ground import RandomGround


def get_terrain_model_class(terrain_type: str) -> ModelWrapper:

    if "flat" == terrain_type:
        return Ground
    elif "random_flat" == terrain_type:
        return RandomGround
    elif "lunar" == terrain_type:
        return LunarSurface


def is_terrain_type_randomizable(terrain_type: str) -> bool:

    if "random_flat" == terrain_type:
        return True
    return False
