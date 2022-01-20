from gym_ignition.scenario.model_wrapper import ModelWrapper

from .ground import Ground
from .lunar_heightmap import LunarHeightmap
from .lunar_surface import LunarSurface
from .random_ground import RandomGround
from .random_lunar_surface import RandomLunarSurface

# TODO: Change to enum


def get_terrain_model_class(terrain_type: str) -> ModelWrapper:

    if "flat" == terrain_type:
        return Ground
    elif "random_flat" == terrain_type:
        return RandomGround
    elif "lunar_heightmap" == terrain_type:
        return LunarHeightmap
    elif "lunar_surface" == terrain_type:
        return LunarSurface
    elif "random_lunar_surface" == terrain_type:
        return RandomLunarSurface


def is_terrain_type_randomizable(terrain_type: str) -> bool:

    return "random_flat" == terrain_type or "random_lunar_surface" == terrain_type
