from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from numpy.random import RandomState
from scenario import core as scenario
from typing import Tuple, Optional
import numpy as np


class RandomSun(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "sun",
        direction_minmax_elevation: Tuple[float, float] = (-0.15, -0.65),
        distance: float = 800.0,
        visual: bool = True,
        radius: float = 20.0,
        color_minmax_r: Tuple[float, float] = (0.95, 1.0),
        color_minmax_g: Tuple[float, float] = (0.95, 1.0),
        color_minmax_b: Tuple[float, float] = (0.95, 1.0),
        specular: float = 1.0,
        attenuation_minmax_range: Tuple[float, float] = (750.0, 15000.0),
        attenuation_minmax_constant: Tuple[float, float] = (0.5, 1.0),
        attenuation_minmax_linear: Tuple[float, float] = (0.001, 0.1),
        attenuation_minmax_quadratic: Tuple[float, float] = (0.0001, 0.01),
        np_random: Optional[RandomState] = None,
        **kwargs,
    ):

        if np_random is None:
            np_random = np.random.default_rng()

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Get random yaw direction
        direction = np_random.uniform(-1.0, 1.0, (2,))
        # Normalize yaw direction
        direction = direction / np.linalg.norm(direction)

        # Get random elevation
        direction = np.append(
            direction,
            np_random.uniform(
                direction_minmax_elevation[0], direction_minmax_elevation[1]
            ),
        )
        # Normalize again
        direction = direction / np.linalg.norm(direction)

        # Initial pose
        initial_pose = scenario.Pose(
            (
                -direction[0] * distance,
                -direction[1] * distance,
                -direction[2] * distance,
            ),
            (1, 0, 0, 0),
        )

        # Create SDF string for the model
        sdf = self.get_sdf(
            model_name=model_name,
            direction=direction,
            visual=visual,
            radius=radius,
            color_minmax_r=color_minmax_r,
            color_minmax_g=color_minmax_g,
            color_minmax_b=color_minmax_b,
            attenuation_minmax_range=attenuation_minmax_range,
            attenuation_minmax_constant=attenuation_minmax_constant,
            attenuation_minmax_linear=attenuation_minmax_linear,
            attenuation_minmax_quadratic=attenuation_minmax_quadratic,
            specular=specular,
            np_random=np_random,
        )

        # Insert the model
        ok_model = world.to_gazebo().insert_model_from_string(
            sdf, initial_pose, model_name
        )
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)

    @classmethod
    def get_sdf(
        self,
        model_name: str,
        direction: Tuple[float, float, float],
        visual: bool,
        radius: float,
        color_minmax_r: Tuple[float, float],
        color_minmax_g: Tuple[float, float],
        color_minmax_b: Tuple[float, float],
        attenuation_minmax_range: Tuple[float, float],
        attenuation_minmax_constant: Tuple[float, float],
        attenuation_minmax_linear: Tuple[float, float],
        attenuation_minmax_quadratic: Tuple[float, float],
        specular: float,
        np_random: RandomState,
    ) -> str:

        # Sample random values for parameters
        color_r = np_random.uniform(color_minmax_r[0], color_minmax_r[1])
        color_g = np_random.uniform(color_minmax_g[0], color_minmax_g[1])
        color_b = np_random.uniform(color_minmax_b[0], color_minmax_b[1])
        attenuation_range = np_random.uniform(
            attenuation_minmax_range[0], attenuation_minmax_range[1]
        )
        attenuation_constant = np_random.uniform(
            attenuation_minmax_constant[0], attenuation_minmax_constant[1]
        )
        attenuation_linear = np_random.uniform(
            attenuation_minmax_linear[0], attenuation_minmax_linear[1]
        )
        attenuation_quadratic = np_random.uniform(
            attenuation_minmax_quadratic[0], attenuation_minmax_quadratic[1]
        )

        return f'''<sdf version="1.9">
                <model name="{model_name}">
                    <static>true</static>
                    <link name="{model_name}_link">
                        <light type="directional" name="{model_name}_light">
                            <direction>{direction[0]} {direction[1]} {direction[2]}</direction>
                            <attenuation>
                                <range>{attenuation_range}</range>
                                <constant>{attenuation_constant}</constant>
                                <linear>{attenuation_linear}</linear>
                                <quadratic>{attenuation_quadratic}</quadratic>
                            </attenuation>
                            <diffuse>{color_r} {color_g} {color_b} 1</diffuse>
                            <specular>{specular*color_r} {specular*color_g} {specular*color_b} 1</specular>
                            <cast_shadows>true</cast_shadows>
                        </light>
                        {
                        f"""
                        <visual name="{model_name}_visual">
                            <geometry>
                                <sphere>
                                    <radius>{radius}</radius>
                                </sphere>
                            </geometry>
                            <material>
                                <emissive>{color_r} {color_g} {color_b} 1</emissive>
                            </material>
                            <cast_shadows>false</cast_shadows>
                        </visual>
                        """ if visual else ""
                        }
                    </link>
                </model>
            </sdf>'''
