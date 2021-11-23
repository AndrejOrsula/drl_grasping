from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List, Tuple
import numpy as np


class Sun(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "sun",
        direction: Tuple[float, float, float] = (0.5, -0.25, -0.75),
        color: List[float] = (1.0, 1.0, 1.0, 1.0),
        distance: float = 800.0,
        visual: bool = True,
        radius: float = 20.0,
        specular: float = 1.0,
        attenuation_range: float = 10000.0,
        attenuation_constant: float = 0.9,
        attenuation_linear: float = 0.01,
        attenuation_quadratic: float = 0.001,
        **kwargs,
    ):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Normalize direction
        direction = np.array(direction)
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
            color=color,
            visual=visual,
            radius=radius,
            specular=specular,
            attenuation_range=attenuation_range,
            attenuation_constant=attenuation_constant,
            attenuation_linear=attenuation_linear,
            attenuation_quadratic=attenuation_quadratic,
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
        color: Tuple[float, float, float, float],
        visual: bool,
        radius: float,
        specular: float,
        attenuation_range: float,
        attenuation_constant: float,
        attenuation_linear: float,
        attenuation_quadratic: float,
    ) -> str:

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
                            <diffuse>{color[0]} {color[1]} {color[2]} 1</diffuse>
                            <specular>{specular*color[0]} {specular*color[1]} {specular*color[2]} 1</specular>
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
                                <emissive>{color[0]} {color[1]} {color[2]} 1</emissive>
                            </material>
                            <cast_shadows>false</cast_shadows>
                        </visual>
                        """ if visual else ""
                        }
                    </link>
                </model>
            </sdf>'''
