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
        distance: float = 1000.0,
        visual: bool = True,
        radius: float = 25.0,
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
    ) -> str:

        return f'''<sdf version="1.9">
                <model name="{model_name}">
                    <static>true</static>
                    <link name="{model_name}_link">
                        <light type="directional" name="{model_name}_light">
                            <direction>{direction[0]} {direction[1]} {direction[2]}</direction>
                            <attenuation>
                                <range>1000</range>
                                <constant>0.9</constant>
                                <linear>0.01</linear>
                                <quadratic>0.001</quadratic>
                            </attenuation>
                            <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
                            <specular>{color[0]} {color[1]} {color[2]} {color[3]}</specular>
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
                                <emissive>{color[0]} {color[1]} {color[2]} {color[3]}</emissive>
                            </material>
                            <cast_shadows>false</cast_shadows>
                        </visual>
                        """ if visual else ""
                        }
                    </link>
                </model>
            </sdf>'''
