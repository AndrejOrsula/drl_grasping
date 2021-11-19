from gym_ignition.scenario import model_wrapper
from gym_ignition.utils import misc
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List


class Box(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "box",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        size: List[float] = (0.05, 0.05, 0.05),
        mass: float = 0.1,
        static: bool = False,
        collision: bool = True,
        friction: float = 1.0,
        visual: bool = True,
        gui_only: bool = False,
        color: List[float] = (0.8, 0.8, 0.8, 1.0),
        **kwargs,
    ):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Create SDF string for the model
        sdf = self.get_sdf(
            model_name=model_name,
            size=size,
            mass=mass,
            static=static,
            collision=collision,
            friction=friction,
            visual=visual,
            gui_only=gui_only,
            color=color,
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
        size: List[float],
        mass: float,
        static: bool,
        collision: bool,
        friction: float,
        visual: bool,
        gui_only: bool,
        color: List[float],
    ) -> str:
        return f'''<sdf version="1.7">
                <model name="{model_name}">
                    <static>{"true" if static else "false"}</static>
                    <link name="{model_name}_link">
                        {
                        f"""
                        <collision name="{model_name}_collision">
                            <geometry>
                                <box>
                                    <size>{size[0]} {size[1]} {size[2]}</size>
                                </box>
                            </geometry>
                            <surface>
                                <friction>
                                    <ode>
                                        <mu>{friction}</mu>
                                        <mu2>{friction}</mu2>
                                        <fdir1>0 0 0</fdir1>
                                        <slip1>0.0</slip1>
                                        <slip2>0.0</slip2>
                                    </ode>
                                </friction>
                            </surface>
                        </collision>
                        """ if collision else ""
                        }
                        {
                        f"""
                        <visual name="{model_name}_visual">
                            <geometry>
                                <box>
                                    <size>{size[0]} {size[1]} {size[2]}</size>
                                </box>
                            </geometry>
                            <material>
                                <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
                                <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
                                <specular>{color[0]} {color[1]} {color[2]} {color[3]}</specular>
                            </material>
                            <transparency>{1.0-color[3]}</transparency>
                            {'<visibility_flags>1</visibility_flags> <cast_shadows>false</cast_shadows>' if gui_only else ''}
                        </visual>
                        """ if visual else ""
                        }
                        <inertial>
                            <mass>{mass}</mass>
                            <inertia>
                                <ixx>{(size[1]**2 + size[2]**2)*mass/12}</ixx>
                                <iyy>{(size[0]**2 + size[2]**2)*mass/12}</iyy>
                                <izz>{(size[0]**2 + size[1]**2)*mass/12}</izz>
                                <ixy>0.0</ixy>
                                <ixz>0.0</ixz>
                                <iyz>0.0</iyz>
                            </inertia>
                        </inertial>
                    </link>
                </model>
            </sdf>'''
