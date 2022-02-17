import os
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from numpy.random import RandomState
from scenario import core as scenario

from drl_grasping.envs.models.utils import ModelCollectionRandomizer


class RandomLunarRock(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "rock",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        mass_range: Tuple[float, float] = (0.2, 0.4),
        friction_range: Tuple[float, float] = (5.0, 10.0),
        np_random: Optional[RandomState] = None,
        **kwargs,
    ):

        if np_random is None:
            np_random = np.random.default_rng()

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get path to the model and the important directories
        model_path = ModelCollectionRandomizer.get_collection_paths(
            owner="AndrejOrsula", collection="", model_name="lunar_rock"
        )[0]

        visual_mesh_dir = os.path.join(model_path, "meshes", "visual")
        collision_mesh_dir = os.path.join(model_path, "meshes", "collision")
        texture_dir = os.path.join(model_path, "materials", "textures")

        # Get list of the available meshes
        visual_meshes = os.listdir(visual_mesh_dir)
        # Choose a random mesh from these
        mesh_path_visual = os.path.join(
            visual_mesh_dir, np_random.choice(visual_meshes)
        )
        # Get a matching collision geometry
        mesh_basename = os.path.basename(mesh_path_visual).split(".")[0]
        mesh_path_collision = os.path.join(
            collision_mesh_dir,
            f"{mesh_basename}.stl",
        )
        # Make sure that it exists
        if not os.path.exists(mesh_path_collision):
            raise ValueError(
                f"Collision mesh '{mesh_path_collision}' for RandomLunarSurface model is not a valid file."
            )

        # Find random PBR texture
        albedo_map = None
        normal_map = None
        roughness_map = None
        metalness_map = None
        if texture_dir:
            # Get list of the available textures
            textures = os.listdir(texture_dir)

            # Choose a random texture from these
            random_texture_dir = os.path.join(texture_dir, np_random.choice(textures))

            # List all files
            texture_files = os.listdir(random_texture_dir)

            # Extract the appropriate files
            for texture in texture_files:
                texture_lower = texture.lower()
                if "basecolor" in texture_lower or "albedo" in texture_lower:
                    albedo_map = os.path.join(random_texture_dir, texture)
                elif "normal" in texture_lower:
                    normal_map = os.path.join(random_texture_dir, texture)
                elif "roughness" in texture_lower:
                    roughness_map = os.path.join(random_texture_dir, texture)
                elif "specular" in texture_lower or "metalness" in texture_lower:
                    metalness_map = os.path.join(random_texture_dir, texture)

        material_intensity = np_random.uniform(low=0.2, high=0.5)
        material_intensity_specular = material_intensity / 100000.0

        # Estimate inertial properties (with random mass)
        mesh = trimesh.load(mesh_path_visual, force="mesh", ignore_materials=True)
        mass = np_random.uniform(low=mass_range[0], high=mass_range[1])
        mesh.density = mass / mesh.volume
        inertia = mesh.moment_inertia
        center_mass = mesh.center_mass

        # Randomize friction
        friction = np_random.uniform(low=friction_range[0], high=friction_range[1])

        # Create SDF string for the model
        sdf = f"""<sdf version="1.9">
            <model name="{model_name}">
                <link name="{model_name}_link">
                    <collision name="{model_name}_collision">
                        <geometry>
                            <mesh>
                                <scale>1.0 1.0 1.0</scale>
                                <uri>{mesh_path_collision}</uri>
                            </mesh>
                        </geometry>
                        <surface>
                            <friction>
                                <ode>
                                    <mu>{friction}</mu>
                                    <mu2>{friction}</mu2>
                                    <fdir1>1 0 0</fdir1>
                                    <slip1>0.0</slip1>
                                    <slip2>0.0</slip2>
                                </ode>
                            </friction>
                        </surface>
                    </collision>
                    <visual name="{model_name}_visual">
                        <geometry>
                            <mesh>
                                <scale>1.0 1.0 1.0</scale>
                                <uri>{mesh_path_visual}</uri>
                            </mesh>
                        </geometry>
                        <material>
                            <diffuse>{material_intensity} {material_intensity} {material_intensity} 1</diffuse>
                            <specular>{material_intensity_specular} {material_intensity_specular} {material_intensity_specular} 1</specular>
                            <pbr>
                                <metal>
                                    {"<albedo_map>%s</albedo_map>"
                                        % albedo_map if albedo_map else ""}
                                    {"<normal_map>%s</normal_map>"
                                        % normal_map if normal_map else ""}
                                    {"<roughness_map>%s</roughness_map>"
                                        % roughness_map if roughness_map else ""}
                                    {"<metalness_map>%s</metalness_map>"
                                        % metalness_map if metalness_map else ""}
                                </metal>
                            </pbr>
                        </material>
                    </visual>
                    <inertial>
                        <pose>{center_mass[0]} {center_mass[1]} {center_mass[2]} 0 0 0</pose>
                        <mass>{mass}</mass>
                        <inertia>
                            <ixx>{inertia[0][0]}</ixx>
                            <ixy>{inertia[0][1]}</ixy>
                            <ixz>{inertia[0][2]}</ixz>
                            <iyy>{inertia[1][1]}</iyy>
                            <iyz>{inertia[1][2]}</iyz>
                            <izz>{inertia[2][2]}</izz>
                        </inertia>
                    </inertial>
                </link>
            </model>
        </sdf>"""

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
