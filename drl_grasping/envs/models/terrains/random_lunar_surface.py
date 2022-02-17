import os
from typing import List, Optional, Tuple

import numpy as np
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from numpy.random import RandomState
from scenario import core as scenario

from drl_grasping.envs.models.utils.model_collection_randomizer import (
    ModelCollectionRandomizer,
)


class RandomLunarSurface(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: str = "lunar_surface",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        friction_range: Tuple[float, float] = (7.5, 15.0),
        include_pebbles: bool = True,
        np_random: Optional[RandomState] = None,
        **kwargs,
    ):

        if np_random is None:
            np_random = np.random.default_rng()

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Setup initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get path to the model and the important directories
        model_path = ModelCollectionRandomizer.get_collection_paths(
            owner="AndrejOrsula", collection="", model_name="lunar_surface"
        )[0]
        visual_mesh_dir = os.path.join(model_path, "meshes", "visual")
        collision_mesh_dir = os.path.join(model_path, "meshes", "collision")
        pebbles_visual_mesh_dir = os.path.join(model_path, "meshes", "pebbles")
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

        # Get also mesh for pebbles (if desired)
        if include_pebbles:
            mesh_path_pebbles = os.path.join(
                pebbles_visual_mesh_dir,
                f"{mesh_basename}.dae",
            )
            # Make sure that it exists
            if not os.path.exists(mesh_path_pebbles):
                raise ValueError(
                    f"Pebbles mesh '{mesh_path_pebbles}' for RandomLunarSurface model is not a valid file."
                )
            pebbles_material_intensity = np_random.uniform(low=0.05, high=0.3)
            pebbles_material_intensity_specular = pebbles_material_intensity / 10000.0

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
                if "color" in texture_lower or "albedo" in texture_lower:
                    albedo_map = os.path.join(random_texture_dir, texture)
                elif "normal" in texture_lower:
                    normal_map = os.path.join(random_texture_dir, texture)
                elif "roughness" in texture_lower:
                    roughness_map = os.path.join(random_texture_dir, texture)
                elif "specular" in texture_lower or "metalness" in texture_lower:
                    metalness_map = os.path.join(random_texture_dir, texture)

        material_intensity = np_random.uniform(low=0.4, high=1.0)
        material_intensity_specular = material_intensity / 100000.0

        # Randomize friction
        friction = np_random.uniform(low=friction_range[0], high=friction_range[1])

        # Create SDF string for the model
        sdf = f"""<sdf version="1.9">
            <model name="{model_name}">
                <static>true</static>
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
                    {
                    f'''<visual name="{model_name}_pebbles">
                        <geometry>
                            <mesh>
                                <scale>1.0 1.0 1.0</scale>
                                <uri>{mesh_path_pebbles}</uri>
                            </mesh>
                        </geometry>
                        <material>
                            <diffuse>{pebbles_material_intensity} {pebbles_material_intensity} {pebbles_material_intensity} 1</diffuse>
                            <specular>{pebbles_material_intensity_specular} {pebbles_material_intensity_specular} {pebbles_material_intensity_specular} 1</specular>
                        </material>
                    </visual>''' if include_pebbles else ""
                    }
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
