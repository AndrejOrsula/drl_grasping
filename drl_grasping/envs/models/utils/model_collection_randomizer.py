import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from gym_ignition.utils import logger
from numpy.random import RandomState
from pcg_gazebo.parsers import parse_sdf
from pcg_gazebo.parsers.sdf import create_sdf_element
from scenario import gazebo as scenario_gazebo

# Note: only models with mesh geometry are supported


class ModelCollectionRandomizer:

    _class_model_paths = None
    __sdf_base_name = "model.sdf"
    __configured_sdf_base_name = "model_modified.sdf"
    __blacklisted_base_name = "BLACKLISTED"
    __collision_mesh_dir = "meshes/collision/"
    __collision_mesh_file_type = "stl"
    __original_scale_base_name = "original_scale.txt"

    def __init__(
        self,
        model_paths=None,
        owner="GoogleResearch",
        collection="Google Scanned Objects",
        server="https://fuel.ignitionrobotics.org",
        server_version="1.0",
        unique_cache=False,
        reset_collection=False,
        enable_blacklisting=True,
        np_random: Optional[RandomState] = None,
    ):

        # If enabled, the newly created objects of this class will use its own individual cache
        # for model paths and must discover/download them on its own
        self._unique_cache = unique_cache

        # Flag that determines if models that cannot be used are blacklisted
        self._enable_blacklisting = enable_blacklisting

        # If enabled, the cache of the class used to store model paths among instances will be reset
        if reset_collection and not self._unique_cache:
            self._class_model_paths = None

        # Get file path to all models from
        # a) `model_paths` arg
        # b) local cache owner (if `owner` has some models, i.e `collection` is already downloaded)
        # c) Fuel collection (if `owner` has no models in local cache)
        if model_paths is not None:
            # Use arg
            if self._unique_cache:
                self._model_paths = model_paths
            else:
                self._class_model_paths = model_paths
        else:
            # Use local cache or Fuel
            if self._unique_cache:
                self._model_paths = self.get_collection_paths(
                    owner=owner,
                    collection=collection,
                    server=server,
                    server_version=server_version,
                )
            elif self._class_model_paths is None:
                # Executed only once, unless the paths are reset with `reset_collection` arg
                self._class_model_paths = self.get_collection_paths(
                    owner=owner,
                    collection=collection,
                    server=server,
                    server_version=server_version,
                )

        # Initialise rng with (with seed is desired)
        if np_random is not None:
            self.np_random = np_random
        else:
            self.np_random = np.random.default_rng()

    @classmethod
    def get_collection_paths(
        cls,
        owner="GoogleResearch",
        collection="Google Scanned Objects",
        server="https://fuel.ignitionrobotics.org",
        server_version="1.0",
        model_name: str = "",
    ) -> List[str]:

        # First check the local cache (for performance)
        # Note: This unfortunately does not check if models belong to the specified collection
        # TODO: Make sure models belong to the collection if sampled from local cache
        model_paths = scenario_gazebo.get_local_cache_model_paths(
            owner=owner, name=model_name
        )
        if len(model_paths) > 0:
            return model_paths

        # Else download the models from Fuel and then try again
        if collection:
            download_uri = "%s/%s/%s/collections/%s" % (
                server,
                server_version,
                owner,
                collection,
            )
        elif model_name:
            download_uri = "%s/%s/%s/models/%s" % (
                server,
                server_version,
                owner,
                model_name,
            )
        download_command = 'ign fuel download -v 3 -t model -j %s -u "%s"' % (
            os.cpu_count(),
            download_uri,
        )
        os.system(download_command)

        model_paths = scenario_gazebo.get_local_cache_model_paths(
            owner=owner, name=model_name
        )
        if 0 == len(model_paths):
            logger.error(
                'URI "%s" is not valid and does not contain any models that are \
                          owned by the owner of the collection'
                % download_uri
            )
            pass

        return model_paths

    def random_model(
        self,
        min_scale=0.125,
        max_scale=0.175,
        min_mass=0.05,
        max_mass=0.25,
        min_friction=0.75,
        max_friction=1.5,
        decimation_fraction_of_visual=0.25,
        decimation_min_faces=40,
        decimation_max_faces=200,
        max_faces=40000,
        max_vertices=None,
        component_min_faces_fraction=0.1,
        component_max_volume_fraction=0.35,
        fix_mtl_texture_paths=True,
        skip_blacklisted=True,
        return_sdf_path=True,
    ) -> str:

        # Loop until a model is found, checked for validity, configured and returned
        # If any of these steps fail, sample another model and try again
        # Note: Due to this behaviour, the function could stall if all models are invalid
        # TODO: Add a simple timeout to random sampling of valid model (# of attempts or time-based)
        while True:

            # Get path to a random model from the collection
            model_path = self.get_random_model_path()

            # Check if the model is already blacklisted and skip if desired
            if skip_blacklisted and self.is_blacklisted(model_path):
                continue

            # Check is the model is already configured
            if self.is_configured(model_path):
                # If so, break the loop
                break

            # Process the model and break loop only if it is valid
            if self.process_model(
                model_path,
                decimation_fraction_of_visual=decimation_fraction_of_visual,
                decimation_min_faces=decimation_min_faces,
                decimation_max_faces=decimation_max_faces,
                max_faces=max_faces,
                max_vertices=max_vertices,
                component_min_faces_fraction=component_min_faces_fraction,
                component_max_volume_fraction=component_max_volume_fraction,
                fix_mtl_texture_paths=fix_mtl_texture_paths,
            ):
                break

        # Apply randomization
        self.randomize_configured_model(
            model_path,
            min_scale=min_scale,
            max_scale=max_scale,
            min_friction=min_friction,
            max_friction=max_friction,
            min_mass=min_mass,
            max_mass=max_mass,
        )

        if return_sdf_path:
            # Return path to the configured SDF file
            return self.get_configured_sdf_path(model_path)
        else:
            # Return path to the model directory
            return model_path

    def process_all_models(
        self,
        decimation_fraction_of_visual=0.025,
        decimation_min_faces=8,
        decimation_max_faces=400,
        max_faces=40000,
        max_vertices=None,
        component_min_faces_fraction=0.1,
        component_max_volume_fraction=0.35,
        fix_mtl_texture_paths=True,
    ):
        if self._unique_cache:
            model_paths = self._model_paths
        else:
            model_paths = self._class_model_paths

        blacklist_model_counter = 0
        for i in range(len(model_paths)):
            if not self.process_model(
                model_paths[i],
                decimation_fraction_of_visual=decimation_fraction_of_visual,
                decimation_min_faces=decimation_min_faces,
                decimation_max_faces=decimation_max_faces,
                max_faces=max_faces,
                max_vertices=max_vertices,
                component_min_faces_fraction=component_min_faces_fraction,
                component_max_volume_fraction=component_max_volume_fraction,
                fix_mtl_texture_paths=fix_mtl_texture_paths,
            ):
                blacklist_model_counter += 1

            print('Processed model %i/%i "%s"' % (i, len(model_paths), model_paths[i]))

        print("Number of blacklisted models: %i" % blacklist_model_counter)

    def process_model(
        self,
        model_path,
        decimation_fraction_of_visual=0.25,
        decimation_min_faces=40,
        decimation_max_faces=200,
        max_faces=40000,
        max_vertices=None,
        component_min_faces_fraction=0.1,
        component_max_volume_fraction=0.35,
        fix_mtl_texture_paths=True,
    ) -> bool:

        # Parse the SDF of the model
        sdf = parse_sdf(self.get_sdf_path(model_path))

        # Process the model(s) contained in the SDF
        for model in sdf.models:

            # Process the link(s) of each model
            for link in model.links:

                # Get rid of the existing collisions prior to simplifying it
                link.collisions.clear()

                # Values for the total inertial properties of current link
                # These values will be updated for each body that the link contains
                total_mass = 0.0
                total_inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                common_centre_of_mass = [0.0, 0.0, 0.0]

                # Go through the visuals and process them
                for visual in link.visuals:

                    # Get path to the mesh of the link's visual
                    mesh_path = self.get_mesh_path(model_path, visual)

                    # If desired, fix texture path in 'mtl' files for '.obj' mesh format
                    if fix_mtl_texture_paths:
                        self.fix_mtl_texture_paths(
                            model_path, mesh_path, model.attributes["name"]
                        )

                    # Load the mesh (without materials)
                    mesh = trimesh.load(mesh_path, force="mesh", skip_materials=True)

                    # Check if model has too much geometry (blacklist if needed)
                    if not self.check_excessive_geometry(
                        mesh, model_path, max_faces=max_faces, max_vertices=max_vertices
                    ):
                        return False

                    # Check if model has disconnected geometry/components (blacklist if needed)
                    if not self.check_disconnected_components(
                        mesh,
                        model_path,
                        component_min_faces_fraction=component_min_faces_fraction,
                        component_max_volume_fraction=component_max_volume_fraction,
                    ):
                        return False

                    # Compute inertial properties for this mesh
                    (
                        total_mass,
                        total_inertia,
                        common_centre_of_mass,
                    ) = self.sum_inertial_properties(
                        mesh, total_mass, total_inertia, common_centre_of_mass
                    )

                    # Add decimated collision geometry to the SDF
                    self.add_collision(
                        mesh,
                        link,
                        model_path,
                        fraction_of_visual=decimation_fraction_of_visual,
                        min_faces=decimation_min_faces,
                        max_faces=decimation_max_faces,
                    )

                    # Write original scale (size) into the SDF
                    # This is used for later reference during randomization (for scale limits)
                    self.write_original_scale(mesh, model_path)

                # Make sure the link has valid inertial properties (blacklist if needed)
                if not self.check_inertial_properties(
                    model_path, total_mass, total_inertia
                ):
                    return False

                # Write inertial properties to the SDF of the link
                self.write_inertial_properties(
                    link, total_mass, total_inertia, common_centre_of_mass
                )

        # Write the configured SDF into a file
        sdf.export_xml(self.get_configured_sdf_path(model_path))
        return True

    def add_collision(
        self,
        mesh,
        link,
        model_path,
        fraction_of_visual=0.05,
        min_faces=8,
        max_faces=750,
        friction=1.0,
    ):

        # Determine name of path to the collistion geometry
        collision_name = (
            link.attributes["name"] + "_collision_" + str(len(link.collisions))
        )
        collision_mesh_path = self.get_collision_mesh_path(model_path, collision_name)

        # Determine number of faces to keep after the decimation
        face_count = min(
            max(fraction_of_visual * len(mesh.faces), min_faces), max_faces
        )

        # Simplify mesh via decimation
        collision_mesh = mesh.simplify_quadratic_decimation(face_count)

        # Export the collision mesh to the appropriate location
        os.makedirs(os.path.dirname(collision_mesh_path), exist_ok=True)
        collision_mesh.export(
            collision_mesh_path, file_type=self.__collision_mesh_file_type
        )

        # Create collision SDF element
        collision = create_sdf_element("collision")

        # Add collision geometry to the SDF
        collision.geometry.mesh = create_sdf_element("mesh")
        collision.geometry.mesh.uri = os.path.relpath(
            collision_mesh_path, start=model_path
        )

        # Add surface friction to the SDF of collision (default to 1 and randomize later)
        collision.surface = create_sdf_element("surface")
        collision.surface.friction = create_sdf_element("friction", "surface")
        collision.surface.friction.ode = create_sdf_element("ode", "collision")
        collision.surface.friction.ode.mu = friction
        collision.surface.friction.ode.mu2 = friction

        # Add it to the SDF of the link
        collision_name = os.path.basename(collision_mesh_path).split(".")[0]
        link.add_collision(collision_name, collision)

    def sum_inertial_properties(
        self, mesh, total_mass, total_inertia, common_centre_of_mass, density=1.0
    ) -> Tuple[float, float, float]:

        # Arbitrary density is used here
        # The mass will be randomized once it is fully computed for a link
        mesh.density = density

        # Tmp variable to store the mass of all previous geometry, used to determine centre of mass
        mass_of_others = total_mass

        # For each additional mesh, simply add the mass and inertia
        total_mass += mesh.mass
        total_inertia += mesh.moment_inertia

        # Compute a common centre of mass between all previous geometry and the new mesh
        common_centre_of_mass = [
            mass_of_others * common_centre_of_mass[0] + mesh.mass * mesh.center_mass[0],
            mass_of_others * common_centre_of_mass[1] + mesh.mass * mesh.center_mass[1],
            mass_of_others * common_centre_of_mass[2] + mesh.mass * mesh.center_mass[2],
        ] / total_mass

        return total_mass, total_inertia, common_centre_of_mass

    def randomize_configured_model(
        self,
        model_path,
        min_scale=0.05,
        max_scale=0.25,
        min_mass=0.1,
        max_mass=3.0,
        min_friction=0.75,
        max_friction=1.5,
    ):

        # Get path to the configured SDF file
        configured_sdf_path = self.get_configured_sdf_path(model_path)

        # Parse the configured SDF that needs to be randomized
        sdf = parse_sdf(configured_sdf_path)

        # Process the model(s) contained in the SDF
        for model in sdf.models:

            # Process the link(s) of each model
            for link in model.links:

                # Randomize scale of the link
                self.randomize_scale(
                    model_path, link, min_scale=min_scale, max_scale=max_scale
                )

                # Randomize inertial properties of the link
                self.randomize_inertial(link, min_mass=min_mass, max_mass=max_mass)

                # Randomize friction of the link
                self.randomize_friction(
                    link, min_friction=min_friction, max_friction=max_friction
                )

        # Overwrite the configured SDF file with randomized values
        sdf.export_xml(configured_sdf_path)

    def randomize_scale(self, model_path, link, min_scale=0.05, max_scale=0.25):

        # Note: This function currently supports only scaling of links with single mesh geometry
        if len(link.visuals) > 1:
            return False

        # Get a random scale for the size of mesh
        random_scale = self.np_random.uniform(min_scale, max_scale)

        # Determine a scale factor that will result in such scale for the size of mesh
        original_mesh_scale = self.read_original_scale(model_path)
        scale_factor = random_scale / original_mesh_scale

        # Determine scale factor for inertial properties based on random scale and current scale
        current_scale = link.visuals[0].geometry.mesh.scale.value[0]
        inertial_scale_factor = scale_factor / current_scale

        # Write scale factor into SDF for visual and collision geometry
        link.visuals[0].geometry.mesh.scale = [scale_factor] * 3
        link.collisions[0].geometry.mesh.scale = [scale_factor] * 3

        # Recompute inertial properties acording to the scale
        link.inertial.pose.x *= inertial_scale_factor
        link.inertial.pose.y *= inertial_scale_factor
        link.inertial.pose.z *= inertial_scale_factor

        # Mass is scaled n^3
        link.mass = link.mass.value * inertial_scale_factor ** 3

        # Inertia is scaled n^5
        inertial_scale_factor_n5 = inertial_scale_factor ** 5
        link.inertia.ixx = link.inertia.ixx.value * inertial_scale_factor_n5
        link.inertia.iyy = link.inertia.iyy.value * inertial_scale_factor_n5
        link.inertia.izz = link.inertia.izz.value * inertial_scale_factor_n5
        link.inertia.ixy = link.inertia.ixy.value * inertial_scale_factor_n5
        link.inertia.ixz = link.inertia.ixz.value * inertial_scale_factor_n5
        link.inertia.iyz = link.inertia.iyz.value * inertial_scale_factor_n5

    def randomize_inertial(
        self, link, min_mass=0.1, max_mass=3.0
    ) -> Tuple[float, float]:

        random_mass = self.np_random.uniform(min_mass, max_mass)
        mass_scale_factor = random_mass / link.mass.value

        link.mass = random_mass
        link.inertia.ixx = link.inertia.ixx.value * mass_scale_factor
        link.inertia.iyy = link.inertia.iyy.value * mass_scale_factor
        link.inertia.izz = link.inertia.izz.value * mass_scale_factor
        link.inertia.ixy = link.inertia.ixy.value * mass_scale_factor
        link.inertia.ixz = link.inertia.ixz.value * mass_scale_factor
        link.inertia.iyz = link.inertia.iyz.value * mass_scale_factor

    def randomize_friction(self, link, min_friction=0.75, max_friction=1.5):

        for collision in link.collisions:
            random_friction = self.np_random.uniform(min_friction, max_friction)

            collision.surface.friction.ode.mu = random_friction
            collision.surface.friction.ode.mu2 = random_friction

    def write_inertial_properties(self, link, mass, inertia, centre_of_mass):

        link.mass = mass

        link.inertia.ixx = inertia[0][0]
        link.inertia.iyy = inertia[1][1]
        link.inertia.izz = inertia[2][2]
        link.inertia.ixy = inertia[0][1]
        link.inertia.ixz = inertia[0][2]
        link.inertia.iyz = inertia[1][2]

        link.inertial.pose = [
            centre_of_mass[0],
            centre_of_mass[1],
            centre_of_mass[2],
            0.0,
            0.0,
            0.0,
        ]

    def write_original_scale(self, mesh, model_path):

        file = open(self.get_original_scale_path(model_path), "w")
        file.write(str(mesh.scale))
        file.close()

    def read_original_scale(self, model_path) -> float:

        file = open(self.get_original_scale_path(model_path), "r")
        original_scale = file.read()
        file.close()

        return float(original_scale)

    def check_excessive_geometry(
        self, mesh, model_path, max_faces=40000, max_vertices=None
    ) -> bool:

        if max_faces is not None:
            num_faces = len(mesh.faces)
            if num_faces > max_faces:
                self.blacklist_model(
                    model_path, reason="Excessive geometry (%d faces)" % num_faces
                )
                return False

        if max_vertices is not None:
            num_vertices = len(mesh.vertices)
            if num_vertices > max_vertices:
                self.blacklist_model(
                    model_path, reason="Excessive geometry (%d vertices)" % num_vertices
                )
                return False

        return True

    def check_disconnected_components(
        self,
        mesh,
        model_path,
        component_min_faces_fraction=0.05,
        component_max_volume_fraction=0.1,
    ) -> bool:

        # Get a list of all connected componends inside the mesh
        # Consider components only with `component_min_faces_fraction` percent faces
        min_faces = round(component_min_faces_fraction * len(mesh.faces))
        connected_components = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=min_faces
        )

        # If more than 1 objects were detected, consider also relative volume of the meshes
        if len(connected_components) > 1:
            total_volume = mesh.volume

            large_component_counter = 0
            for component in connected_components:
                submesh = mesh.copy()
                mask = np.zeros(len(mesh.faces), dtype=np.bool)
                mask[component] = True
                submesh.update_faces(mask)

                volume_fraction = submesh.volume / total_volume
                if volume_fraction > component_max_volume_fraction:
                    large_component_counter += 1

                if large_component_counter > 1:
                    self.blacklist_model(
                        model_path,
                        reason="Disconnected components (%d instances)"
                        % len(connected_components),
                    )
                    return False

        return True

    def check_inertial_properties(self, model_path, mass, inertia) -> bool:

        if (
            mass < 1e-10
            or inertia[0][0] < 1e-10
            or inertia[1][1] < 1e-10
            or inertia[2][2] < 1e-10
        ):
            self.blacklist_model(model_path, reason="Invalid inertial properties")
            return False

        return True

    def get_random_model_path(self) -> str:

        if self._unique_cache:
            return self.np_random.choice(self._model_paths)
        else:
            return self.np_random.choice(self._class_model_paths)

    def get_collision_mesh_path(self, model_path, collision_name) -> str:

        return os.path.join(
            model_path,
            self.__collision_mesh_dir,
            collision_name + "." + self.__collision_mesh_file_type,
        )

    def get_sdf_path(self, model_path) -> str:

        return os.path.join(model_path, self.__sdf_base_name)

    def get_configured_sdf_path(self, model_path) -> str:

        return os.path.join(model_path, self.__configured_sdf_base_name)

    def get_blacklisted_path(self, model_path) -> str:

        return os.path.join(model_path, self.__blacklisted_base_name)

    def get_mesh_path(self, model_path, visual_or_collision) -> str:

        # TODO: This might need fixing for certain collections/models
        mesh_uri = visual_or_collision.geometry.mesh.uri.value
        return os.path.join(model_path, mesh_uri)

    def get_original_scale_path(self, model_path) -> str:

        return os.path.join(model_path, self.__original_scale_base_name)

    def blacklist_model(self, model_path, reason="Unknown"):

        if self._enable_blacklisting:
            bl_file = open(self.get_blacklisted_path(model_path), "w")
            bl_file.write(reason)
            bl_file.close()
        logger.warn(
            '%s model "%s". Reason: %s.'
            % (
                "Blacklisting" if self._enable_blacklisting else "Skipping",
                model_path,
                reason,
            )
        )

    def is_blacklisted(self, model_path) -> bool:

        return os.path.isfile(self.get_blacklisted_path(model_path))

    def is_configured(self, model_path) -> bool:

        return os.path.isfile(self.get_configured_sdf_path(model_path))

    def fix_mtl_texture_paths(self, model_path, mesh_path, model_name):

        # The `.obj` files use mtl
        if mesh_path.endswith(".obj"):

            # Find all textures located in the model path, used later to relative linking
            texture_files = glob.glob(os.path.join(model_path, "**", "textures", "*.*"))

            # Find location ot mtl file, if any
            mtllib_file = None
            with open(mesh_path, "r") as file:
                for line in file:
                    if "mtllib" in line:
                        mtllib_file = line.split(" ")[-1].strip()
                        break

            if mtllib_file is not None:
                mtllib_file = os.path.join(os.path.dirname(mesh_path), mtllib_file)

                fin = open(mtllib_file, "r")
                data = fin.read()
                for line in data.splitlines():
                    if "map_" in line:
                        # Find the name of the texture/map in the mtl
                        map_file = line.split(" ")[-1].strip()

                        # Find the first match of the texture/map file
                        for texture_file in texture_files:
                            if os.path.basename(
                                texture_file
                            ) == map_file or os.path.basename(
                                texture_file
                            ) == os.path.basename(
                                map_file
                            ):

                                # Make the file unique to the model (unless it already is)
                                if model_name in texture_file:
                                    new_texture_file_name = texture_file
                                else:
                                    new_texture_file_name = texture_file.replace(
                                        map_file, model_name + "_" + map_file
                                    )

                                os.rename(texture_file, new_texture_file_name)

                                # Apply the correct relative path
                                data = data.replace(
                                    map_file,
                                    os.path.relpath(
                                        new_texture_file_name,
                                        start=os.path.dirname(mesh_path),
                                    ),
                                )
                                break
                fin.close()

                # Write in the correct data
                fout = open(mtllib_file, "w")
                fout.write(data)
                fout.close()
