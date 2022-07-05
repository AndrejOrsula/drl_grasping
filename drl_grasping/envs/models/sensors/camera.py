import os
from threading import Thread
from typing import List, Optional, Union

from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario

from drl_grasping.envs.models.utils import ModelCollectionRandomizer


class Camera(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: Union[str, None] = None,
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        static: bool = True,
        camera_type: str = "rgbd_camera",
        width: int = 212,
        height: int = 120,
        image_format: str = "R8G8B8",
        update_rate: int = 15,
        horizontal_fov: float = 1.567821,
        vertical_fov: float = 1.022238,
        clip_color: List[float] = (0.02, 1000.0),
        clip_depth: List[float] = (0.02, 10.0),
        noise_mean: float = None,
        noise_stddev: float = None,
        ros2_bridge_color: bool = False,
        ros2_bridge_depth: bool = False,
        ros2_bridge_points: bool = False,
        visibility_mask: int = 0,
        visual: Optional[str] = None,
        # visual: Optional[str] = "intel_realsense_d435",
    ):

        # Get a unique model name
        if name is not None:
            model_name = get_unique_model_name(world, name)
        else:
            model_name = get_unique_model_name(world, camera_type)
        self._model_name = model_name

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get resources for visual (if enabled)
        if visual:
            use_mesh: bool = False
            if "intel_realsense_d435" == visual:
                use_mesh = True

                # Get path to the model and the important directories
                model_path = ModelCollectionRandomizer.get_collection_paths(
                    owner="OpenRobotics",
                    collection="",
                    model_name="Intel RealSense D435",
                )[0]

                mesh_dir = os.path.join(model_path, "meshes")
                texture_dir = os.path.join(model_path, "materials", "textures")

                # Get path to the mesh
                mesh_path_visual = os.path.join(mesh_dir, "realsense.dae")
                # Make sure that it exists
                if not os.path.exists(mesh_path_visual):
                    raise ValueError(
                        f"Visual mesh '{mesh_path_visual}' for Camera model is not a valid file."
                    )

                # Find PBR textures
                albedo_map = None
                normal_map = None
                roughness_map = None
                metalness_map = None
                if texture_dir:
                    # List all files
                    texture_files = os.listdir(texture_dir)

                    # Extract the appropriate files
                    for texture in texture_files:
                        texture_lower = texture.lower()
                        if "basecolor" in texture_lower or "albedo" in texture_lower:
                            albedo_map = os.path.join(texture_dir, texture)
                        elif "normal" in texture_lower:
                            normal_map = os.path.join(texture_dir, texture)
                        elif "roughness" in texture_lower:
                            roughness_map = os.path.join(texture_dir, texture)
                        elif (
                            "specular" in texture_lower or "metalness" in texture_lower
                        ):
                            metalness_map = os.path.join(texture_dir, texture)

                if not (albedo_map and normal_map and roughness_map and metalness_map):
                    raise ValueError(f"Not all textures for Camera model were found.")

        # Create SDF string for the model
        sdf = f'''<sdf version="1.9">
            <model name="{model_name}">
                <static>{static}</static>
                <link name="{self.link_name}">
                    <sensor name="camera" type="{camera_type}">
                        <topic>{model_name}</topic>
                        <always_on>true</always_on>
                        <update_rate>{update_rate}</update_rate>
                        <camera name="{model_name}_camera">
                            <image>
                                <width>{width}</width>
                                <height>{height}</height>
                                <format>{image_format}</format>
                            </image>
                            <horizontal_fov>{horizontal_fov}</horizontal_fov>
                            <vertical_fov>{vertical_fov}</vertical_fov>
                            <clip>
                                <near>{clip_color[0]}</near>
                                <far>{clip_color[1]}</far>
                            </clip>
                            {
                            f"""<depth_camera>
                                <clip>
                                    <near>{clip_depth[0]}</near>
                                    <far>{clip_depth[1]}</far>
                                </clip>
                            </depth_camera>""" if "rgbd" in model_name else ""
                            }
                            {
                            f"""<noise>
                                <type>gaussian</type>
                                <mean>{noise_mean}</mean>
                                <stddev>{noise_stddev}</stddev>
                            </noise>""" if noise_mean is not None and noise_stddev is not None else ""
                            }
                            <visibility_mask>{visibility_mask}</visibility_mask>
                        </camera>
                        <visualize>true</visualize>
                    </sensor>
                    {
                        f"""
                        <visual name="{model_name}_visual_lens">
                            <pose>-0.01 0 0 0 1.5707963 0</pose>
                            <geometry>
                                <cylinder>
                                    <radius>0.02</radius>
                                    <length>0.02</length>
                                </cylinder>
                            </geometry>
                            <material>
                                <ambient>0.0 0.8 0.0</ambient>
                                <diffuse>0.0 0.8 0.0</diffuse>
                                <specular>0.0 0.8 0.0</specular>
                            </material>
                        </visual>
                        <visual name="{model_name}_visual_body">
                            <pose>-0.05 0 0 0 0 0</pose>
                            <geometry>
                                <box>
                                    <size>0.06 0.05 0.05</size>
                                </box>
                            </geometry>
                            <material>
                                <ambient>0.0 0.8 0.0</ambient>
                                <diffuse>0.0 0.8 0.0</diffuse>
                                <specular>0.0 0.8 0.0</specular>
                            </material>
                        </visual>
                        """ if visual and not use_mesh else ""
                        }
                        {
                        f"""
                        <inertial>
                            <mass>0.0615752</mass>
                            <inertia>
                                <ixx>9.108e-05</ixx>
                                <ixy>0.0</ixy>
                                <ixz>0.0</ixz>
                                <iyy>2.51e-06</iyy>
                                <iyz>0.0</iyz>
                                <izz>8.931e-05</izz>
                            </inertia>
                        </inertial>
                        <visual name="{model_name}_visual">
                            <pose>0 0 0 0 0 1.5707963</pose>
                            <geometry>
                                <mesh>
                                    <uri>{mesh_path_visual}</uri>
                                    <submesh>
                                        <name>RealSense</name>
                                        <center>false</center>
                                    </submesh>
                                </mesh>
                            </geometry>
                            <material>
                                <diffuse>1 1 1 1</diffuse>
                                <specular>1 1 1 1</specular>
                                <pbr>
                                    <metal>
                                        <albedo_map>{albedo_map}</albedo_map>
                                        <normal_map>{normal_map}</normal_map>
                                        <roughness_map>{roughness_map}</roughness_map>
                                        <metalness_map>{metalness_map}</metalness_map>
                                    </metal>
                                </pbr>
                            </material>
                        </visual>
                        """ if visual and use_mesh else ""
                        }
                </link>
            </model>
        </sdf>'''

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

        if ros2_bridge_color or ros2_bridge_depth or ros2_bridge_points:
            self.__threads = []
            if ros2_bridge_color:
                self.__threads.append(
                    Thread(
                        target=self.construct_ros2_bridge,
                        args=(
                            self.color_topic,
                            "sensor_msgs/msg/Image",
                            "ignition.msgs.Image",
                        ),
                        daemon=True,
                    )
                )

            if ros2_bridge_depth:
                self.__threads.append(
                    Thread(
                        target=self.construct_ros2_bridge,
                        args=(
                            self.depth_topic,
                            "sensor_msgs/msg/Image",
                            "ignition.msgs.Image",
                        ),
                        daemon=True,
                    )
                )

            if ros2_bridge_points:
                self.__threads.append(
                    Thread(
                        target=self.construct_ros2_bridge,
                        args=(
                            self.points_topic,
                            "sensor_msgs/msg/PointCloud2",
                            "ignition.msgs.PointCloudPacked",
                        ),
                        daemon=True,
                    )
                )

            for thread in self.__threads:
                thread.start()

    def __del__(self):
        if hasattr(self, "__threads"):
            for thread in self.__threads:
                thread.join()

    @classmethod
    def construct_ros2_bridge(self, topic: str, ros_msg: str, ign_msg: str):
        node_name = "parameter_bridge" + topic.replace("/", "_")
        command = (
            f"ros2 run ros_ign_bridge parameter_bridge {topic}@{ros_msg}[{ign_msg} "
            + f"--ros-args --remap __node:={node_name} --ros-args -p use_sim_time:=true"
        )
        os.system(command)

    @classmethod
    def get_frame_id(cls, model_name: str) -> str:
        return f"{model_name}/{model_name}_link/camera"

    @property
    def frame_id(self) -> str:
        return self.get_frame_id(self._model_name)

    @classmethod
    def get_color_topic(cls, model_name: str) -> str:
        return f"/{model_name}/image" if "rgbd" in model_name else f"/{model_name}"

    @property
    def color_topic(self) -> str:
        return self.get_color_topic(self._model_name)

    @classmethod
    def get_depth_topic(cls, model_name: str) -> str:
        return (
            f"/{model_name}/depth_image" if "rgbd" in model_name else f"/{model_name}"
        )

    @property
    def depth_topic(self) -> str:
        return self.get_depth_topic(self._model_name)

    @classmethod
    def get_points_topic(cls, model_name: str) -> str:
        return f"/{model_name}/points"

    @property
    def points_topic(self) -> str:
        return self.get_points_topic(self._model_name)

    @classmethod
    def get_link_name(cls, model_name: str) -> str:
        return f"{model_name}_link"

    @property
    def link_name(self) -> str:
        return self.get_link_name(self._model_name)
