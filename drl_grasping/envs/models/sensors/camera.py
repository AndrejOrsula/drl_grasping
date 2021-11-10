from gym_ignition.scenario import model_wrapper
from gym_ignition.utils import misc
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List, Union
import os


class Camera(model_wrapper.ModelWrapper):
    def __init__(
        self,
        world: scenario.World,
        name: Union[str, None] = None,
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        camera_type: str = "rgbd_camera",
        width: int = 212,
        height: int = 120,
        update_rate: int = 15,
        horizontal_fov: float = 1.567821,
        vertical_fov: float = 1.022238,
        clip_color: List[float] = (0.01, 1000.0),
        clip_depth: List[float] = (0.01, 10.0),
        noise_mean: float = None,
        noise_stddev: float = None,
        ros2_bridge_color: bool = False,
        ros2_bridge_depth: bool = False,
        ros2_bridge_points: bool = False,
        visibility_mask: int = 2,
    ):

        # Get a unique model name
        if name is not None:
            model_name = get_unique_model_name(world, name)
        else:
            model_name = get_unique_model_name(world, camera_type)
        self._model_name = model_name

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Create SDF string for the model
        sdf = f'''<sdf version="1.7">
            <model name="{model_name}">
                <static>true</static>
                <link name="{model_name}_link">
                    <sensor name="camera" type="{camera_type}">
                        <topic>{model_name}</topic>
                        <always_on>1</always_on>
                        <update_rate>{update_rate}</update_rate>
                        <camera name="{model_name}_camera">
                            <image>
                                <width>{width}</width>
                                <height>{height}</height>
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
                            </depth_camera>""" if self.is_rgbd() else ""
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
                    </sensor>
                </link>
            </model>
        </sdf>'''

        # Convert it into a file
        sdf_file = misc.string_to_file(sdf)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(sdf_file, initial_pose, model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)

        if ros2_bridge_color or ros2_bridge_depth or ros2_bridge_points:
            from threading import Thread

            # Note: This unfortunately hinders use of SIGINT

            threads = []
            if ros2_bridge_color:
                thread = Thread(
                    target=self.construct_ros2_bridge,
                    args=(
                        self.color_topic(),
                        "sensor_msgs/msg/Image",
                        "ignition.msgs.Image",
                    ),
                    daemon=True,
                )
                threads.append(thread)

            if ros2_bridge_depth:
                thread = Thread(
                    target=self.construct_ros2_bridge,
                    args=(
                        self.depth_topic(),
                        "sensor_msgs/msg/Image",
                        "ignition.msgs.Image",
                    ),
                    daemon=True,
                )
                threads.append(thread)

            if ros2_bridge_points:
                thread = Thread(
                    target=self.construct_ros2_bridge,
                    args=(
                        self.points_topic(),
                        "sensor_msgs/msg/PointCloud2",
                        "ignition.msgs.PointCloudPacked",
                    ),
                    daemon=True,
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

    @classmethod
    def construct_ros2_bridge(self, topic: str, ros_msg: str, ign_msg: str):
        node_name = "parameter_bridge" + topic.replace("/", "_")
        command = (
            f"ros2 run ros_ign_bridge parameter_bridge {topic}@{ros_msg}[{ign_msg} "
            + f"--ros-args --remap __node:={node_name} --ros-args -p use_sim_time:=true"
        )
        os.system(command)

    @classmethod
    def frame_id_name(self, model_name: str) -> str:
        return f"{model_name}/{model_name}_link/camera"

    def frame_id(self) -> str:
        return self.frame_id_name(self._model_name)

    def color_topic(self) -> str:
        return (
            f"/{self._model_name}/image" if self.is_rgbd() else f"/{self._model_name}"
        )

    def depth_topic(self) -> str:
        return (
            f"/{self._model_name}/depth_image"
            if self.is_rgbd()
            else f"/{self._model_name}"
        )

    def points_topic(self) -> str:
        return f"/{self._model_name}/points"

    def is_rgbd(self) -> bool:
        if "rgbd" in self._model_name:
            return True
        else:
            return False
