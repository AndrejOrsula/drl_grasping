from geometry_msgs.msg import Transform
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformListener, Buffer
import rclpy


class Tf2Listener:
    def __init__(
        self,
        node: Node,
    ):

        # Create tf2 buffer and listener for transform lookup
        self.__tf2_buffer = Buffer()
        TransformListener(buffer=self.__tf2_buffer, node=node)

    def lookup_transform_sync(self, target_frame: str, source_frame: str) -> Transform:

        try:
            return self.__tf2_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time(),
            ).transform
        except:
            while rclpy.ok():
                if self.__tf2_buffer.can_transform(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    time=rclpy.time.Time(),
                    timeout=rclpy.time.Duration(seconds=1, nanoseconds=0),
                ):
                    return self.__tf2_buffer.lookup_transform(
                        target_frame=target_frame,
                        source_frame=source_frame,
                        time=rclpy.time.Time(),
                    ).transform

                print(
                    f'Lookup of transform from "{source_frame}"'
                    f' to "{target_frame}" failed, retrying...'
                )


class Tf2ListenerStandalone(Node, Tf2Listener):
    def __init__(
        self,
        node_name: str = "drl_grasping_tf_listener",
        use_sim_time: bool = True,
    ):

        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys

                sys.exit("ROS 2 could not be initialised")

        Node.__init__(self, node_name)
        self.set_parameters(
            [Parameter("use_sim_time", type_=Parameter.Type.BOOL, value=use_sim_time)]
        )

        Tf2Listener.__init__(self, node=self)
