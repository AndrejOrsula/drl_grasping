from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import StaticTransformBroadcaster
from typing import Tuple
import rclpy


class Tf2Broadcaster:
    def __init__(
        self,
        node: Node,
    ):

        self._node = node
        self.__tf2_broadcaster = StaticTransformBroadcaster(node=self._node)
        self._transform_stamped = TransformStamped()

    def broadcast_tf(
        self,
        parent_frame_id: str,
        child_frame_id: str,
        translation: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
        xyzw: bool = True,
    ):
        """
        Broadcast transformation of the camera
        """

        self._transform_stamped.header.frame_id = parent_frame_id
        self._transform_stamped.child_frame_id = child_frame_id

        self._transform_stamped.header.stamp = self._node.get_clock().now().to_msg()

        self._transform_stamped.transform.translation.x = float(translation[0])
        self._transform_stamped.transform.translation.y = float(translation[1])
        self._transform_stamped.transform.translation.z = float(translation[2])

        if xyzw:
            self._transform_stamped.transform.rotation.x = float(rotation[0])
            self._transform_stamped.transform.rotation.y = float(rotation[1])
            self._transform_stamped.transform.rotation.z = float(rotation[2])
            self._transform_stamped.transform.rotation.w = float(rotation[3])
        else:
            self._transform_stamped.transform.rotation.w = float(rotation[0])
            self._transform_stamped.transform.rotation.x = float(rotation[1])
            self._transform_stamped.transform.rotation.y = float(rotation[2])
            self._transform_stamped.transform.rotation.z = float(rotation[3])

        self.__tf2_broadcaster.sendTransform(self._transform_stamped)


class Tf2BroadcasterStandalone(Node, Tf2Broadcaster):
    def __init__(
        self,
        node_name: str = "drl_grasping_tf_broadcaster",
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

        Tf2Broadcaster.__init__(self, node=self)
