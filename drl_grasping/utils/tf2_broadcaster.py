from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from tf2_ros import StaticTransformBroadcaster
from typing import Tuple
import rclpy


class Tf2Broadcaster(Node):
    def __init__(self,
                 parent_frame_id: str = "world",
                 child_frame_id: str = "unknown_child_id",
                 use_sim_time: bool = True,
                 node_name: str = 'drl_grasping_camera_tf_broadcaster'):

        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")

        Node.__init__(self, node_name)
        self.set_parameters([Parameter('use_sim_time',
                                       type_=Parameter.Type.BOOL,
                                       value=use_sim_time)])

        qos = QoSProfile(durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_ALL)
        self._tf2_broadcaster = StaticTransformBroadcaster(self, qos=qos)

        self._transform_stamped = TransformStamped()
        self.set_parent_frame_id(parent_frame_id)
        self.set_child_frame_id(child_frame_id)

    def set_parent_frame_id(self, parent_frame_id: str):

        self._transform_stamped.header.frame_id = parent_frame_id

    def set_child_frame_id(self, child_frame_id: str):

        self._transform_stamped.child_frame_id = child_frame_id

    def broadcast_tf(self,
                     translation: Tuple[float, float, float],
                     rotation: Tuple[float, float, float, float],
                     xyzw: bool = True,
                     parent_frame_id: str = None,
                     child_frame_id: str = None):
        """
        Broadcast transformation of the camera
        """

        transform = self._transform_stamped

        if parent_frame_id is not None:
            transform.header.frame_id = parent_frame_id

        if child_frame_id is not None:
            transform.child_frame_id = child_frame_id

        transform.header.stamp = self.get_clock().now().to_msg()

        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        if xyzw:
            transform.transform.rotation.x = float(rotation[0])
            transform.transform.rotation.y = float(rotation[1])
            transform.transform.rotation.z = float(rotation[2])
            transform.transform.rotation.w = float(rotation[3])
        else:
            transform.transform.rotation.w = float(rotation[0])
            transform.transform.rotation.x = float(rotation[1])
            transform.transform.rotation.y = float(rotation[2])
            transform.transform.rotation.z = float(rotation[3])

        self._tf2_broadcaster.sendTransform(transform)
