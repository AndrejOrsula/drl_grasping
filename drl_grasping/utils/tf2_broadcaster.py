from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from tf2_ros import StaticTransformBroadcaster
import rclpy


class Tf2Broadcaster(Node):
    def __init__(self, use_sim_time=True):

        self._create_tf_broadcaster()

        self.set_parameters(
            [Parameter('use_sim_time', type_=Parameter.Type.BOOL, value=use_sim_time)])

    def _create_tf_broadcaster(self,
                               parent_frame_id="world",
                               child_frame_id="unknown_child_id"):
        """Add PointCloud subscriber and spin it in another thread"""

        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")

        Node.__init__(self, "drl_grasping_camera_tf_broadcaster")

        qos = QoSProfile(durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)
        self._tf2_broadcaster = StaticTransformBroadcaster(self, qos=qos)

        self._transform_stamped = TransformStamped()
        self.set_parent_frame_id(parent_frame_id)
        self.set_child_frame_id(child_frame_id)

    def set_parent_frame_id(self, parent_frame_id):

        self._transform_stamped.header.frame_id = parent_frame_id

    def set_child_frame_id(self, child_frame_id):

        self._transform_stamped.child_frame_id = child_frame_id

    def broadcast_tf(self, translation, rotation, parent_frame_id=None, child_frame_id=None):
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

        transform.transform.rotation.w = float(rotation[0])
        transform.transform.rotation.x = float(rotation[1])
        transform.transform.rotation.y = float(rotation[2])
        transform.transform.rotation.z = float(rotation[3])

        self._tf2_broadcaster.sendTransform(transform)
