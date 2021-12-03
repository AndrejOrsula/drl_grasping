from copy import deepcopy
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from std_srvs.srv import Trigger
from typing import Tuple, Optional
import numpy as np


class MoveIt2Servo:
    def __init__(
        self,
        node: Node,
        frame_id: str,
        linear_speed: float = 0.5,
        angular_speed: float = 10.0 * np.pi / 180.0,
        enable_at_init: bool = True,
        callback_group: Optional[CallbackGroup] = None,
    ):

        self._node = node

        # Create publisher
        self.__twist_pub = self._node.create_publisher(
            msg_type=TwistStamped,
            topic="delta_twist_cmds",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_ALL,
            ),
            callback_group=callback_group,
        )

        # Create service clients
        self.__start_service = self._node.create_client(
            srv_type=Trigger,
            srv_name="/servo_node/start_servo",
            callback_group=callback_group,
        )
        self.__stop_service = self._node.create_client(
            srv_type=Trigger,
            srv_name="/servo_node/stop_servo",
            callback_group=callback_group,
        )
        self.__trigger_req = Trigger.Request()
        self.__is_enabled = False

        # Initialize message based on passed arguments
        self.__twist_msg = TwistStamped()
        self.__twist_msg.header.frame_id = frame_id
        self.__twist_msg.twist.linear.x = linear_speed
        self.__twist_msg.twist.linear.y = linear_speed
        self.__twist_msg.twist.linear.z = linear_speed
        self.__twist_msg.twist.angular.x = angular_speed
        self.__twist_msg.twist.angular.y = angular_speed
        self.__twist_msg.twist.angular.z = angular_speed

        # Enable servo immediately, if desired
        if enable_at_init:
            self.enable()

    def __call__(
        self,
        linear: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):

        self.servo(linear=linear, angular=angular)

    def servo(
        self,
        linear: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):

        twist_msg = deepcopy(self.__twist_msg)
        twist_msg.header.stamp = self._node.get_clock().now().to_msg()
        twist_msg.twist.linear.x *= linear[0]
        twist_msg.twist.linear.y *= linear[1]
        twist_msg.twist.linear.z *= linear[2]
        twist_msg.twist.angular.x *= angular[0]
        twist_msg.twist.angular.y *= angular[1]
        twist_msg.twist.angular.z *= angular[2]
        self.__twist_pub.publish(twist_msg)

    def enable(self):

        while not self.__start_service.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Service '{self.__start_service.srv_name}' is not yet available..."
            )
        self.__start_service.call_async(self.__trigger_req)
        self.__is_enabled = True

    def disable(self):

        while not self.__stop_service.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Service '{self.__stop_service.srv_name}' is not yet available..."
            )
        self.__stop_service.call_async(self.__trigger_req)
        self.__is_enabled = False

    @property
    def is_enabled(self) -> bool:

        return self.__is_enabled

    @property
    def frame_id(self) -> str:

        return self.__twist_msg.header.frame_id

    @frame_id.setter
    def frame_id(self, value: str):

        self.__twist_msg.header.frame_id = value

    @property
    def linear_speed(self) -> float:

        return self.__twist_msg.twist.linear.x

    @linear_speed.setter
    def linear_speed(self, value: float):

        self.__twist_msg.twist.linear.x = value
        self.__twist_msg.twist.linear.y = value
        self.__twist_msg.twist.linear.z = value

    @property
    def angular_speed(self) -> float:

        return self.__twist_msg.twist.angular.x

    @angular_speed.setter
    def angular_speed(self, value: float):

        self.__twist_msg.twist.angular.x = value
        self.__twist_msg.twist.angular.y = value
        self.__twist_msg.twist.angular.z = value
