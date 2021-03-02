from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, Image
from threading import Thread, Lock
from typing import Union
import rclpy


class CameraSubscriber(Node):
    def __init__(self,
                 topic: str,
                 is_point_cloud: bool,
                 use_sim_time: bool = True,
                 node_name: str = 'drl_grasping_camera_sub'):

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

        # Prepare the subscriber
        if is_point_cloud:
            observation_type = PointCloud2
        else:
            observation_type = Image
        self.__observation = observation_type()
        self.__observation_sub = self.create_subscription(msg_type=observation_type,
                                                          topic=topic,
                                                          callback=self.observation_callback,
                                                          qos_profile=QoSProfile(durability=QoSDurabilityPolicy.SYSTEM_DEFAULT,
                                                                                 reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                                                                 history=QoSHistoryPolicy.SYSTEM_DEFAULT))
        self.__observation_mutex = Lock()
        self.__new_observation_available = False

        # Spin the node in a separate thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = Thread(
            target=self._executor.spin, args=(), daemon=True)
        self._executor_thread.start()

    def observation_callback(self, msg):
        """
        Callback for getting observation.
        """

        self.__observation_mutex.acquire()
        self.__observation = msg
        self.__new_observation_available = True
        self.__observation_mutex.release()

    def get_observation(self) -> Union[PointCloud2, Image]:
        """
        Get the last received observation.
        """

        self.__observation_mutex.acquire()
        observation = self.__observation
        self.__observation_mutex.release()
        return observation

    def new_observation_available(self):
        """
        Check if new observation is available since `self.reset_new_observation_checker()` was called
        """

        return self.__new_observation_available

    def reset_new_observation_checker(self):
        """
        Reset checker of new observations, i.e. `self.new_observation_available()`
        """

        self.__new_observation_available = False
