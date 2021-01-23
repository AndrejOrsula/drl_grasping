from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from threading import Thread, Lock
import rclpy


class PointCloudSub(Node):
    def __init__(self,
                 topic="rgbd_camera/points"):
        self.create_point_cloud_sub(topic)

    def create_point_cloud_sub(self, topic):
        """Add PointCloud subscriber and spin it in another thread"""
        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")

        Node.__init__(self, "drl_grasping_point_cloud_sub")

        self.__point_cloud = PointCloud2()
        self.__point_cloud_mutex = Lock()
        self.__point_cloud_sub = self.create_subscription(PointCloud2,
                                                          topic,
                                                          self.point_cloud_callback, 1)

        self._realsense_executor = MultiThreadedExecutor(1)
        self._realsense_executor.add_node(self)
        thread = Thread(target=self._realsense_executor.spin, args=())
        thread.daemon = True
        thread.start()

    def point_cloud_callback(self, msg):
        """
        Callback for getting point cloud.
        """
        self.__point_cloud_mutex.acquire()
        self.__point_cloud = msg
        self.__point_cloud_mutex.release()

    def get_point_cloud(self) -> PointCloud2:
        """
        Get the last received point cloud.
        """
        self.__point_cloud_mutex.acquire()
        point_cloud = self.__point_cloud
        self.__point_cloud_mutex.release()
        return point_cloud
