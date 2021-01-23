from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from threading import Thread, Lock
import rclpy


class ImageSub(Node):
    def __init__(self,
                 topic="rgbd_camera/image"):
        self.create_image_sub(topic)

    def create_image_sub(self, topic):
        """Add Image subscriber and spin it in another thread"""
        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")

        Node.__init__(self, "drl_grasping_image_sub")

        self.__image = Image()
        self.__image_mutex = Lock()
        self.__is_new_available = False
        self.__image_sub = self.create_subscription(Image,
                                                    topic,
                                                    self.image_callback, 1)

        self._realsense_executor = MultiThreadedExecutor(1)
        self._realsense_executor.add_node(self)
        thread = Thread(target=self._realsense_executor.spin, args=())
        thread.daemon = True
        thread.start()

    def image_callback(self, msg):
        """
        Callback for getting point cloud.
        """
        self.__image_mutex.acquire()
        self.__image = msg
        self.__is_new_available = True
        self.__image_mutex.release()

    def get_image(self) -> Image:
        """
        Get the last received point cloud.
        """
        self.__image_mutex.acquire()
        image = self.__image
        self.__is_new_available = False
        self.__image_mutex.release()
        return image

    def is_new_available(self) -> bool:
        return self.__is_new_available
