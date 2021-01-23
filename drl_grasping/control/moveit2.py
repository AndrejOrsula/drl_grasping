from moveit2 import MoveIt2Interface
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import rclpy


class MoveIt2(MoveIt2Interface):
    def __init__(self):
        self.__add_moveit2_interface()

    def __add_moveit2_interface(self):
        """Add MoveIt2 interface and spin it in another thread"""
        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")

        super().__init__()

        self._moveit2_executor = MultiThreadedExecutor(1)
        self._moveit2_executor.add_node(self)
        thread = Thread(target=self._moveit2_executor.spin, args=())
        thread.daemon = True
        thread.start()
