from moveit2 import MoveIt2Interface
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import rclpy


class MoveIt2(MoveIt2Interface):
    def __init__(
        self,
        robot_model: str,
        separate_gripper_controller: bool = True,
        use_sim_time: bool = True,
        node_name: str = "ign_moveit2_py",
    ):
        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys

                sys.exit("ROS 2 could not be initialised")

        if "lunalab_summit_xl_gen" == robot_model:
            super().__init__(
                robot_model="kinova_j2s7s300",
                namespace="/lunalab_summit_xl_gen",
                prefix="robot_",
                arm_group_name="arm",
                gripper_group_name="gripper",
                separate_gripper_controller=separate_gripper_controller,
                use_sim_time=use_sim_time,
                node_name=node_name,
            )
        else:
            super().__init__(
                robot_model=robot_model,
                separate_gripper_controller=separate_gripper_controller,
                use_sim_time=use_sim_time,
                node_name=node_name,
            )

        self._moveit2_executor = MultiThreadedExecutor(1)
        self._moveit2_executor.add_node(self)
        thread = Thread(target=self._moveit2_executor.spin, args=())
        thread.daemon = True
        thread.start()
