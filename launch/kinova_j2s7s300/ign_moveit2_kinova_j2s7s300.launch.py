"""Forwarded launch of ign_moveit2 (with RViz2 enabled)"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    config_rviz2 = LaunchConfiguration('config_rviz2', default=os.path.join(get_package_share_directory('drl_grasping'),
                                                                            'launch', 'kinova_j2s7s300', 'rviz_kinova_j2s7s300.rviz'))
    log_level = LaunchConfiguration('log_level', default='fatal')

    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description="If true, use simulated clock"),
        DeclareLaunchArgument(
            'config_rviz2',
            default_value=config_rviz2,
            description="Path to config for RViz2. If empty, RViz2 will be disabled"),
        DeclareLaunchArgument(
            'log_level',
            default_value=log_level,
            description="Log level of all nodes launched by this script"),

        # MoveIt2 move_group action server with necessary ROS2 <-> Ignition bridges
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory('ign_moveit2'),
                              'launch', 'ign_moveit2_kinova_j2s7s300.launch.py')]),
            launch_arguments=[('use_sim_time', use_sim_time),
                              ('config_rviz2', config_rviz2),
                              ('log_level', log_level)]),

        # JointTrajectory bridge for gripper (ROS2 -> IGN)
        Node(package='ros_ign_bridge',
             executable='parameter_bridge',
             name='parameter_bridge_gripper_trajectory',
             output='screen',
             arguments=['/gripper_trajectory@trajectory_msgs/msg/JointTrajectory]ignition.msgs.JointTrajectory',
                        '--ros-args', '--log-level', log_level],
             parameters=[{'use_sim_time': use_sim_time}])
    ])
