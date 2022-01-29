#!/usr/bin/env -S ros2 launch
"""Configure and setup interface with real Summit XL-GEN (LunaLab variant)"""

from os import path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:

    # Declare all launch arguments
    declared_arguments = generate_declared_arguments()

    # Get substitution for all arguments
    robot_name = LaunchConfiguration("robot_name")
    prefix = LaunchConfiguration("prefix")
    enable_rviz = LaunchConfiguration("enable_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    # List of included launch descriptions
    launch_descriptions = [
        # Launch move_group of MoveIt 2
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare(["lunalab_summit_xl_gen_moveit_config"]),
                        "launch",
                        "move_group_ros1_controllers.launch.py",
                    ]
                )
            ),
            launch_arguments=[
                ("name", robot_name),
                ("prefix", prefix),
                ("enable_rviz", enable_rviz),
                ("rviz_config", rviz_config),
                ("use_sim_time", use_sim_time),
                ("log_level", log_level),
            ],
        ),
    ]

    # List of nodes to be launched
    nodes = [
        # Static tf for world
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                ["drl_grasping_world"],
                "--child-frame-id",
                [prefix, "summit_xl_base_footprint"],
                "--ros-args",
                "--log-level",
                log_level,
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        # Static tf for camera
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--x",
                "0.0",
                "--y",
                "0.0",
                "--z",
                "0.0",
                ## RPY
                "--roll",
                "0.0",
                "--pitch",
                "0.0",
                "--yaw",
                "0.0",
                # ## Quat
                # "--qx",
                # "0.0",
                # "--qy",
                # "0.0",
                # "--qz",
                # "0.0",
                # "--qw",
                # "0.0",
                "--frame-id",
                [prefix, "j2s7s300_link_base"],
                "--child-frame-id",
                ["rs_d455"],
                "--ros-args",
                "--log-level",
                log_level,
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]

    # List for logging
    logs = [
        LogInfo(
            msg=[
                "Configuring drl_grasping for real Summit XL-GEN (LunaLab variant)",
            ],
        )
    ]

    return LaunchDescription(declared_arguments + launch_descriptions + nodes + logs)


def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    """
    Generate list of all launch arguments that are declared for this launch script.
    """

    return [
        # Naming of the world and robot
        DeclareLaunchArgument(
            "robot_name",
            default_value="lunalab_summit_xl_gen",
            description="Name of the robot.",
        ),
        DeclareLaunchArgument(
            "prefix",
            default_value="robot_",
            description="Prefix for all robot entities. If modified, then joint names in the configuration of controllers must also be updated.",
        ),
        # Miscellaneous
        DeclareLaunchArgument(
            "enable_rviz", default_value="true", description="Flag to enable RViz2."
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=path.join(
                get_package_share_directory("drl_grasping"),
                "rviz",
                "drl_grasping_real_evaluation.rviz",
            ),
            description="Path to configuration for RViz2.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="If true, use simulated clock.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="warn",
            description="The level of logging that is applied to all ROS 2 nodes launched by this script.",
        ),
    ]
