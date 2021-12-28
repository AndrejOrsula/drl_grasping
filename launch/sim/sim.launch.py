#!/usr/bin/env -S ros2 launch
"""Configure and setup interface with a simulated robot inside Ignition Gazebo"""

from os import path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:

    # Declare all launch arguments
    declared_arguments = generate_declared_arguments()

    # Get substitution for all arguments
    world_name = LaunchConfiguration("world_name")
    robot_model = LaunchConfiguration("robot_model")
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
                        FindPackageShare([robot_model, "_moveit_config"]),
                        "launch",
                        "move_group.launch.py",
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
        # Launch ROS<->IGN bridges
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("lunalab_summit_xl_gen_ign"),
                        "launch",
                        "bridge.launch.py",
                    ]
                )
            ),
            launch_arguments=[
                ("world_name", world_name),
                ("robot_name", robot_name),
                ("prefix", prefix),
                ("use_sim_time", use_sim_time),
                ("log_level", log_level),
            ],
            # TODO: Make ROS<->IGN bridges more general to support all robots
            # condition=LaunchConfigurationEquals("robot_model", "lunalab_summit_xl_gen"),
        ),
    ]

    # List of nodes to be launched
    nodes = []

    # List for logging
    logs = [
        LogInfo(
            msg=[
                "Configuring drl_grasping for Ignition Gazebo world ",
                world_name,
                "\n\tRobot model: ",
                robot_name,
                "\n\tPrefix: ",
                prefix,
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
            "world_name",
            default_value="drl_grasping_world",
            description="Name of the Ignition Gazebo world, which affects some of the Ignition topic names.",
        ),
        DeclareLaunchArgument(
            "robot_model",
            default_value="lunalab_summit_xl_gen",
            description="Name of the robot to use. Supported options are: 'panda', 'ur5_rg2', 'kinova_j2s7s300' and 'lunalab_summit_xl_gen'.",
        ),
        DeclareLaunchArgument(
            "robot_name",
            default_value=LaunchConfiguration("robot_model"),
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
                get_package_share_directory("drl_grasping"), "rviz", "drl_grasping.rviz"
            ),
            description="Path to configuration for RViz2.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="If true, use simulated clock.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="error",
            description="The level of logging that is applied to all ROS 2 nodes launched by this script.",
        ),
    ]
