#!/usr/bin/env -S ros2 launch
"""Evaluate an RL agent"""

from os import path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:

    # Declare all launch arguments
    declared_arguments = generate_declared_arguments()

    # Get substitution for all arguments
    robot_model = LaunchConfiguration("robot_model")
    robot_name = LaunchConfiguration("robot_name")
    prefix = LaunchConfiguration("prefix")
    env = LaunchConfiguration("env")
    env_kwargs = LaunchConfiguration("env_kwargs")
    algo = LaunchConfiguration("algo")
    num_threads = LaunchConfiguration("num_threads")
    n_episodes = LaunchConfiguration("n_episodes")
    seed = LaunchConfiguration("seed")
    log_folder = LaunchConfiguration("log_folder")
    exp_id = LaunchConfiguration("exp_id")
    load_best = LaunchConfiguration("load_best")
    load_checkpoint = LaunchConfiguration("load_checkpoint")
    stochastic = LaunchConfiguration("stochastic")
    reward_log = LaunchConfiguration("reward_log")
    norm_reward = LaunchConfiguration("norm_reward")
    no_render = LaunchConfiguration("no_render")
    verbose = LaunchConfiguration("verbose")
    enable_rviz = LaunchConfiguration("enable_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    # List of included launch descriptions
    launch_descriptions = [
        # Configure and setup interface with simulated robots inside Ignition Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("drl_grasping"),
                        "launch",
                        "sim",
                        "sim.launch.py",
                    ]
                )
            ),
            launch_arguments=[
                ("robot_model", robot_model),
                ("robot_name", robot_name),
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
        # Evaluation node
        Node(
            package="drl_grasping",
            executable="enjoy.py",
            output="log",
            arguments=[
                "--env",
                env,
                "--env-kwargs",
                env_kwargs,
                # Make sure `robot_model` is specified (with priority)
                "--env-kwargs",
                ['robot_model:"', robot_model, '"'],
                "--algo",
                algo,
                "--seed",
                "--num-threads",
                num_threads,
                "--n-episodes",
                n_episodes,
                "--seed",
                seed,
                "--log-folder",
                log_folder,
                "--exp-id",
                exp_id,
                "--load-best",
                load_best,
                "--load-checkpoint",
                load_checkpoint,
                "--stochastic",
                stochastic,
                "--reward-log",
                reward_log,
                "--norm-reward",
                norm_reward,
                "--no-render",
                no_render,
                "--verbose",
                verbose,
                "--ros-args",
                "--log-level",
                log_level,
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]

    return LaunchDescription(declared_arguments + launch_descriptions + nodes)


def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    """
    Generate list of all launch arguments that are declared for this launch script.
    """

    return [
        # Robot model and its name
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
        # Environment and its parameters
        DeclareLaunchArgument(
            "env",
            default_value="GraspPlanetary-OctreeWithColor-Gazebo-v0",
            description="Environment ID",
        ),
        DeclareLaunchArgument(
            "env_kwargs",
            default_value=['robot_model:"', LaunchConfiguration("robot_model"), '"'],
            description="Optional keyword argument to pass to the env constructor.",
        ),
        # Algorithm
        DeclareLaunchArgument(
            "algo",
            default_value="tqc",
            description="RL algorithm that was used during the training.",
        ),
        DeclareLaunchArgument(
            "num_threads",
            default_value="-1",
            description="Number of threads for PyTorch (-1 to use default).",
        ),
        # Test duration
        DeclareLaunchArgument(
            "n_episodes",
            default_value="200",
            description="Number of evaluation episodes.",
        ),
        # Random seed
        DeclareLaunchArgument(
            "seed",
            default_value="-1",
            description="Random generator seed.",
        ),
        # Model to test
        DeclareLaunchArgument(
            "log_folder",
            default_value="logs",
            description="Path to the log directory.",
        ),
        DeclareLaunchArgument(
            "exp_id",
            default_value="0",
            description="Experiment ID (default: 0: latest, -1: no exp folder).",
        ),
        DeclareLaunchArgument(
            "load_best",
            default_value="False",
            description="Load best model instead of last model if available.",
        ),
        DeclareLaunchArgument(
            "load_checkpoint",
            default_value="0",
            description="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it.",
        ),
        # Deterministic/stochastic actions
        DeclareLaunchArgument(
            "stochastic",
            default_value="False",
            description="Use stochastic actions instead of deterministic.",
        ),
        # Logging
        DeclareLaunchArgument(
            "reward_log",
            default_value="reward_logs",
            description="Where to log reward.",
        ),
        DeclareLaunchArgument(
            "norm_reward",
            default_value="False",
            description="Normalize reward if applicable (trained with VecNormalize)",
        ),
        # Disable render
        DeclareLaunchArgument(
            "no_render",
            default_value="False",
            description="Do not render the environment (useful for tests).",
        ),
        # Verbosity
        DeclareLaunchArgument(
            "verbose",
            default_value="1",
            description="Verbose mode (0: no output, 1: INFO).",
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
