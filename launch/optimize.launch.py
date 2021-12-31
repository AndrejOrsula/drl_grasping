#!/usr/bin/env -S ros2 launch
"""Optimize hyperparameters for RL training with Optuna"""

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
    n_timesteps = LaunchConfiguration("n_timesteps")
    num_threads = LaunchConfiguration("num_threads")
    seed = LaunchConfiguration("seed")
    preload_replay_buffer = LaunchConfiguration("preload_replay_buffer")
    log_folder = LaunchConfiguration("log_folder")
    tensorboard_log = LaunchConfiguration("tensorboard_log")
    log_interval = LaunchConfiguration("log_interval")
    uuid = LaunchConfiguration("uuid")
    sampler = LaunchConfiguration("sampler")
    pruner = LaunchConfiguration("pruner")
    n_trials = LaunchConfiguration("n_trials")
    n_startup_trials = LaunchConfiguration("n_startup_trials")
    n_evaluations = LaunchConfiguration("n_evaluations")
    n_jobs = LaunchConfiguration("n_jobs")
    storage = LaunchConfiguration("storage")
    study_name = LaunchConfiguration("study_name")
    eval_episodes = LaunchConfiguration("eval_episodes")
    verbose = LaunchConfiguration("verbose")
    truncate_last_trajectory = LaunchConfiguration("truncate_last_trajectory")
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
        # Train node
        Node(
            package="drl_grasping",
            executable="train.py",
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
                n_timesteps,
                "--num-threads",
                num_threads,
                "--preload-replay-buffer",
                seed,
                "--n-timesteps",
                preload_replay_buffer,
                "--log-folder",
                log_folder,
                "--tensorboard-log",
                tensorboard_log,
                "--log-interval",
                log_interval,
                "--uuid",
                uuid,
                "--optimize-hyperparameters",
                "True",
                "--sampler",
                sampler,
                "--pruner",
                pruner,
                "--n-trials",
                n_trials,
                "--n-startup-trials",
                n_startup_trials,
                "--n-evaluations",
                n_evaluations,
                "--n-jobs",
                n_jobs,
                "--storage",
                storage,
                "--study-name",
                study_name,
                "--eval-episodes",
                eval_episodes,
                "--verbose",
                verbose,
                "--truncate-last-trajectory",
                truncate_last_trajectory,
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
        DeclareLaunchArgument(
            "vec_env",
            default_value="dummy",
            description="Type of VecEnv to use (dummy or subproc).",
        ),
        # Algorithm and optimization
        DeclareLaunchArgument(
            "algo",
            default_value="tqc",
            description="RL algorithm to use during the optimization.",
        ),
        DeclareLaunchArgument(
            "n_timesteps",
            default_value="-1",
            description="Overwrite the number of timesteps.",
        ),
        DeclareLaunchArgument(
            "num_threads",
            default_value="-1",
            description="Number of threads for PyTorch (-1 to use default).",
        ),
        # Random seed
        DeclareLaunchArgument(
            "seed",
            default_value="-1",
            description="Random generator seed.",
        ),
        # Pre-load a replay buffer and start optimization on it
        DeclareLaunchArgument(
            "preload_replay_buffer",
            default_value="",
            description="Path to a replay buffer that should be preloaded before starting the optimization process.",
        ),
        # Logging
        DeclareLaunchArgument(
            "log_folder",
            default_value="logs",
            description="Path to the log directory.",
        ),
        DeclareLaunchArgument(
            "tensorboard_log",
            default_value="tensorboard_logs",
            description="Tensorboard log dir.",
        ),
        DeclareLaunchArgument(
            "log_interval",
            default_value="-1",
            description="Override log interval (default: -1, no change).",
        ),
        DeclareLaunchArgument(
            "uuid",
            default_value="False",
            description="Ensure that the run has a unique ID.",
        ),
        # Hyperparameter optimization
        DeclareLaunchArgument(
            "sampler",
            default_value="tpe",
            description="Sampler to use when optimizing hyperparameters (random, tpe or skopt).",
        ),
        DeclareLaunchArgument(
            "pruner",
            default_value="median",
            description="Pruner to use when optimizing hyperparameters (halving, median or none).",
        ),
        DeclareLaunchArgument(
            "n_trials",
            default_value="10",
            description="Number of trials for optimizing hyperparameters.",
        ),
        DeclareLaunchArgument(
            "n_startup_trials",
            default_value="5",
            description="Number of trials before using optuna sampler.",
        ),
        DeclareLaunchArgument(
            "n_evaluations",
            default_value="2",
            description="Number of evaluations for hyperparameter optimization.",
        ),
        DeclareLaunchArgument(
            "n_jobs",
            default_value="1",
            description="Number of parallel jobs when optimizing hyperparameters.",
        ),
        DeclareLaunchArgument(
            "storage",
            default_value="",
            description="Database storage path if distributed optimization should be used.",
        ),
        DeclareLaunchArgument(
            "study_name",
            default_value="",
            description="Study name for distributed optimization.",
        ),
        # Evaluation
        DeclareLaunchArgument(
            "eval_episodes",
            default_value="5",
            description="Number of episodes to use for evaluation.",
        ),
        # Verbosity
        DeclareLaunchArgument(
            "verbose",
            default_value="1",
            description="Verbose mode (0: no output, 1: INFO).",
        ),
        # HER specifics
        DeclareLaunchArgument(
            "truncate_last_trajectory",
            default_value="True",
            description="When using HER with online sampling the last trajectory in the replay buffer will be truncated after reloading the replay buffer.",
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
