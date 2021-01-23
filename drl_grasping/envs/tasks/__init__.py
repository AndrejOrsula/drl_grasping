from .grasping import Grasping

from gym.envs.registration import register

register(
    id='Grasping-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=40,
    kwargs={'task_cls': Grasping,
            'agent_rate': 4,
            'physics_rate': 100,
            'real_time_factor': 15.0,
            # TODO: Remove the necessity of world in task (currently due to world plugins)
            'world': '/home/andrej/uni/repos/drl_grasping/drl_grasping/envs/worlds/training_grounds.sdf'
            })
