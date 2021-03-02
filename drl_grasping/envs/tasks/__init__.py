from gym.envs.registration import register

from .reach import Reach, ReachColorImage, ReachDepthImage, ReachOctree
from .grasp import Grasp, GraspOctree

# Reach
REACH_MAX_EPISODE_STEPS: int = 100
REACH_AGENT_RATE: float = 2.5
REACH_PHYSICS_RATE: float = 200.0
REACH_RTF: float = 15.0
register(
    id='Reach-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'task_cls': Reach,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            })
register(
    id='Reach-ColorImage-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'task_cls': ReachColorImage,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            })
register(
    id='Reach-DepthImage-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'task_cls': ReachDepthImage,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            })
register(
    id='Reach-Octree-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'task_cls': ReachOctree,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            })

# Grasp
GRASP_MAX_EPISODE_STEPS: int = 250
GRASP_AGENT_RATE: float = 2.5
GRASP_PHYSICS_RATE: float = 250.0
GRASP_RTF: float = 10.0
register(
    id='Grasp-Octree-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={'task_cls': GraspOctree,
            'agent_rate': GRASP_AGENT_RATE,
            'physics_rate': GRASP_PHYSICS_RATE,
            'real_time_factor': GRASP_RTF,
            })
