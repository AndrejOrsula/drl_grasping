from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from typing import List, Optional


def init_joint_state(
    joint_names: List[str],
    joint_positions: Optional[List[str]] = None,
    joint_velocities: Optional[List[str]] = None,
    joint_effort: Optional[List[str]] = None,
) -> JointState:

    joint_state = JointState()

    joint_state.name = joint_names
    joint_state.position = (
        joint_positions if joint_positions is not None else [0.0] * len(joint_names)
    )
    joint_state.velocity = (
        joint_velocities if joint_velocities is not None else [0.0] * len(joint_names)
    )
    joint_state.effort = (
        joint_effort if joint_effort is not None else [0.0] * len(joint_names)
    )

    return joint_state


def init_follow_joint_trajectory_goal(
    joint_trajectory: JointTrajectory,
) -> Optional[FollowJointTrajectory.Goal]:

    if joint_trajectory is None:
        return None

    follow_joint_trajectory_goal = FollowJointTrajectory.Goal()

    follow_joint_trajectory_goal.trajectory = joint_trajectory
    # follow_joint_trajectory_goal.multi_dof_trajectory = "Ignored"
    # follow_joint_trajectory_goal.path_tolerance = "Ignored"
    # follow_joint_trajectory_goal.component_path_tolerance = "Ignored"
    # follow_joint_trajectory_goal.goal_tolerance = "Ignored"
    # follow_joint_trajectory_goal.component_goal_tolerance = "Ignored"
    # follow_joint_trajectory_goal.goal_time_tolerance = "Ignored"

    return follow_joint_trajectory_goal


def init_dummy_joint_trajectory_from_state(joint_state: JointState) -> JointTrajectory:

    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = joint_state.name

    point = JointTrajectoryPoint()
    point.positions = joint_state.position
    point.velocities = joint_state.velocity
    point.accelerations = [0.0] * len(joint_trajectory.joint_names)
    point.effort = joint_state.effort
    joint_trajectory.points.append(point)

    return joint_trajectory
