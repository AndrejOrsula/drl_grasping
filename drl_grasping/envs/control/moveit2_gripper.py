from .common import (
    init_dummy_joint_trajectory_from_state,
    init_follow_joint_trajectory_goal,
)
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from typing import List, Optional


class MoveIt2Gripper:
    def __init__(
        self,
        node: Node,
        frame_id: str,
        gripper_joint_names: List[str],
        open_gripper_joint_positions: List[float],
        closed_gripper_joint_positions: List[float],
        gripper_group_name: str = "gripper",
        use_planning_service: bool = True,
        plan_once: bool = True,
        ignore_new_calls_while_executing: bool = True,
        callback_group: Optional[CallbackGroup] = None,
    ):

        self._node = node

        # Create action client for move action
        if not (use_planning_service and plan_once):
            self.__move_action_client = ActionClient(
                node=self._node,
                action_type=MoveGroup,
                action_name="move_action",
                goal_service_qos_profile=QoSProfile(
                    durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                ),
                result_service_qos_profile=QoSProfile(
                    durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=5,
                ),
                cancel_service_qos_profile=QoSProfile(
                    durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=5,
                ),
                feedback_sub_qos_profile=QoSProfile(
                    durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                ),
                status_sub_qos_profile=QoSProfile(
                    durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                ),
                callback_group=callback_group,
            )

        # Create action client for trajectory execution
        self.__follow_joint_trajectory_action_client = ActionClient(
            node=self._node,
            action_type=FollowJointTrajectory,
            action_name="gripper_trajectory_controller/follow_joint_trajectory",
            goal_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            result_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            cancel_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            feedback_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            status_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=callback_group,
        )

        # If desired, a separate service will be used to plan paths instead of relying onmove action server
        # This service seems to be much faster than communicating with move action server
        self.__use_planning_service = use_planning_service
        if self.__use_planning_service:
            self.__plan_kinematic_path_service = self._node.create_client(
                srv_type=GetMotionPlan,
                srv_name="plan_kinematic_path",
                callback_group=callback_group,
            )
            self.__kinematic_path_request = GetMotionPlan.Request()

        self.__move_action_goal = self.__init_move_action_goal(
            frame_id=frame_id, gripper_group_name=gripper_group_name
        )
        self.__open_joint_state = self.__init_gripper_joint_state(
            gripper_joint_names=gripper_joint_names,
            joint_positions=open_gripper_joint_positions,
        )
        self.__closed_joint_state = self.__init_gripper_joint_state(
            gripper_joint_names=gripper_joint_names,
            joint_positions=closed_gripper_joint_positions,
        )
        self.__open_goal_constraints = self.__init_gripper_joint_contraint(
            gripper_joint_names=gripper_joint_names,
            joint_positions=open_gripper_joint_positions,
        )
        self.__closed_goal_constraints = self.__init_gripper_joint_contraint(
            gripper_joint_names=gripper_joint_names,
            joint_positions=closed_gripper_joint_positions,
        )
        self.__open_reset_dummy_trajectory_goal = init_follow_joint_trajectory_goal(
            joint_trajectory=init_dummy_joint_trajectory_from_state(
                self.__open_joint_state
            )
        )
        self.__closed_reset_dummy_trajectory_goal = init_follow_joint_trajectory_goal(
            joint_trajectory=init_dummy_joint_trajectory_from_state(
                self.__closed_joint_state
            )
        )

        # Flag that determines whether a new goal can be send while the previous one is being executed
        self.__ignore_new_calls_while_executing = ignore_new_calls_while_executing
        if self.__ignore_new_calls_while_executing:
            self.__is_executing = False

        # If desired, the trajectory will be planned only once and all subsequent execution will communicate directly with the controller
        self.__plan_once = plan_once

        # Initialize additional variables
        self.__is_open = True
        self.__open_follow_joint_trajectory_req = None
        self.__close_follow_joint_trajectory_req = None

    def __call__(self):

        self.toggle()

    def toggle(self):

        if self.is_open:
            self.close()
        else:
            self.open()

    def open(self):

        # Don't do anything if open already
        if self.is_open:
            return

        # Plan only once and execute the same trajectory on each subsequent call
        if self.__plan_once:
            # Plan trajectory if needed (first time only)
            if self.__open_follow_joint_trajectory_req is None:
                self.__prepare_move_action_goal_open()
                if self.__use_planning_service:
                    joint_trajectory = self.__plan_kinematic_path()
                else:
                    joint_trajectory = self.__send_goal_move_action_plan_only()
                self.__open_follow_joint_trajectory_req = (
                    init_follow_joint_trajectory_goal(joint_trajectory=joint_trajectory)
                )
                self.__try_destroy_move_action_client()

            # Execute
            self.__send_goal_async_follow_joint_trajectory(
                self.__open_follow_joint_trajectory_req
            )

        else:
            # Plan and execute with move action
            self.__prepare_move_action_goal_open()
            self.__send_goal_async_move_action()

    def close(self):

        # Don't do anything if closed already
        if self.is_closed:
            return

        # Plan only once and execute the same trajectory on each subsequent call
        if self.__plan_once:
            # Plan trajectory if needed (first time only)
            if self.__close_follow_joint_trajectory_req is None:
                self.__prepare_move_action_goal_close()
                if self.__use_planning_service:
                    joint_trajectory = self.__plan_kinematic_path()
                else:
                    joint_trajectory = self.__send_goal_move_action_plan_only()
                self.__close_follow_joint_trajectory_req = (
                    init_follow_joint_trajectory_goal(joint_trajectory=joint_trajectory)
                )
                self.__try_destroy_move_action_client()

            # Execute
            self.__send_goal_async_follow_joint_trajectory(
                self.__close_follow_joint_trajectory_req
            )

        else:
            # Plan and execute with move action
            self.__prepare_move_action_goal_close()
            self.__send_goal_async_move_action()

    @property
    def is_open(self) -> bool:

        return self.__is_open

    @property
    def is_closed(self) -> bool:

        return not self.is_open

    def reset_open(self):
        """
        Reset controller to a given `joint_state` by sending a dummy joint trajectory.
        This is useful for simulated robots that allow instantaneous reset of joints.
        """

        self.__send_goal_async_follow_joint_trajectory(
            goal=self.__open_reset_dummy_trajectory_goal
        )

        self.__is_open = True

    def reset_closed(self):
        """
        Reset controller to a given `joint_state` by sending a dummy joint trajectory.
        This is useful for simulated robots that allow instantaneous reset of joints.
        """

        self.__send_goal_async_follow_joint_trajectory(
            goal=self.__closed_reset_dummy_trajectory_goal
        )

        self.__is_open = False

    def __toggle_internal_gripper_state(self):

        self.__is_open = not self.__is_open

    def __send_goal_move_action_plan_only(self) -> Optional[JointTrajectory]:

        # Set action goal to only do planning without execution
        original_plan_only = self.__move_action_goal.planning_options.plan_only
        self.__move_action_goal.planning_options.plan_only = True

        stamp = self._node.get_clock().now().to_msg()
        self.__move_action_goal.request.workspace_parameters.header.stamp = stamp

        if not self.__move_action_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Action server {self.__move_action_client._action_name} is not yet ready. Better luck next time!"
            )
            return None

        move_action_result = self.__move_action_client.send_goal(
            goal=self.__move_action_goal,
            feedback_callback=None,
        )

        # Revert back to original planning/execution mode
        self.__move_action_goal.planning_options.plan_only = original_plan_only

        if move_action_result.status == GoalStatus.STATUS_SUCCEEDED:
            return move_action_result.result.planned_trajectory.joint_trajectory
        else:
            return None

    def __plan_kinematic_path(self) -> Optional[JointTrajectory]:

        # Re-use request from move action goal
        self.__kinematic_path_request.motion_plan_request = (
            self.__move_action_goal.request
        )

        stamp = self._node.get_clock().now().to_msg()
        self.__kinematic_path_request.motion_plan_request.workspace_parameters.header.stamp = (
            stamp
        )

        if not self.__plan_kinematic_path_service.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Service '{self.__plan_kinematic_path_service.srv_name}' is not yet available. Better luck next time!"
            )
            return None

        res = self.__plan_kinematic_path_service.call(
            self.__kinematic_path_request
        ).motion_plan_response

        if MoveItErrorCodes.SUCCESS == res.error_code.val:
            return res.trajectory.joint_trajectory
        else:
            self._node.get_logger().warn(
                f"Planning failed! Error Code: {res.error_code.val}"
            )
            return None

    def __send_goal_async_move_action(self):

        if self.__ignore_new_calls_while_executing:
            if self.__is_executing:
                return

        stamp = self._node.get_clock().now().to_msg()
        self.__move_action_goal.request.workspace_parameters.header.stamp = stamp

        if not self.__move_action_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Action server {self.__move_action_client._action_name} is not yet ready. Better luck next time!"
            )
            return

        self.__send_goal_future_move_action = self.__move_action_client.send_goal_async(
            goal=self.__move_action_goal,
            feedback_callback=None,
        )

        self.__send_goal_future_move_action.add_done_callback(
            self.__response_callback_move_action
        )

    def __response_callback_move_action(self, response):

        goal_handle = response.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn(
                f"Action '{self.__move_action_client._action_name}' was rejected"
            )
            return

        if self.__ignore_new_calls_while_executing:
            self.__is_executing = True

        self.__get_result_future_move_action = goal_handle.get_result_async()
        self.__get_result_future_move_action.add_done_callback(
            self.__result_callback_move_action
        )

    def __result_callback_move_action(self, res):

        if res.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.__toggle_internal_gripper_state()
        else:
            self._node.get_logger().error(
                f"Action '{self.__move_action_client._action_name}' was unsuccessful: {res.result().status}"
            )

        if self.__ignore_new_calls_while_executing:
            self.__is_executing = False

    def __send_goal_async_follow_joint_trajectory(self, goal: FollowJointTrajectory):

        if self.__ignore_new_calls_while_executing:
            if self.__is_executing:
                return

        if not self.__follow_joint_trajectory_action_client.wait_for_server(
            timeout_sec=1.0
        ):
            self._node.get_logger().warn(
                f"Action server {self.__follow_joint_trajectory_action_client._action_name} is not yet ready. Better luck next time!"
            )
            return

        action_result = self.__follow_joint_trajectory_action_client.send_goal_async(
            goal=goal,
            feedback_callback=None,
        )

        action_result.add_done_callback(
            self.__response_callback_follow_joint_trajectory
        )

    def __response_callback_follow_joint_trajectory(self, response):

        goal_handle = response.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn(
                f"Action '{self.__follow_joint_trajectory_action_client._action_name}' was rejected"
            )
            return

        if self.__ignore_new_calls_while_executing:
            self.__is_executing = True

        self.__get_result_future_follow_joint_trajectory = (
            goal_handle.get_result_async()
        )
        self.__get_result_future_follow_joint_trajectory.add_done_callback(
            self.__result_callback_follow_joint_trajectory
        )

    def __result_callback_follow_joint_trajectory(self, res):

        if res.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.__toggle_internal_gripper_state()
        else:
            self._node.get_logger().error(
                f"Action '{self.__follow_joint_trajectory_action_client._action_name}' was unsuccessful: {res.result().status}"
            )

        if self.__ignore_new_calls_while_executing:
            self.__is_executing = False

    def __prepare_move_action_goal_close(self):

        self.__move_action_goal.request.goal_constraints = (
            self.__closed_goal_constraints
        )
        self.__move_action_goal.request.start_state.joint_state = (
            self.__open_joint_state
        )

    def __prepare_move_action_goal_open(self):

        self.__move_action_goal.request.goal_constraints = self.__open_goal_constraints
        self.__move_action_goal.request.start_state.joint_state = (
            self.__closed_joint_state
        )

    @classmethod
    def __init_gripper_joint_state(
        cls,
        gripper_joint_names: List[str],
        joint_positions: List[float],
    ) -> JointState:

        joint_state = JointState()

        joint_state.name = gripper_joint_names
        joint_state.position = joint_positions
        joint_state.velocity = [0.0] * len(gripper_joint_names)
        joint_state.effort = [0.0] * len(gripper_joint_names)

        return joint_state

    @classmethod
    def __init_gripper_joint_contraint(
        cls,
        gripper_joint_names: List[str],
        joint_positions: List[float],
        tolerance: float = 0.1,
    ) -> List[Constraints]:

        goal_constrants = Constraints()

        for i in range(len(gripper_joint_names)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = gripper_joint_names[i]
            joint_constraint.position = joint_positions[i]
            joint_constraint.tolerance_above = tolerance
            joint_constraint.tolerance_below = tolerance
            joint_constraint.weight = 1.0
            goal_constrants.joint_constraints.append(joint_constraint)

        return [goal_constrants]

    @classmethod
    def __init_move_action_goal(
        cls, frame_id: str, gripper_group_name: str
    ) -> MoveGroup.Goal:

        move_action_goal = MoveGroup.Goal()
        move_action_goal.request.workspace_parameters.header.frame_id = frame_id
        # move_action_goal.request.workspace_parameters.header.stamp = "Set during request"
        move_action_goal.request.workspace_parameters.min_corner.x = -1.0
        move_action_goal.request.workspace_parameters.min_corner.y = -1.0
        move_action_goal.request.workspace_parameters.min_corner.z = -1.0
        move_action_goal.request.workspace_parameters.max_corner.x = 1.0
        move_action_goal.request.workspace_parameters.max_corner.y = 1.0
        move_action_goal.request.workspace_parameters.max_corner.z = 1.0
        # move_action_goal.request.start_state = "Set during request"
        # move_action_goal.request.goal_constraints = "Set during request"
        # move_action_goal.request.path_constraints = "Ignored"
        # move_action_goal.request.trajectory_constraints = "Ignored"
        # move_action_goal.request.reference_trajectories = "Ignored"
        # move_action_goal.request.pipeline_id = "Ignored"
        # move_action_goal.request.planner_id = "Ignored"
        move_action_goal.request.group_name = gripper_group_name
        move_action_goal.request.num_planning_attempts = 2
        move_action_goal.request.allowed_planning_time = 0.1
        move_action_goal.request.max_velocity_scaling_factor = 0.0
        move_action_goal.request.max_acceleration_scaling_factor = 0.0
        # move_action_goal.request.cartesian_speed_end_effector_link = "Ignored"
        move_action_goal.request.max_cartesian_speed = 0.0

        # move_action_goal.planning_options.planning_scene_diff = "Ignored"
        move_action_goal.planning_options.plan_only = False
        # move_action_goal.planning_options.look_around = "Ignored"
        # move_action_goal.planning_options.look_around_attempts = "Ignored"
        # move_action_goal.planning_options.max_safe_execution_cost = "Ignored"
        # move_action_goal.planning_options.replan = "Ignored"
        # move_action_goal.planning_options.replan_attempts = "Ignored"
        # move_action_goal.planning_options.replan_delay = "Ignored"

        return move_action_goal

    def __try_destroy_move_action_client(self):

        if (
            self.__open_follow_joint_trajectory_req is None
            or self.__close_follow_joint_trajectory_req is None
        ):
            return

        if not (self.__use_planning_service and self.__plan_once):
            self.__move_action_client.destroy()
            del self.__move_action_client
        del self.__move_action_goal
        del self.__open_joint_state
        del self.__closed_joint_state
        del self.__open_goal_constraints
        del self.__closed_goal_constraints
