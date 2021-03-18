from __future__ import annotations
from stable_baselines3.common import logger
from typing import Union, Type, Dict
import enum


@enum.unique
class GraspStage(enum.Enum):
    """
    Ordered enum representing the different stages of grasp curriculum.
    Each subsequent stage is entered only after the previous one reaches the desired success rate.
    """

    REACH = enum.auto()
    TOUCH = enum.auto()
    GRASP = enum.auto()
    LIFT = enum.auto()

    @classmethod
    def first(self) -> Type[GraspStage]:

        return GraspStage(1)

    @classmethod
    def last(self) -> Type[GraspStage]:

        return GraspStage(len(GraspStage))

    def next(self) -> Union[Type[GraspStage], None]:

        next_value = self.value + 1

        if next_value > self.last().value:
            return None
        else:
            return GraspStage(next_value)

    def previous(self) -> Union[Type[GraspStage], None]:

        previous_value = self.value - 1

        if previous_value < self.first().value:
            return None
        else:
            return GraspStage(previous_value)


class GraspCurriculum():
    """
    Curriculum learning implementation for grasp task that provides termination (success/fail) and reward for each stage of the task.
    """

    # TODO: (Maybe) Add curriculum learning also for the goals that need to be reached at each stage (e.g. specify as range that would be adjusted according to the success rate)
    # This might just make it unnecessarily complicated with little benefit, since it would be very similar to having a dense (shaped) reward

    def __init__(self,
                 task,
                 enabled: bool,
                 sparse_reward: bool,
                 required_reach_distance: float,
                 required_lift_height: float,
                 act_quick_reward: float,
                 success_rate_threshold: float,
                 success_rate_rolling_average_n: int,
                 stage_reward_multiplier: float,
                 restart_every_n_steps: int,
                 min_workspace_scale: float,
                 scale_negative_rewards: bool,
                 verbose: bool = False):

        # Grasp task/environment that will be used to extract information from the scene
        self.task = task

        # Parameters
        self._enabled = enabled
        self._sparse_reward = sparse_reward
        self._success_rate_threshold = success_rate_threshold
        self._success_rate_moving_average_n = success_rate_rolling_average_n
        self._stage_reward_multiplier = stage_reward_multiplier
        self._restart_every_n_steps = restart_every_n_steps
        self._min_workspace_scale = min_workspace_scale
        self._scale_negative_rewards = scale_negative_rewards
        self._verbose = verbose

        # Requirements
        self._required_reach_distance = required_reach_distance
        self._required_lift_height = required_lift_height

        # Small constant negative reward designed to act quickly (value is subtracted)
        self._act_quick_reward = act_quick_reward if act_quick_reward >= 0.0 else -act_quick_reward

        # Current stage of curriculum
        self._stage: GraspStage = GraspStage.first() if self._enabled else GraspStage.last()
        # Dict of bools determining if stage has been completed in the current episode (dict excludes the last stage)
        self._stage_completed: Dict[GraspStage, bool] = {GraspStage(stage): False
                                                         for stage in range(GraspStage.first().value,
                                                                            GraspStage.last().value + 1)}
        # Success rate (moving average over last N episodes)
        self._success_rate: float = 0.0
        # Flag that determines if the current episode ended with success
        self._is_success: bool = False
        # Flag that determines if the current episode got terminated due to failure
        self._is_failure: bool = False

        # Distance to closest object in the previous step (or after reset)
        # Used for REACH sub-task
        self._previous_min_distance: float = None
        # Heights of objects in the scene in the previous step (or after reset)
        # Used for LIFT sub-task
        self._previous_object_heights: Dict[str, float] = {}

        # Counter of steps
        self._reset_step_counter = self._restart_every_n_steps

        self._restart_exploration = False

    def update_success_rate(self, is_success: bool):
        """
        Update success rate using moving average over last N episodes.
        """

        self._success_rate = ((self._success_rate_moving_average_n - 1) * self._success_rate +
                              float(is_success)) / self._success_rate_moving_average_n

        if self._verbose:
            print(f"Average success rate (n={self._success_rate_moving_average_n}) "
                  f"= {self._success_rate}")

        scale = max(self._min_workspace_scale,
                    self._success_rate / self._success_rate_threshold)
        self.task.update_workspace_size(scale)

    def maybe_next_stage(self) -> bool:
        """
        If success rate is high enough, transition into the next stage.
        """

        if GraspStage.last() == self._stage:
            return False

        if self._success_rate > self._success_rate_threshold:
            print("Curriculum stage change:\n"
                  f"\tAverage success rate ({self._success_rate}) is now higher than the threshold ({self._success_rate_threshold}). "
                  f"Moving onto the next stage ({self._stage.next()}). "
                  f"Success rate reset to {0.0}.")
            self._stage = self._stage.next()
            self._success_rate = 0.0
            self._restart_exploration = True
            return True

        return False

    def maybe_restart_to_first_stage(self) -> bool:
        """
        If enabled (restart_every_n_steps is positive), start again from the first stage once in a while every X episodes.
        The countdown begins once second episode has been reached.
        The initial success rate will be reset to `0.5*success_rate_threshold` (fewer samples are needed).
        """

        if self._reset_step_counter <= 0:
            print("Curriculum stage change:\n"
                  f"\tRestarting to first stage ({GraspStage.first()}) after {self._restart_every_n_steps} episodes "
                  f"have elapsed since reaching the second stage ({GraspStage(GraspStage.first().value+1)}). "
                  f"Success rate reset to {0.5*self._success_rate_threshold}.")
            self._stage = GraspStage.first()
            self._success_rate = 0.5*self._success_rate_threshold
            self._reset_step_counter = self._restart_every_n_steps
            self._restart_exploration = True
            return True

        return False

    def get_reward(self) -> float:
        """
        Get reward for the current stage, as well as a discounted reward for previous stages (determined by stage_reward_multiplier arg)
        Should be called with inside `self.task.get_reward()`
        """

        reward = 0.0

        # Determine the stage at which to start computing reward (do not compute reward for stages that already succeded in this episode)
        first_stage_to_process = GraspStage.last().value
        for stage in range(GraspStage.first().value, GraspStage.last().value):
            if not self._stage_completed[GraspStage(stage)]:
                first_stage_to_process = stage
                break

        # Data used in multiple stages is pushed to kwargs
        kwargs = {}
        # object_positions are needed for ALL (+ REACH and LIFT)
        kwargs['object_positions'] = self.task.get_object_positions()

        # Iterate over all stages that the reward needs to be computed for
        for stage in range(first_stage_to_process, GraspStage.last().value + 1):

            # grasped_objects are needed for GRASP and LIFT, add once reached
            if GraspStage.GRASP == GraspStage(stage):
                kwargs['grasped_objects'] = self.task.get_grasped_objects()

            # Reward multiplier based on stage
            reward_multiplier = self._stage_reward_multiplier**(stage-1)

            # Compute and add the reward for this stage
            reward += reward_multiplier * \
                self.GET_REWARD[GraspStage(stage)](self, **kwargs)

            # Break if currently computed stage is not completed, as the next stage will not get any reward anyway
            if not self._stage_completed[GraspStage(stage)]:
                break

        # Compute also (negative) reward that is common in each episode
        reward_multiplier = 1.0
        if self._scale_negative_rewards:
            reward_multiplier = self._stage_reward_multiplier**(
                self._stage.value - 1)
        reward += reward_multiplier * self._get_reward_ALL(**kwargs)

        return reward

    def is_done(self) -> bool:
        """
        Should be called inside `self.task.is_done()`
        """

        if self._is_success:
            self.update_success_rate(is_success=True)
            return True
        elif self._is_failure:
            self.update_success_rate(is_success=False)
            return True
        else:
            return False

    def get_info(self) -> Dict:
        """
        Should be called inside `self.task.get_info()`
        """

        if self._stage != GraspStage.first() and self._restart_every_n_steps > 0:
            self._reset_step_counter -= 1
            self.maybe_restart_to_first_stage()

        info = {'curriculum.restart_exploration': self._restart_exploration}
        self._restart_exploration = False

        return info

    def reset_task(self):
        """
        Should be called with inside `self.task.reset_task()`
        """

        # Update with a failed attempt if episode was reset due to reaching timeout (max episode steps)
        if not (self._is_success or self._is_failure):
            self.update_success_rate(is_success=False)

        # Check if there is a need to change curriculum stage
        self.maybe_next_stage()

        # Add curriculum-specific data to logs
        self._log_curriculum()

        self._is_success = False
        self._is_failure = False
        for stage in range(GraspStage.first().value, GraspStage.last().value + 1):
            self._stage_completed[GraspStage(stage)] = False

        if not self._sparse_reward:
            # Get current positions of all objects in the scene
            object_positions = self.task.get_object_positions()

            # Get distance to the closest object after the reset
            # Part of REACH, therefore always enabled
            self._previous_min_distance = self.task.get_closest_object_distance(
                object_positions)

            # Get height of all objects in the scene after the reset
            self._previous_object_heights.clear()
            for object_name, object_position in object_positions.items():
                self._previous_object_heights[object_name] = object_position[2]

    def _get_reward_REACH(self, **kwargs) -> float:

        object_positions = kwargs['object_positions']
        current_min_distance = self.task.get_closest_object_distance(
            object_positions)
        if current_min_distance < self._required_reach_distance:
            self._is_success = True if GraspStage.REACH == self._stage else self._is_success
            self._stage_completed[GraspStage.REACH] = True
            if self._sparse_reward:
                return 1.0

        if self._sparse_reward:
            # Sparse reward without success
            return 0.0
        else:
            # Dense reward
            reward = self._previous_min_distance - current_min_distance
            self._previous_min_distance = current_min_distance
            return reward

    def _get_reward_TOUCH(self, **kwargs) -> float:
        # TODO: If needed, engineer a dense reward for TOUCH sub-task

        touched_objects = self.task.get_touched_objects()
        if len(touched_objects) > 0:
            self._is_success = True if GraspStage.TOUCH == self._stage else self._is_success
            self._stage_completed[GraspStage.TOUCH] = True
            return 1.0
        else:
            return 0.0

    def _get_reward_GRASP(self, **kwargs) -> float:
        # TODO: If needed, engineer a dense reward for GRASP sub-task

        grasped_objects = kwargs['grasped_objects']
        if len(grasped_objects) > 0:
            self._is_success = True if GraspStage.GRASP == self._stage else self._is_success
            self._stage_completed[GraspStage.GRASP] = True
            if self._verbose:
                print(f"Object(s) grasped: {grasped_objects}")
            return 1.0
        else:
            return 0.0

    def _get_reward_LIFT(self, **kwargs) -> float:

        grasped_objects = kwargs['grasped_objects']
        if 0 == len(grasped_objects):
            return 0.0

        reward = 0.0

        object_positions = kwargs['object_positions']
        for grasped_object in grasped_objects:
            if object_positions[grasped_object][2] > self._required_lift_height:
                self._is_success = True if GraspStage.LIFT == self._stage else self._is_success
                self._stage_completed[GraspStage.LIFT] = True
                if self._sparse_reward:
                    reward += 1.0

            if not self._sparse_reward:
                reward += object_positions[grasped_object][2] - \
                    self._previous_object_heights[grasped_object]

        if not self._sparse_reward:
            # Update all object heights
            for object_name, object_position in object_positions.items():
                self._previous_object_heights[object_name] = object_position[2]

        return reward

    def _get_reward_ALL(self, **kwargs) -> float:

        # Subtract a small reward each step to provide incentive to act quickly
        reward = -self._act_quick_reward

        # Terminate and return reward of -1.0 if robot collides with the ground plane
        if self.task.check_ground_collision():
            self._is_failure = True
            reward -= 1.0
            if self._verbose:
                print("Robot collided with the ground plane. Terminating episode...")
            return reward

        # If all objects are outside of workspace, terminate and return -1.0
        object_positions = kwargs['object_positions']
        if self.task.check_all_objects_outside_workspace(object_positions):
            self._is_failure = True
            reward -= 1.0
            if self._verbose:
                print("All objects are outside of the workspace. Terminating episode...")
            return reward

        return reward

    def _log_curriculum(self):
        logger.record("curriculum/current_stage",
                      self._stage, exclude="tensorboard")
        logger.record("curriculum/current_stage_id",
                      self._stage.value, exclude="stdout")
        logger.record("curriculum/success_rate",
                      self._success_rate)
        if self._restart_every_n_steps > 0:
            logger.record("curriculum/steps_until_reset",
                          self._reset_step_counter)

    GET_REWARD = {
        GraspStage.REACH: _get_reward_REACH,
        GraspStage.TOUCH: _get_reward_TOUCH,
        GraspStage.GRASP: _get_reward_GRASP,
        GraspStage.LIFT: _get_reward_LIFT,
    }
