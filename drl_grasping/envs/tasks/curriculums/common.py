from __future__ import annotations

import enum
import math
from typing import Callable, Dict, Optional, Tuple, Type

from gym_ignition.base.task import Task
from gym_ignition.utils.typing import Reward
from stable_baselines3.common import logger as stable_baselines3_logger
from tf2_ros.buffer_interface import TypeException


@enum.unique
class CurriculumStage(enum.Enum):
    """
    Ordered enum that represents stages of a curriculum for RL task.
    """

    @classmethod
    def first(self) -> CurriculumStage:

        return self(1)

    @classmethod
    def last(self) -> CurriculumStage:

        return self(len(self))

    def next(self) -> Optional[CurriculumStage]:

        next_value = self.value + 1

        if next_value > self.last().value:
            return None
        else:
            return self(next_value)

    def previous(self) -> Optional[CurriculumStage]:

        previous_value = self.value - 1

        if previous_value < self.first().value:
            return None
        else:
            return self(previous_value)


class StageRewardCurriculum:
    """
    Curriculum that begins to compute rewards for a stage once all previous stages are complete.
    """

    PERSISTENT_ID: str = "PERSISTENT"

    def __init__(
        self,
        curriculum_stage: Type[CurriculumStage],
        stage_reward_multiplier: float = 1.0,
        dense_reward: bool = True,
        enable_logger_sb3: bool = True,
        **kwargs,
    ):

        if 0 == len(curriculum_stage):
            raise TypeException(f"{curriculum_stage} has length of 0")

        self.__use_dense_reward = dense_reward

        # Setup internals
        self._stage_type = curriculum_stage
        self._stages_completed_this_episode: Dict[curriculum_stage, bool] = {
            curriculum_stage(stage): False for stage in iter(curriculum_stage)
        }
        self._stage_reward_functions: Dict[curriculum_stage, Callable] = {
            curriculum_stage(stage): getattr(self, f"get_reward_{stage.name}")
            for stage in iter(curriculum_stage)
        }
        self.__stage_reward_multipliers: Dict[curriculum_stage, float] = {
            curriculum_stage(stage): stage_reward_multiplier ** (stage.value - 1)
            for stage in iter(curriculum_stage)
        }
        self.__episode_succeeded = False
        self.__episode_failed = False

        # Setup stable-baselines3 logger if desired
        self.__enable_logger_sb3 = enable_logger_sb3
        if enable_logger_sb3:
            self.__logger_sb3 = stable_baselines3_logger.Logger(
                folder=None, output_formats=[stable_baselines3_logger.KVWriter]
            )
            self.__logger_sb3_root_stage_reward = "curriculum/stage_reward/"
            self.__stages_rewards_this_episode: Dict[curriculum_stage, float] = {
                curriculum_stage(stage): 0.0 for stage in iter(curriculum_stage)
            }

    def get_reward(self, **kwargs) -> Reward:

        reward = 0.0

        # Determine the stage at which to start computing reward [performance - done stages give no reward]
        for stage in iter(self._stage_type):
            if not self._stages_completed_this_episode[stage]:
                first_stage_to_process = stage
                break

        # Iterate over all stages that might need to be processed
        for stage in range(first_stage_to_process.value, len(self._stage_type) + 1):
            stage = self._stage_type(stage)

            # Compute reward for the current stage
            stage_reward = self._stage_reward_functions[stage](**kwargs)
            # Multiply by the reward multiplier
            stage_reward *= self.__stage_reward_multipliers[stage]
            # Add to the total step reward
            reward += stage_reward
            # Add reward to the list for logger (if enabled)
            if self.__enable_logger_sb3:
                self.__stages_rewards_this_episode[stage] = stage_reward

            # Break if stage is not yet completed [performance - next stages won't give any reward]
            if not self._stages_completed_this_episode[stage]:
                break

        # If the last stage is complete, the episode has succeeded
        self.__episode_succeeded = self._stages_completed_this_episode[
            self._stage_type.last()
        ]

        # Add persistent reward that is added regardless of the episode
        persistent_reward = self.get_persistent_reward(**kwargs)
        # Add to the total step reward
        reward += persistent_reward
        # Add reward to the list for logger (if enabled)
        if self.__enable_logger_sb3:
            self.__stages_rewards_this_episode[self.PERSISTENT_ID] = persistent_reward

        # Perform logging (if enabled)
        self.__log_get_reward()

        return reward

    def is_done(self) -> bool:

        if self.__episode_succeeded:
            # The episode ended with success
            self.on_episode_success()
            return True
        elif self.__episode_failed:
            # The episode ended due to failure
            self.on_episode_failure()
            return True
        else:
            # Otherwise, the episode is not yet done
            return False

    def get_info(self) -> Dict:

        info = {
            "is_success": self.__episode_succeeded,
        }
        return info

    def reset_task(self):

        if not (self.__episode_succeeded or self.__episode_failed):
            # The episode ended due to timeout
            self.on_episode_timeout()

        # Perform logging (if enabled)
        self.__log_reset_task()

        # Reset internals
        self._stage_reward_functions = dict.fromkeys(
            self._stage_reward_functions, False
        )
        self.__episode_succeeded = False
        self.__episode_failed = False
        if self.__enable_logger_sb3:
            self.__stages_rewards_this_episode = dict.fromkeys(
                self.__stages_rewards_this_episode, 0.0
            )

    def __log_get_reward(self):

        if self.__enable_logger_sb3:
            # Reward of each stage + persistent reward
            for stage, reward in self.__stages_rewards_this_episode.items():
                self.__logger_sb3.record_mean(
                    f"{self.__logger_sb3_root_stage_reward}reward_mean_{stage}",
                    reward,
                )

    def __log_reset_task(self):

        if self.__enable_logger_sb3:
            # Reached stage
            for stage in iter(self._stage_type):
                re_stageached = stage
                if self._stages_completed_this_episode[stage]:
                    break
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root_stage_reward}reached_stage_name",
                re_stageached.name,
                exclude="tensorboard",
            )
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root_stage_reward}reached_stage_value",
                re_stageached.value,
                exclude="stdout",
            )

    @property
    def episode_succeeded(self) -> bool:

        return self.__episode_succeeded

    @episode_succeeded.setter
    def episode_succeeded(self, value: bool):

        self.__episode_succeeded = value

    @property
    def episode_failed(self) -> bool:

        return self.__episode_failed

    @episode_failed.setter
    def episode_failed(self, value: bool):

        self.__episode_failed = value

    @property
    def use_dense_reward(self) -> bool:

        return self.__use_dense_reward

    def get_persistent_reward(self, **kwargs) -> float:
        """
        Virtual method.
        """

        reward = 0.0

        return reward

    def on_episode_success(self):
        """
        Virtual method.
        """

        pass

    def on_episode_failure(self):
        """
        Virtual method.
        """

        pass

    def on_episode_timeout(self):
        """
        Virtual method.
        """

        pass


class SuccessRateImpl:
    """
    Moving average over the success rate of last N episodes.
    """

    def __init__(
        self,
        initial_success_rate: float = 0.0,
        rolling_average_n: int = 100,
        enable_logger_sb3: bool = True,
        **kwargs,
    ):

        self.__success_rate = initial_success_rate
        self.__rolling_average_n = rolling_average_n

        # Setup internals
        self.__previous_success_rate_weight: int = 0
        self.__collected_samples: int = 0

        # Setup stable-baselines3 logger if desired
        self.__enable_logger_sb3 = enable_logger_sb3
        if enable_logger_sb3:
            self.__logger_sb3 = stable_baselines3_logger.Logger(
                folder=None, output_formats=[stable_baselines3_logger.KVWriter]
            )
            self.__logger_sb3_root_success_rate = "curriculum/"

    def update_success_rate(self, is_success: bool):

        # Until `rolling_average_n` is reached, use number of collected samples during computations
        if self.__collected_samples < self.__rolling_average_n:
            self.__previous_success_rate_weight = self.__collected_samples
            self.__collected_samples += 1

        self.__success_rate = (
            self.__previous_success_rate_weight * self.__success_rate
            + float(is_success)
        ) / self.__collected_samples

        # Perform logging (if enabled)
        self.__log_success_rate()

    def __log_success_rate(self):

        if self.__enable_logger_sb3:
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root_success_rate}success_rate",
                self.__success_rate,
            )

    @property
    def success_rate(self) -> float:

        return self.__success_rate


class WorkspaceScaleCurriculum:
    """
    Curriculum that increases the workspace size as the success rate increases.
    """

    def __init__(
        self,
        task: Task,
        success_rate_impl: SuccessRateImpl,
        min_workspace_scale: float,
        max_workspace_volume: Tuple[float, float, float],
        max_workspace_scale_success_rate_threshold: float,
        enable_logger_sb3: bool = True,
        **kwargs,
    ):

        self.__task = task
        self.__success_rate_impl = success_rate_impl
        self.__min_workspace_scale = min_workspace_scale
        self.__max_workspace_volume = max_workspace_volume
        self.__max_workspace_scale_success_rate_threshold = (
            max_workspace_scale_success_rate_threshold
        )

        # Setup stable-baselines3 logger if desired
        self.__enable_logger_sb3 = enable_logger_sb3
        if enable_logger_sb3:
            self.__logger_sb3 = stable_baselines3_logger.Logger(
                folder=None, output_formats=[stable_baselines3_logger.KVWriter]
            )
            self.__logger_sb3_root = "curriculum/"

    def reset_task(self):

        # Update workspace size
        self.__update_workspace_size()

        # Perform logging (if enabled)
        self.__log_workspace_scale()

    def __update_workspace_size(self):

        self.__workspace_scale = min(
            1.0,
            max(
                self.__min_workspace_scale,
                self.__success_rate_impl.success_rate
                / self.__max_workspace_scale_success_rate_threshold,
            ),
        )

        workspace_volume_new = (
            self.__workspace_scale * self.__max_workspace_volume[0],
            self.__workspace_scale * self.__max_workspace_volume[1],
            # Z workspace is currently kept the same on purpose
            self.__max_workspace_volume[2],
        )
        workspace_volume_half_new = (
            workspace_volume_new[0] / 2,
            workspace_volume_new[1] / 2,
            workspace_volume_new[2] / 2,
        )
        workspace_min_bound_new = (
            self.__task.workspace_centre[0] - workspace_volume_half_new[0],
            self.__task.workspace_centre[1] - workspace_volume_half_new[1],
            self.__task.workspace_centre[2] - workspace_volume_half_new[2],
        )
        workspace_max_bound_new = (
            self.__task.workspace_centre[0] + workspace_volume_half_new[0],
            self.__task.workspace_centre[1] + workspace_volume_half_new[1],
            self.__task.workspace_centre[2] + workspace_volume_half_new[2],
        )

        self.__task.add_task_parameter_overrides(
            {
                "workspace_volume": workspace_volume_new,
                "workspace_min_bound": workspace_min_bound_new,
                "workspace_max_bound": workspace_max_bound_new,
            }
        )

    def __log_workspace_scale(self):

        if self.__enable_logger_sb3:
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root}workspace_scale",
                self.__workspace_scale,
            )


class ObjectSpawnVolumeScaleCurriculum:
    """
    Curriculum that increases the object spawn volume as the success rate increases.
    """

    def __init__(
        self,
        task: Task,
        success_rate_impl: SuccessRateImpl,
        min_object_spawn_volume_scale: float,
        max_object_spawn_volume: Tuple[float, float, float],
        max_object_spawn_volume_scale_success_rate_threshold: float,
        enable_logger_sb3: bool = True,
        **kwargs,
    ):

        self.__task = task
        self.__success_rate_impl = success_rate_impl
        self.__min_object_spawn_volume_scale = min_object_spawn_volume_scale
        self.__max_object_spawn_volume = max_object_spawn_volume
        self.__max_object_spawn_volume_scale_success_rate_threshold = (
            max_object_spawn_volume_scale_success_rate_threshold
        )

        # Setup stable-baselines3 logger if desired
        self.__enable_logger_sb3 = enable_logger_sb3
        if enable_logger_sb3:
            self.__logger_sb3 = stable_baselines3_logger.Logger(
                folder=None, output_formats=[stable_baselines3_logger.KVWriter]
            )
            self.__logger_sb3_root = "curriculum/"

    def reset_task(self):

        # Update object_spawn_volume size
        self.__update_object_spawn_volume_size()

        # Perform logging (if enabled)
        self.__log_object_spawn_volume_scale()

    def __update_object_spawn_volume_size(self):

        self.__object_spawn_volume_scale = min(
            1.0,
            max(
                self.__min_object_spawn_volume_scale,
                self.__success_rate_impl.success_rate
                / self.__max_object_spawn_volume_scale_success_rate_threshold,
            ),
        )

        object_spawn_volume_volume_new = (
            self.__object_spawn_volume_scale * self.__max_object_spawn_volume[0],
            self.__object_spawn_volume_scale * self.__max_object_spawn_volume[1],
            self.__object_spawn_volume_scale * self.__max_object_spawn_volume[2],
        )

        self.__task.add_task_parameter_overrides(
            {
                "_object_random_spawn_volume": object_spawn_volume_volume_new,
            }
        )

    def __log_object_spawn_volume_scale(self):

        if self.__enable_logger_sb3:
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root}object_spawn_volume_scale",
                self.__object_spawn_volume_scale,
            )


class ObjectCountCurriculum:
    """
    Curriculum that increases the number of objects as the success rate increases.
    """

    def __init__(
        self,
        task: Task,
        success_rate_impl: SuccessRateImpl,
        object_count_min: int,
        object_count_max: int,
        max_object_count_success_rate_threshold: float,
        enable_logger_sb3: bool = True,
        **kwargs,
    ):

        self.__task = task
        self.__success_rate_impl = success_rate_impl
        self.__object_count_min = object_count_min
        self.__object_count_max = object_count_max
        self.__max_object_count_success_rate_threshold = (
            max_object_count_success_rate_threshold
        )

        self.__object_count_min_max_diff = object_count_max - object_count_min
        if self.__object_count_min_max_diff < 0:
            raise Exception(
                "'object_count_min' cannot be larger than 'object_count_max'"
            )

        # Setup stable-baselines3 logger if desired
        self.__enable_logger_sb3 = enable_logger_sb3
        if enable_logger_sb3:
            self.__logger_sb3 = stable_baselines3_logger.Logger(
                folder=None, output_formats=[stable_baselines3_logger.KVWriter]
            )
            self.__logger_sb3_root = "curriculum/"

    def reset_task(self):

        # Update object count
        self.__update_object_count()

        # Perform logging (if enabled)
        self.__log_object_count()

    def __update_object_count(self):

        self.__object_count = min(
            self.__object_count_max,
            math.floor(
                self.__object_count_min
                + (
                    self.__success_rate_impl.success_rate
                    / self.__max_object_count_success_rate_threshold
                )
                * self.__object_count_min_max_diff
            ),
        )

        self.__task.add_randomizer_parameter_overrides(
            {
                "object_count": self.__object_count,
            }
        )

    def __log_object_count(self):

        if self.__enable_logger_sb3:
            self.__logger_sb3.record(
                f"{self.__logger_sb3_root}object_count",
                self.__object_count,
            )
