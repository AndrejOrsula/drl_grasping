from typing import Dict, List, Tuple

from gym_ignition.base.task import Task

from drl_grasping.envs.tasks.curriculums.common import *
from drl_grasping.envs.utils.math import distance_to_nearest_point


class GraspStage(CurriculumStage):
    """
    Ordered enum that represents stages of a curriculum for Grasp (and GraspPlanetary) task.
    """

    REACH = 1
    TOUCH = 2
    GRASP = 3
    LIFT = 4


class GraspCurriculum(
    StageRewardCurriculum,
    SuccessRateImpl,
    WorkspaceScaleCurriculum,
    ObjectSpawnVolumeScaleCurriculum,
    ObjectCountCurriculum,
):
    """
    Curriculum learning implementation for grasp task that provides termination (success/fail) and reward for each stage of the task.
    """

    def __init__(
        self,
        task: Task,
        stages_base_reward: float,
        reach_required_distance: float,
        lift_required_height: float,
        persistent_reward_each_step: float,
        persistent_reward_terrain_collision: float,
        persistent_reward_all_objects_outside_workspace: float,
        enable_workspace_scale_curriculum: bool,
        enable_stage_reward_curriculum: bool,
        enable_object_spawn_volume_scale_curriculum: bool,
        enable_object_count_curriculum: bool,
        **kwargs,
    ):

        StageRewardCurriculum.__init__(self, curriculum_stage=GraspStage, **kwargs)
        SuccessRateImpl.__init__(self, **kwargs)
        WorkspaceScaleCurriculum.__init__(
            self, task=task, success_rate_impl=self, **kwargs
        )
        ObjectSpawnVolumeScaleCurriculum.__init__(
            self, task=task, success_rate_impl=self, **kwargs
        )
        ObjectCountCurriculum.__init__(
            self, task=task, success_rate_impl=self, **kwargs
        )

        # Grasp task/environment that will be used to extract information from the scene
        self.__task = task

        # Parameters
        self.__stages_base_reward = stages_base_reward
        self.__reach_required_distance = reach_required_distance
        self.__lift_required_height = lift_required_height
        self.__persistent_reward_each_step = persistent_reward_each_step
        self.__persistent_reward_terrain_collision = persistent_reward_terrain_collision
        self.__persistent_reward_all_objects_outside_workspace = (
            persistent_reward_all_objects_outside_workspace
        )
        self.__enable_stage_reward_curriculum = enable_stage_reward_curriculum
        self.__enable_workspace_scale_curriculum = enable_workspace_scale_curriculum
        self.__enable_object_spawn_volume_scale_curriculum = (
            enable_object_spawn_volume_scale_curriculum
        )
        self.__enable_object_count_curriculum = enable_object_count_curriculum

        # Make sure that the persistent rewards for each step are negative
        if self.__persistent_reward_each_step > 0.0:
            self.__persistent_reward_each_step *= -1.0
        if self.__persistent_reward_terrain_collision > 0.0:
            self.__persistent_reward_terrain_collision *= -1.0

    def get_reward(self) -> Reward:

        if self.__enable_stage_reward_curriculum:
            # Try to get reward from each stage
            return StageRewardCurriculum.get_reward(
                self,
                ee_position=self.__task.get_ee_position(),
                object_positions=self.__task.get_object_positions(),
                touched_objects=self.__task.get_touched_objects(),
                grasped_objects=self.__task.get_grasped_objects(),
            )
        else:
            # If curriculum is disabled, compute reward only for the last stage
            return self.get_reward_LIFT(
                object_positions=self.__task.get_object_positions(),
                grasped_objects=self.__task.get_grasped_objects(),
            )

    def is_done(self) -> bool:

        return StageRewardCurriculum.is_done(self)

    def get_info(self) -> Dict:

        return StageRewardCurriculum.get_info(self)

    def reset_task(self):

        StageRewardCurriculum.reset_task(self)
        if self.__enable_workspace_scale_curriculum:
            WorkspaceScaleCurriculum.reset_task(self)
        if self.__enable_object_spawn_volume_scale_curriculum:
            ObjectSpawnVolumeScaleCurriculum.reset_task(self)
        if self.__enable_object_count_curriculum:
            ObjectCountCurriculum.reset_task(self)

    def on_episode_success(self):

        self.update_success_rate(is_success=True)

    def on_episode_failure(self):

        self.update_success_rate(is_success=False)

    def on_episode_timeout(self):

        self.update_success_rate(is_success=False)

    def get_reward_REACH(
        self,
        ee_position: Tuple[float, float, float],
        object_positions: Dict[str, Tuple[float, float, float]],
        **kwargs,
    ) -> float:

        if not object_positions:
            return 0.0

        nearest_object_distance = distance_to_nearest_point(
            origin=ee_position, points=list(object_positions.values())
        )

        self.__task.get_logger().debug(
            f"[Curriculum] Distance to nearest object: {nearest_object_distance}"
        )
        if nearest_object_distance < self.__reach_required_distance:
            self.__task.get_logger().info(
                f"[Curriculum] An object is now closer than the required distance of {self.__reach_required_distance}"
            )
            self.stages_completed_this_episode[GraspStage.REACH] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_TOUCH(self, touched_objects: List[str], **kwargs) -> float:

        if touched_objects:
            self.__task.get_logger().info(
                f"[Curriculum] Touched objects: {touched_objects}"
            )
            self.stages_completed_this_episode[GraspStage.TOUCH] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_GRASP(self, grasped_objects: List[str], **kwargs) -> float:

        if grasped_objects:
            self.__task.get_logger().info(
                f"[Curriculum] Grasped objects: {grasped_objects}"
            )
            self.stages_completed_this_episode[GraspStage.GRASP] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_LIFT(
        self,
        object_positions: Dict[str, Tuple[float, float, float]],
        grasped_objects: List[str],
        **kwargs,
    ) -> float:

        if not (grasped_objects or object_positions):
            return 0.0

        for grasped_object in grasped_objects:
            grasped_object_height = object_positions[grasped_object][2]

            self.__task.get_logger().debug(
                f"[Curriculum] Height of grasped object '{grasped_objects}': {grasped_object_height}"
            )
            if grasped_object_height > self.__lift_required_height:
                self.__task.get_logger().info(
                    f"[Curriculum] Lifted object: {grasped_object}"
                )
                self.stages_completed_this_episode[GraspStage.LIFT] = True
                return self.__stages_base_reward

        return 0.0

    def get_persistent_reward(
        self, object_positions: Dict[str, Tuple[float, float, float]], **kwargs
    ) -> float:

        # Subtract a small reward each step to provide incentive to act quickly
        reward = self.__persistent_reward_each_step

        # Negative reward for colliding with terrain
        if self.__persistent_reward_terrain_collision:
            if self.__task.check_terrain_collision():
                reward += self.__persistent_reward_terrain_collision

        # Negative reward for having all objects outside of the workspace
        if self.__persistent_reward_all_objects_outside_workspace:
            if self.__task.check_all_objects_outside_workspace(
                object_positions=object_positions
            ):
                reward += self.__persistent_reward_all_objects_outside_workspace
                self.episode_failed = True

        return reward
