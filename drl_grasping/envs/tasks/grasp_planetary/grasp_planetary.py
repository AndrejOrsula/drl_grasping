import abc

from drl_grasping.envs.models.robots import LunalabSummitXlGen
from drl_grasping.envs.tasks.grasp import Grasp


class GraspPlanetary(Grasp, abc.ABC):
    def __init__(
        self,
        **kwargs,
    ):

        # Initialize the Task base class
        Grasp.__init__(
            self,
            **kwargs,
        )

        # Overwrite initial joint positions for certain robots
        if LunalabSummitXlGen == self.robot_model_class:
            # self.initial_arm_joint_positions = [
            #     1.0471975511965976,
            #     2.356194490192345,
            #     0.0,
            #     4.71238898038469,
            #     0.0,
            #     2.356194490192345,
            #     0.0,
            # ]
            self.initial_arm_joint_positions = [
                0.0,
                2.356194490192345,
                0.0,
                4.71238898038469,
                0.0,
                2.356194490192345,
                0.0,
            ]
