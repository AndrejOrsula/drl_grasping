import abc

from drl_grasping.envs.tasks.grasp import Grasp

# Note: GraspPlanetary task is currently identical to the Grasp task


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
