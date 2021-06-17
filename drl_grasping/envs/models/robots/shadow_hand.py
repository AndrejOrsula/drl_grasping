from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple
from os import path


class ShadowHand(model_wrapper.ModelWrapper,
                 model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'shadow_hand',
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_file: str = None,
                 use_fuel: bool = True,
                 initial_joint_positions: List[float] = (0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0,
                                                         0.0)):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(fuel=use_fuel)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Set initial joint configuration
        self.__set_initial_joint_positions(initial_joint_positions)
        if not model.to_gazebo().reset_joint_positions(self.get_initial_joint_positions(),
                                                       self.get_joint_names()):
            raise RuntimeError("Failed to set initial robot joint positions")

        assert model.set_joint_control_mode(scenario.JointControlMode_force)
        assert model.set_controller_period(900.0)

        # # Add JointStatePublisher to shadow_hand
        # self.__add_joint_state_publisher(model)

        # # Add JointTrajectoryController to shadow_hand
        # self.__add_joint_trajectory_controller(model)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=True) -> str:
        # Warning: Model with fixed forearm and wrist joints should be used
        
        # if fuel:
        #     return scenario_gazebo.get_model_file_from_fuel(
        #         "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/shadow_hand")
        # else:
        return "shadow_hand"


    @classmethod
    def get_joint_names(self) -> List[str]:
        return ["thumb_joint1",
                "thumb_joint2",
                "thumb_joint3",
                "thumb_joint4",
                "thumb_joint5",
                "index_finger_joint1",
                "index_finger_joint2",
                "index_finger_joint3",
                "index_finger_joint4",
                "middle_finger_joint1",
                "middle_finger_joint2",
                "middle_finger_joint3",
                "middle_finger_joint4",
                "ring_finger_joint1",
                "ring_finger_joint2",
                "ring_finger_joint3",
                "ring_finger_joint4",
                "little_finger_joint1",
                "little_finger_joint2",
                "little_finger_joint3",
                "little_finger_joint4",
                "little_finger_joint5"]

    @classmethod
    def get_base_link_name(self) -> str:
        return "forearm"

    def get_initial_joint_positions(self) -> List[float]:
        return self.__initial_joint_positions

    def __set_initial_joint_positions(self, initial_joint_positions):
        self.__initial_joint_positions = initial_joint_positions

    def __add_joint_state_publisher(self, model) -> bool:
        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-state-publisher-system.so",
            "ignition::gazebo::systems::JointStatePublisher",
            self.__get_joint_state_publisher_config()
        )

    @classmethod
    def __get_joint_state_publisher_config(self) -> str:
        return \
            """
            <sdf version="1.7">
            %s
            </sdf>
            """ \
            % " ".join(("<joint_name>" + joint + "</joint_name>" for joint in self.get_joint_names()))

    def __add_joint_trajectory_controller(self, model) -> bool:
        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-trajectory-controller-system.so",
            "ignition::gazebo::systems::JointTrajectoryController",
            self.__get_joint_trajectory_controller_config()
        )

    def __get_joint_trajectory_controller_config(self) -> str:
        # TODO: refactor into something more sensible
        return \
            """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>
            
            <joint_name>forearm_joint</joint_name>
                <position_cmd_min>-10</position_cmd_min>
                <position_cmd_max>10</position_cmd_max>

                <joint_name>wrist_joint</joint_name>
                <position_cmd_min>-5</position_cmd_min>
                <position_cmd_max>5</position_cmd_max>

                <joint_name>thumb_joint1</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>thumb_joint2</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>thumb_joint3</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>thumb_joint4</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>thumb_joint5</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>index_finger_joint1</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>index_finger_joint2</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>index_finger_joint3</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>index_finger_joint4</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>middle_finger_joint1</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>middle_finger_joint2</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>middle_finger_joint3</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>middle_finger_joint4</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>ring_finger_joint1</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>ring_finger_joint2</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>ring_finger_joint3</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>ring_finger_joint4</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>little_finger_joint1</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>little_finger_joint2</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>little_finger_joint3</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>little_finger_joint4</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>

                <joint_name>little_finger_joint5</joint_name>
                <position_cmd_min>-4</position_cmd_min>
                <position_cmd_max>4</position_cmd_max>
            </sdf>
            """
