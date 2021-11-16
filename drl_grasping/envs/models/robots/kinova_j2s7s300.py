from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple
from os import path


class KinovaJ2s7s300(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):
    def __init__(
        self,
        world: scenario.World,
        name: str = "j2s7s300",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_file: str = None,
        use_fuel: bool = True,
        arm_collision: bool = True,
        hand_collision: bool = True,
        separate_gripper_controller: bool = True,
        initial_joint_positions: List[float] = (
            3.14159,
            3.14159,
            3.14159,
            3.14159,
            3.14159,
            3.14159,
            3.14159,
            0.0,
            0.0,
            0.0,
        ),
        **kwargs,
    ):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(fuel=use_fuel)

        if not arm_collision or not hand_collision:
            model_file = self.disable_collision(
                model_file=model_file,
                arm_collision=arm_collision,
                hand_collision=hand_collision,
            )

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file, initial_pose, model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        self.__separate_gripper_controller = separate_gripper_controller

        # Set initial joint configuration
        self.__set_initial_joint_positions(initial_joint_positions)
        if not model.to_gazebo().reset_joint_positions(
            self.get_initial_joint_positions(), self.get_joint_names()
        ):
            raise RuntimeError("Failed to set initial robot joint positions")

        # Add JointStatePublisher to Panda
        self.__add_joint_state_publisher(model)

        # Add JointTrajectoryController to Panda
        self.__add_joint_trajectory_controller(model)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=True) -> str:
        if fuel:
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/kinova_j2s7s300"
            )
        else:
            return "kinova_j2s7s300"

    @classmethod
    def get_joint_names(self) -> List[str]:
        return [
            "j2s7s300_joint_1",
            "j2s7s300_joint_2",
            "j2s7s300_joint_3",
            "j2s7s300_joint_4",
            "j2s7s300_joint_5",
            "j2s7s300_joint_6",
            "j2s7s300_joint_7",
            "j2s7s300_joint_finger_1",
            "j2s7s300_joint_finger_2",
            "j2s7s300_joint_finger_3",
        ]

    @classmethod
    def get_joint_limits(self) -> List[Tuple[float, float]]:
        return [
            (-31.415927, 31.415927),
            (0.820304748437, 5.46288055874),
            (-31.415927, 31.415927),
            (0.5235987755980001, 5.75958653158),
            (-31.415927, -31.415927),
            (1.1344640138, 5.14872129338),
            (-31.415927, 31.415927),
            (0.0, 1.3),
            (0.0, 1.3),
            (0.0, 1.3),
        ]

    @classmethod
    def get_base_link_name(self) -> str:
        return "j2s7s300_link_base"

    @classmethod
    def get_ee_link_name(self) -> str:
        return "j2s7s300_end_effector"

    @classmethod
    def get_gripper_link_names(self) -> List[str]:
        return [
            "j2s7s300_link_finger_1",
            "j2s7s300_link_finger_2",
            "j2s7s300_link_finger_3",
        ]

    @classmethod
    def get_finger_count(self) -> int:
        return 3

    def get_initial_joint_positions(self) -> List[float]:
        return self.__initial_joint_positions

    def __set_initial_joint_positions(self, initial_joint_positions):
        self.__initial_joint_positions = initial_joint_positions

    def __add_joint_state_publisher(self, model) -> bool:
        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-state-publisher-system.so",
            "ignition::gazebo::systems::JointStatePublisher",
            self.__get_joint_state_publisher_config(),
        )

    @classmethod
    def __get_joint_state_publisher_config(self) -> str:
        return """
            <sdf version="1.7">
            %s
            </sdf>
            """ % " ".join(
            (
                "<joint_name>" + joint + "</joint_name>"
                for joint in self.get_joint_names()
            )
        )

    def __add_joint_trajectory_controller(self, model) -> bool:
        """Add JointTrajectoryController"""
        if self.__separate_gripper_controller:
            model.to_gazebo().insert_model_plugin(
                "libignition-gazebo-joint-trajectory-controller-system.so",
                "ignition::gazebo::systems::JointTrajectoryController",
                self.__get_joint_trajectory_controller_config_joints_only(),
            )
            model.to_gazebo().insert_model_plugin(
                "libignition-gazebo-joint-trajectory-controller-system.so",
                "ignition::gazebo::systems::JointTrajectoryController",
                self.__get_joint_trajectory_controller_config_gripper_only(),
            )
        else:
            model.to_gazebo().insert_model_plugin(
                "libignition-gazebo-joint-trajectory-controller-system.so",
                "ignition::gazebo::systems::JointTrajectoryController",
                self.__get_joint_trajectory_controller_config(),
            )

    def __get_joint_trajectory_controller_config(self) -> str:
        # TODO: refactor into something more sensible
        return """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>500</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>2000</position_i_gain>
            <position_i_min>-7</position_i_min>
            <position_i_max>7</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>1100</position_p_gain>
            <position_d_gain>25</position_d_gain>
            <position_i_gain>5000</position_i_gain>
            <position_i_min>-9</position_i_min>
            <position_i_max>9</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>950</position_p_gain>
            <position_d_gain>17.5</position_d_gain>
            <position_i_gain>3500</position_i_gain>
            <position_i_min>-8</position_i_min>
            <position_i_max>8</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s1.57</initial_position>
            <position_p_gain>800</position_p_gain>
            <position_d_gain>12.5</position_d_gain>
            <position_i_gain>2500</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>-5</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>750</position_p_gain>
            <position_d_gain>7.5</position_d_gain>
            <position_i_gain>2000</position_i_gain>
            <position_i_min>-6</position_i_min>
            <position_i_max>6</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>600</position_p_gain>
            <position_d_gain>6</position_d_gain>
            <position_i_gain>1800</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>5</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>400</position_p_gain>
            <position_d_gain>4</position_d_gain>
            <position_i_gain>1200</position_i_gain>
            <position_i_min>-3.5</position_i_min>
            <position_i_max>3.5</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>
            </sdf>
            """ % (
            self.get_joint_names()[0],
            str(self.get_initial_joint_positions()[0]),
            self.get_joint_names()[1],
            str(self.get_initial_joint_positions()[1]),
            self.get_joint_names()[2],
            str(self.get_initial_joint_positions()[2]),
            self.get_joint_names()[3],
            str(self.get_initial_joint_positions()[3]),
            self.get_joint_names()[4],
            str(self.get_initial_joint_positions()[4]),
            self.get_joint_names()[5],
            str(self.get_initial_joint_positions()[5]),
            self.get_joint_names()[6],
            str(self.get_initial_joint_positions()[6]),
            self.get_joint_names()[7],
            str(self.get_initial_joint_positions()[7]),
            self.get_joint_names()[8],
            str(self.get_initial_joint_positions()[8]),
            self.get_joint_names()[9],
            str(self.get_initial_joint_positions()[9]),
        )

    def __get_joint_trajectory_controller_config_joints_only(self) -> str:
        # TODO: refactor into something more sensible
        return """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>500</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>2000</position_i_gain>
            <position_i_min>-7</position_i_min>
            <position_i_max>7</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>1100</position_p_gain>
            <position_d_gain>25</position_d_gain>
            <position_i_gain>5000</position_i_gain>
            <position_i_min>-9</position_i_min>
            <position_i_max>9</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>950</position_p_gain>
            <position_d_gain>17.5</position_d_gain>
            <position_i_gain>3500</position_i_gain>
            <position_i_min>-8</position_i_min>
            <position_i_max>8</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s1.57</initial_position>
            <position_p_gain>800</position_p_gain>
            <position_d_gain>12.5</position_d_gain>
            <position_i_gain>2500</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>-5</position_i_max>
            <position_cmd_min>-30.5</position_cmd_min>
            <position_cmd_max>30.5</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>750</position_p_gain>
            <position_d_gain>7.5</position_d_gain>
            <position_i_gain>2000</position_i_gain>
            <position_i_min>-6</position_i_min>
            <position_i_max>6</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>600</position_p_gain>
            <position_d_gain>6</position_d_gain>
            <position_i_gain>1800</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>5</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>400</position_p_gain>
            <position_d_gain>4</position_d_gain>
            <position_i_gain>1200</position_i_gain>
            <position_i_min>-3.5</position_i_min>
            <position_i_max>3.5</position_i_max>
            <position_cmd_min>-13.6</position_cmd_min>
            <position_cmd_max>13.6</position_cmd_max>
            </sdf>
            """ % (
            self.get_joint_names()[0],
            str(self.get_initial_joint_positions()[0]),
            self.get_joint_names()[1],
            str(self.get_initial_joint_positions()[1]),
            self.get_joint_names()[2],
            str(self.get_initial_joint_positions()[2]),
            self.get_joint_names()[3],
            str(self.get_initial_joint_positions()[3]),
            self.get_joint_names()[4],
            str(self.get_initial_joint_positions()[4]),
            self.get_joint_names()[5],
            str(self.get_initial_joint_positions()[5]),
            self.get_joint_names()[6],
            str(self.get_initial_joint_positions()[6]),
        )

    def __get_joint_trajectory_controller_config_gripper_only(self) -> str:
        # TODO: refactor into something more sensible
        return """
            <sdf version="1.7">
            <topic>gripper_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>8</position_p_gain>
            <position_d_gain>0.35</position_d_gain>
            <position_i_gain>105</position_i_gain>
            <position_i_min>-3</position_i_min>
            <position_i_max>3</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>
            </sdf>
            """ % (
            self.get_joint_names()[7],
            str(self.get_initial_joint_positions()[7]),
            self.get_joint_names()[8],
            str(self.get_initial_joint_positions()[8]),
            self.get_joint_names()[9],
            str(self.get_initial_joint_positions()[9]),
        )

    @classmethod
    def disable_collision(
        self, model_file: str, arm_collision: bool, hand_collision: bool
    ) -> str:

        new_model_file = path.join(
            path.dirname(model_file), "model_without_arm_collision.sdf"
        )

        # Remove collision geometry
        with open(model_file, "r") as original_sdf_file:
            with open(new_model_file, "w") as new_sdf_file:
                while True:
                    # Read a new line and make sure it is not the end of the file
                    line = original_sdf_file.readline()
                    if not line.rstrip():
                        break

                    if not arm_collision:
                        if (
                            '<collision name="j2s7s300_link' in line
                            and not '<collision name="j2s7s300_link_finger' in line
                        ):
                            line = original_sdf_file.readline()
                            while not "</collision>" in line:
                                line = original_sdf_file.readline()
                            continue

                    if not hand_collision:
                        if '<collision name="j2s7s300_link_finger' in line:
                            line = original_sdf_file.readline()
                            while not "</collision>" in line:
                                line = original_sdf_file.readline()
                            continue

                    # Write all other lines into the new file
                    new_sdf_file.write(line)

        # Return path to the new file
        return new_model_file
