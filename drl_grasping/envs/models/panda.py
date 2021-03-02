from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple


class Panda(model_wrapper.ModelWrapper,
            model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'panda',
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_file: str = None,
                 separate_gripper_controller: bool = True,
                 initial_joint_positions: List[float] = (0, 0, 0, -1.57, 0, 1.57, 0.79, 0, 0)):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(fuel=False)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        self.__separate_gripper_controller = separate_gripper_controller

        # Set initial joint configuration
        self.__set_initial_joint_positions(initial_joint_positions)
        if not model.to_gazebo().reset_joint_positions(self.get_initial_joint_positions(),
                                                       self.get_joint_names()):
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
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/panda")
        else:
            return "panda"

    @classmethod
    def get_joint_names(self) -> List[str]:
        return ["panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2"]

    @classmethod
    def get_joint_limits(self) -> List[Tuple[float, float]]:
        return [(-2.897246558310587, 2.897246558310587),
                (-1.762782544514273, 1.762782544514273),
                (-2.897246558310587, 2.897246558310587),
                (-3.07177948351002, -0.06981317007977318),
                (-2.897246558310587, -2.897246558310587),
                (-0.0174532925199433, 3.752457891787809),
                (-2.897246558310587, 2.897246558310587),
                (0.0, 0.04),
                (0.0, 0.04)]

    @classmethod
    def get_base_link_name(self) -> str:
        return "panda_link0"

    @classmethod
    def get_ee_link_name(self) -> str:
        return "panda_link8"

    @classmethod
    def get_gripper_link_names(self) -> List[str]:
        return ["panda_leftfinger",
                "panda_rightfinger"]

    @classmethod
    def get_finger_count(self) -> int:
        return 2

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
        if self.__separate_gripper_controller:
            model.to_gazebo().insert_model_plugin(
                "libignition-gazebo-joint-trajectory-controller-system.so",
                "ignition::gazebo::systems::JointTrajectoryController",
                self.__get_joint_trajectory_controller_config_joints_only()
            )
            model.to_gazebo().insert_model_plugin(
                "libignition-gazebo-joint-trajectory-controller-system.so",
                "ignition::gazebo::systems::JointTrajectoryController",
                self.__get_joint_trajectory_controller_config_gripper_only()
            )
        else:
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
            
            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>1650</position_i_gain>
            <position_i_min>-15</position_i_min>
            <position_i_max>15</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>9500</position_p_gain>
            <position_d_gain>47.5</position_d_gain>
            <position_i_gain>5225</position_i_gain>
            <position_i_min>-47.5</position_i_min>
            <position_i_max>47.5</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>6500</position_p_gain>
            <position_d_gain>32.5</position_d_gain>
            <position_i_gain>3575</position_i_gain>
            <position_i_min>-32.5</position_i_min>
            <position_i_max>32.5</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s1.57</initial_position>
            <position_p_gain>6000</position_p_gain>
            <position_d_gain>30</position_d_gain>
            <position_i_gain>3300</position_i_gain>
            <position_i_min>-30</position_i_min>
            <position_i_max>30</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2750</position_p_gain>
            <position_d_gain>2.75</position_d_gain>
            <position_i_gain>1515</position_i_gain>
            <position_i_min>-6.88</position_i_min>
            <position_i_max>6.88</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2500</position_p_gain>
            <position_d_gain>2.5</position_d_gain>
            <position_i_gain>1375</position_i_gain>
            <position_i_min>-6.25</position_i_min>
            <position_i_max>6.25</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2000</position_p_gain>
            <position_d_gain>2</position_d_gain>
            <position_i_gain>1100</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>5</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>250</position_p_gain>
            <position_d_gain>0.2</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>250</position_p_gain>
            <position_d_gain>0.2</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>
            </sdf>
            """ % \
            (self.get_joint_names()[0],
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
             str(self.get_initial_joint_positions()[8]))

    def __get_joint_trajectory_controller_config_joints_only(self) -> str:
        # TODO: refactor into something more sensible
        return \
            """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>
            
            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>1650</position_i_gain>
            <position_i_min>-15</position_i_min>
            <position_i_max>15</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>9500</position_p_gain>
            <position_d_gain>47.5</position_d_gain>
            <position_i_gain>5225</position_i_gain>
            <position_i_min>-47.5</position_i_min>
            <position_i_max>47.5</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>6500</position_p_gain>
            <position_d_gain>32.5</position_d_gain>
            <position_i_gain>3575</position_i_gain>
            <position_i_min>-32.5</position_i_min>
            <position_i_max>32.5</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s1.57</initial_position>
            <position_p_gain>6000</position_p_gain>
            <position_d_gain>30</position_d_gain>
            <position_i_gain>3300</position_i_gain>
            <position_i_min>-30</position_i_min>
            <position_i_max>30</position_i_max>
            <position_cmd_min>-87</position_cmd_min>
            <position_cmd_max>87</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2750</position_p_gain>
            <position_d_gain>2.75</position_d_gain>
            <position_i_gain>1515</position_i_gain>
            <position_i_min>-6.88</position_i_min>
            <position_i_max>6.88</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2500</position_p_gain>
            <position_d_gain>2.5</position_d_gain>
            <position_i_gain>1375</position_i_gain>
            <position_i_min>-6.25</position_i_min>
            <position_i_max>6.25</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>2000</position_p_gain>
            <position_d_gain>2</position_d_gain>
            <position_i_gain>1100</position_i_gain>
            <position_i_min>-5</position_i_min>
            <position_i_max>5</position_i_max>
            <position_cmd_min>-12</position_cmd_min>
            <position_cmd_max>12</position_cmd_max>
            </sdf>
            """ % \
            (self.get_joint_names()[0],
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
             str(self.get_initial_joint_positions()[6]))

    def __get_joint_trajectory_controller_config_gripper_only(self) -> str:
        # TODO: refactor into something more sensible
        return \
            """
            <sdf version="1.7">
            <topic>gripper_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>250</position_p_gain>
            <position_d_gain>0.2</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>250</position_p_gain>
            <position_d_gain>0.2</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-20</position_cmd_min>
            <position_cmd_max>20</position_cmd_max>
            </sdf>
            """ % \
            (self.get_joint_names()[7],
             str(self.get_initial_joint_positions()[7]),
             self.get_joint_names()[8],
             str(self.get_initial_joint_positions()[8]))
