from ament_index_python.packages import get_package_share_directory
from drl_grasping.envs.models.utils import xacro2sdf
from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from os import path
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple, Optional, Dict

# TODO: Use description package for ur5_rg2 with xacro support


class UR5RG2(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):

    ROBOT_MODEL_NAME: str = "ur5_rg2"
    DEFAULT_PREFIX: str = ""
    __PREFIX_MANIPULATOR: str = ""

    # __DESCRIPTION_PACKAGE = ROBOT_MODEL_NAME + "_description"
    # __DEFAULT_XACRO_FILE = path.join(
    #     get_package_share_directory(__DESCRIPTION_PACKAGE),
    #     "urdf",
    #     ROBOT_MODEL_NAME + ".urdf.xacro",
    # )
    # __DEFAULT_XACRO_MAPPINGS: Dict = {
    #     "name": ROBOT_MODEL_NAME,
    #     "prefix": DEFAULT_PREFIX,
    # }
    # __XACRO_MODEL_PATH_REMAP: Tuple[str, str] = (
    #     __DESCRIPTION_PACKAGE,
    #     ROBOT_MODEL_NAME,
    # )

    DEFAULT_ARM_JOINT_POSITIONS: List[float] = (
        0.0,
        0.0,
        1.57,
        0.0,
        -1.57,
        -1.57,
    )
    OPEN_GRIPPER_JOINT_POSITIONS: List[float] = (
        0.52359878,
        0.52359878,
    )
    CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = (
        0.0,
        0.0,
    )
    DEFAULT_GRIPPER_JOINT_POSITIONS: List[float] = OPEN_GRIPPER_JOINT_POSITIONS

    def __init__(
        self,
        world: scenario.World,
        name: str = ROBOT_MODEL_NAME,
        prefix: str = DEFAULT_PREFIX,
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_file: str = None,
        use_fuel: bool = True,
        use_xacro: bool = False,
        # xacro_file: str = __DEFAULT_XACRO_FILE,
        # xacro_mappings: Dict = __DEFAULT_XACRO_MAPPINGS,
        initial_arm_joint_positions: List[float] = DEFAULT_ARM_JOINT_POSITIONS,
        initial_gripper_joint_positions: List[float] = OPEN_GRIPPER_JOINT_POSITIONS,
        # TODO: Expose the rest of the parameters for ur5_rg2 in xacro once it is available
        arm_collision: bool = False,
        hand_collision: bool = True,
        separate_gripper_controller: bool = True,
        **kwargs,
    ):

        # Store params that are needed internally
        self.__prefix = prefix
        self.__initial_arm_joint_positions = initial_arm_joint_positions
        self.__initial_gripper_joint_positions = initial_gripper_joint_positions
        self.__separate_gripper_controller = separate_gripper_controller

        # Allow passing of custom model file as an argument
        if model_file is None:
            if use_xacro:
                raise NotADirectoryError
                # Generate SDF from xacro
                mappings = self.__DEFAULT_XACRO_MAPPINGS
                mappings.update(kwargs)
                mappings.update(xacro_mappings)
                mappings.update({"prefix": prefix})
                model_file = xacro2sdf(
                    input_file_path=xacro_file,
                    mappings=mappings,
                    model_path_remap=self.__XACRO_MODEL_PATH_REMAP,
                )
            else:
                # Otherwise, use the default SDF file (local or fuel)
                model_file = self.get_model_file(fuel=use_fuel)

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Setup initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Determine whether to insert from string or file
        if use_xacro:
            insert_fn = scenario_gazebo.World.insert_model_from_string
        else:
            insert_fn = scenario_gazebo.World.insert_model_from_file

        # For non-xacro model files, disable hand collisions manually
        if not use_xacro and (not arm_collision or not hand_collision):
            model_file = self.disable_collision(
                model_file=model_file,
                arm_collision=arm_collision,
                hand_collision=hand_collision,
            )

        # Insert the model
        ok_model = insert_fn(world.to_gazebo(), model_file, initial_pose, model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Set initial joint configuration
        self.set_initial_joint_positions(model)

        # Add JointStatePublisher
        self.__add_joint_state_publisher(model)

        # Add JointTrajectoryController
        self.__add_joint_trajectory_controller(model)

        # Initialize base class
        super().__init__(model=model)

    def set_initial_joint_positions(self, model):

        model = model.to_gazebo()
        if not model.reset_joint_positions(
            self.initial_arm_joint_positions, self.arm_joint_names
        ):
            raise RuntimeError("Failed to set initial positions of arm's joints")
        if not model.reset_joint_positions(
            self.initial_gripper_joint_positions, self.gripper_joint_names
        ):
            raise RuntimeError("Failed to set initial positions of gripper's joints")

    @classmethod
    def get_model_file(cls, fuel=False) -> str:

        if fuel:
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/"
                + cls.ROBOT_MODEL_NAME
            )
        else:
            return cls.ROBOT_MODEL_NAME

    # Meta information #
    @property
    def is_mobile(self) -> bool:

        return False

    # Prefix #
    @property
    def prefix(self) -> str:

        return self.__prefix

    # Joints #
    @property
    def joint_names(self) -> List[str]:

        return self.move_base_joint_names + self.manipulator_joint_names

    @property
    def move_base_joint_names(self) -> List[str]:

        return []

    @property
    def manipulator_joint_names(self) -> List[str]:

        return self.arm_joint_names + self.gripper_joint_names

    @property
    def arm_joint_names(self) -> List[str]:

        return [
            self.prefix + self.__PREFIX_MANIPULATOR + "shoulder_pan_joint",
            self.prefix + self.__PREFIX_MANIPULATOR + "shoulder_lift_joint",
            self.prefix + self.__PREFIX_MANIPULATOR + "elbow_joint",
            self.prefix + self.__PREFIX_MANIPULATOR + "wrist_1_joint",
            self.prefix + self.__PREFIX_MANIPULATOR + "wrist_2_joint",
            self.prefix + self.__PREFIX_MANIPULATOR + "wrist_3_joint",
        ]

    @property
    def gripper_joint_names(self) -> List[str]:

        return [
            self.prefix + self.__PREFIX_MANIPULATOR + "rg2_finger_joint1",
            self.prefix + self.__PREFIX_MANIPULATOR + "rg2_finger_joint2",
        ]

    @property
    def move_base_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return None

    @property
    def arm_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return [
            (-6.28319, 6.28319),
            (-6.28319, 6.28319),
            (-6.28319, 6.28319),
            (-6.28319, 6.28319),
            (-6.28319, 6.28319),
            (-6.28319, 6.28319),
        ]

    @property
    def gripper_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return [
            (0.0, 0.52359878),
            (0.0, 0.52359878),
        ]

    @property
    def gripper_joints_close_towards_positive(self) -> bool:

        return (
            self.OPEN_GRIPPER_JOINT_POSITIONS[0]
            < self.CLOSED_GRIPPER_JOINT_POSITIONS[0]
        )

    @property
    def initial_arm_joint_positions(self) -> List[float]:

        return self.__initial_arm_joint_positions

    @property
    def initial_gripper_joint_positions(self) -> List[float]:

        return self.__initial_gripper_joint_positions

    # Passive joints #
    @property
    def passive_joint_names(self) -> List[str]:

        return self.manipulator_passive_joint_names + self.move_base_passive_joint_names

    @property
    def move_base_passive_joint_names(self) -> List[str]:

        return []

    @property
    def manipulator_passive_joint_names(self) -> List[str]:

        return self.arm_passive_joint_names + self.gripper_passive_joint_names

    @property
    def arm_passive_joint_names(self) -> List[str]:

        return []

    @property
    def gripper_passive_joint_names(self) -> List[str]:

        return []

    # Links #
    @classmethod
    def get_robot_base_link_name(cls, prefix: str = "") -> str:

        return cls.get_arm_base_link_name(prefix)

    @property
    def robot_base_link_name(self) -> str:

        return self.get_robot_base_link_name(self.prefix)

    @classmethod
    def get_arm_base_link_name(cls, prefix: str = "") -> str:

        # Same as `self.arm_link_names[0]``
        return prefix + cls.__PREFIX_MANIPULATOR + "base_link"

    @property
    def arm_base_link_name(self) -> str:

        return self.get_arm_base_link_name(self.prefix)

    @classmethod
    def get_ee_link_name(cls, prefix: str = "") -> str:

        return prefix + cls.__PREFIX_MANIPULATOR + "tool0"

    @property
    def ee_link_name(self) -> str:

        return self.get_ee_link_name(self.prefix)

    @classmethod
    def get_wheel_link_names(cls, prefix: str = "") -> List[str]:

        return []

    @property
    def wheel_link_names(self) -> List[str]:

        return self.get_wheel_link_names(self.prefix)

    @classmethod
    def get_arm_link_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "base_link",
            prefix + cls.__PREFIX_MANIPULATOR + "shoulder_link",
            prefix + cls.__PREFIX_MANIPULATOR + "upper_arm_link",
            prefix + cls.__PREFIX_MANIPULATOR + "forearm_link",
            prefix + cls.__PREFIX_MANIPULATOR + "wrist_1_link",
            prefix + cls.__PREFIX_MANIPULATOR + "wrist_2_link",
            prefix + cls.__PREFIX_MANIPULATOR + "wrist_3_link",
            prefix + cls.__PREFIX_MANIPULATOR + "rg2_hand",
        ]

    @property
    def arm_link_names(self) -> List[str]:

        return self.get_arm_link_names(self.prefix)

    @classmethod
    def get_gripper_link_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "rg2_leftfinger",
            prefix + cls.__PREFIX_MANIPULATOR + "rg2_rightfinger",
        ]

    @property
    def gripper_link_names(self) -> List[str]:

        return self.get_gripper_link_names(self.prefix)

    # TODO: Replace all custom functions below with ur5_rg2 xacros
    def __add_joint_state_publisher(self, model) -> bool:

        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-state-publisher-system.so",
            "ignition::gazebo::systems::JointStatePublisher",
            self.__get_joint_state_publisher_config(),
        )

    def __get_joint_state_publisher_config(self) -> str:

        return """
            <sdf version="1.7">
            %s
            </sdf>
            """ % " ".join(
            ("<joint_name>" + joint + "</joint_name>" for joint in self.joint_names)
        )

    def __add_joint_trajectory_controller(self, model) -> bool:

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

        return """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>1650</position_i_gain>
            <position_i_min>-15</position_i_min>
            <position_i_max>15</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>9500</position_p_gain>
            <position_d_gain>47.5</position_d_gain>
            <position_i_gain>5225</position_i_gain>
            <position_i_min>-47.5</position_i_min>
            <position_i_max>47.5</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>6500</position_p_gain>
            <position_d_gain>32.5</position_d_gain>
            <position_i_gain>3575</position_i_gain>
            <position_i_min>-32.5</position_i_min>
            <position_i_max>32.5</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>5000</position_p_gain>
            <position_d_gain>20</position_d_gain>
            <position_i_gain>1200</position_i_gain>
            <position_i_min>-30</position_i_min>
            <position_i_max>30</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>4250</position_p_gain>
            <position_d_gain>10</position_d_gain>
            <position_i_gain>250</position_i_gain>
            <position_i_min>-6.88</position_i_min>
            <position_i_max>6.88</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>2.5</position_d_gain>
            <position_i_gain>775</position_i_gain>
            <position_i_min>-6.25</position_i_min>
            <position_i_max>6.25</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>300</position_p_gain>
            <position_d_gain>0.5</position_d_gain>
            <position_i_gain>100</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-10.6</position_cmd_min>
            <position_cmd_max>10.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>300</position_p_gain>
            <position_d_gain>0.5</position_d_gain>
            <position_i_gain>100</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-10.6</position_cmd_min>
            <position_cmd_max>10.6</position_cmd_max>
            </sdf>
            """ % (
            self.arm_joint_names[0],
            str(self.initial_arm_joint_positions[0]),
            self.arm_joint_names[1],
            str(self.initial_arm_joint_positions[1]),
            self.arm_joint_names[2],
            str(self.initial_arm_joint_positions[2]),
            self.arm_joint_names[3],
            str(self.initial_arm_joint_positions[3]),
            self.arm_joint_names[4],
            str(self.initial_arm_joint_positions[4]),
            self.arm_joint_names[5],
            str(self.initial_arm_joint_positions[5]),
            self.gripper_joint_names[0],
            str(self.initial_gripper_joint_positions[0]),
            self.gripper_joint_names[1],
            str(self.initial_gripper_joint_positions[1]),
        )

    def __get_joint_trajectory_controller_config_joints_only(self) -> str:

        return """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>15</position_d_gain>
            <position_i_gain>1650</position_i_gain>
            <position_i_min>-15</position_i_min>
            <position_i_max>15</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>9500</position_p_gain>
            <position_d_gain>47.5</position_d_gain>
            <position_i_gain>5225</position_i_gain>
            <position_i_min>-47.5</position_i_min>
            <position_i_max>47.5</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>6500</position_p_gain>
            <position_d_gain>32.5</position_d_gain>
            <position_i_gain>3575</position_i_gain>
            <position_i_min>-32.5</position_i_min>
            <position_i_max>32.5</position_i_max>
            <position_cmd_min>-150</position_cmd_min>
            <position_cmd_max>150</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>5000</position_p_gain>
            <position_d_gain>20</position_d_gain>
            <position_i_gain>1200</position_i_gain>
            <position_i_min>-30</position_i_min>
            <position_i_max>30</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>4250</position_p_gain>
            <position_d_gain>10</position_d_gain>
            <position_i_gain>250</position_i_gain>
            <position_i_min>-6.88</position_i_min>
            <position_i_max>6.88</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>3000</position_p_gain>
            <position_d_gain>2.5</position_d_gain>
            <position_i_gain>775</position_i_gain>
            <position_i_min>-6.25</position_i_min>
            <position_i_max>6.25</position_i_max>
            <position_cmd_min>-28</position_cmd_min>
            <position_cmd_max>28</position_cmd_max>
            </sdf>
            """ % (
            self.arm_joint_names[0],
            str(self.initial_arm_joint_positions[0]),
            self.arm_joint_names[1],
            str(self.initial_arm_joint_positions[1]),
            self.arm_joint_names[2],
            str(self.initial_arm_joint_positions[2]),
            self.arm_joint_names[3],
            str(self.initial_arm_joint_positions[3]),
            self.arm_joint_names[4],
            str(self.initial_arm_joint_positions[4]),
            self.arm_joint_names[5],
            str(self.initial_arm_joint_positions[5]),
        )

    def __get_joint_trajectory_controller_config_gripper_only(self) -> str:

        return """
            <sdf version="1.7">
            <topic>gripper_trajectory</topic>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>150</position_p_gain>
            <position_d_gain>0.05</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-10.6</position_cmd_min>
            <position_cmd_max>10.6</position_cmd_max>

            <joint_name>%s</joint_name>
            <initial_position>%s</initial_position>
            <position_p_gain>150</position_p_gain>
            <position_d_gain>0.05</position_d_gain>
            <position_i_gain>50</position_i_gain>
            <position_i_min>-10</position_i_min>
            <position_i_max>10</position_i_max>
            <position_cmd_min>-10.6</position_cmd_min>
            <position_cmd_max>10.6</position_cmd_max>
            </sdf>
            """ % (
            self.gripper_joint_names[0],
            str(self.initial_gripper_joint_positions[0]),
            self.gripper_joint_names[1],
            str(self.initial_gripper_joint_positions[1]),
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

                    # Once `<collision>` for lower links is encountered, skip that and all lines until `</collision>` is reached
                    if not arm_collision:
                        if (
                            '<collision name="base_link_collision"' in line
                            or '<collision name="shoulder_link_collision"' in line
                            or '<collision name="upper_arm_link_collision"' in line
                            or '<collision name="forearm_link_collision"' in line
                            or '<collision name="wrist_1_link_collision"' in line
                            or '<collision name="wrist_2_link_collision"' in line
                        ):
                            line = original_sdf_file.readline()
                            while not "</collision>" in line:
                                line = original_sdf_file.readline()
                            continue

                    # Same as for arm, but check for hand and both fingers
                    if not hand_collision:
                        if (
                            '<collision name="rg2_hand_collision"' in line
                            or '<collision name="rg2_leftfinger_collision"' in line
                            or '<collision name="rg2_rightfinger_collision"' in line
                        ):
                            line = original_sdf_file.readline()
                            while not "</collision>" in line:
                                line = original_sdf_file.readline()
                            continue

                    # Write all other lines into the new file
                    new_sdf_file.write(line)

        # Return path to the new file
        return new_model_file
