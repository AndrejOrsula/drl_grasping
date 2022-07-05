from os import path
from typing import Dict, List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from gym_ignition.scenario import model_with_file, model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo

from drl_grasping.envs.models.utils import xacro2sdf


class LunalabSummitXlGen(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):

    ROBOT_MODEL_NAME: str = "lunalab_summit_xl_gen"
    DEFAULT_PREFIX: str = "robot_"
    __PREFIX_MOBILE_BASE: str = "summit_xl_"
    __PREFIX_MANIPULATOR: str = "j2s7s300_"

    __DESCRIPTION_PACKAGE = ROBOT_MODEL_NAME + "_description"
    __DEFAULT_XACRO_FILE = path.join(
        get_package_share_directory(__DESCRIPTION_PACKAGE),
        "urdf",
        ROBOT_MODEL_NAME + ".urdf.xacro",
    )
    __DEFAULT_XACRO_MAPPINGS: Dict[str, any] = {
        "name": ROBOT_MODEL_NAME,
        "prefix": DEFAULT_PREFIX,
        "safety_limits": True,
        "safety_soft_limit_margin": 0.17453293,
        "safety_k_position": 20,
        "collision_chassis": False,
        "collision_wheels": True,
        "collision_arm": False,
        "collision_gripper": True,
        "high_quality_mesh": True,
        "mimic_gripper_joints": False,
        "ros2_control": True,
        "ros2_control_plugin": "ign",
        "ros2_control_command_interface": "effort",
        "gazebo_preserve_fixed_joint": True,
        "gazebo_self_collide": False,
        "gazebo_self_collide_fingers": True,
        "gazebo_diff_drive": True,
        "gazebo_joint_trajectory_controller": False,
        "gazebo_joint_state_publisher": False,
        "gazebo_pose_publisher": True,
    }
    __XACRO_MODEL_PATH_REMAP: Tuple[str, str] = (
        __DESCRIPTION_PACKAGE,
        ROBOT_MODEL_NAME,
    )

    DEFAULT_ARM_JOINT_POSITIONS: List[float] = (
        0.0,
        3.141592653589793,
        0.0,
        4.71238898038469,
        0.0,
        1.5707963267948966,
        0.0,
    )
    OPEN_GRIPPER_JOINT_POSITIONS: List[float] = (
        0.2,
        0.2,
        0.2,
    )
    CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = (
        1.3,
        1.3,
        1.3,
    )
    DEFAULT_GRIPPER_JOINT_POSITIONS: List[float] = OPEN_GRIPPER_JOINT_POSITIONS

    # Offset from `base_arm_link` to `base_footprint`
    BASE_LINK_Z_OFFSET: float = -0.22

    def __init__(
        self,
        world: scenario.World,
        name: str = ROBOT_MODEL_NAME,
        prefix: str = DEFAULT_PREFIX,
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_file: str = None,
        use_fuel: bool = False,
        use_xacro: bool = True,
        xacro_file: str = __DEFAULT_XACRO_FILE,
        xacro_mappings: Dict[str, any] = __DEFAULT_XACRO_MAPPINGS,
        initial_arm_joint_positions: List[float] = DEFAULT_ARM_JOINT_POSITIONS,
        initial_gripper_joint_positions: List[float] = OPEN_GRIPPER_JOINT_POSITIONS,
        **kwargs,
    ):

        # Store params that are needed internally
        self.__prefix = prefix
        self.__initial_arm_joint_positions = initial_arm_joint_positions
        self.__initial_gripper_joint_positions = initial_gripper_joint_positions

        # Allow passing of custom model file as an argument
        if model_file is None:
            if use_xacro:
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

        # Insert the model
        ok_model = insert_fn(world.to_gazebo(), model_file, initial_pose, model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Set initial joint configuration
        self.set_initial_joint_positions(model)

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
    def get_model_file(cls, fuel: bool = False) -> str:

        if fuel:
            # TODO: Add "lunalab_summit_xl_gen" to Fuel
            raise NotImplementedError
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/"
                + cls.ROBOT_MODEL_NAME
            )
        else:
            return cls.ROBOT_MODEL_NAME

    # Meta information #
    @property
    def is_mobile(self) -> bool:

        return True

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

        return [
            self.prefix + self.__PREFIX_MOBILE_BASE + "back_left_wheel_joint",
            self.prefix + self.__PREFIX_MOBILE_BASE + "back_right_wheel_joint",
            self.prefix + self.__PREFIX_MOBILE_BASE + "front_left_wheel_joint",
            self.prefix + self.__PREFIX_MOBILE_BASE + "front_right_wheel_joint",
        ]

    @property
    def manipulator_joint_names(self) -> List[str]:

        return self.arm_joint_names + self.gripper_joint_names

    @classmethod
    def get_arm_joint_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "joint_1",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_2",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_3",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_4",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_5",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_6",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_7",
        ]

    @property
    def arm_joint_names(self) -> List[str]:

        return self.get_arm_joint_names(self.prefix)

    @classmethod
    def get_gripper_joint_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "joint_finger_1",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_finger_2",
            prefix + cls.__PREFIX_MANIPULATOR + "joint_finger_3",
        ]

    @property
    def gripper_joint_names(self) -> List[str]:

        return self.get_gripper_joint_names(self.prefix)

    @property
    def move_base_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return None

    @property
    def arm_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return [
            (-6.283185307179586, 6.283185307179586),
            (0.8203047484373349, 5.462880558742252),
            (-6.283185307179586, 6.283185307179586),
            (0.5235987755982988, 5.759586531581287),
            (-6.283185307179586, 6.283185307179586),
            (1.1344640137963142, 5.148721293383272),
            (-6.283185307179586, 6.283185307179586),
        ]

    @property
    def gripper_joint_limits(self) -> Optional[List[Tuple[float, float]]]:

        return [
            (0.0, 1.51),
            (0.0, 1.51),
            (0.0, 1.51),
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

        return [
            self.prefix + self.__PREFIX_MANIPULATOR + "joint_finger_tip_1",
            self.prefix + self.__PREFIX_MANIPULATOR + "joint_finger_tip_2",
            self.prefix + self.__PREFIX_MANIPULATOR + "joint_finger_tip_3",
        ]

    # Links #
    @classmethod
    def get_robot_base_link_name(cls, prefix: str = "") -> str:

        return prefix + cls.__PREFIX_MOBILE_BASE + "base_footprint"

    @property
    def robot_base_link_name(self) -> str:

        return self.get_robot_base_link_name(self.prefix)

    @classmethod
    def get_arm_base_link_name(cls, prefix: str = "") -> str:

        # Same as `self.arm_link_names[0]``
        return prefix + cls.__PREFIX_MANIPULATOR + "link_base"

    @property
    def arm_base_link_name(self) -> str:

        return self.get_arm_base_link_name(self.prefix)

    @classmethod
    def get_ee_link_name(cls, prefix: str = "") -> str:

        return prefix + cls.__PREFIX_MANIPULATOR + "end_effector"

    @property
    def ee_link_name(self) -> str:

        return self.get_ee_link_name(self.prefix)

    @classmethod
    def get_wheel_link_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MOBILE_BASE + "back_left_wheel",
            prefix + cls.__PREFIX_MOBILE_BASE + "back_right_wheel",
            prefix + cls.__PREFIX_MOBILE_BASE + "front_left_wheel",
            prefix + cls.__PREFIX_MOBILE_BASE + "front_right_wheel",
        ]

    @property
    def wheel_link_names(self) -> List[str]:

        return self.get_wheel_link_names(self.prefix)

    @classmethod
    def get_arm_link_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "link_base",
            prefix + cls.__PREFIX_MANIPULATOR + "link_1",
            prefix + cls.__PREFIX_MANIPULATOR + "link_2",
            prefix + cls.__PREFIX_MANIPULATOR + "link_3",
            prefix + cls.__PREFIX_MANIPULATOR + "link_4",
            prefix + cls.__PREFIX_MANIPULATOR + "link_5",
            prefix + cls.__PREFIX_MANIPULATOR + "link_6",
            prefix + cls.__PREFIX_MANIPULATOR + "link_7",
        ]

    @property
    def arm_link_names(self) -> List[str]:

        return self.get_arm_link_names(self.prefix)

    @classmethod
    def get_gripper_link_names(cls, prefix: str = "") -> List[str]:

        return [
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_1",
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_2",
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_3",
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_tip_1",
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_tip_2",
            prefix + cls.__PREFIX_MANIPULATOR + "link_finger_tip_3",
        ]

    @property
    def gripper_link_names(self) -> List[str]:

        return self.get_gripper_link_names(self.prefix)
