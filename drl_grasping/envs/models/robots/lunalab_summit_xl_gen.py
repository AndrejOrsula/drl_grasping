from ament_index_python.packages import get_package_share_directory
from drl_grasping.envs.models.utils import xacro2sdf
from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from os import path
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple, Dict


class LunalabSummitXlGen(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):

    DEFAULT_XACRO_FILE = path.join(
        get_package_share_directory("lunalab_summit_xl_gen_description"),
        "urdf",
        "lunalab_summit_xl_gen.urdf.xacro",
    )
    DEFAULT_XACRO_MAPPINGS = {
        "name": "lunalab_summit_xl_gen",
        "prefix": "robot_",
        "safety_limits": True,
        "safety_soft_limit_margin": 0.17453293,
        "safety_k_position": 20,
        "high_quality_mesh": True,
        "ros2_control": True,
        "gazebo_diff_drive": True,
        "gazebo_joint_trajectory_controller": True,
        "gazebo_joint_state_publisher": True,
        "gazebo_pose_publisher": True,
        # TODO: All of these could also be part of xacro_mapping (needs to be added to xacro in description package)
        # "wheels_collisions": True,
        # "arm_collision": True,
        # "hand_collision": True,
        # "separate_gripper_controller": True,
    }
    XACRO_MODEL_PATH_REMAP = (
        "lunalab_summit_xl_gen_description",
        "lunalab_summit_xl_gen",
    )

    PREFIX_SUMMIT_XL = "summit_xl_"
    PREFIX_MANIPULATOR = "j2s7s300_"

    def __init__(
        self,
        world: scenario.World,
        name: str = "lunalab_summit_xl_gen",
        prefix: str = "robot_",
        position: List[float] = (0, 0, 0),
        orientation: List[float] = (1, 0, 0, 0),
        model_file: str = None,
        use_fuel: bool = False,
        use_xacro: bool = True,
        xacro_file: str = DEFAULT_XACRO_FILE,
        xacro_mappings: Dict = {},
        initial_joint_positions: List[float] = (
            0.0,
            3.14159265359,
            0.0,
            3.14159265359,
            0.0,
            3.14159265359,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ):

        # Store params that are needed internally
        self._prefix = prefix
        self._initial_joint_positions = initial_joint_positions

        # Allow passing of custom model file as an argument
        if model_file is None:
            if use_xacro:
                # Generate SDF from xacro
                mappings = self.DEFAULT_XACRO_MAPPINGS
                mappings.update(xacro_mappings)
                model_file = xacro2sdf(
                    input_file_path=xacro_file,
                    mappings=mappings,
                    model_path_remap=self.XACRO_MODEL_PATH_REMAP,
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
        if not model.to_gazebo().reset_joint_positions(
            self.get_initial_joint_positions(), self.get_joint_names()
        ):
            raise RuntimeError("Failed to set initial robot joint positions")

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=False) -> str:
        if fuel:
            # TODO: Add "lunalab_summit_xl_gen" to Fuel
            raise NotImplementedError
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/lunalab_summit_xl_gen"
            )
        else:
            return "lunalab_summit_xl_gen"

    @classmethod
    def get_joint_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_MANIPULATOR + "joint_1",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_2",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_3",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_4",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_5",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_6",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_7",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_1",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_2",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_3",
        ]

    @classmethod
    def get_passive_joint_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_tip_1",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_tip_2",
            self._prefix + self.PREFIX_MANIPULATOR + "joint_finger_tip_3",
        ]

    @classmethod
    def get_wheel_joint_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_SUMMIT_XL + "back_left_wheel_joint",
            self._prefix + self.PREFIX_SUMMIT_XL + "back_right_wheel_joint",
            self._prefix + self.PREFIX_SUMMIT_XL + "front_left_wheel_joint",
            self._prefix + self.PREFIX_SUMMIT_XL + "front_right_wheel_joint",
        ]

    @classmethod
    def get_joint_limits(self) -> List[Tuple[float, float]]:
        return [
            (-6.283185307179586, 6.283185307179586),
            (0.8203047484373349, 5.462880558742252),
            (-6.283185307179586, 6.283185307179586),
            (0.5235987755982988, 5.759586531581287),
            (-6.283185307179586, 6.283185307179586),
            (1.1344640137963142, 5.148721293383272),
            (-6.283185307179586, 6.283185307179586),
            (0.0, 1.51),
            (0.0, 1.51),
            (0.0, 1.51),
        ]

    @classmethod
    def get_finger_count(self) -> int:
        return 3

    @classmethod
    def get_base_footprint_name(self) -> str:
        return self._prefix + self.PREFIX_SUMMIT_XL + "base_footprint"

    @classmethod
    def get_wheel_link_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_SUMMIT_XL + "back_left_wheel",
            self._prefix + self.PREFIX_SUMMIT_XL + "back_right_wheel",
            self._prefix + self.PREFIX_SUMMIT_XL + "front_left_wheel",
            self._prefix + self.PREFIX_SUMMIT_XL + "front_right_wheel",
        ]

    @classmethod
    def get_chassis_link_names(self) -> List[str]:
        return [self._prefix + self.PREFIX_SUMMIT_XL + "base_link"]

    @classmethod
    def get_arm_link_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_MANIPULATOR + "link_base",
            self._prefix + self.PREFIX_MANIPULATOR + "link_1",
            self._prefix + self.PREFIX_MANIPULATOR + "link_2",
            self._prefix + self.PREFIX_MANIPULATOR + "link_3",
            self._prefix + self.PREFIX_MANIPULATOR + "link_4",
            self._prefix + self.PREFIX_MANIPULATOR + "link_5",
            self._prefix + self.PREFIX_MANIPULATOR + "link_6",
            self._prefix + self.PREFIX_MANIPULATOR + "link_7",
        ]

    @classmethod
    def get_gripper_link_names(self) -> List[str]:
        return [
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_1",
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_2",
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_3",
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_tip_1",
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_tip_2",
            self._prefix + self.PREFIX_MANIPULATOR + "link_finger_tip_3",
        ]

    @classmethod
    def get_base_link_name(self) -> str:
        return self.get_arm_link_names()[0]

    @classmethod
    def get_ee_link_name(self) -> str:
        return (self._prefix + self.PREFIX_MANIPULATOR + "end_effector",)

    def get_initial_joint_positions(self) -> List[float]:
        return self._initial_joint_positions
