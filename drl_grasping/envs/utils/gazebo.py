from typing import Tuple, Union

from gym_ignition.scenario.model_wrapper import ModelWrapper
from numpy import exp
from scenario.bindings.gazebo import Link, World
from scipy.spatial.transform import Rotation

from drl_grasping.envs.utils.conversions import quat_to_wxyz, quat_to_xyzw
from drl_grasping.envs.utils.math import quat_mul


def get_model_pose(
    world: World,
    model: Union[ModelWrapper, str],
    link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Return pose of model's link. Orientation is represented as wxyz quaternion or xyzw based on the passed argument `xyzw`.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.get_link(link_name=model.link_names()[0])
    elif isinstance(link, str):
        # Get reference to the link from its name if string
        link = model.get_link(link_name=link)

    # Get position and orientation
    position = link.position()
    quat = link.orientation()

    # Convert to xyzw order if desired
    if xyzw:
        quat = quat_to_xyzw(quat)

    # Return pose of the model's link
    return (
        position,
        quat,
    )


def get_model_position(
    world: World,
    model: Union[ModelWrapper, str],
    link: Union[Link, str, None] = None,
) -> Tuple[float, float, float]:
    """
    Return position of model's link.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.get_link(link_name=model.link_names()[0])
    elif isinstance(link, str):
        # Get reference to the link from its name if string
        link = model.get_link(link_name=link)

    # Return position of the model's link
    return link.position()


def get_model_orientation(
    world: World,
    model: Union[ModelWrapper, str],
    link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Return orientation of model's link that is represented as wxyz quaternion or xyzw based on the passed argument `xyzw`.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.get_link(link_name=model.link_names()[0])
    elif isinstance(link, str):
        # Get reference to the link from its name if string
        link = model.get_link(link_name=link)

    # Get orientation
    quat = link.orientation()

    # Convert to xyzw order if desired
    if xyzw:
        quat = quat_to_xyzw(quat)

    # Return orientation of the model's link
    return quat


def transform_move_to_model_pose(
    world: World,
    position: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Transform such that original `position` and `quat` are represented with respect to `target_model::target_link`.
    The resulting pose is still represented in world coordinate system.
    """

    target_frame_position, target_frame_quat = get_model_pose(
        world,
        model=target_model,
        link=target_link,
        xyzw=True,
    )

    transformed_position = Rotation.from_quat(target_frame_quat).apply(position)
    transformed_position = (
        transformed_position[0] + target_frame_position[0],
        transformed_position[1] + target_frame_position[1],
        transformed_position[2] + target_frame_position[2],
    )

    if not xyzw:
        target_frame_quat = quat_to_wxyz(target_frame_quat)
    transformed_quat = quat_mul(quat, target_frame_quat, xyzw=xyzw)

    return (transformed_position, transformed_quat)


def transform_move_to_model_position(
    world: World,
    position: Tuple[float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

    target_frame_position, target_frame_quat_xyzw = get_model_pose(
        world,
        model=target_model,
        link=target_link,
        xyzw=True,
    )

    transformed_position = Rotation.from_quat(target_frame_quat_xyzw).apply(position)
    transformed_position = (
        target_frame_position[0] + transformed_position[0],
        target_frame_position[1] + transformed_position[1],
        target_frame_position[2] + transformed_position[2],
    )

    return transformed_position


def transform_move_to_model_orientation(
    world: World,
    quat: Tuple[float, float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

    target_frame_quat = get_model_orientation(
        world,
        model=target_model,
        link=target_link,
        xyzw=xyzw,
    )

    transformed_quat = quat_mul(quat, target_frame_quat, xyzw=xyzw)

    return transformed_quat


def transform_change_reference_frame_pose(
    world: World,
    position: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Change reference frame of original `position` and `quat` from world coordinate system to `target_model::target_link` coordinate system.
    """

    target_frame_position, target_frame_quat = get_model_pose(
        world,
        model=target_model,
        link=target_link,
        xyzw=True,
    )

    transformed_position = (
        position[0] - target_frame_position[0],
        position[1] - target_frame_position[1],
        position[2] - target_frame_position[2],
    )
    transformed_position = Rotation.from_quat(target_frame_quat).apply(
        transformed_position, inverse=True
    )

    if not xyzw:
        target_frame_quat = quat_to_wxyz(target_frame_quat)
    transformed_quat = quat_mul(target_frame_quat, quat, xyzw=xyzw)

    return (tuple(transformed_position), transformed_quat)


def transform_change_reference_frame_position(
    world: World,
    position: Tuple[float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

    target_frame_position, target_frame_quat_xyzw = get_model_pose(
        world,
        model=target_model,
        link=target_link,
        xyzw=True,
    )

    transformed_position = (
        position[0] - target_frame_position[0],
        position[1] - target_frame_position[1],
        position[2] - target_frame_position[2],
    )
    transformed_position = Rotation.from_quat(target_frame_quat_xyzw).apply(
        transformed_position, inverse=True
    )

    return tuple(transformed_position)


def transform_change_reference_frame_orientation(
    world: World,
    quat: Tuple[float, float, float, float],
    target_model: Union[ModelWrapper, str],
    target_link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:

    target_frame_quat = get_model_orientation(
        world,
        model=target_model,
        link=target_link,
        xyzw=xyzw,
    )

    transformed_quat = quat_mul(target_frame_quat, quat, xyzw=xyzw)

    return transformed_quat
