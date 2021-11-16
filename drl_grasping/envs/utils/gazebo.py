from drl_grasping.envs.utils.conversions import quat_to_xyzw, quat_to_wxyz
from gym_ignition.scenario.model_wrapper import ModelWrapper
from scenario.bindings.gazebo import World, Link
from typing import Tuple, Union
import numpy as np


def get_model_pose(
    world: World,
    model: Union[ModelWrapper, str],
    link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return pose of model's link. Orientation is represented as wxyz quaternion or xyzw based on the passed argument `xyzw`.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.link_names()[0]
    elif isinstance(link, str):
        # Get reference to the link from its name if string is name
        link = model.get_link(link_name=link)

    # Return pose of the model's link
    return (
        get_model_position(world, model, link),
        get_model_orientation(world, model, link, xyzw),
    )


def get_model_position(
    world: World, model: Union[ModelWrapper, str], link: Union[Link, str, None] = None
) -> np.ndarray:
    """
    Return position of model's link.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.link_names()[0]
    elif isinstance(link, str):
        # Get reference to the link from its name if string is name
        link = model.get_link(link_name=link)

    # Return position of the model's link
    return np.array(link.position())


def get_model_orientation(
    world: World,
    model: Union[ModelWrapper, str],
    link: Union[Link, str, None] = None,
    xyzw: bool = False,
) -> np.ndarray:
    """
    Return orientation of model's link that is represented as wxyz quaternion or xyzw based on the passed argument `xyzw`.
    """

    if isinstance(model, str):
        # Get reference to the model from its name if string is passed
        model = world.to_gazebo().get_model(model).to_gazebo()

    if link is None:
        # Use the first link if not specified
        link = model.link_names()[0]
    elif isinstance(link, str):
        # Get reference to the link from its name if string is name
        link = model.get_link(link_name=link)

    # Get orientation and convert to ndarray
    quat = np.array(link.orientation())

    # Convert to xyzw order if desired
    if xyzw:
        quat = quat_to_xyzw(quat)

    # Return orientation of the model's link
    return quat
