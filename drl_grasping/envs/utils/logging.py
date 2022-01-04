from typing import Union

from gym import logger as gym_logger
from gym_ignition.utils import logger as gym_ign_logger


def set_log_level(log_level: Union[int, str]):
    """
    Set log level for (Gym) Ignition.
    """

    if not isinstance(log_level, int):
        log_level = getattr(gym_logger, str(log_level).upper())

    gym_ign_logger.set_level(
        level=log_level,
        scenario_level=log_level,
    )
