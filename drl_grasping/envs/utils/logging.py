from typing import Union

from gym import logger as gym_logger
from gym_ignition.utils import logger as gym_ign_logger


def set_log_level(log_level: Union[int, str]):
    """
    Set log level for (Gym) Ignition.
    """

    if not isinstance(log_level, int):
        log_level = str(log_level).upper()

        if "WARNING" == log_level:
            log_level = "WARN"
        elif not log_level in ["DEBUG", "INFO", "WARN", "ERROR", "DISABLED"]:
            log_level = "DISABLED"

        log_level = getattr(gym_logger, log_level)

    gym_ign_logger.set_level(
        level=log_level,
        scenario_level=log_level,
    )
