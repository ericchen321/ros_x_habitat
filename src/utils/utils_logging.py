# logging utility built upon code by 'eos87' from
# https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

import logging
import sys
import numpy as np
from habitat.tasks.nav.nav import (
    merge_sim_episode_config,
    SimulatorTaskAction,
    MoveForwardAction,
    TurnLeftAction,
    TurnRightAction,
    StopAction,
)
import math
from habitat.utils.geometry_utils import angle_between_quaternions


def setup_logger(name, log_file=None, level=logging.INFO):
    r"""
    To setup as many loggers as you want.
    :param name: name of the logger
    :param log_file: name of the file to export log. If not
        supplied then log to stdout
    :param level: level to log messages
    """

    handler = None
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def close_logger(logger):
    r"""
    Close a logger. Assume only one handler exists.
    """

    handler = logger.handlers[0]
    logger.removeHandler(handler)


def log_continuous_actuation(
    action,
    new_position,
    current_position,
    new_rotation,
    current_rotation,
    dataframe,
    filename):
    r"""
    Log the actuation of a discrete action in the continuous action space.
    """
    displacement = np.linalg.norm(new_position - current_position)
    # NOTE: to get angle between quarternions, use angle_between_quaternions()
    # from geometry_utils in habitat
    angle_diff = math.degrees(
        angle_between_quaternions(new_rotation, current_rotation)
    )
    if isinstance(action, TurnLeftAction):
        dataframe = dataframe.append(
            {
                "action": "TurnLeft",
                "desired_value": 10.0,
                "actual_value": angle_diff,
            },
            ignore_index=True,
        )
    elif isinstance(action, TurnRightAction):
        dataframe = dataframe.append(
            {
                "action": "TurnRight",
                "desired_value": 10.0,
                "actual_value": angle_diff,
            },
            ignore_index=True,
        )
    elif isinstance(action, MoveForwardAction):
        dataframe = dataframe.append(
            {
                "action": "MoveForward",
                "desired_value": 0.25,
                "actual_value": displacement,
            },
            ignore_index=True,
        )
    else:
        pass

    if not dataframe.empty:
        print("Updating csv: " + filename)
        dataframe.to_csv(filename, index=False)
