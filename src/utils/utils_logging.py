# logging utility built upon code by 'eos87' from
# https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

import logging
import sys


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
