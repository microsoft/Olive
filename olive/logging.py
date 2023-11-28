# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys
from datetime import datetime


def get_olive_logger():
    return logging.getLogger(__name__.split(".", maxsplit=1)[0])


def set_verbosity(verbose):
    get_olive_logger().setLevel(verbose)


def set_verbosity_info():
    set_verbosity(logging.INFO)


def set_verbosity_warning():
    set_verbosity(logging.WARNING)


def set_verbosity_debug():
    set_verbosity(logging.DEBUG)


def set_verbosity_error():
    set_verbosity(logging.ERROR)


def set_verbosity_critical():
    set_verbosity(logging.CRITICAL)


def get_logger_level(level):
    """Get Python logging level for the integer level.

    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    level_map = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR, 4: logging.CRITICAL}
    # check if level is valid
    if level not in level_map:
        raise ValueError(f"Invalid level {level}, should be one of {list(level_map.keys())}")

    return level_map[level]


def set_default_logger_severity(level):
    """Set log level for olive package.

    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    # set logger level
    set_verbosity(get_logger_level(level))


def set_ort_logger_severity(level):
    """Set log level for onnxruntime package.

    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    logger_level = get_logger_level(level)

    # set logger level
    ort_logger = logging.getLogger("onnxruntime")
    ort_logger.setLevel(logger_level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [onnxruntime]-[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    ort_logger.addHandler(stream_handler)


def enable_filelog(level):
    olive_logger = get_olive_logger()

    file_handler = logging.FileHandler("olive-{:%Y-%m-%d-%H-%M}.log".format(datetime.now()))
    file_handler.setLevel(get_logger_level(level))
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
    file_handler.setFormatter(formatter)
    olive_logger.addHandler(file_handler)
