# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys


def set_verbosity(verbose):
    logging.getLogger(__name__.split(".", maxsplit=1)[0]).setLevel(verbose)


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


def set_default_logger_severity(level):
    """Set log level for olive package.

    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    # mapping from level to logging level
    level_map = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR, 4: logging.CRITICAL}

    # check if level is valid
    if level not in level_map:
        raise ValueError(f"Invalid level {level}, should be one of {list(level_map.keys())}")

    # set logger level
    set_verbosity(level_map[level])


def set_ort_logger_severity(level):
    """Set log level for onnxruntime package.

    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    # mapping from level to logging level
    level_map = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR, 4: logging.CRITICAL}

    # check if level is valid
    if level not in level_map:
        raise ValueError(f"Invalid level {level}, should be one of {list(level_map.keys())}")

    # set logger level
    ort_logger = logging.getLogger("onnxruntime")
    ort_logger.setLevel(level_map[level])
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [onnxruntime]-[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    ort_logger.addHandler(stream_handler)
