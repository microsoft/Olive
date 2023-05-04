# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_sc = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
_sc.setFormatter(_formatter)
_logger.addHandler(_sc)
_logger.propagate = False

__version__ = "0.1.0"


def set_default_logger_severity(level):
    """
    Set log level for olive package.
    :param level: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL
    """
    _logger.setLevel(level * 10)
