# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging


def set_verbosity(verbose):
    logging.getLogger(__name__.split(".")[0]).setLevel(verbose)


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
