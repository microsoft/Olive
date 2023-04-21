# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum

from olive.common.config_utils import ConfigBase


class PackagingType(str, Enum):
    """
    Output Artifacts type
    """

    Zipfile = "Zipfile"


class PackagingConfig(ConfigBase):
    """
    Olive output artifacts generatio config
    """

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
