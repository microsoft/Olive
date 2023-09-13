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
    Olive output artifacts generation config
    """

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
    export_in_mlflow_format: bool = False
