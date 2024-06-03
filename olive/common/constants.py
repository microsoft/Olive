# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum

DEFAULT_WORKFLOW_ID = "default_workflow"


class OS(str, Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"
