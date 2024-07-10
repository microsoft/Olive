# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum


class OS(str, Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"


############# Engine #############

DEFAULT_WORKFLOW_ID = "default_workflow"


############# Packaging #############

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04"
