# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum


class OS(str, Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"


##### AzureML system #####

WORKFLOW_CONFIG = "workflow_config"
WORKFLOW_ARTIFACTS = "workflow_artifacts"
HF_LOGIN = "HF_LOGIN"
KEYVAULT_NAME = "KEYVAULT_NAME"


############# Engine #############

DEFAULT_WORKFLOW_ID = "default_workflow"


############# Packaging #############

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
