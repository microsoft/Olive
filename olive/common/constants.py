# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.utils import StrEnumBase


class OS(StrEnumBase):
    WINDOWS = "Windows"
    LINUX = "Linux"


############# Engine #############

DEFAULT_WORKFLOW_ID = "default_workflow"
DEFAULT_CACHE_DIR = ".olive-cache"


############# Packaging #############

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"

############# HF #############

DEFAULT_HF_TASK = "text-generation-with-past"


########### Model ###########

LOCAL_INPUT_MODEL_ID = "local_input_model"


########### Cache ###########

ACCOUNT_URL_TEMPLATE = "https://{account_name}.blob.core.windows.net"
