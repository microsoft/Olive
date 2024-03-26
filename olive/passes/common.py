# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


def get_qualcomm_env_config():
    return {
        "use_olive_env": PassConfigParam(
            type_=bool,
            default_value=True,
            description=(
                "Whether to use the Olive built-in environment. Usually, if you do not prepare the environment with"
                "Olive's `python -m olive.platform_sdk.qualcomm.configure --py_version 3.8 --sdk qnn/snpe`"
                " you should set `use_olive_env` to False."
                " If set to True, only QNN_SDK_ROOT/SNPE_ROOT need to be set,"
                " other environment variables will be set by Olive. If set to False,"
                " QNN_SDK_ROOT/SNPE_ROOOT, LD_LIBRARY_PATH, PYTHONPATH and PATH need to be set as:"
                " QNN_SDK_ROOT: the path to the QNN SDK directory;"
                " LD_LIBRARY_PATH: $QNN_SDK_ROOT/lib/<target_arch>;"
                " PYTHONPATH: $QNN_SDK_ROOT/lib/python;"
                " PATH: $QNN_SDK_ROOT/bin/<target_arch>."
                " <target_arch> is the target architecture in"
                " olive.platform_sdk.qualcomm.constants.SDKTargetDevice."
            ),
        ),
    }
