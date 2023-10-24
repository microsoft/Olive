# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.olive_pass import FullPassConfig, Pass
from olive.passes.onnx import *  # noqa: F403
from olive.passes.openvino import *  # noqa: F403
from olive.passes.pass_config import PassParamDefault
from olive.passes.pytorch import *  # noqa: F403
from olive.passes.snpe import *  # noqa: F403

REGISTRY = Pass.registry

__all__ = [
    "Pass",
    "PassParamDefault",
    "FullPassConfig",
]
