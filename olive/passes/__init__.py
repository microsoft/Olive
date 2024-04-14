# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.olive_pass import FullPassConfig, Pass
from olive.passes.pass_config import PassModuleConfig, PassParamDefault

REGISTRY = Pass.registry

__all__ = [
    "Pass",
    "PassParamDefault",
    "PassModuleConfig",
    "FullPassConfig",
]
