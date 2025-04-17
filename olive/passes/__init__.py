# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.olive_pass import FullPassConfig, Pass

REGISTRY = Pass.registry

__all__ = [
    "REGISTRY",
    "FullPassConfig",
    "Pass",
]
