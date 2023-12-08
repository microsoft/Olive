# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict


class IoConfigMixin:
    """Provide the following model get io config functionalites.

    Each model handler could choose to override the behavior.
    For example, both PyTorch model and ONNX model handler choose to override the default behavior.
    """

    def get_io_config(self) -> Dict[str, Any]:
        return self.io_config
