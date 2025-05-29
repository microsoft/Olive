# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any


class IoConfigMixin:
    """Provide the following model get io config functionalities.

    Each model handler could choose to override the behavior.
    For example, both PyTorch model and ONNX model handler choose to override the default behavior.
    """

    @property
    def io_config(self) -> dict[str, Any]:
        return self._io_config
