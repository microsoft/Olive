# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict


class IoConfigMixin:
    def get_io_config(self) -> Dict[str, Any]:
        return self.io_config
