# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pydantic v2 native API module.

This module now uses pydantic v2 native APIs directly.
"""

# pylint: disable=redefined-builtin, wildcard-import, unused-wildcard-import

from pydantic import *  # noqa: F403
from pydantic import ConfigDict, RootModel, field_validator, model_validator  # noqa: F401
