# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pydantic v2 native API module.

This module now uses pydantic v2 native APIs directly.
"""

# pylint: disable=redefined-builtin, wildcard-import, unused-wildcard-import

from pydantic import *  # noqa: F403, F401
from pydantic import ConfigDict, RootModel, field_validator, model_validator  # noqa: F401


# Compatibility wrapper for root_validator -> model_validator
def root_validator(*_, pre=False, skip_on_failure=False, allow_reuse=False, **__):
    """Compatibility wrapper for pydantic v1's root_validator using v2's model_validator."""
    mode = "before" if pre else "after"
    return model_validator(mode=mode)


# Compatibility wrapper for validator -> field_validator
def validator(*field_names, pre=False, always=False, allow_reuse=False, **kwargs):
    """Compatibility wrapper for pydantic v1's validator using v2's field_validator."""
    mode = "before" if pre else "after"
    return field_validator(*field_names, mode=mode)
