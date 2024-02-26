# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pydantic v1 compatibility module.

Pydantic v2 has breaking changes that are not compatible with the current version of Olive.
Migration Guide: https://docs.pydantic.dev/latest/migration/.

In order to support both versions of Pydantic, we use this module to access pydantic's v1 API.
"""

# pylint: disable=redefined-builtin, wildcard-import, unused-wildcard-import

try:
    # pydantic v2
    from pydantic.v1 import *  # noqa: F403
except ImportError:
    # pydantic v1
    from pydantic import *  # noqa: F403
