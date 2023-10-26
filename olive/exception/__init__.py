# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
class OliveError(Exception):
    """Base class for Olive exceptions."""


class OlivePassError(OliveError):
    """Base class for Olive pass exceptions."""


class OliveEvaluationError(OliveError):
    """Base class for Olive evaluation exceptions."""


EXCEPTIONS_TO_RAISE = (AssertionError, AttributeError, ImportError, TypeError, ValueError)
