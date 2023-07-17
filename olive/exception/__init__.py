# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
class OliveException(Exception):
    """
    Base class for Olive exceptions.
    """

    pass


class OlivePassException(OliveException):
    """
    Base class for Olive pass exceptions.
    """

    pass


class OliveEvaluationException(OliveException):
    """
    Base class for Olive evaluation exceptions.
    """

    pass
