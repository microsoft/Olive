# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Protocol, runtime_checkable


@runtime_checkable
class EvaluationSignal(Protocol):
    """Protocol for evaluation results fed back to search.

    Decouples the search subsystem from the concrete MetricResult class.
    Any object supporting dict-like access (``signal[key]``) where values
    have a ``.value`` attribute satisfies this protocol.
    """

    def __getitem__(self, key: str): ...
