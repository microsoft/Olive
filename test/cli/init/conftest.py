# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_questionary(monkeypatch):
    """Replace ``questionary`` with a MagicMock in every module that imports it."""
    mock_q = MagicMock()

    # Modules that do ``import questionary`` at the top level.
    for mod in (
        "olive.cli.init.wizard",
        "olive.cli.init.onnx_flow",
        "olive.cli.init.pytorch_flow",
        "olive.cli.init.diffusers_flow",
    ):
        monkeypatch.setattr(f"{mod}.questionary", mock_q, raising=False)
    monkeypatch.setitem(sys.modules, "questionary", mock_q)
