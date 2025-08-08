# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from unittest.mock import patch

from olive.systems.utils.available_providers_runner import main as available_providers_main


@patch("onnxruntime.get_available_providers")
def test_available_providers_runner(mock_get_providers, tmp_path):
    mock_get_providers.return_value = ["DummyExecutionProvider"]
    output_path = tmp_path / "available_eps.json"

    # command
    args = ["--output_path", str(output_path)]

    # execute
    available_providers_main(args)

    # assert
    assert output_path.exists()
    mock_get_providers.assert_called_once()
    with output_path.open("r") as f:
        assert json.load(f) == ["DummyExecutionProvider"]
