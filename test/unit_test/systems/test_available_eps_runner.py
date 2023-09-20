# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from unittest.mock import patch

from olive.systems.available_eps_runner import main as available_eps_main


@patch("onnxruntime.get_available_providers")
def test_available_eps_script(mock_get_providers, tmp_path):
    dummy_eps = ["DummyExecutionProvider"]
    mock_get_providers.return_value = dummy_eps

    # command
    args = ["--output_path", str(tmp_path)]

    # execute
    available_eps_main(args)

    # assert
    available_eps_json_path = tmp_path / "available_eps.json"
    assert available_eps_json_path.exists()
    mock_get_providers.assert_called_once()
    with open(available_eps_json_path, "r") as f:
        assert json.load(f) == dummy_eps
