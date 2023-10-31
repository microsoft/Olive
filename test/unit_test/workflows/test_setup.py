# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

from olive.workflows.run.config import RunConfig
from olive.workflows.run.run import dependency_setup


def test_dependency_setup():
    user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"
    run_config = RunConfig.parse_file(user_script_config_file)
    dependency_setup(run_config)
