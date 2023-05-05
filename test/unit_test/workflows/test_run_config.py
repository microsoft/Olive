# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path

import pytest

from olive.workflows.run.config import RunConfig


class TestRunConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.transformer_dataset_config_file = Path(__file__).parent / "mock_data" / "transformer_dataset.json"
        self.user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"

    def test_transformer_dataset_config_file(self):
        run_config = RunConfig.parse_file(self.transformer_dataset_config_file)
        for dc in run_config.data_config.values():
            dc.to_data_container().create_dataloader()

    def test_user_script_config_file(self):
        assert RunConfig.parse_file(self.user_script_config_file)
