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
        self.run_config_file = Path(__file__).parent / "mock_data" / "bert_config.json"

    def test_run_config_init(self):
        run_config = RunConfig.parse_file(self.run_config_file)
        for dc in run_config.data_container.values():
            dc.to_data_container().create_dataloader()
