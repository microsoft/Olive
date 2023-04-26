# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from olive.data_container.config import DataContainerConfig


class TestDataContainerConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = DataContainerConfig()

    def test_property(self):
        assert self.dc_config.dataloader
        assert self.dc_config.dataset
        assert self.dc_config.pre_process
        assert self.dc_config.post_process
