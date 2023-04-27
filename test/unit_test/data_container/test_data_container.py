# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import get_data_container_config, get_huggingface_data_container_config

import pytest

from olive.data_container.config import DataContainerConfig
from olive.data_container.container.base_container import BaseContainer


class TestDataContainerConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = get_data_container_config()
        self.dc = BaseContainer(config=self.dc_config)

    def test_constructor(self):
        dc_config = DataContainerConfig()
        dc = BaseContainer(config=dc_config)
        assert dc.config
        assert dc

    def test_huggingface_constructor(self):
        dc_config = get_huggingface_data_container_config()
        dc = dc_config.to_data_container()
        assert dc.config.dataset.__name__.startswith("huggingface")

    def test_dc_runner(self):
        try:
            self.dc.dataset()
            self.dc.dataloader()
            self.dc.pre_process()
            self.dc.post_process()
        except Exception as e:
            pytest.fail(f"Failed to run data container: {e}")
