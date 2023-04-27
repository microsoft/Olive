# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import get_data_container_config

import pytest

from olive.data_container.config import DataContainerConfig
from olive.data_container.registry import Registry


class TestDataContainerConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = get_data_container_config()

    def test_default_property(self):
        dc_config = DataContainerConfig()
        for k, _ in dc_config.default_components.items():
            assert k in dc_config.components
        assert dc_config.dataloader
        assert dc_config.dataset
        assert dc_config.pre_process
        assert dc_config.post_process

    def test_customized_property(self):
        # function name
        assert self.dc_config.dataset.__name__ == "_test_dataset"
        assert self.dc_config.dataloader.__name__ == "_test_dataloader"
        assert self.dc_config.pre_process.__name__ == Registry.get_default_pre_process_component().__name__
        assert self.dc_config.post_process.__name__ == Registry.get_default_post_process_component().__name__
