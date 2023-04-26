# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from olive.data_container.config import DataContainerConfig
from olive.data_container.container.base_container import BaseContainer


class TestDataContainerConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = DataContainerConfig()
        self.dc = BaseContainer(config=self.dc_config)

    def test_constructor(self):
        assert self.dc.config
        assert self.dc
