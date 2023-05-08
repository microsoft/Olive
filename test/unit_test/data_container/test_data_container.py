# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import get_data_config, get_dc_params_config, get_glue_huggingface_data_config

import pytest

from olive.data.config import DataConfig
from olive.data.container.base_container import BaseContainer


class TestDataConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = get_data_config()
        self.dc = BaseContainer(config=self.dc_config)

    def test_constructor(self):
        dc_config = DataConfig()
        dc = BaseContainer(config=dc_config)
        assert dc.config
        assert dc

    def test_params_override(self):
        dc_config = get_dc_params_config()
        assert dc_config.components["dataset"].params["batch_size"] == 10
        assert "label_from_params_config" in dc_config.components["dataset"].params["label_cols"]

    def test_huggingface_constructor(self):
        dc_config = DataConfig(type="HuggingfaceContainer")
        dc = dc_config.to_data_container()
        assert dc.config.dataset.__name__.startswith("huggingface")

    def test_huggingface_dc_runner(self):
        dc_config = get_glue_huggingface_data_config()
        # override the default components from task_type
        assert dc_config.components["post_process"].type == "text_classification_post_process"
        dc = dc_config.to_data_container()
        dc.create_dataloader()
        dc.create_calibration_dataloader()

    def test_dc_runner(self):
        try:
            dataset = self.dc.dataset()
            self.dc.pre_process(dataset)
        except Exception as e:
            pytest.fail(f"Failed to run get pre_process from data config: {e}")
