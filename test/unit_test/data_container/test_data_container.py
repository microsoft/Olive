# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import (
    create_raw_data,
    get_data_config,
    get_dc_params_config,
    get_glue_huggingface_data_config,
)

import numpy as np
import pytest

from olive.data.config import DataConfig
from olive.data.container.data_container import DataContainer


class TestDataConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dc_config = get_data_config()
        self.dc = DataContainer(config=self.dc_config)

    def test_constructor(self):
        dc_config = DataConfig()
        dc = DataContainer(config=dc_config)
        assert dc.config
        assert dc

    def test_params_override(self):
        dc_config = get_dc_params_config()
        assert dc_config.components["load_dataset"].params["batch_size"] == 10
        assert "label_from_params_config" in dc_config.components["load_dataset"].params["label_cols"]

    def test_huggingface_constructor(self):
        dc_config = DataConfig(type="HuggingfaceContainer")
        dc = dc_config.to_data_container()
        assert dc.config.load_dataset.__name__.startswith("huggingface")

    def test_huggingface_dc_runner(self):
        dc_config = get_glue_huggingface_data_config()
        # override the default components from task_type
        assert dc_config.components["post_process_data"].type == "text_classification_post_process"
        dc = dc_config.to_data_container()
        dc.create_dataloader(data_root_path=None)
        dc.create_calibration_dataloader(data_root_path=None)

    def test_raw_data_constructor(self):
        dc_config = DataConfig(type="RawDataContainer")
        dc = dc_config.to_data_container()
        assert dc.config.load_dataset.__name__.startswith("raw_dataset")

    def test_raw_data_runner(self, tmpdir):
        input_names = ["float_input", "int_input"]
        input_shapes = [[1, 3], [1, 2]]
        input_types = ["float32", "int32"]
        data = create_raw_data(tmpdir, input_names, input_shapes, input_types)

        dc_config = DataConfig(
            type="RawDataContainer",
            params_config={
                "data_dir": str(tmpdir),
                "input_names": input_names,
                "input_shapes": input_shapes,
                "input_types": input_types,
            },
        )
        dc = dc_config.to_data_container()

        # check the dataset
        dataset = dc.load_dataset()
        assert len(dataset) == 1
        for input_name in input_names:
            input_data, _ = dataset[0]
            assert input_name in input_data
            assert input_data[input_name].shape == tuple(input_shapes[input_names.index(input_name)])
            assert input_data[input_name].dtype == input_types[input_names.index(input_name)]
            assert np.array_equal(input_data[input_name], data[input_name][0])

        dc.create_dataloader(data_root_path=None)
        dc.create_calibration_dataloader(data_root_path=None)

    def test_dc_runner(self):
        try:
            dataset = self.dc.load_dataset()
            self.dc.pre_process(dataset)
        except Exception as e:
            pytest.fail(f"Failed to run get pre_process from data config: {e}")
