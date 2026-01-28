# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import patch

import pytest

import olive.data.template as data_config_template
from olive.data.config import DataComponentConfig
from test.utils import create_raw_data


class TestDataConfigTemplate:
    @patch("datasets.load_dataset")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_huggingface_template(self, mock_from_pretrained, mock_load_dataset):
        dataloader = data_config_template.huggingface_data_config_template(
            model_name="Intel/bert-base-uncased-mrpc",
            task="text-classification",
            load_dataset_config=DataComponentConfig(
                params={
                    "data_name": "nyu-mll/glue",
                    "subset": "mrpc",
                    "split": "train",
                }
            ),
            pre_process_data_config=DataComponentConfig(
                params={
                    "input_cols": ["sentence1", "sentence2"],
                    "label_col": "label",
                }
            ),
            dataloader_config=DataComponentConfig(
                params={
                    "batch_size": 32,
                }
            ),
        )
        dataloader = dataloader.to_data_container().create_dataloader()
        assert dataloader is not None, "Failed to create dataloader from huggingface template."

    @pytest.mark.parametrize(
        "input_names",
        [None, ["input_ids", "attention_mask", "token_type_ids"]],
    )
    def test_dummy_template(self, input_names):
        dataloader = data_config_template.dummy_data_config_template(
            input_names=input_names,
            input_shapes=[[1, 128], [1, 128], [1, 128]],
            input_types=["int64", "int64", "int64"],
        )
        dummy_inputs, _ = dataloader.to_data_container().get_first_batch()
        if input_names:
            assert isinstance(dummy_inputs, dict), "Failed to create dummy dict dataset from dummy template."
        else:
            assert isinstance(dummy_inputs, list), "Failed to create dummy list input from dummy template."

    def test_raw_data_template(self, tmpdir):
        input_names = ["float_input", "int_input"]
        input_shapes = [[1, 3], [1, 2]]
        input_types = ["float32", "int32"]
        create_raw_data(tmpdir, input_names, input_shapes, input_types)

        dc = data_config_template.raw_data_config_template(
            data_dir=str(tmpdir), input_names=input_names, input_shapes=input_shapes, input_types=input_types
        )
        input_data, _ = dc.to_data_container().get_first_batch()
        assert isinstance(input_data, dict), "Failed to create raw data dict dataset from raw data template."
