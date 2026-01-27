# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
import torch

from olive.common.hf.model_io import get_model_dummy_input, get_model_io_config
from olive.common.hf.utils import load_model_from_task


def test_load_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "text-classification"
    model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    model = load_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    ("exceptions", "expected_exception", "expected_message"),
    [
        ([None], None, None),
        ([FileNotFoundError("file not found error")], FileNotFoundError, "file not found error"),
        ([ValueError("value error")], ValueError, "value error"),
        ([ImportError("import error")], ImportError, "import error"),
        (
            [ImportError("import error"), ValueError("value error")],
            ImportError,
            "import error",
        ),
        ([None, ImportError("import error")], None, None),
        ([None, ValueError("value error")], None, None),
        ([ValueError("value error"), None], None, None),
        ([ValueError("value error"), ImportError("import error")], ImportError, "import error"),
        ([ValueError("value error 1"), ValueError("value error 2")], ValueError, "value error 2"),
    ],
)
@patch("olive.common.hf.utils.get_model_config")
def test_load_model_from_task_exception_handling(_, exceptions, expected_exception, expected_message):
    with patch("transformers.pipelines.check_task") as mock_check_task:
        mocked_model_classes = []
        for exception in exceptions:
            mock_class = MagicMock()
            if exception is None:
                mock_class.from_pretrained = MagicMock(return_value=MagicMock(spec=torch.nn.Module))
            else:
                mock_class.from_pretrained = MagicMock(side_effect=exception)
            mocked_model_classes.append(mock_class)

        mock_check_task.return_value = ("text-classification", {"pt": tuple(mocked_model_classes)}, None)

        if expected_exception is None:
            model = load_model_from_task("text-classification", "dummy-model-name")
            assert isinstance(model, torch.nn.Module)
        else:
            with pytest.raises(expected_exception, match=expected_message):
                _ = load_model_from_task("text-classification", "dummy-model-name")


def get_model_name_task(with_past: bool):
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    task = "text-generation"
    if with_past:
        task = "text-generation-with-past"
    return model_name, task


@pytest.mark.parametrize("with_past", [True, False])
def test_get_model_dummy_input(with_past):
    dummy_input = get_model_dummy_input(*get_model_name_task(with_past))
    expected_keys = ["input_ids", "attention_mask", "position_ids"]
    if with_past:
        expected_keys.append("past_key_values")
    assert set(dummy_input.keys()) == set(expected_keys)


@pytest.mark.parametrize("use_cache", [True, False, None])
@pytest.mark.parametrize("with_past", [True, False])
def test_get_model_io_config(use_cache, with_past):
    model_name, task = get_model_name_task(with_past)
    kwargs = {"use_cache": use_cache} if use_cache is not None else {}
    model = load_model_from_task(task, model_name, **kwargs)
    io_config = get_model_io_config(model_name, task, model, **kwargs)
    expected_keys = ["input_names", "output_names", "dynamic_axes", "dynamic_shapes"]
    assert set(io_config.keys()) == set(expected_keys)
    expected_input_names = ["input_ids", "attention_mask", "position_ids"]
    expected_output_names = ["logits"]
    for layer_id in range(model.config.num_hidden_layers):
        if use_cache is None or use_cache:
            expected_output_names.extend([f"present.{layer_id}.key", f"present.{layer_id}.value"])
        if with_past:
            expected_input_names.extend([f"past_key_values.{layer_id}.key", f"past_key_values.{layer_id}.value"])
    assert io_config["input_names"] == expected_input_names
    assert io_config["output_names"] == expected_output_names
    assert set(io_config["dynamic_axes"].keys()) == set(expected_input_names + expected_output_names)
