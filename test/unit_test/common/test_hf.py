# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.onnx import OnnxConfig

from olive.common.hf.utils import load_model_from_task
from olive.common.hf.model_io import get_onnx_config


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
def test_load_model_from_task_exception_handling(exceptions, expected_exception, expected_message):
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


@pytest.mark.parametrize(
    ("model_name", "task", "feature"),
    [
        ("hf-internal-testing/tiny-random-BertForSequenceClassification", "text-classification", "default"),
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "text-generation", "default"),
    ],
)
def test_get_onnx_config(model_name, task, feature):
    onnx_config = get_onnx_config(model_name, task, feature)
    assert isinstance(onnx_config, OnnxConfig)
