# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BertConfig, GPT2Config, Qwen3Config

from olive.common.hf.model_io import get_model_dummy_input, get_model_io_config
from olive.common.hf.utils import (
    TEST_MODEL_MARKER_FILE,
    _apply_test_model_config,
    _load_test_model,
    load_model_from_task,
)


def test_load_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "text-classification"
    model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    model = load_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    ("model_config", "hidden_layers_attr"),
    [
        (BertConfig(num_hidden_layers=12), "num_hidden_layers"),  # pylint: disable=unexpected-keyword-arg
        (GPT2Config(n_layer=12), "n_layer"),  # pylint: disable=unexpected-keyword-arg
    ],
)
def test_load_model_from_task_test_model_config(model_config, hidden_layers_attr):
    created_model = MagicMock(spec=torch.nn.Module)

    with (
        patch("transformers.pipelines.check_task") as mock_check_task,
        patch("olive.common.hf.utils.from_pretrained", return_value=model_config) as mock_from_pretrained,
    ):
        mock_model_class = MagicMock()
        mock_model_class.from_config.return_value = created_model
        mock_check_task.return_value = ("text-classification", {"pt": (mock_model_class,)}, None)

        model = load_model_from_task("text-classification", "dummy-model", test_model_config={"hidden_layers": 2})

    assert model is created_model
    mock_from_pretrained.assert_called_once()
    mock_model_class.from_config.assert_called_once()
    assert getattr(mock_model_class.from_config.call_args.args[0], hidden_layers_attr) == 2


def test_load_model_from_task_test_model_config_fails_without_fallback():
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg

    with (
        patch("transformers.pipelines.check_task") as mock_check_task,
        patch("olive.common.hf.utils.from_pretrained", return_value=model_config),
    ):
        first_model_class = MagicMock()
        first_model_class.from_config.side_effect = ValueError("unexpected architecture")
        second_model_class = MagicMock()
        second_model_class.from_config.return_value = MagicMock(spec=torch.nn.Module)
        mock_check_task.return_value = ("text-classification", {"pt": (first_model_class, second_model_class)}, None)

        with pytest.raises(ValueError, match="unexpected architecture"):
            load_model_from_task("text-classification", "dummy-model", test_model_config={"hidden_layers": 2})

    first_model_class.from_config.assert_called_once()
    second_model_class.from_config.assert_not_called()


def test_load_model_from_task_test_model_config_saves_model(tmp_path):
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    created_model = MagicMock()
    test_model_path = tmp_path / "saved_test_model"

    with (
        patch("transformers.pipelines.check_task") as mock_check_task,
        patch("olive.common.hf.utils.from_pretrained", return_value=model_config),
    ):
        mock_model_class = MagicMock()
        mock_model_class.from_config.return_value = created_model
        mock_check_task.return_value = ("text-classification", {"pt": (mock_model_class,)}, None)

        model = load_model_from_task(
            "text-classification",
            "dummy-model",
            test_model_config={"num_hidden_layers": 2},
            test_model_path=str(test_model_path),
        )

    assert model is created_model
    mock_model_class.from_config.assert_called_once()
    created_model.save_pretrained.assert_called_once_with(str(test_model_path))
    assert json.loads((test_model_path / TEST_MODEL_MARKER_FILE).read_text())["type"] == "olive_hf_test_model"


def test_load_model_from_task_test_model_config_reuses_saved_model(tmp_path):
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    test_model_path = tmp_path / "saved_test_model"
    test_model_path.mkdir()
    (test_model_path / "config.json").write_text("{}")
    (test_model_path / TEST_MODEL_MARKER_FILE).write_text(json.dumps({"type": "olive_hf_test_model"}))
    loaded_model = MagicMock(spec=torch.nn.Module)

    with (
        patch("transformers.pipelines.check_task") as mock_check_task,
        patch(
            "olive.common.hf.utils.from_pretrained", side_effect=[model_config, loaded_model]
        ) as mock_from_pretrained,
    ):
        mock_model_class = MagicMock()
        mock_check_task.return_value = ("text-classification", {"pt": (mock_model_class,)}, None)

        model = load_model_from_task(
            "text-classification",
            "dummy-model",
            test_model_config={"num_hidden_layers": 2},
            test_model_path=str(test_model_path),
        )

    assert model is loaded_model
    mock_model_class.from_config.assert_not_called()
    assert mock_from_pretrained.call_args_list[1].args[1] == str(test_model_path)


def test_load_model_from_task_test_model_config_rejects_non_test_model_dir(tmp_path):
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    test_model_path = tmp_path / "saved_test_model"
    test_model_path.mkdir()
    (test_model_path / "config.json").write_text("{}")

    with (
        patch("transformers.pipelines.check_task") as mock_check_task,
        patch("olive.common.hf.utils.from_pretrained", return_value=model_config),
    ):
        mock_model_class = MagicMock()
        mock_check_task.return_value = ("text-classification", {"pt": (mock_model_class,)}, None)

        with pytest.raises(ValueError, match="is not an Olive test model directory"):
            load_model_from_task(
                "text-classification",
                "dummy-model",
                test_model_config={"num_hidden_layers": 2},
                test_model_path=str(test_model_path),
            )

    mock_model_class.from_config.assert_not_called()


def test_apply_test_model_config_updates_qwen3_layer_types():
    model_config = Qwen3Config()
    model_config.num_hidden_layers = 4
    model_config.layer_types = model_config.layer_types[:4]

    updated_config = _apply_test_model_config(model_config, {"hidden_layers": 2})

    assert updated_config.num_hidden_layers == 2
    assert updated_config.layer_types == model_config.layer_types[:2]
    reloaded_config = Qwen3Config(**updated_config.to_dict())
    assert reloaded_config.num_hidden_layers == 2
    assert len(reloaded_config.layer_types) == 2
    assert reloaded_config.layer_types == model_config.layer_types[:2]


def test_load_test_model_omits_unsupported_trust_remote_code_kwarg():
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    captured = {}

    class MockModelClass:
        @staticmethod
        def from_config(config):
            captured["config"] = config
            return config

    created_model = _load_test_model(MockModelClass, model_config, trust_remote_code=True)

    assert created_model is model_config
    assert captured == {"config": model_config}


def test_load_test_model_omits_none_trust_remote_code_kwarg():
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    captured = {}

    class MockModelClass:
        @staticmethod
        def from_config(config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs
            return config

    created_model = _load_test_model(MockModelClass, model_config)

    assert created_model is model_config
    assert captured == {"config": model_config, "kwargs": {}}


def test_load_test_model_passes_supported_trust_remote_code_kwarg():
    model_config = BertConfig(num_hidden_layers=12)  # pylint: disable=unexpected-keyword-arg
    captured = {}

    class MockModelClass:
        @staticmethod
        def from_config(config, trust_remote_code=None):
            captured["config"] = config
            captured["trust_remote_code"] = trust_remote_code
            return config

    created_model = _load_test_model(MockModelClass, model_config, trust_remote_code=True)

    assert created_model is model_config
    assert captured == {"config": model_config, "trust_remote_code": True}


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
