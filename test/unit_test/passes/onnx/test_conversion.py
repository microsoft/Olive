# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
import shutil
import sys
from itertools import chain
from pathlib import Path
from test.unit_test.utils import ONNX_MODEL_PATH, get_hf_model, get_onnx_model, get_pytorch_model, pytorch_model_loader
from unittest.mock import patch

import pytest
import torch

from olive.common.config_utils import validate_config
from olive.common.constants import OS
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.config import IoConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion, OnnxOpVersionConversion


@pytest.mark.skipif(sys.version_info > (3, 8), reason="Failed with Python 3.10, need to investigate.")
@pytest.mark.parametrize(
    ("input_model", "use_dynamo_exporter"),
    [(get_pytorch_model(), True), (get_hf_model(), True), (get_pytorch_model(), False), (get_hf_model(), False)],
)
def test_onnx_conversion_pass_with_exporters(input_model, use_dynamo_exporter, tmp_path):
    # setup
    p = create_pass_from_dict(OnnxConversion, {"use_dynamo_exporter": use_dynamo_exporter}, disable_search=True)
    output_folder = str(tmp_path / "onnx")
    # The conversion need torch version > 1.13.1, otherwise, it will complain
    # Unsupported ONNX opset version: 18
    onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


# TODO(team): Failed in pipeline (linux gpu). Need to investigate.
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available() or True,
    reason="bitsandbytes requires Linux GPU.",
)
@pytest.mark.parametrize("add_quantized_modules", [True, False])
def test_onnx_conversion_pass_quant_model(add_quantized_modules, tmp_path):
    # setup
    quantized_modules = ["v_proj", "k_proj", "fc_in", "fc_out", "out_proj", "q_proj"]
    input_model = HfModelHandler(
        model_path="hf-internal-testing/tiny-random-gptj",
        load_kwargs={
            "quantization_method": "bitsandbytes",
            "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
        },
        model_attributes={"quantized_modules": quantized_modules} if add_quantized_modules else None,
    )
    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    assert set(onnx_model.model_attributes["quantized_modules"]) == set(quantized_modules)
    assert onnx_model.model_attributes["quantization_config"]["load_in_4bit"] is True
    assert onnx_model.model_attributes["quantization_config"]["bnb_4bit_quant_type"] == "nf4"


@pytest.mark.parametrize("target_opset", [9, 10, 16])
def test_onnx_op_version_conversion_pass(target_opset, tmp_path):
    input_model = get_onnx_model()
    # setup
    p = create_pass_from_dict(
        OnnxOpVersionConversion,
        {"target_opset": target_opset},
        disable_search=True,
    )
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, output_folder)

    # assert
    assert onnx_model.load_model().opset_import[0].version == target_opset


def get_io_config_phi2(model):
    input_names = [
        "input_ids",
        "attention_mask",
        *list(chain.from_iterable((f"past_key_values.{i}",) for i in range(32))),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present_key_values.{i}",) for i in range(32))),
    ]
    return {
        "input_names": input_names,
        "output_names": output_names,
    }


def get_dummy_inputs_phi2(model):
    def get_past_kv_inputs(batch_size: int, past_seq_len: int):
        num_heads, head_size = 31, 80
        torch_dtype = torch.float32
        return [(torch.rand(batch_size, past_seq_len, 1, num_heads, head_size, dtype=torch_dtype),) for _ in range(32)]

    input_ids = torch.randint(low=0, high=51200, size=(2, 8), dtype=torch.int64)
    attention_mask = torch.ones(2, 16, dtype=torch.int64)
    past_key_values = get_past_kv_inputs(2, 16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }


def get_io_config_llama2(model):
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(chain.from_iterable((f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(32))),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(32))),
    ]
    return {
        "input_names": input_names,
        "output_names": output_names,
    }


def get_dummy_inputs_llama2(_):
    def get_past_kv_inputs(batch_size: int, past_seq_len: int):
        num_heads = 32
        head_size = 80
        torch_dtype = torch.float32
        return [
            (
                torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
                torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            )
            for _ in range(32)
        ]

    input_ids = torch.randint(low=0, high=51200, size=(2, 8), dtype=torch.int64)
    attention_mask = torch.ones(2, 16, dtype=torch.int64)
    past_key_values = get_past_kv_inputs(2, 16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }


@pytest.mark.parametrize(
    ("io_config_func", "dummy_inputs_func"),
    [
        (get_io_config_llama2, get_dummy_inputs_llama2),
        (get_io_config_phi2, get_dummy_inputs_phi2),
    ],
)
@patch("torch.onnx.export")
def test_onnx_conversion_with_past_key_values(mock_onnx_export, tmp_path, io_config_func, dummy_inputs_func):
    dummy_inputs = None

    def mock_onnx_export_func(*args, **kwargs):
        nonlocal dummy_inputs
        _, dummy_inputs, output_path = args
        shutil.copyfile(ONNX_MODEL_PATH, output_path)

    output_folder = tmp_path / "onnx"
    output_folder.mkdir(parents=True, exist_ok=True)
    input_model = PyTorchModelHandler(
        model_loader=pytorch_model_loader,
        model_path=None,
        io_config=io_config_func,
        dummy_inputs_func=dummy_inputs_func,
    )
    mock_onnx_export.side_effect = mock_onnx_export_func
    # setup
    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    _ = p.run(input_model, str(output_folder))
    assert "past_key_values" in dummy_inputs  # pylint: disable=unsupported-membership-test


@pytest.mark.parametrize(
    "dynamic_shapes",
    [
        [{"0": ["axis_batch", 0, 1024], "1": ["x_axis", 0, 8]}, {"0": ["axis_batch", 0, 1024], "1": ["y_axis", 0, 6]}],
        {
            "input_x": {"0": ["axis_batch", 0, 1024], "1": ["x_axis", 0, 8]},
            "input_y": {"0": ["axis_batch", 0, 1024], "1": ["y_axis", 0, 6]},
        },
    ],
)
def test_dynamic_shapes_passes_validate_io_config_with_both_list_and_dict_format(dynamic_shapes):
    config = {"input_names": ["input_x", "input_y"], "output_names": ["logits"]}
    config["dynamic_shapes"] = dynamic_shapes
    io_config = validate_config(config, IoConfig)
    assert io_config.dynamic_shapes == dynamic_shapes


def _get_simulate_torch_float_tensor_inputs(return_tuple: bool = False):
    if return_tuple:
        return (
            torch.ones(5),
            (torch.zeros(5), torch.ones(5)),
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
    return {
        "w": torch.ones(5),
        "x": (torch.zeros(5), torch.ones(5)),
        "y": {"a": torch.zeros(5), "b": torch.ones(5)},
        "z": torch.ones(4),
    }


@pytest.mark.parametrize(
    ("dynamic_shapes", "expected_dynamic_shapes", "inputs"),
    [
        (
            [
                {"0": ["axis_batch", 0, 1024], "1": ["x_axis", 0, 8]},
                [{"1": ["x_axis", 0, 8]}, {"0": ["axis_batch", 0, 1024]}],
                {"a": {"0": ["axis_batch", 0, 1024]}, "b": {"1": ["x_axis", 0, 8]}},
                None,
            ],
            (
                {0: ["axis_batch", 0, 1024], 1: ["x_axis", 0, 8]},
                ({1: ["x_axis", 0, 8]}, {0: ["axis_batch", 0, 1024]}),
                {"a": {0: ["axis_batch", 0, 1024]}, "b": {1: ["x_axis", 0, 8]}},
                None,
            ),
            _get_simulate_torch_float_tensor_inputs(return_tuple=True),
        ),
        (
            {
                "w": {"0": ["axis_batch", 0, 1024], "1": ["x_axis", 0, 8]},
                "x": [{"1": ["x_axis", 0, 8]}, {"0": ["axis_batch", 0, 1024]}],
                "y": {"a": {"0": ["axis_batch", 0, 1024]}, "b": {"1": ["x_axis", 0, 8]}},
                "z": None,
            },
            {
                "w": {0: ["axis_batch", 0, 1024], 1: ["x_axis", 0, 8]},
                "x": ({1: ["x_axis", 0, 8]}, {0: ["axis_batch", 0, 1024]}),
                "y": {"a": {0: ["axis_batch", 0, 1024]}, "b": {1: ["x_axis", 0, 8]}},
                "z": None,
            },
            _get_simulate_torch_float_tensor_inputs(return_tuple=False),
        ),
    ],
    ids=["in_nested_tuple_inputs", "in_nested_dict_format"],
)
def test___validate_dynamic_shapes_follow_input_format(dynamic_shapes, expected_dynamic_shapes, inputs):
    from olive.passes.onnx.conversion import _validate_dynamic_shapes

    converted_dynamic_shapes = _validate_dynamic_shapes(dynamic_shapes, inputs)
    assert converted_dynamic_shapes == expected_dynamic_shapes
