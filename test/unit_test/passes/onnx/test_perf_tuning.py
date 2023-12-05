# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import create_dataloader, get_onnx_model
from unittest.mock import patch

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OrtPerfTuning
from olive.passes.onnx.perf_tuning import generate_test_name


@pytest.mark.parametrize(
    "config",
    [
        {"input_names": ["input"], "input_shapes": [[1, 1]]},
        {},
        {"dataloader_func": create_dataloader},
        {"dataloader_func": create_dataloader, "dataloader_func_kwargs": {"dummy_kwarg": 1}},
    ],
)
def test_ort_perf_tuning_pass(config, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


@patch("olive.passes.onnx.OrtPerfTuning._run_for_config")
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"input_names": ["input"], "input_shapes": [[1, 1]]},
        {"providers_list": ["CPUExecutionProvider", "CUDAExecutionProvider"], "device": "gpu"},
    ],
)
def test_ort_perf_tuning_with_customized_configs(mock_run, config):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)

    # execute
    p.run(input_model, None, None)

    # assert
    if "providers_list" not in config:
        assert mock_run.call_args.args[2]["providers_list"] == [
            "CPUExecutionProvider"
        ], "providers_list is not set correctly as ['CPUExecutionProvider'] by default when user does not specify it"
    if "device" not in config:
        assert (
            mock_run.call_args.args[2]["device"] == "cpu"
        ), "device is not set correctly as cpu by default when user does not specify it"
    for k, v in config.items():
        assert mock_run.call_args.args[2][k] == v, f"{k} is not set correctly as {v}"


@patch("olive.model.ONNXModel.get_io_config")
def test_ort_perf_tuning_pass_with_dynamic_shapes(mock_get_io_config, tmp_path):
    mock_get_io_config.return_value = {
        "input_names": ["input"],
        "input_shapes": [["input_0", "input_1"]],
        "input_types": ["float32", "float32"],
        "output_names": ["output"],
        "output_shapes": [["input_0", 10]],
        "output_types": ["float32", "float32"],
    }

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(TypeError) as e:
        # execute
        p.run(input_model, None, output_folder)
    assert "ones() received an invalid combination of arguments" in str(e.value)


@patch("olive.passes.onnx.perf_tuning.threads_num_binary_search")
def test_ort_perf_tuning_pass_with_import_error(mock_threads_num_binary_search, tmp_path):
    mock_threads_num_binary_search.side_effect = ModuleNotFoundError("test")

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(ModuleNotFoundError) as e:
        # execute
        p.run(input_model, None, output_folder)

    assert "test" in str(e.value)


def test_generate_test_name():
    test_params = {
        "execution_provider": [("CPUExecutionProvider", {})],
        "session_options": {
            "execution_mode": 1,
            "extra_session_config": None,
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 8,
        },
    }

    name = generate_test_name(test_params, True)
    assert name == (
        "cpu-{'execution_mode': 1, "
        "'extra_session_config': None, 'inter_op_num_threads': 1, 'intra_op_num_threads': 8}"
        "-{'io_bind': True}"
    )

    test_params = {
        "execution_provider": [("TensorrtExecutionProvider", {"trt_fp16_enable": True})],
        "session_options": {
            "execution_mode": 1,
            "extra_session_config": {
                "session.intra_op_thread_affinities": "0;1;2;3;4;5;6;7",
            },
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 8,
            "graph_optimization_level": 99,
        },
    }
    name = generate_test_name(test_params, False)
    assert name == (
        "('tensorrt', {'trt_fp16_enable': True})-"
        "{'execution_mode': 1, "
        "'extra_session_config': {'session.intra_op_thread_affinities': '0;1;2;3;4;5;6;7'}, "
        "'inter_op_num_threads': 1, 'intra_op_num_threads': 8, 'graph_optimization_level': 99}"
    )
