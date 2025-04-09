# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from unittest.mock import patch

import onnx
import pytest
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

from olive.common.pydantic_v1 import ValidationError
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.quantization import (
    OnnxMatMul4Quantizer,
    OnnxQuantization,
    OnnxQuantizationPreprocess,
    OnnxStaticQuantization,
)
from test.unit_test.utils import get_onnx_model, get_pytorch_model_dummy_input


class DummyCalibrationDataReader(CalibrationDataReader):
    # pylint: disable=W0223
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.sample_counter = 500

    def get_next(self) -> dict:
        if self.sample_counter <= 0:
            return None

        data = get_pytorch_model_dummy_input()
        try:
            item = {"input": data}
            self.sample_counter -= 1
            return item
        except Exception:
            return None


@Registry.register_dataloader()
def _test_quant_dataloader(dataset, batch_size, **kwargs):
    return DummyCalibrationDataReader(batch_size=batch_size)


@pytest.mark.parametrize("quant_format", ["QOperator", "QDQ"])
def test_static_quantization(quant_format, tmp_path):
    input_model = get_onnx_model()
    config = {
        "quant_mode": "static",
        "calibrate_method": "MinMax",
        "quant_format": quant_format,
        "per_channel": True,
        "reduce_range": True,
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "weight_type": "QUInt8",
        "activation_type": "QUInt8",
        "quant_preprocess": True,
    }
    p = create_pass_from_dict(OnnxQuantization, config, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out is not None


def test_dynamic_quantization(tmp_path):
    input_model = get_onnx_model()
    config = {"quant_mode": "dynamic"}
    p = create_pass_from_dict(OnnxQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)
    assert out is not None


def test_quantization_preprocess(tmp_path):
    input_model = get_onnx_model()
    config = {"skip_optimization": True, "skip_onnx_shape": False, "skip_symbolic_shape": True}
    p = create_pass_from_dict(OnnxQuantizationPreprocess, config, disable_search=True)

    out = p.run(input_model, tmp_path)
    assert out is not None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"op_types_to_quantize": ["Gemm", "Sigmoid"]}),
        ({"op_types_to_quantize": ["Gemm"]}, {"op_types_to_quantize": ["Gemm"]}),
        ({"op_types_to_exclude": ["Gemm"]}, {"op_types_to_quantize": ["Sigmoid"], "nodes_to_exclude": ["/fc1/Gemm"]}),
        (
            # this node does not exist in the model but using this instead of "/fc1/Gemm"
            # there is only one Gemm node so op_types_to_quantize differs after 1.21.0
            {"nodes_to_exclude": ["/fc2/Gemm"]},
            {"op_types_to_quantize": ["Gemm", "Sigmoid"], "nodes_to_exclude": ["/fc2/Gemm"]},
        ),
    ],
)
@pytest.mark.parametrize("is_qnn", [True, False])
@patch("onnxruntime.quantization.quantize_static")
def test_nodes_and_ops(mock_quantize_static, tmp_path, kwargs, expected, is_qnn):
    if not is_qnn and version.parse(OrtVersion) < version.parse("1.21.0"):
        pytest.skip("prepare_qdq_config is only supported in onnxruntime>=1.21.0")
    input_model = get_onnx_model()
    config = {
        "quant_format": "QDQ",
        "prepare_qdq_config": True,
        "weight_symmetric": True,
        "activation_symmetric": True,
        "min_real_range": 5e-4,
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        **kwargs,
    }
    accelerator_spec = (
        AcceleratorSpec(
            accelerator_type="NPU",
            execution_provider="QNNExecutionProvider",
        )
        if is_qnn
        else None
    )

    def dummy_quantize_static(model_input, model_output, **kwargs):
        onnx.save(onnx.load(model_input), model_output)

    mock_quantize_static.side_effect = dummy_quantize_static

    p = create_pass_from_dict(OnnxStaticQuantization, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, tmp_path)
    assert out is not None

    mocked_kwargs = mock_quantize_static.call_args.kwargs
    for key in ["op_types_to_quantize", "nodes_to_exclude"]:
        assert set(mocked_kwargs[key]) == set(expected.get(key, []))
    extra_options = mocked_kwargs.get("extra_options", {})
    assert extra_options.get("MinimumRealRange") == (1e-4 if is_qnn else 5e-4)
    assert extra_options.get("WeightSymmetric") is True
    assert extra_options.get("ActivationSymmetric") is True


@pytest.mark.parametrize(
    ("algorithm", "weight_only_quant_configs"),
    [
        (None, None),
        ("RTN", {"ratios": {}}),
        ("DEFAULT", None),
        ("HQQ", None),
    ],
)
def test_matmul_4bit_quantization_without_dataloader(tmp_path, algorithm, weight_only_quant_configs):
    input_model = get_onnx_model()
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 4,
        "algorithm": algorithm,
        "weight_only_quant_configs": weight_only_quant_configs,
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    p = create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, tmp_path)
    assert out is not None


def test_matmul_4bits_gptq_with_dataloader(tmp_path, caplog):
    input_model = get_onnx_model()
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 4,
        "algorithm": "GPTQ",
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "weight_only_quant_configs": {"percdamp": 0.01, "block_size": 128, "use_less_config": 1},
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    # capture log
    logger = logging.getLogger("olive")
    logger.propagate = True

    p = create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, tmp_path)
    assert out is not None
    assert "Invalid weight_only_quant_configs: {'use_less_config'} for algorithm GPTQ" in caplog.text
    assert (
        "The pass config parameter block_size's value 32 is different from the algorithm config's value 128. The"
        " algorithm config's value will be used." in caplog.text
    )


def test_invalid_config_for_matmul_4bits():
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 5,
        "algorithm": "TE",
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    with pytest.raises(ValidationError):
        create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
