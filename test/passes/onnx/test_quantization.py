# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import patch

import onnx
import pytest
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

from olive.constants import Precision
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.quantization import (
    OnnxQuantization,
    OnnxQuantizationPreprocess,
    OnnxStaticQuantization,
)
from test.utils import get_onnx_model, get_pytorch_model_dummy_input


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
        "precision": Precision.UINT8,
        "activation_type": Precision.UINT8,
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


# Note: With dynamo export, node names are like "node_linear" instead of "/fc1/Gemm"
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"op_types_to_quantize": ["Gemm", "Sigmoid"]}),
        ({"op_types_to_quantize": ["Gemm"]}, {"op_types_to_quantize": ["Gemm"]}),
        ({"op_types_to_exclude": ["Gemm"]}, {"op_types_to_quantize": ["Sigmoid"], "nodes_to_exclude": ["node_linear"]}),
        (
            # this node does not exist in the model but using this instead of "node_linear"
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
        "extra_options": {"CalibStrideMinMax": 1},
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
    assert extra_options.get("CalibStrideMinMax") == 1
