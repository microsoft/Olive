# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model, pytorch_model_loader

import onnx
import onnxruntime
import pytest
import torch
from packaging import version

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.bnb_quantization import OnnxBnb4Quantization

# pylint: disable=protected-access


def get_onnx_matmul_model(model_path, model_attributes=None):
    pytorch_model = pytorch_model_loader(model_path=None)
    # need 3D input for MatMul, otherwise it will be converted to Gemm
    dummy_input = torch.randn(1, 1, 1)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    return ONNXModelHandler(model_path, model_attributes=model_attributes)


def get_onnx_gemm_model(model_path=None, model_attributes=None):
    model = get_onnx_model()
    model.model_attributes = model_attributes
    return model


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) < version.parse("1.16.2"),
    reason="OnnxBnb4Quantization requires ORT >= 1.16.2",
)
@pytest.mark.parametrize(
    ("pass_config", "model_attributes", "expected_error"),
    [
        ({"quant_type": "fp4"}, None, None),  # quant_type from config is fp4
        ({"quant_type": "invalid"}, None, AssertionError),  # quant_type from config is invalid
        (None, None, AssertionError),  # quant_type is not specified
        (
            None,
            {"quantization_config": {"bnb_4bit_quant_type": "nf4"}},
            None,
        ),  # quant_type from model_attributes is nf4
        (
            None,
            {"quantization_config": {"bnb_4bit_quant_type": "invalid"}},
            AssertionError,
        ),  # quant_type from model_attributes is invalid
        (
            None,
            {"quantization_config": {"load_in_8bit": True}},
            ValueError,
        ),  # load_in_8bit is True
        (None, {"quantization_config": None}, ValueError),  # quantization_config is None
        (
            None,
            {"quantization_config": {"load_in_4bit": True}},
            ValueError,
        ),  # quantization_config does not have bnb_4bit_quant_type
        (None, {"dummy_attribute": None}, ValueError),  # quantization_config is not specified
    ],
)
def test_validate_quant_type(pass_config, model_attributes, expected_error, tmp_path):
    input_model = get_onnx_gemm_model(model_attributes=model_attributes)

    p = create_pass_from_dict(OnnxBnb4Quantization, pass_config, disable_search=True)
    if expected_error:
        with pytest.raises(expected_error):
            p.run(input_model, str(tmp_path / "model.onnx"))
    else:
        p.run(input_model, str(tmp_path / "model.onnx"))


@pytest.mark.parametrize(("model_generator", "expected_count"), [(get_onnx_matmul_model, 1), (get_onnx_gemm_model, 0)])
def test__find_matmul_nodes(tmp_path, model_generator, expected_count):
    onnx_model = model_generator(str(tmp_path / "model.onnx"))
    matmul_nodes = OnnxBnb4Quantization._find_matmul_nodes(onnx_model.load_model().graph)
    assert len(matmul_nodes) == expected_count
    if expected_count:
        assert "fc1" in matmul_nodes[0]


def count_matmulbnb4_nodes(model: onnx.ModelProto):
    count = 0
    for node in model.graph.node:
        if node.op_type == "MatMulBnb4":
            count += 1
    return count


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) < version.parse("1.16.2"),
    reason="MatMulBnb4Quantizer is only supported in onnxruntime >= 1.16.2",
)
@pytest.mark.parametrize(
    ("model_generator", "quantized_modules", "expected_count"),
    [
        (get_onnx_gemm_model, None, 0),  # no matmul nodes in the graph
        (get_onnx_matmul_model, None, 1),  # all matmul nodes
        (get_onnx_matmul_model, ["fc1"], 1),  # only fc1
        (get_onnx_matmul_model, ["fc2"], 0),  # only fc2, not in the graph
        (get_onnx_matmul_model, ["fc1", "fc2"], 1),  # fc1 and fc2
    ],
)
def test_quantized_modules(tmp_path, model_generator, quantized_modules, expected_count):
    input_model = model_generator(str(tmp_path / "model.onnx"))
    p = create_pass_from_dict(OnnxBnb4Quantization, {"quant_type": "nf4", "quantized_modules": quantized_modules})
    output_model = p.run(input_model, (tmp_path / "output_model.onnx"))
    assert count_matmulbnb4_nodes(output_model.load_model()) == expected_count
