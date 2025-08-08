# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform

import numpy as np
import onnx
import pytest
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader

from olive.common.constants import OS
from olive.constants import Precision
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.aimet_quantization import AimetQuantization
from test.unit_test.utils import get_pytorch_model_dummy_input

IS_LINUX = platform.system() == OS.LINUX
CUDA_AVAILABLE = torch.cuda.is_available()


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


def dummy_onnx_model(model_path):
    matmul_node = onnx.helper.make_node("MatMul", inputs=["input", "weight"], outputs=["matmul_out"], name="matmul")
    softmax_node = onnx.helper.make_node("Softmax", inputs=["matmul_out"], outputs=["output"], name="softmax")
    weight = onnx.numpy_helper.from_array(np.random.randn(1, 1).astype(np.float32), name="weight")
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1])
    graph = onnx.helper.make_graph(
        [matmul_node, softmax_node],
        "dummy_graph",
        initializer=[weight],
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = onnx.helper.make_model(graph, ir_version=10, opset_imports=[onnx.helper.make_operatorsetid("", 22)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path)


def dummy_quantized_onnx_model(model_path):
    matmul_node = onnx.helper.make_node("MatMul", inputs=["input", "weight_dq"], outputs=["matmul_out"], name="matmul")
    softmax_node = onnx.helper.make_node("Softmax", inputs=["matmul_out"], outputs=["output"], name="softmax")
    dequantize_node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["weight", "weight_scale", "weight_offset"],
        outputs=["weight_dq"],
        name="weight_dequantizer",
    )
    weight = onnx.numpy_helper.from_array(np.random.randint(-128, 127, size=(1, 1), dtype=np.int8), name="weight")
    weight_scale = onnx.numpy_helper.from_array(np.array(0.1).astype(np.float32), name="weight_scale")
    weight_offset = onnx.numpy_helper.from_array(np.array(0).astype(np.int8), name="weight_offset")
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1])
    graph = onnx.helper.make_graph(
        [dequantize_node, matmul_node, softmax_node],
        "dummy_graph",
        initializer=[weight, weight_scale, weight_offset],
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = onnx.helper.make_model(graph, ir_version=10, opset_imports=[onnx.helper.make_operatorsetid("", 22)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path)


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
@pytest.mark.parametrize(
    "precisions",
    [
        ("int8", "uint8"),
        ("int4", "uint16"),
        ("int16", "uint16"),
    ],
)
def test_aimet_quantization_uses_provided_precisions(tmp_path, precisions):
    precision, act_type = precisions
    input_model = dummy_onnx_model(tmp_path / "dummy_model.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "precision": precision,
        "activation_type": act_type,
        "quant_scheme": "min_max",
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)

    assert out is not None

    model = onnx.load(out.model_path)

    initializer_dict = {tensor.name: tensor for tensor in model.graph.initializer}
    tensor_to_quantizer = {
        node.input[0]: node for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    }

    # Weight should be symmetrically quantized with precision type
    weight_quantizer = tensor_to_quantizer["weight"]
    weight_offset = onnx.numpy_helper.to_array(initializer_dict[weight_quantizer.input[2]])
    assert np.all(weight_offset == 0)
    # Note: int4 weights are packed into int8 data type
    assert weight_offset.dtype == np.dtype("int8") if precision == "int4" else np.dtype(precision)

    # Activations should be quantized with activation_type
    activation_tensors = {"input", "matmul_out"}
    for tensor in activation_tensors:
        quantizer = tensor_to_quantizer[tensor]
        offset = onnx.numpy_helper.to_array(initializer_dict[quantizer.input[2]])
        assert offset.dtype == np.dtype(act_type)


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_adheres_to_custom_config(tmp_path):
    input_model = dummy_onnx_model(tmp_path / "dummy_model.onnx")
    quantsim_config = {
        "defaults": {
            "hw_version": "V66",
            "ops": {"is_output_quantized": "True"},
            "params": {"is_quantized": "True", "is_symmetric": "False"},
            "per_channel_quantization": "True",
            "strict_symmetric": "False",
            "unsigned_symmetric": "False",
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {"MatMul": {"is_output_quantized": "False"}},
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {"is_output_quantized": "True"},
    }
    config_file = tmp_path / "aimet_config.json"
    with open(config_file, "w") as f:
        json.dump(quantsim_config, f)

    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "precision": Precision.INT8,
        "activation_type": Precision.UINT8,
        "config_file": str(config_file),
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)

    assert out is not None
    model = onnx.load(out.model_path)

    tensor_to_quantizer = {
        node.input[0]: node for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    }
    assert "matmul_out" not in tensor_to_quantizer


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_raises_error_with_prequantized_model(tmp_path):
    input_model = dummy_quantized_onnx_model(tmp_path / "dummy_model.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "precision": Precision.INT8,
        "activation_type": Precision.UINT8,
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    with pytest.raises(NotImplementedError):
        p.run(input_model, tmp_path)


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
@pytest.mark.parametrize(
    "pass_config",
    [
        {"precision": Precision.UINT8, "activation_type": Precision.UINT8},
        {"precision": Precision.INT8, "activation_type": Precision.INT8},
        {"precision": Precision.INT8, "activation_type": Precision.INT4},
    ],
)
def test_validate_config_returns_false_for_unsupported_configurations(pass_config):
    pass_config.update(
        {
            "data_config": DataConfig(
                name="test_quant_dc_config",
                load_dataset_config=DataComponentConfig(type="simple_dataset"),
                dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
            ),
        }
    )

    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )

    config = AimetQuantization.generate_config(accelerator_spec, pass_config, disable_search=True)
    assert not AimetQuantization.validate_config(config, accelerator_spec)
