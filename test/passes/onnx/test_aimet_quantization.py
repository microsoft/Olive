# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
from collections.abc import Iterable
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader

from olive.common.constants import OS
from olive.common.onnx_io import get_kv_info
from olive.constants import Precision
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.aimet_quantization import AimetQuantization
from olive.passes.onnx.conversion import OnnxConversion
from test.utils import make_local_tiny_llama

IS_LINUX = platform.system() == OS.LINUX
CUDA_AVAILABLE = torch.cuda.is_available()


class DummyCalibrationDataReader(CalibrationDataReader):
    # pylint: disable=W0223
    def __init__(self, batch_size: int = 16, input_shape=1):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.sample_counter = 500

    def get_next(self) -> dict:
        if self.sample_counter <= 0:
            return None

        data = torch.randn(1, self.input_shape)
        try:
            item = {"input": data}
            self.sample_counter -= 1
            return item
        except Exception:
            return None


class DummyDataGenerator(CalibrationDataReader):
    # pylint: disable=W0223
    def __init__(self, model):
        super().__init__()
        self.sample_counter = 1
        self.model = model

    def get_next(self) -> dict:
        if self.sample_counter <= 0:
            return None

        self.sample_counter -= 1
        return self.model.get_dummy_inputs()


@Registry.register_dataloader()
def _test_quant_dataloader(dataset, batch_size, **kwargs):
    return DummyCalibrationDataReader(batch_size=batch_size)


@Registry.register_dataloader()
def _test_quant_dataloader_len_16(dataset, batch_size, **kwargs):
    return DummyCalibrationDataReader(batch_size=batch_size, input_shape=16)


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


def dummy_onnx_matmul_model(model_path):
    matmul_node = onnx.helper.make_node("MatMul", inputs=["input", "weight"], outputs=["matmul_out"], name="matmul")
    softmax_node = onnx.helper.make_node("Softmax", inputs=["matmul_out"], outputs=["output"], name="softmax")
    weight = onnx.numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="weight")
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 16])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16])
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
        node.input[0].removesuffix("_q"): node
        for node in model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
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
def test_aimet_quantization_applies_precision_overrides(tmp_path):
    input_model = dummy_onnx_model(tmp_path / "dummy_model.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "tensor_precision_overrides": {"output": Precision.UINT16},
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)

    model = onnx.load(out.model_path)

    initializer_dict = {tensor.name: tensor for tensor in model.graph.initializer}
    tensor_to_quantizer = {
        node.output[0]: node for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    }
    output_offset = onnx.numpy_helper.to_array(initializer_dict[tensor_to_quantizer["output"].input[2]])
    assert output_offset.dtype == np.dtype("uint16")


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
def test_aimet_quantization_applies_lpbq(tmp_path):
    block_size = 4
    input_model = dummy_onnx_matmul_model(tmp_path / "dummy_model_mm.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_len_16"),
        ),
        "precision": "int8",
        "activation_type": "uint8",
        "quant_scheme": "min_max",
        "techniques": [{"name": "lpbq", "block_size": block_size, "op_types": ("MatMul",)}],
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)

    assert out is not None

    model = onnx.load(out.model_path)

    tensor_to_quantizer = {node.input[0]: node for node in model.graph.node if node.op_type in ("QuantizeLinear",)}
    for name, quantizer in tensor_to_quantizer.items():
        block_size_attr = [attr for attr in quantizer.attribute if attr.name == "block_size"]
        if name == "weight":
            # MatMul weight quantizer should be blockwise
            assert len(block_size_attr) == 1
            assert block_size_attr[0].i == block_size
        else:
            # All other quantizers should not be blockwise
            assert not block_size_attr


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_applies_adaround(tmp_path):
    input_model = dummy_onnx_matmul_model(tmp_path / "dummy_model_mm.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_len_16"),
        ),
        "techniques": [{"name": "adaround", "num_iterations": 5}],
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    with patch("aimet_onnx.apply_adaround") as mock_adaround:
        out = p.run(input_model, tmp_path)
        assert mock_adaround.call_count == 1

        (_, data, num_iterations, nodes_to_include), _ = mock_adaround.call_args
        assert isinstance(data, Iterable)
        assert num_iterations == 5
        assert nodes_to_include is None

    assert out is not None


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_excludes_adaround_nodes(tmp_path):
    input_model = dummy_onnx_matmul_model(tmp_path / "dummy_model_mm.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_len_16"),
        ),
        "techniques": [
            {
                "name": "adaround",
                "nodes_to_exclude": ["matmul"],
                "data_config": DataConfig(
                    name="test_quant_dc_config",
                    load_dataset_config=DataComponentConfig(type="simple_dataset"),
                    dataloader_config=DataComponentConfig(type="_test_quant_dataloader_len_16"),
                ),
            }
        ],
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    with patch("aimet_onnx.apply_adaround") as mock_adaround:
        p.run(input_model, tmp_path)
        assert mock_adaround.call_count == 1
        (_, _, _, nodes_to_include), _ = mock_adaround.call_args
        assert not nodes_to_include


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_applies_seq_mse(tmp_path):
    input_model = dummy_onnx_matmul_model(tmp_path / "dummy_model_mm.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_len_16"),
        ),
        "precision": "int4",
        "techniques": [
            {
                "name": "seqmse",
                "num_candidates": 5,
            }
        ],
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    with patch("aimet_onnx.apply_seq_mse") as mock_seq_mse:
        out = p.run(input_model, tmp_path)
        assert mock_seq_mse.call_count == 1

        (_, data, num_candidates), _ = mock_seq_mse.call_args
        assert isinstance(data, Iterable)
        assert num_candidates == 5

    assert out is not None


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
@pytest.mark.parametrize(
    ("op_types", "disabled_quantizers"),
    [
        (["Softmax"], ["output"]),
        (["MatMul"], ["weight", "input"]),
        (["MatMul", "Softmax"], ["weight", "input", "matmul_out", "output"]),
    ],
)
def test_aimet_quantization_excludes_op_types(tmp_path, op_types, disabled_quantizers):
    input_model = dummy_onnx_model(tmp_path / "dummy_model.onnx")
    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
        "precision": "int8",
        "activation_type": "uint8",
        "quant_scheme": "min_max",
        "op_types_to_exclude": op_types,
    }
    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)

    out = p.run(input_model, tmp_path)

    assert out is not None

    model = onnx.load(out.model_path)

    tensor_to_quantizer = {
        tensor.removesuffix("_q"): node
        for node in model.graph.node
        for tensor in (node.input[0], node.output[0])
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    }

    for tensor_name in ("weight", "input", "matmul_out", "output"):
        assert (tensor_name in disabled_quantizers) == (tensor_name not in tensor_to_quantizer)


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_preserves_quantization_in_prequantized_model(tmp_path):
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

    out = p.run(input_model, tmp_path)

    model = onnx.load(out.model_path)

    tensor_to_quantizer = {
        node.input[0].removesuffix("_q"): node
        for node in model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    }

    weight_quantizer = tensor_to_quantizer["weight_dq"]
    weight_scale = [t for t in model.graph.initializer if t.name == weight_quantizer.input[1]]
    weight_scale = onnx.numpy_helper.to_array(weight_scale[0])
    assert weight_scale == np.array(0.1).astype(np.float32)
    assert "input" in tensor_to_quantizer


@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
@pytest.mark.parametrize(
    "pass_config",
    [
        {"precision": Precision.UINT8, "activation_type": Precision.UINT8},
        {"precision": Precision.INT8, "activation_type": Precision.INT8},
        {"precision": Precision.INT8, "activation_type": Precision.INT4},
        {"techniques": [{"name": "lpbq", "unsupported_arg": 0}]},
        {"techniques": [{"block_size": 64}]},
        {"techniques": [{"name": "unsupported"}]},
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


@pytest.mark.skip(reason="Dynamo export fails for Llama, need fix")
@pytest.mark.skipif(not IS_LINUX, reason="Only run on linux")
@pytest.mark.skipif(CUDA_AVAILABLE, reason="Only run on cpu tests")
def test_aimet_quantization_ties_kv_io_quantizers(tmp_path):
    model = make_local_tiny_llama(tmp_path / "input_model")
    onnx_model = create_pass_from_dict(OnnxConversion, {}, disable_search=True).run(model, tmp_path / "onnx_model")

    @Registry.register_dataloader()
    def _test_quant_dataloader_llm(*args, **kwargs):
        return DummyDataGenerator(model)

    config = {
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_llm"),
        ),
    }

    p = create_pass_from_dict(AimetQuantization, config, disable_search=True)
    out = p.run(onnx_model, tmp_path)

    output_model = onnx.load(out.model_path)

    # Map from tensor name to its quantization scale tensor name
    tensor_to_scale_offset = {}
    for node in output_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            scale = node.input[1]
            offset = node.input[2] if len(node.input) > 2 else None
            tensor_to_scale_offset[node.input[0]] = (scale, offset)
            tensor_to_scale_offset[node.output[0]] = (scale, offset)

    initializer_dict = {init.name: onnx.numpy_helper.to_array(init) for init in output_model.graph.initializer}

    kv_info = get_kv_info(out.io_config)
    # Verify that all present key/value quantization scales are equal
    for present, past in kv_info["present_to_past"].items():
        present_scale, present_offset = tensor_to_scale_offset[present]
        past_scale, past_offset = tensor_to_scale_offset[past]
        assert np.array_equal(initializer_dict[present_scale], initializer_dict[past_scale])
        if present_offset:
            assert np.array_equal(initializer_dict[present_offset], initializer_dict[past_offset])
