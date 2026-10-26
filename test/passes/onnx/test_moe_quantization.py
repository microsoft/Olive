# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.moe_quantization import OnnxMoEQuantization


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16])
def test_quantize_experts_uses_stored_scale_precision(bits, dtype):
    weights = np.random.default_rng(1234).normal(size=(2, 64, 128)).astype(dtype)
    packed, scales = OnnxMoEQuantization.quantize_experts(weights, bits, 64)
    pack = 8 // bits
    codes = np.empty(weights.shape, dtype=np.uint8)
    for index in range(pack):
        codes[..., index::pack] = (packed >> (index * bits)) & ((1 << bits) - 1)
    midpoint = 1 << (bits - 1)
    blocks = weights.astype(np.float32).reshape(2, 64, 2, 64)
    expected = (
        np.clip(np.rint(blocks / scales.astype(np.float32)[..., None]), -midpoint, midpoint - 1).reshape(weights.shape)
        + midpoint
    )
    np.testing.assert_array_equal(codes, expected)
    unrounded_scales = np.max(np.abs(blocks), axis=-1) / (midpoint - 1)
    incorrect = (
        np.clip(np.rint(blocks / unrounded_scales[..., None]), -midpoint, midpoint - 1).reshape(weights.shape)
        + midpoint
    )
    assert np.any(codes != incorrect)


def test_quantize_experts_zero_blocks_are_exact():
    packed, scales = OnnxMoEQuantization.quantize_experts(np.zeros((2, 3, 128), np.float16), 4, 128)
    np.testing.assert_array_equal(packed, 0x88)
    np.testing.assert_array_equal(scales, 1)


@pytest.mark.parametrize(
    "weights",
    [np.zeros((2, 64), np.float16), np.zeros((2, 3, 65), np.float16), np.full((2, 3, 128), np.nan, np.float16)],
)
def test_quantize_experts_rejects_invalid_banks(weights):
    with pytest.raises(ValueError, match="Expert weights must"):
        OnnxMoEQuantization.quantize_experts(weights, 4, 128)


def test_quantization_shares_banks_and_preserves_external_routing(tmp_path):
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT16), shape=ir.Shape(["n", 128]))
    routers = [ir.Value(name=f"router_{i}", type=x.type, shape=ir.Shape(["n", 2])) for i in range(2)]
    weights = [
        ir.Value(name=name, const_value=ir.tensor(np.ones(shape, np.float16)))
        for name, shape in [("gate_up", (2, 256, 128)), ("down", (2, 128, 128))]
    ]
    nodes = [
        ir.node(
            "MoE",
            domain="com.microsoft",
            name=f"route_{i}",
            inputs=[x, router, weights[0], None, weights[1]],
            attributes={"k": 1, "normalize_routing_weights": 1, "swiglu_fusion": 1, "activation_type": "swiglu"},
        )
        for i, router in enumerate(routers)
    ]
    for i, node in enumerate(nodes):
        node.outputs[0].name = f"y{i}"
        node.outputs[0].type, node.outputs[0].shape = x.type, x.shape
    source = tmp_path / "model.onnx"
    ir.save(
        ir.Model(
            ir.Graph(
                [x, *routers],
                [n.outputs[0] for n in nodes],
                nodes=nodes,
                initializers=weights,
                opset_imports={"": 24, "com.microsoft": 1},
            ),
            ir_version=10,
        ),
        source,
    )
    quantizer = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4, "block_size": 128}, disable_search=True)
    result = quantizer.run(ONNXModelHandler(model_path=source), str(tmp_path / "quant.onnx"))
    output = result.load_ir_model()
    first, second = list(output.graph)
    assert first.op_type == second.op_type == "QMoE"
    assert first.domain == second.domain == "com.microsoft"
    assert first.inputs[2] is second.inputs[2]
    assert first.inputs[5] is second.inputs[5]
    assert len(output.graph.initializers) == 4
    assert [first.inputs[1].name, second.inputs[1].name] == ["router_0", "router_1"]
    assert first.attributes.get_int("k") == first.attributes.get_int("normalize_routing_weights") == 1
    assert first.attributes.get_int("weights_prepacked") == 0


@pytest.mark.parametrize("separate_fc3", [False, True])
def test_quantization_preserves_biases_and_skips_separate_fc3(tmp_path, separate_fc3, caplog):
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT16), shape=ir.Shape(["n", 128]))
    router = ir.Value(name="router", type=x.type, shape=ir.Shape(["n", 2]))
    shapes = [(2, 64, 128), (2, 128, 64), (2, 64, 128)]
    banks = [
        ir.Value(name=f"w{i}", const_value=ir.tensor(np.ones(shape, np.float16))) for i, shape in enumerate(shapes)
    ]
    biases = [
        ir.Value(name=f"b{i}", const_value=ir.tensor(np.full(shape[:2], i + 1, np.float16)))
        for i, shape in enumerate(shapes)
    ]
    node = ir.node(
        "MoE",
        domain="com.microsoft",
        inputs=[x, router, banks[0], biases[0], banks[1], biases[1]] + ([banks[2], biases[2]] if separate_fc3 else []),
        attributes={"k": 1, "activation_type": "silu"},
    )
    node.outputs[0].name, node.outputs[0].type, node.outputs[0].shape = "y", x.type, x.shape
    source = tmp_path / "model.onnx"
    ir.save(
        ir.Model(
            ir.Graph(
                [x, router],
                node.outputs,
                nodes=[node],
                initializers=[*banks, *biases],
                opset_imports={"": 24, "com.microsoft": 1},
            ),
            ir_version=10,
        ),
        source,
    )
    quantizer = create_pass_from_dict(OnnxMoEQuantization, {"block_size": 64}, disable_search=True)
    output = quantizer.run(ONNXModelHandler(model_path=source), str(tmp_path / "quant.onnx")).load_ir_model()
    quantized = next(iter(output.graph))
    if separate_fc3:
        assert quantized.op_type == "MoE"
        assert "does not support separate FC3" in caplog.text
        for index, original in enumerate(banks):
            actual = quantized.inputs[2 + 2 * index]
            assert actual.dtype == original.const_value.dtype
            np.testing.assert_array_equal(actual.const_value.numpy(), original.const_value.numpy())
            np.testing.assert_array_equal(
                quantized.inputs[3 + 2 * index].const_value.numpy(), biases[index].const_value.numpy()
            )
        return
    assert quantized.op_type == "QMoE"
    for index, original in enumerate(biases[:2]):
        actual = quantized.inputs[4 + 3 * index]
        assert actual.name == original.name
        np.testing.assert_array_equal(actual.const_value.numpy(), original.const_value.numpy())
        assert quantized.inputs[2 + 3 * index].dtype == ir.DataType.UINT8
        assert list(quantized.inputs[3 + 3 * index].shape) == [*shapes[index][:2], shapes[index][-1] // 64]


@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers()
    or tuple(int(part) for part in ort.__version__.split(".")[:2]) < (1, 31),
    reason="Requires CUDA ORT >= 1.31 with BF16 raw-weight QMoE prepacking.",
)
@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_quantization_matches_nonzero_lossless_experts(tmp_path, bits):
    rng = np.random.default_rng(93)
    experts, hidden = 8, 128
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["n", hidden]))
    router = ir.Value(name="router", type=x.type, shape=ir.Shape(["n", experts]))
    casts = [ir.node("Cast", inputs=[value], attributes={"to": int(ir.DataType.BFLOAT16)}) for value in (x, router)]
    banks = []
    maximum = (1 << (bits - 1)) - 1
    for index, shape in enumerate(((experts, 2 * hidden, hidden), (experts, hidden, hidden))):
        values = rng.integers(-maximum, maximum + 1, shape)
        values[..., -1] = maximum
        banks.append(ir.Value(name=f"w{index}", const_value=ir.tensor((values / 256).astype(ml_dtypes.bfloat16))))
    moe = ir.node(
        "MoE",
        domain="com.microsoft",
        name="experts",
        inputs=[casts[0].outputs[0], casts[1].outputs[0], banks[0], None, banks[1]],
        attributes={"k": 1, "normalize_routing_weights": 1, "swiglu_fusion": 1, "activation_type": "swiglu"},
    )
    readout = ir.node("Cast", inputs=moe.outputs, attributes={"to": int(ir.DataType.FLOAT)})
    readout.outputs[0].name, readout.outputs[0].type, readout.outputs[0].shape = "y", x.type, x.shape
    source = tmp_path / "model.onnx"
    ir.save(
        ir.Model(
            ir.Graph(
                [x, router],
                readout.outputs,
                nodes=[*casts, moe, readout],
                initializers=banks,
                opset_imports={"": 24, "com.microsoft": 1},
            ),
            ir_version=10,
        ),
        source,
    )
    quantizer = create_pass_from_dict(OnnxMoEQuantization, {"bits": bits, "block_size": 128}, disable_search=True)
    result = quantizer.run(ONNXModelHandler(model_path=source), str(tmp_path / "quant.onnx"))
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sessions = [
        ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
        for path in (source, result.model_path)
    ]
    for rows in (1, 17):
        feeds = {
            "x": (rng.integers(-7, 8, (rows, hidden)) / 64).astype(np.float32),
            "router": np.eye(experts, dtype=np.float32)[np.arange(rows) % experts],
        }
        expected, actual = (session.run(None, feeds)[0] for session in sessions)
        assert np.any(expected != 0)
        np.testing.assert_array_equal(actual, expected)
