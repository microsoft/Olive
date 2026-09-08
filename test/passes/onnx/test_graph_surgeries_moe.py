# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for weight-aware MoE graph surgeries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnxscript import ir

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.graph_surgery import Surgeon
from olive.passes.onnx.graph_surgery import moe as moe_surgeries

HIDDEN = 32
INTERMEDIATE = 16
EXPERTS = 4
TOP_K = 2
BLOCK_SIZE = 16
BITS = 4


def _initializer(graph: ir.Graph, name: str, array: np.ndarray, dtype: ir.DataType) -> ir.Value:
    value = ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(dtype),
        const_value=ir.tensor(array, name=name, dtype=dtype),
    )
    graph.register_initializer(value)
    return value


def _constant_int(nodes: list[ir.Node], name: str, value: int) -> ir.Value:
    node = ir.node("Constant", inputs=[], attributes={"value_int": value}, num_outputs=1, name=name)
    node.outputs[0].name = f"{name}.out"
    nodes.append(node)
    return node.outputs[0]


def _count(model: ir.Model, op_type: str) -> int:
    return sum(node.op_type == op_type for node in model.graph)


def _run_surgery(
    model: ir.Model,
    tmp_path: Path,
    surgeon: str,
    *,
    input_external: bool = False,
    output_external: bool = False,
    **parameters,
) -> ir.Model:
    input_path = tmp_path / "input.onnx"
    proto = ir.to_proto(model)
    if input_external:
        onnx.save_model(
            proto,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
        )
    else:
        onnx.save_model(proto, input_path)

    config = {
        "surgeries": [{"surgeon": surgeon, **parameters}],
        "remove_duplicate_initializers": False,
        "save_as_external_data": output_external,
        "size_threshold": 0,
    }
    graph_surgeries = create_pass_from_dict(GraphSurgeries, config, disable_search=True)
    output = graph_surgeries.run(ONNXModelHandler(model_path=str(input_path)), str(tmp_path / "output"))
    if output_external:
        assert (Path(output.model_path).parent / f"{Path(output.model_path).name}.data").exists()
    return ir.from_proto(output.load_model())


class _QuantizedWeight:
    def __init__(self, rng: np.random.Generator, out_features: int, in_features: int, dtype: ir.DataType):
        blocks = in_features // BLOCK_SIZE
        codes = rng.integers(0, 16, size=(out_features, in_features), dtype=np.uint8)
        weight = np.zeros((out_features, blocks, BLOCK_SIZE // 2), dtype=np.uint8)
        for block in range(blocks):
            values = codes[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE]
            weight[:, block] = values[:, 0::2] | values[:, 1::2] << 4
        zero_point_codes = rng.integers(0, 16, size=(out_features, blocks), dtype=np.uint8)
        zero_points = np.zeros((out_features, (blocks + 1) // 2), dtype=np.uint8)
        for block in range(blocks):
            zero_points[:, block // 2] |= zero_point_codes[:, block] << (4 * (block % 2))
        self.weight = weight
        self.scales = rng.random((out_features, blocks), dtype=np.float32).astype(dtype.numpy())
        self.zero_points = zero_points
        self.out_features = out_features
        self.in_features = in_features
        self.dtype = dtype


def _dequantize(weight: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
    out_features = weight.shape[0]
    packed_block_size = BLOCK_SIZE // 2
    blocks = weight.shape[1] // packed_block_size if weight.ndim == 2 else weight.shape[1]
    weight = weight.reshape(out_features, blocks, packed_block_size)
    dense = np.empty((out_features, blocks * packed_block_size * 2), dtype=np.float32)
    for block in range(blocks):
        packed = weight[:, block].astype(np.int32)
        codes = np.empty((out_features, BLOCK_SIZE), dtype=np.int32)
        codes[:, 0::2] = packed & 0xF
        codes[:, 1::2] = packed >> 4
        zero_point = (zero_points[:, block // 2].astype(np.int32) >> (4 * (block % 2))) & 0xF
        dense[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE] = (codes - zero_point[:, None]) * scales[
            :, block : block + 1
        ].astype(np.float32)
    return dense


def _matmul_nbits(
    graph: ir.Graph,
    nodes: list[ir.Node],
    name: str,
    value: ir.Value,
    weight: _QuantizedWeight,
    *,
    bits: int = BITS,
    block_size: int = BLOCK_SIZE,
    include_zero_points: bool = True,
) -> ir.Value:
    packed = _initializer(graph, f"{name}.weight", weight.weight, ir.DataType.UINT8)
    scales = _initializer(graph, f"{name}.scales", weight.scales, weight.dtype)
    zero_points = _initializer(graph, f"{name}.zero_points", weight.zero_points, ir.DataType.UINT8)
    inputs = [value, packed, scales]
    if include_zero_points:
        inputs.append(zero_points)
    node = ir.node(
        "MatMulNBits",
        inputs=inputs,
        attributes={
            "K": weight.in_features,
            "N": weight.out_features,
            "bits": bits,
            "block_size": block_size,
        },
        domain="com.microsoft",
        num_outputs=1,
        name=name,
    )
    node.outputs[0].name = f"{name}.out"
    nodes.append(node)
    return node.outputs[0]


def _build_qmoe_graph(
    *,
    activation_dtype: ir.DataType = ir.DataType.FLOAT16,
    hidden_typed: bool = True,
    legacy_activation: bool = False,
    router_cast: bool = False,
    topk_value_cast: bool = False,
    block_size: int = BLOCK_SIZE,
    include_zero_points: bool = True,
    mask_cast_op: str = "CastLike",
) -> tuple[ir.Model, dict[str, _QuantizedWeight]]:
    rng = np.random.default_rng(0)
    weights: dict[str, _QuantizedWeight] = {}
    nodes: list[ir.Node] = []
    hidden = ir.Value(
        name="hidden",
        shape=ir.Shape(["tokens", HIDDEN]),
        type=ir.TensorType(activation_dtype) if hidden_typed else None,
    )
    graph = ir.Graph([hidden], [], nodes=[], name="dense_qmoe")

    def quant(name: str, out_features: int, in_features: int) -> _QuantizedWeight:
        weights[name] = _QuantizedWeight(rng, out_features, in_features, activation_dtype)
        return weights[name]

    router = _matmul_nbits(
        graph,
        nodes,
        "router",
        hidden,
        quant("router", EXPERTS, HIDDEN),
        block_size=block_size,
        include_zero_points=include_zero_points,
    )
    logits = router
    if router_cast:
        cast = ir.node(
            "Cast",
            inputs=[router],
            attributes={"to": activation_dtype.value},
            num_outputs=1,
            name="router.cast",
        )
        cast.outputs[0].name = "router.cast.out"
        nodes.append(cast)
        logits = cast.outputs[0]

    topk_k = _initializer(graph, "topk.k", np.array([TOP_K], dtype=np.int64), ir.DataType.INT64)
    topk = ir.node("TopK", inputs=[logits, topk_k], attributes={"axis": -1}, num_outputs=2, name="topk")
    topk.outputs[0].name = "topk.values"
    topk.outputs[1].name = "topk.indices"
    nodes.append(topk)
    softmax_input = topk.outputs[0]
    if topk_value_cast:
        cast = ir.node(
            "Cast",
            inputs=[softmax_input],
            attributes={"to": activation_dtype.value},
            num_outputs=1,
            name="topk.cast",
        )
        cast.outputs[0].name = "topk.cast.out"
        nodes.append(cast)
        softmax_input = cast.outputs[0]
    softmax = ir.node("Softmax", inputs=[softmax_input], attributes={"axis": -1}, num_outputs=1, name="softmax")
    softmax.outputs[0].name = "softmax.out"
    nodes.append(softmax)
    axes = _initializer(graph, "axes", np.array([-1], dtype=np.int64), ir.DataType.INT64)

    routed = None
    for expert in range(EXPERTS):
        expert_id = _constant_int(nodes, f"expert_id_{expert}", expert)
        equal = ir.node("Equal", inputs=[topk.outputs[1], expert_id], num_outputs=1, name=f"equal_{expert}")
        nodes.append(equal)
        if mask_cast_op == "CastLike":
            cast = ir.node(
                "CastLike",
                inputs=[equal.outputs[0], softmax.outputs[0]],
                num_outputs=1,
                name=f"mask_cast_{expert}",
            )
        else:
            cast = ir.node(
                "Cast",
                inputs=[equal.outputs[0]],
                attributes={"to": activation_dtype.value},
                num_outputs=1,
                name=f"mask_cast_{expert}",
            )
        nodes.append(cast)
        weighted_mask = ir.node(
            "Mul",
            inputs=[softmax.outputs[0], cast.outputs[0]],
            num_outputs=1,
            name=f"weighted_mask_{expert}",
        )
        nodes.append(weighted_mask)
        reduce_sum = ir.node(
            "ReduceSum",
            inputs=[weighted_mask.outputs[0], axes],
            attributes={"keepdims": 1},
            num_outputs=1,
            name=f"reduce_{expert}",
        )
        nodes.append(reduce_sum)

        gate = _matmul_nbits(
            graph,
            nodes,
            f"expert{expert}.gate",
            hidden,
            quant(f"gate{expert}", INTERMEDIATE, HIDDEN),
            include_zero_points=include_zero_points,
        )
        if legacy_activation:
            sigmoid = ir.node("Sigmoid", inputs=[gate], num_outputs=1, name=f"expert{expert}.sigmoid")
            nodes.append(sigmoid)
            activation = ir.node("Mul", inputs=[gate, sigmoid.outputs[0]], num_outputs=1, name=f"expert{expert}.silu")
        else:
            activation = ir.node("Swish", inputs=[gate], num_outputs=1, name=f"expert{expert}.silu")
        nodes.append(activation)
        up = _matmul_nbits(
            graph,
            nodes,
            f"expert{expert}.up",
            hidden,
            quant(f"up{expert}", INTERMEDIATE, HIDDEN),
            include_zero_points=include_zero_points,
        )
        product = ir.node("Mul", inputs=[activation.outputs[0], up], num_outputs=1, name=f"expert{expert}.product")
        nodes.append(product)
        down = _matmul_nbits(
            graph,
            nodes,
            f"expert{expert}.down",
            product.outputs[0],
            quant(f"down{expert}", HIDDEN, INTERMEDIATE),
            block_size=block_size,
            include_zero_points=include_zero_points,
        )
        contribution = ir.node(
            "Mul", inputs=[down, reduce_sum.outputs[0]], num_outputs=1, name=f"expert{expert}.contribution"
        )
        nodes.append(contribution)
        if routed is None:
            routed = contribution.outputs[0]
        else:
            add = ir.node("Add", inputs=[routed, contribution.outputs[0]], num_outputs=1, name=f"sum_{expert}")
            nodes.append(add)
            routed = add.outputs[0]

    shared = ir.node("Identity", inputs=[hidden], num_outputs=1, name="shared_expert")
    nodes.append(shared)
    final = ir.node("Add", inputs=[routed, shared.outputs[0]], num_outputs=1, name="final_add")
    final.outputs[0].name = "output"
    final.outputs[0].shape = hidden.shape
    final.outputs[0].type = ir.TensorType(activation_dtype)
    nodes.append(final)
    for node in nodes:
        graph.append(node)
    graph.outputs.append(final.outputs[0])
    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = 21
    model.opset_imports["com.microsoft"] = 1
    return model, weights


def _make_native_weight(rng: np.random.Generator, out_features: int, in_features: int, fmt: str) -> np.ndarray:
    block_elements, block_bytes = moe_surgeries._NATIVE_BLOCK_FORMATS[fmt]
    blocks = (in_features + block_elements - 1) // block_elements
    return rng.integers(0, 256, size=(out_features, blocks, block_bytes), dtype=np.uint8)


def _block_matmul(
    graph: ir.Graph,
    nodes: list[ir.Node],
    name: str,
    value: ir.Value,
    weight: np.ndarray,
    fmt: str,
    *,
    bias: bool = False,
) -> ir.Value:
    packed = _initializer(graph, f"{name}.weight", weight, ir.DataType.UINT8)
    inputs = [value, packed]
    if bias:
        inputs.append(_initializer(graph, f"{name}.bias", np.zeros(weight.shape[0], np.float32), ir.DataType.FLOAT))
    node = ir.node(
        "BlockQuantizedMatMul",
        inputs=inputs,
        attributes={"K": HIDDEN, "N": weight.shape[0], "format": fmt, "block_layout_version": 1},
        domain="pkg.nxrt",
        num_outputs=1,
        name=name,
    )
    node.outputs[0].name = f"{name}.out"
    node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    nodes.append(node)
    return node.outputs[0]


def _build_block_moe_graph(
    *,
    gate_format: str = "iq4_xs",
    up_format: str = "iq4_xs",
    down_format: str = "iq4_xs",
    dtype: ir.DataType = ir.DataType.FLOAT,
    legacy_activation: bool = False,
    corrupt_gate_format: tuple[int, str] | None = None,
    biased_expert: int | None = None,
    broken_expert: int | None = None,
    alien_routing_expert: int | None = None,
    mask_cast_op: str = "CastLike",
) -> tuple[ir.Model, dict[str, np.ndarray]]:
    rng = np.random.default_rng(1)
    weights: dict[str, np.ndarray] = {}
    nodes: list[ir.Node] = []
    hidden = ir.Value(name="hidden", shape=ir.Shape(["tokens", HIDDEN]), type=ir.TensorType(dtype))
    graph = ir.Graph([hidden], [], nodes=[], name="dense_block_moe")

    topk_k = _initializer(graph, "topk.k", np.array([TOP_K], dtype=np.int64), ir.DataType.INT64)
    router_weight = _initializer(
        graph, "router.weight", rng.standard_normal((HIDDEN, EXPERTS)).astype(np.float32), ir.DataType.FLOAT
    )
    router = ir.node("MatMul", inputs=[hidden, router_weight], num_outputs=1, name="router")
    nodes.append(router)
    sigmoid = ir.node("Sigmoid", inputs=[router.outputs[0]], num_outputs=1, name="router.sigmoid")
    nodes.append(sigmoid)
    topk = ir.node("TopK", inputs=[sigmoid.outputs[0], topk_k], attributes={"axis": -1}, num_outputs=2, name="topk")
    topk.outputs[0].shape = ir.Shape(["tokens", TOP_K])
    topk.outputs[1].shape = ir.Shape(["tokens", TOP_K])
    nodes.append(topk)
    axes = _initializer(graph, "axes", np.array([-1], dtype=np.int64), ir.DataType.INT64)
    total = ir.node(
        "ReduceSum",
        inputs=[topk.outputs[0], axes],
        attributes={"keepdims": 1},
        num_outputs=1,
        name="routing.total",
    )
    nodes.append(total)
    routing_weights = ir.node("Div", inputs=[topk.outputs[0], total.outputs[0]], num_outputs=1, name="routing")
    routing_weights.outputs[0].shape = ir.Shape(["tokens", TOP_K])
    routing_weights.outputs[0].type = ir.TensorType(dtype)
    nodes.append(routing_weights)

    def cast_in(value: ir.Value, name: str) -> ir.Value:
        if dtype == ir.DataType.FLOAT:
            return value
        cast = ir.node(
            "Cast",
            inputs=[value],
            attributes={"to": ir.DataType.FLOAT.value},
            num_outputs=1,
            name=f"{name}.cast_in",
        )
        nodes.append(cast)
        return cast.outputs[0]

    def cast_out(value: ir.Value, name: str) -> ir.Value:
        if dtype == ir.DataType.FLOAT:
            return value
        cast = ir.node("Cast", inputs=[value], attributes={"to": dtype.value}, num_outputs=1, name=f"{name}.cast_out")
        cast.outputs[0].type = ir.TensorType(dtype)
        nodes.append(cast)
        return cast.outputs[0]

    routed = None
    for expert in range(EXPERTS):
        expert_id = _constant_int(nodes, f"expert_id_{expert}", expert)
        equal = ir.node("Equal", inputs=[topk.outputs[1], expert_id], num_outputs=1, name=f"equal_{expert}")
        nodes.append(equal)
        if mask_cast_op == "CastLike":
            cast = ir.node(
                "CastLike",
                inputs=[equal.outputs[0], routing_weights.outputs[0]],
                num_outputs=1,
                name=f"mask_cast_{expert}",
            )
        else:
            cast = ir.node(
                "Cast",
                inputs=[equal.outputs[0]],
                attributes={"to": dtype.value},
                num_outputs=1,
                name=f"mask_cast_{expert}",
            )
        nodes.append(cast)
        expert_routing_weights = routing_weights.outputs[0]
        if alien_routing_expert == expert:
            alien = ir.node(
                "Identity", inputs=[routing_weights.outputs[0]], num_outputs=1, name=f"alien_routing_{expert}"
            )
            nodes.append(alien)
            expert_routing_weights = alien.outputs[0]
        weighted_mask = ir.node(
            "Mul", inputs=[expert_routing_weights, cast.outputs[0]], num_outputs=1, name=f"weighted_mask_{expert}"
        )
        nodes.append(weighted_mask)
        reduce_sum = ir.node(
            "ReduceSum",
            inputs=[weighted_mask.outputs[0], axes],
            attributes={"keepdims": 1},
            num_outputs=1,
            name=f"reduce_{expert}",
        )
        nodes.append(reduce_sum)

        expert_gate_format = (
            corrupt_gate_format[1]
            if corrupt_gate_format is not None and corrupt_gate_format[0] == expert
            else gate_format
        )
        gate_weight = _make_native_weight(rng, INTERMEDIATE, HIDDEN, expert_gate_format)
        up_weight = _make_native_weight(rng, INTERMEDIATE, HIDDEN, up_format)
        down_weight = _make_native_weight(rng, HIDDEN, INTERMEDIATE, down_format)
        weights[f"gate{expert}"] = gate_weight
        weights[f"up{expert}"] = up_weight
        weights[f"down{expert}"] = down_weight
        gate = _block_matmul(
            graph,
            nodes,
            f"expert{expert}.gate",
            cast_in(hidden, f"expert{expert}.gate"),
            gate_weight,
            expert_gate_format,
        )
        gate = cast_out(gate, f"expert{expert}.gate")
        if legacy_activation:
            activation_sigmoid = ir.node("Sigmoid", inputs=[gate], num_outputs=1, name=f"expert{expert}.sigmoid")
            nodes.append(activation_sigmoid)
            activation = ir.node(
                "Mul",
                inputs=[gate, activation_sigmoid.outputs[0]],
                num_outputs=1,
                name=f"expert{expert}.silu",
            )
        else:
            activation = ir.node("Swish", inputs=[gate], num_outputs=1, name=f"expert{expert}.silu")
        nodes.append(activation)
        up = _block_matmul(
            graph, nodes, f"expert{expert}.up", cast_in(hidden, f"expert{expert}.up"), up_weight, up_format
        )
        up = cast_out(up, f"expert{expert}.up")
        product = ir.node("Mul", inputs=[activation.outputs[0], up], num_outputs=1, name=f"expert{expert}.product")
        nodes.append(product)
        if broken_expert == expert:
            down = ir.node("Identity", inputs=[cast_in(product.outputs[0], f"expert{expert}.down")], num_outputs=1)
            nodes.append(down)
            down_output = cast_out(down.outputs[0], f"expert{expert}.down")
        else:
            down_output = _block_matmul(
                graph,
                nodes,
                f"expert{expert}.down",
                cast_in(product.outputs[0], f"expert{expert}.down"),
                down_weight,
                down_format,
                bias=biased_expert == expert,
            )
            down_output = cast_out(down_output, f"expert{expert}.down")
        contribution = ir.node(
            "Mul",
            inputs=[down_output, reduce_sum.outputs[0]],
            num_outputs=1,
            name=f"expert{expert}.contribution",
        )
        contribution.outputs[0].type = ir.TensorType(dtype)
        nodes.append(contribution)
        if routed is None:
            routed = contribution.outputs[0]
        else:
            add = ir.node("Add", inputs=[routed, contribution.outputs[0]], num_outputs=1, name=f"sum_{expert}")
            add.outputs[0].type = ir.TensorType(dtype)
            nodes.append(add)
            routed = add.outputs[0]

    shared = ir.node("Identity", inputs=[hidden], num_outputs=1, name="shared_expert")
    nodes.append(shared)
    final = ir.node("Add", inputs=[routed, shared.outputs[0]], num_outputs=1, name="final_add")
    final.outputs[0].name = "output"
    final.outputs[0].shape = hidden.shape
    final.outputs[0].type = ir.TensorType(dtype)
    nodes.append(final)
    for node in nodes:
        graph.append(node)
    graph.outputs.append(final.outputs[0])
    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = 21
    model.opset_imports["pkg.nxrt"] = 1
    return model, weights


def test_moe_surgeries_register_on_explicit_module_import():
    assert Surgeon.registry["fusedensemoetoqmoe"] is moe_surgeries.FuseDenseMoEToQMoE
    assert Surgeon.registry["fuseblockquantizedmoe"] is moe_surgeries.FuseBlockQuantizedMoE


def test_fuse_dense_moe_to_qmoe_preserves_bytes_attributes_and_shared_output(tmp_path):
    model, weights = _build_qmoe_graph()
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    assert _count(rewritten, "QMoE") == 1
    assert _count(rewritten, "MatMulNBits") == 1
    assert _count(rewritten, "Equal") == 0
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    assert qmoe.domain == "com.microsoft"
    assert len(qmoe.inputs) == 15
    assert qmoe.inputs[4] is None
    assert qmoe.inputs[7] is None
    assert [qmoe.attributes[name].value for name in ("k", "expert_weight_bits", "block_size")] == [
        TOP_K,
        BITS,
        BLOCK_SIZE,
    ]
    assert qmoe.attributes["activation_type"].value == "swiglu"
    assert qmoe.attributes["normalize_routing_weights"].value == 1
    assert qmoe.attributes["swiglu_fusion"].value == 2
    assert qmoe.attributes["quant_type"].value == "int"
    assert qmoe.attributes["weights_prepacked"].value == 0
    fc1_weights = qmoe.inputs[2].const_value.numpy()
    fc2_weights = qmoe.inputs[5].const_value.numpy()
    fc1_zero_points = qmoe.inputs[11].const_value.numpy()
    fc2_zero_points = qmoe.inputs[12].const_value.numpy()
    for expert in range(EXPERTS):
        np.testing.assert_array_equal(
            fc1_weights[expert],
            np.concatenate(
                [
                    weights[f"gate{expert}"].weight.reshape(INTERMEDIATE, -1),
                    weights[f"up{expert}"].weight.reshape(INTERMEDIATE, -1),
                ]
            ),
        )
        np.testing.assert_array_equal(fc2_weights[expert], weights[f"down{expert}"].weight.reshape(HIDDEN, -1))
        np.testing.assert_array_equal(
            fc1_zero_points[expert],
            np.concatenate([weights[f"gate{expert}"].zero_points, weights[f"up{expert}"].zero_points]),
        )
        np.testing.assert_array_equal(fc2_zero_points[expert], weights[f"down{expert}"].zero_points)
    final_add = next(node for node in rewritten.graph if node.name == "final_add")
    assert any(value.producer().op_type == "QMoE" for value in final_add.inputs)
    assert any(value.producer().name == "shared_expert" for value in final_add.inputs)


def test_fuse_dense_moe_to_qmoe_preserves_dense_forward_semantics(tmp_path):
    model, weights = _build_qmoe_graph(activation_dtype=ir.DataType.FLOAT)
    hidden = np.random.default_rng(2).standard_normal((5, HIDDEN)).astype(np.float32)

    router_logits = (
        hidden
        @ _dequantize(
            weights["router"].weight,
            weights["router"].scales,
            weights["router"].zero_points,
        ).T
    )
    selected = np.argsort(-router_logits, axis=-1, kind="stable")[:, :TOP_K]
    selected_logits = np.take_along_axis(router_logits, selected, axis=-1)
    routing_weights = np.exp(selected_logits - selected_logits.max(axis=-1, keepdims=True))
    routing_weights /= routing_weights.sum(axis=-1, keepdims=True)
    expected = hidden.copy()
    for expert in range(EXPERTS):
        gate = (
            hidden
            @ _dequantize(
                weights[f"gate{expert}"].weight,
                weights[f"gate{expert}"].scales,
                weights[f"gate{expert}"].zero_points,
            ).T
        )
        up = (
            hidden
            @ _dequantize(
                weights[f"up{expert}"].weight,
                weights[f"up{expert}"].scales,
                weights[f"up{expert}"].zero_points,
            ).T
        )
        activated = gate / (1 + np.exp(-gate)) * up
        down = (
            activated
            @ _dequantize(
                weights[f"down{expert}"].weight,
                weights[f"down{expert}"].scales,
                weights[f"down{expert}"].zero_points,
            ).T
        )
        expected += down * np.where(selected == expert, routing_weights, 0).sum(axis=-1, keepdims=True)

    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    actual = hidden.copy()
    for expert in range(EXPERTS):
        fc1 = _dequantize(
            qmoe.inputs[2].const_value.numpy()[expert],
            qmoe.inputs[3].const_value.numpy()[expert],
            qmoe.inputs[11].const_value.numpy()[expert],
        )
        gate = hidden @ fc1[:INTERMEDIATE].T
        up = hidden @ fc1[INTERMEDIATE:].T
        activated = gate / (1 + np.exp(-gate)) * up
        fc2 = _dequantize(
            qmoe.inputs[5].const_value.numpy()[expert],
            qmoe.inputs[6].const_value.numpy()[expert],
            qmoe.inputs[12].const_value.numpy()[expert],
        )
        down = activated @ fc2.T
        actual += down * np.where(selected == expert, routing_weights, 0).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_fuse_dense_moe_to_qmoe_casts_scales_and_router_to_activation_dtype(tmp_path, dtype):
    model, weights = _build_qmoe_graph(activation_dtype=dtype)
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    assert qmoe.inputs[3].dtype == dtype
    assert qmoe.inputs[6].dtype == dtype
    assert qmoe.inputs[1].producer().attributes["to"].value == dtype.value
    expected = np.concatenate(
        [weights["gate0"].scales.reshape(INTERMEDIATE, -1), weights["up0"].scales.reshape(INTERMEDIATE, -1)]
    ).astype(dtype.numpy())
    np.testing.assert_array_equal(qmoe.inputs[3].const_value.numpy()[0], expected)


def test_fuse_dense_moe_to_qmoe_matches_all_cast_and_activation_variants(tmp_path):
    model, _ = _build_qmoe_graph(
        legacy_activation=True,
        router_cast=True,
        topk_value_cast=True,
        mask_cast_op="Cast",
    )
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    assert _count(rewritten, "QMoE") == 1


def test_fuse_dense_moe_to_qmoe_supports_symmetric_weights_without_zero_points(tmp_path):
    model, _ = _build_qmoe_graph(include_zero_points=False)
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    assert qmoe.inputs[11] is None
    assert qmoe.inputs[12] is None


def test_fuse_dense_moe_to_qmoe_uses_scale_dtype_when_hidden_is_untyped(tmp_path):
    model, _ = _build_qmoe_graph(hidden_typed=False)
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    assert qmoe.inputs[3].dtype == ir.DataType.FLOAT16


def test_fuse_dense_moe_to_qmoe_keeps_unsupported_geometry(tmp_path):
    model, _ = _build_qmoe_graph(block_size=24)
    rewritten = _run_surgery(model, tmp_path, "FuseDenseMoEToQMoE")
    assert _count(rewritten, "QMoE") == 0
    assert _count(rewritten, "TopK") == 1


def test_fuse_dense_moe_to_qmoe_reads_and_writes_external_initializers(tmp_path):
    model, weights = _build_qmoe_graph()
    rewritten = _run_surgery(
        model,
        tmp_path,
        "FuseDenseMoEToQMoE",
        input_external=True,
        output_external=True,
    )
    qmoe = next(node for node in rewritten.graph if node.op_type == "QMoE")
    np.testing.assert_array_equal(
        qmoe.inputs[2].const_value.numpy()[0, :INTERMEDIATE],
        weights["gate0"].weight.reshape(INTERMEDIATE, -1),
    )


def test_fuse_block_quantized_moe_preserves_native_bytes_and_v1_layout(tmp_path):
    model, weights = _build_block_moe_graph()
    rewritten = _run_surgery(model, tmp_path, "FuseBlockQuantizedMoE")
    assert _count(rewritten, "BlockQuantizedMoE") == 1
    assert _count(rewritten, "BlockQuantizedMatMul") == 0
    assert _count(rewritten, "ScatterElements") == 2
    moe = next(node for node in rewritten.graph if node.op_type == "BlockQuantizedMoE")
    assert moe.domain == "pkg.nxrt"
    assert len(moe.inputs) == 9
    assert moe.inputs[3] is None
    assert moe.inputs[5] is None
    assert moe.inputs[6] is None
    assert moe.inputs[7] is None
    assert moe.attributes["format"].value == "iq4_xs"
    assert "block_layout_version" not in moe.attributes
    assert moe.attributes["k"].value == TOP_K
    assert moe.attributes["normalize_routing_weights"].value == 0
    assert moe.attributes["swiglu_fusion"].value == 2
    for expert in range(EXPERTS):
        np.testing.assert_array_equal(
            moe.inputs[2].const_value.numpy()[expert],
            np.concatenate([weights[f"gate{expert}"], weights[f"up{expert}"]], axis=0),
        )
        np.testing.assert_array_equal(moe.inputs[4].const_value.numpy()[expert], weights[f"down{expert}"])
    final_add = next(node for node in rewritten.graph if node.name == "final_add")
    assert any(value.producer().op_type == "BlockQuantizedMoE" for value in final_add.inputs)


def test_fuse_block_quantized_moe_supports_unfused_swiglu_layout_v2_schema(tmp_path):
    model, weights = _build_block_moe_graph(
        gate_format="iq1_s", up_format="iq3_xxs", down_format="iq4_xs", legacy_activation=True, mask_cast_op="Cast"
    )
    rewritten = _run_surgery(
        model,
        tmp_path,
        "FuseBlockQuantizedMoE",
        _allow_perproj_v2_schema=True,
    )
    moe = next(node for node in rewritten.graph if node.op_type == "BlockQuantizedMoE")
    assert moe.attributes["block_layout_version"].value == 2
    assert moe.attributes["fc1_format"].value == "iq1_s"
    assert moe.attributes["fc2_format"].value == "iq4_xs"
    assert moe.attributes["fc3_format"].value == "iq3_xxs"
    assert moe.attributes["swiglu_fusion"].value == 0
    for expert in range(EXPERTS):
        np.testing.assert_array_equal(moe.inputs[2].const_value.numpy()[expert], weights[f"gate{expert}"])
        np.testing.assert_array_equal(moe.inputs[4].const_value.numpy()[expert], weights[f"down{expert}"])
        np.testing.assert_array_equal(moe.inputs[6].const_value.numpy()[expert], weights[f"up{expert}"])


def test_fuse_block_quantized_moe_restores_non_float_output_dtype(tmp_path):
    model, _ = _build_block_moe_graph(dtype=ir.DataType.FLOAT16)
    rewritten = _run_surgery(model, tmp_path, "FuseBlockQuantizedMoE")
    moe = next(node for node in rewritten.graph if node.op_type == "BlockQuantizedMoE")
    consumer = next(node for node, _ in moe.outputs[0].uses())
    assert consumer.op_type == "Cast"
    assert consumer.attributes["to"].value == ir.DataType.FLOAT16.value


@pytest.mark.parametrize(
    ("builder_parameters", "message"),
    [
        ({"gate_format": "iq1_s", "up_format": "iq1_s", "down_format": "iq4_xs"}, "block_layout_version=2"),
        ({"corrupt_gate_format": (1, "iq1_m")}, "mixed native formats"),
        ({"biased_expert": 1}, "unsupported bias"),
        ({"broken_expert": EXPERTS - 1}, "did not trace"),
        ({"alien_routing_expert": EXPERTS - 1}, "did not trace"),
    ],
)
def test_fuse_block_quantized_moe_fails_closed_atomically(tmp_path, builder_parameters, message):
    model, _ = _build_block_moe_graph(**builder_parameters)
    input_path = tmp_path / "input.onnx"
    onnx.save_model(ir.to_proto(model), input_path)
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {
            "surgeries": [{"surgeon": "FuseBlockQuantizedMoE"}],
            "remove_duplicate_initializers": False,
        },
        disable_search=True,
    )
    with pytest.raises(moe_surgeries.MoEGraphSurgeryError, match=message):
        graph_surgeries.run(ONNXModelHandler(model_path=str(input_path)), str(tmp_path / "output"))
    unchanged = ir.from_proto(onnx.load(input_path))
    assert _count(unchanged, "BlockQuantizedMoE") == 0
    assert _count(unchanged, "Equal") == EXPERTS


def test_fuse_block_quantized_moe_allow_dense_opt_in_preserves_fallback(tmp_path):
    model, _ = _build_block_moe_graph(corrupt_gate_format=(1, "iq1_m"))
    rewritten = _run_surgery(model, tmp_path, "FuseBlockQuantizedMoE", allow_dense_moe=True)
    assert _count(rewritten, "BlockQuantizedMoE") == 0
    assert _count(rewritten, "Equal") == EXPERTS


def test_fuse_block_quantized_moe_reads_external_initializers(tmp_path):
    model, weights = _build_block_moe_graph()
    rewritten = _run_surgery(model, tmp_path, "FuseBlockQuantizedMoE", input_external=True)
    moe = next(node for node in rewritten.graph if node.op_type == "BlockQuantizedMoE")
    np.testing.assert_array_equal(
        moe.inputs[2].const_value.numpy()[0, :INTERMEDIATE],
        weights["gate0"],
    )
