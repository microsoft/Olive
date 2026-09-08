# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Weight-aware graph surgeries for dense-fallback Mixture-of-Experts graphs."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from onnxscript import ir

from olive.passes.onnx.graph_surgery import Surgeon

logger = logging.getLogger(__name__)

_MS_DOMAIN = "com.microsoft"
_NXRT_DOMAIN = "pkg.nxrt"
_BLOCK_MATMUL = "BlockQuantizedMatMul"
_BLOCK_MOE = "BlockQuantizedMoE"

_NATIVE_BLOCK_FORMATS = {
    "mxfp4": (32, 17),
    "iq4_nl": (32, 18),
    "iq4_xs": (256, 136),
    "iq3_s": (256, 110),
    "iq3_xxs": (256, 98),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq2_s": (256, 82),
    "iq1_s": (256, 50),
    "iq1_m": (256, 56),
}


class MoEGraphSurgeryError(ValueError):
    """A routed MoE graph cannot be represented by the requested sparse kernel."""


class _UnfusableBlockMoEError(Exception):
    """A native-block MoE layer cannot be represented by one fused node."""


class _UnfusableQMoEError(Exception):
    """A MatMulNBits MoE layer cannot be represented by one fused node."""


def _scalar_int(value: ir.Value | None) -> int | None:
    if value is None:
        return None
    const = value.const_value
    if const is None:
        producer = value.producer()
        if producer is None or producer.op_type != "Constant":
            return None
        for attr in ("value", "value_int", "value_ints"):
            if attr in producer.attributes:
                const = producer.attributes[attr].value
                break
        else:
            return None
    array = np.asarray(const.numpy() if hasattr(const, "numpy") else const).reshape(-1)
    return int(array[0]) if array.size else None


def _consumers_of_type(value: ir.Value, op_type: str) -> list[ir.Node]:
    return [node for node, _ in value.uses() if node.op_type == op_type]


def _single_consumer(value: ir.Value, *op_types: str) -> ir.Node | None:
    for op_type in op_types:
        matches = _consumers_of_type(value, op_type)
        if matches:
            return matches[0]
    return None


def _require_producer(value: ir.Value, op_type: str) -> ir.Node | None:
    producer = value.producer()
    return producer if producer is not None and producer.op_type == op_type else None


def _array(value: ir.Value, *, block_moe: bool = False) -> np.ndarray:
    const = value.const_value
    if const is None:
        message = f"expected an initializer for {value.name!r}"
        if block_moe:
            raise _UnfusableBlockMoEError(message)
        raise ValueError(message)
    return np.asarray(const.numpy())


def _make_initializer(graph: ir.Graph, name: str, array: np.ndarray, dtype: ir.DataType) -> ir.Value:
    tensor = ir.tensor(array, name=name, dtype=dtype)
    value = ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(dtype),
        const_value=tensor,
    )
    graph.register_initializer(value)
    return value


def _remove_dead_nodes(graph: ir.Graph) -> None:
    graph_outputs = set(graph.outputs)
    changed = True
    while changed:
        changed = False
        for node in reversed(list(graph)):
            if any(output in graph_outputs for output in node.outputs):
                continue
            if all(not list(output.uses()) for output in node.outputs):
                graph.remove(node, safe=True)
                changed = True

    used = {
        graph_input.name
        for node in graph
        for graph_input in node.inputs
        if graph_input is not None and graph_input.name is not None
    }
    for name in list(graph.initializers):
        if name not in used:
            del graph.initializers[name]


class _ExpertProjections:
    __slots__ = ("down", "gate", "up")

    def __init__(self, gate: ir.Node, up: ir.Node, down: ir.Node) -> None:
        self.gate = gate
        self.up = up
        self.down = down


def _find_routed_output(contributions: list[ir.Value]) -> ir.Value | None:
    if not contributions:
        return None
    contribution_set = set(contributions)
    accumulation_outputs: set[ir.Value] = set()
    current = contributions[0]
    advanced = True
    while advanced:
        advanced = False
        for node, index in current.uses():
            if node.op_type != "Add":
                continue
            other = node.inputs[1 - index]
            if other in contribution_set or other in accumulation_outputs:
                current = node.outputs[0]
                accumulation_outputs.add(current)
                advanced = True
                break
    return current


# MatMulNBits dense MoE to com.microsoft::QMoE.


def _trace_qmoe_expert(down: ir.Node) -> _ExpertProjections | None:
    activation_mul = _require_producer(down.inputs[0], "Mul")
    if activation_mul is None:
        return None
    first, second = activation_mul.inputs[:2]
    if first.producer() is not None and first.producer().op_type == "MatMulNBits":
        up_output, activation_output = first, second
    elif second.producer() is not None and second.producer().op_type == "MatMulNBits":
        up_output, activation_output = second, first
    else:
        return None

    up = up_output.producer()
    activation = activation_output.producer()
    if activation is None:
        return None
    if activation.op_type == "Swish":
        gate = activation.inputs[0].producer()
        if gate is not None and gate.op_type == "MatMulNBits":
            return _ExpertProjections(gate, up, down)
        return None
    if activation.op_type != "Mul":
        return None

    for gate_output, sigmoid_output in (
        (activation.inputs[0], activation.inputs[1]),
        (activation.inputs[1], activation.inputs[0]),
    ):
        gate = gate_output.producer()
        sigmoid = sigmoid_output.producer()
        if (
            gate is not None
            and gate.op_type == "MatMulNBits"
            and sigmoid is not None
            and sigmoid.op_type == "Sigmoid"
            and sigmoid.inputs[0] is gate_output
        ):
            return _ExpertProjections(gate, up, down)
    return None


def _router_matmul(logits: ir.Value) -> ir.Node | None:
    producer = logits.producer()
    if producer is not None and producer.op_type == "Cast":
        source = producer.inputs[0]
        producer = source.producer() if source is not None else None
    return producer if producer is not None and producer.op_type == "MatMulNBits" else None


def _find_qmoe_anchors(graph: ir.Graph) -> list[ir.Node]:
    anchors = []
    for node in graph:
        if node.op_type != "TopK" or node.inputs[0] is None or _router_matmul(node.inputs[0]) is None:
            continue
        if _consumers_of_type(node.outputs[1], "Equal"):
            anchors.append(node)
    return anchors


class _DenseQMoELayer:
    def __init__(self, topk: ir.Node) -> None:
        self.topk = topk
        self.gate_router = _router_matmul(topk.inputs[0])
        self.hidden = self.gate_router.inputs[0]
        self.logits = topk.inputs[0]
        self.k = _scalar_int(topk.inputs[1])
        softmax = _single_consumer(topk.outputs[0], "Softmax", "Cast")
        if softmax is not None and softmax.op_type == "Cast":
            softmax = _single_consumer(softmax.outputs[0], "Softmax")
        self.softmax = softmax
        self.declared_ids: set[int] = set()
        self.declared_branch_counts: dict[int, int] = {}
        self.experts: dict[int, _ExpertProjections] = {}
        self.contributions: list[ir.Value] = []
        self._collect_experts()
        self.routed_out = _find_routed_output(self.contributions)

    def _collect_experts(self) -> None:
        for equal in _consumers_of_type(self.topk.outputs[1], "Equal"):
            expert_id = _scalar_int(equal.inputs[1])
            if expert_id is None:
                continue
            self.declared_ids.add(expert_id)
            self.declared_branch_counts[expert_id] = self.declared_branch_counts.get(expert_id, 0) + 1
            cast = _single_consumer(equal.outputs[0], "CastLike", "Cast")
            if cast is None:
                continue
            weight_mul = _single_consumer(cast.outputs[0], "Mul")
            reduce_sum = _single_consumer(weight_mul.outputs[0], "ReduceSum") if weight_mul else None
            contribution = _single_consumer(reduce_sum.outputs[0], "Mul") if reduce_sum else None
            if contribution is None:
                continue
            weight_output = reduce_sum.outputs[0]
            expert_output = (
                contribution.inputs[1] if contribution.inputs[0] is weight_output else contribution.inputs[0]
            )
            down = _require_producer(expert_output, "MatMulNBits")
            projections = _trace_qmoe_expert(down) if down is not None else None
            if projections is not None:
                self.experts[expert_id] = projections
                self.contributions.append(contribution.outputs[0])

    def check_fusable(self) -> None:
        if self.k is None or self.softmax is None or self.routed_out is None:
            raise _UnfusableQMoEError("could not trace the router or weighted-sum accumulation")
        missing = sorted(self.declared_ids - set(self.experts))
        if missing:
            raise _UnfusableQMoEError(f"experts {missing} did not trace as MatMulNBits MLPs")
        duplicate_ids = sorted(expert_id for expert_id, count in self.declared_branch_counts.items() if count != 1)
        if duplicate_ids:
            raise _UnfusableQMoEError(f"experts {duplicate_ids} have duplicate routed branches")
        expert_ids = sorted(self.experts)
        if not expert_ids or expert_ids != list(range(len(expert_ids))):
            raise _UnfusableQMoEError(f"routed expert ids are not contiguous 0..E-1: {expert_ids}")

        expected_counts = []
        output_width = self.gate_router.attributes.get("N")
        if output_width is not None:
            expected_counts.append(int(output_width.value))
        router_weight = self.gate_router.inputs[1]
        if router_weight is not None and router_weight.const_value is not None:
            expected_counts.append(int(_array(router_weight).shape[0]))
        if expected_counts and len(set(expected_counts)) != 1:
            raise _UnfusableQMoEError(f"router expert dimensions disagree: {expected_counts}")
        if expected_counts and expert_ids != list(range(expected_counts[0])):
            raise _UnfusableQMoEError(
                f"router declares {expected_counts[0]} experts, but routed branches declare {expert_ids}"
            )


def _qmoe_abi_supported(bits: int, block_size: int) -> bool:
    return bits == 4 and block_size >= 16 and not block_size & (block_size - 1)


def _uniform_qmoe_geometry(nodes: list[ir.Node]) -> tuple[int, int]:
    try:
        geometries = {(int(node.attributes["bits"].value), int(node.attributes["block_size"].value)) for node in nodes}
    except KeyError as error:
        raise _UnfusableQMoEError(f"MatMulNBits projection is missing {error.args[0]!r}") from error
    if len(geometries) != 1:
        raise _UnfusableQMoEError(f"MatMulNBits projections use mismatched quantization geometry: {sorted(geometries)}")
    return next(iter(geometries))


def _consistent_zero_points(nodes: list[ir.Node], bank: str) -> bool:
    presence = {len(node.inputs) > 3 and node.inputs[3] is not None for node in nodes}
    if len(presence) != 1:
        raise _UnfusableQMoEError(f"{bank} projections disagree on zero-point presence")
    return next(iter(presence))


def _pack_qmoe_projection(nodes: list[ir.Node], slot: int) -> np.ndarray:
    arrays = [_array(node.inputs[slot]) for node in nodes]
    return np.stack([array.reshape(array.shape[0], -1) for array in arrays], axis=0)


def _pack_qmoe_scales(nodes: list[ir.Node], slot: int, out_features: int, dtype: ir.DataType) -> np.ndarray:
    return np.stack([_array(node.inputs[slot]).reshape(out_features, -1) for node in nodes], axis=0).astype(
        dtype.numpy()
    )


def _pack_qmoe_zero_points(nodes: list[ir.Node], slot: int, out_features: int) -> np.ndarray:
    return np.stack(
        [_array(node.inputs[slot]).reshape(out_features, -1).astype(np.uint8) for node in nodes],
        axis=0,
    )


@dataclasses.dataclass
class _QMoEFusionPlan:
    layer: _DenseQMoELayer
    prefix: str
    activation_dtype: ir.DataType
    bits: int
    block_size: int
    fc1_weights: np.ndarray
    fc1_scales: np.ndarray
    fc1_zero_points: np.ndarray | None
    fc2_weights: np.ndarray
    fc2_scales: np.ndarray
    fc2_zero_points: np.ndarray | None


def _plan_qmoe_layer(layer: _DenseQMoELayer, index: int, activation_dtype: ir.DataType) -> _QMoEFusionPlan:
    expert_ids = sorted(layer.experts)
    gate_nodes = [layer.experts[expert_id].gate for expert_id in expert_ids]
    up_nodes = [layer.experts[expert_id].up for expert_id in expert_ids]
    down_nodes = [layer.experts[expert_id].down for expert_id in expert_ids]
    bits, block_size = _uniform_qmoe_geometry([*gate_nodes, *up_nodes, *down_nodes])
    fc1_has_zero_points = _consistent_zero_points([*gate_nodes, *up_nodes], "FC1")
    fc2_has_zero_points = _consistent_zero_points(down_nodes, "FC2")

    try:
        intermediate_size = _array(gate_nodes[0].inputs[1]).shape[0]
        hidden_size = _array(down_nodes[0].inputs[1]).shape[0]
        fc1_weights = np.concatenate([_pack_qmoe_projection(gate_nodes, 1), _pack_qmoe_projection(up_nodes, 1)], axis=1)
        fc2_weights = _pack_qmoe_projection(down_nodes, 1)
        fc1_scales = np.concatenate(
            [
                _pack_qmoe_scales(gate_nodes, 2, intermediate_size, activation_dtype),
                _pack_qmoe_scales(up_nodes, 2, intermediate_size, activation_dtype),
            ],
            axis=1,
        )
        fc2_scales = _pack_qmoe_scales(down_nodes, 2, hidden_size, activation_dtype)
        fc1_zero_points = (
            np.concatenate(
                [
                    _pack_qmoe_zero_points(gate_nodes, 3, intermediate_size),
                    _pack_qmoe_zero_points(up_nodes, 3, intermediate_size),
                ],
                axis=1,
            )
            if fc1_has_zero_points
            else None
        )
        fc2_zero_points = _pack_qmoe_zero_points(down_nodes, 3, hidden_size) if fc2_has_zero_points else None
    except ValueError as error:
        raise _UnfusableQMoEError(str(error)) from error

    return _QMoEFusionPlan(
        layer=layer,
        prefix=f"moe.layer{index}",
        activation_dtype=activation_dtype,
        bits=bits,
        block_size=block_size,
        fc1_weights=fc1_weights,
        fc1_scales=fc1_scales,
        fc1_zero_points=fc1_zero_points,
        fc2_weights=fc2_weights,
        fc2_scales=fc2_scales,
        fc2_zero_points=fc2_zero_points,
    )


def _emit_qmoe(graph: ir.Graph, plan: _QMoEFusionPlan) -> None:
    layer = plan.layer
    prefix = plan.prefix

    fc1_weights_value = _make_initializer(graph, f"{prefix}.fc1_experts_weights", plan.fc1_weights, ir.DataType.UINT8)
    fc1_scales_value = _make_initializer(graph, f"{prefix}.fc1_scales", plan.fc1_scales, plan.activation_dtype)
    fc2_weights_value = _make_initializer(graph, f"{prefix}.fc2_experts_weights", plan.fc2_weights, ir.DataType.UINT8)
    fc2_scales_value = _make_initializer(graph, f"{prefix}.fc2_scales", plan.fc2_scales, plan.activation_dtype)
    fc1_zero_points_value = (
        _make_initializer(graph, f"{prefix}.fc1_zero_points", plan.fc1_zero_points, ir.DataType.UINT8)
        if plan.fc1_zero_points is not None
        else None
    )
    fc2_zero_points_value = (
        _make_initializer(graph, f"{prefix}.fc2_zero_points", plan.fc2_zero_points, ir.DataType.UINT8)
        if plan.fc2_zero_points is not None
        else None
    )

    cast = ir.node(
        "Cast",
        inputs=[layer.logits],
        attributes={"to": plan.activation_dtype.value},
        num_outputs=1,
        name=f"{prefix}.router_probs_cast",
    )
    cast.outputs[0].name = f"{prefix}.router_probs"
    qmoe = ir.node(
        "QMoE",
        inputs=[
            layer.hidden,
            cast.outputs[0],
            fc1_weights_value,
            fc1_scales_value,
            None,
            fc2_weights_value,
            fc2_scales_value,
            None,
            None,
            None,
            None,
            fc1_zero_points_value,
            fc2_zero_points_value,
            None,
            None,
        ],
        attributes={
            "activation_type": "swiglu",
            "normalize_routing_weights": 1,
            "k": layer.k,
            "expert_weight_bits": plan.bits,
            "block_size": plan.block_size,
            "swiglu_fusion": 2,
            "quant_type": "int",
            "weights_prepacked": 0,
        },
        domain=_MS_DOMAIN,
        num_outputs=1,
        name=f"{prefix}.qmoe",
    )
    qmoe.outputs[0].name = f"{prefix}.qmoe_output"
    qmoe.outputs[0].type = layer.hidden.type
    qmoe.outputs[0].shape = layer.routed_out.shape
    graph.insert_after(layer.topk, [cast, qmoe])
    if layer.routed_out in graph.outputs:
        qmoe.outputs[0].name = layer.routed_out.name
    layer.routed_out.replace_all_uses_with(qmoe.outputs[0], replace_graph_outputs=True)
    graph.opset_imports[_MS_DOMAIN] = 1


def _fuse_dense_moe_to_qmoe(model: ir.Model) -> int:
    graph = model.graph
    plans: list[_QMoEFusionPlan] = []
    for index, layer in enumerate(_DenseQMoELayer(topk) for topk in _find_qmoe_anchors(graph)):
        try:
            layer.check_fusable()
            nodes = [
                projection for expert in layer.experts.values() for projection in (expert.gate, expert.up, expert.down)
            ]
            bits, block_size = _uniform_qmoe_geometry(nodes)
        except _UnfusableQMoEError as reason:
            raise MoEGraphSurgeryError(f"QMoE graph surgery blocker at {layer.topk.name!r}: {reason}") from reason
        if not _qmoe_abi_supported(bits, block_size):
            logger.warning(
                "Skipping MoE layer at %s: QMoE ABI unsupported (bits=%d, block_size=%d)",
                layer.topk.name,
                bits,
                block_size,
            )
            continue
        down = layer.experts[min(layer.experts)].down
        activation_dtype = layer.hidden.dtype or down.inputs[2].dtype
        if activation_dtype not in {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
            logger.warning(
                "Skipping MoE layer at %s: activation dtype is missing or unsupported (%s)",
                layer.topk.name,
                activation_dtype,
            )
            continue
        try:
            plans.append(_plan_qmoe_layer(layer, index, activation_dtype))
        except _UnfusableQMoEError as reason:
            raise MoEGraphSurgeryError(f"QMoE graph surgery blocker at {layer.topk.name!r}: {reason}") from reason

    for plan in plans:
        _emit_qmoe(graph, plan)
    if plans:
        _remove_dead_nodes(graph)
        graph.sort()
    return len(plans)


class FuseDenseMoEToQMoE(Surgeon):
    """Fuse eligible dense MatMulNBits expert storms into com.microsoft::QMoE."""

    def call_ir(self, model: ir.Model) -> ir.Model:
        _fuse_dense_moe_to_qmoe(model)
        return model


# Native BlockQuantizedMatMul dense MoE to pkg.nxrt::BlockQuantizedMoE.


def _skip_cast(value: ir.Value) -> ir.Value:
    producer = value.producer()
    if producer is not None and producer.op_type == "Cast" and producer.inputs[0] is not None:
        return producer.inputs[0]
    return value


def _producer_through_cast(value: ir.Value, op_type: str) -> ir.Node | None:
    producer = _skip_cast(value).producer()
    return producer if producer is not None and producer.op_type == op_type else None


def _trace_block_expert(down: ir.Node) -> _ExpertProjections | None:
    activation_mul = _producer_through_cast(down.inputs[0], "Mul")
    if activation_mul is None:
        return None
    first, second = activation_mul.inputs[:2]
    if _producer_through_cast(first, _BLOCK_MATMUL) is not None:
        up_output, activation_output = first, second
    elif _producer_through_cast(second, _BLOCK_MATMUL) is not None:
        up_output, activation_output = second, first
    else:
        return None

    up = _producer_through_cast(up_output, _BLOCK_MATMUL)
    activation = _skip_cast(activation_output).producer()
    if activation is None:
        return None
    if activation.op_type == "Swish":
        gate = _producer_through_cast(activation.inputs[0], _BLOCK_MATMUL)
        return _ExpertProjections(gate, up, down) if gate is not None else None
    if activation.op_type != "Mul":
        return None

    for gate_output, sigmoid_output in (
        (activation.inputs[0], activation.inputs[1]),
        (activation.inputs[1], activation.inputs[0]),
    ):
        gate = _producer_through_cast(gate_output, _BLOCK_MATMUL)
        sigmoid = _skip_cast(sigmoid_output).producer()
        if (
            gate is not None
            and sigmoid is not None
            and sigmoid.op_type == "Sigmoid"
            and _producer_through_cast(sigmoid.inputs[0], _BLOCK_MATMUL) is gate
        ):
            return _ExpertProjections(gate, up, down)
    return None


def _find_moe_selectors(graph: ir.Graph) -> list[ir.Value]:
    selectors = []
    seen: set[int] = set()
    for node in graph:
        if node.op_type != "Equal":
            continue
        selected = node.inputs[0]
        if selected is None or _scalar_int(node.inputs[1]) is None:
            continue
        if len(_consumers_of_type(selected, "Equal")) < 2 or id(selected) in seen:
            continue
        seen.add(id(selected))
        selectors.append(selected)
    return selectors


class _DenseBlockMoELayer:
    def __init__(self, selected_experts: ir.Value) -> None:
        self.selected_experts = selected_experts
        self.routing_weights: ir.Value | None = None
        self.declared_ids: set[int] = set()
        self.experts: dict[int, _ExpertProjections] = {}
        self.contributions: list[ir.Value] = []
        self.k: int | None = None
        self._collect_experts()
        self._resolve_k()
        self.routed_out = _find_routed_output(self.contributions)

    def _collect_experts(self) -> None:
        for equal in _consumers_of_type(self.selected_experts, "Equal"):
            expert_id = _scalar_int(equal.inputs[1])
            if expert_id is None:
                continue
            self.declared_ids.add(expert_id)
            cast = _single_consumer(equal.outputs[0], "CastLike", "Cast")
            weight_mul = _single_consumer(cast.outputs[0], "Mul") if cast else None
            reduce_sum = _single_consumer(weight_mul.outputs[0], "ReduceSum") if weight_mul else None
            contribution = _single_consumer(reduce_sum.outputs[0], "Mul") if reduce_sum else None
            if contribution is None:
                continue

            routing_weights = weight_mul.inputs[1] if weight_mul.inputs[0] is cast.outputs[0] else weight_mul.inputs[0]
            weighted_output = reduce_sum.outputs[0]
            expert_output = (
                contribution.inputs[1] if contribution.inputs[0] is weighted_output else contribution.inputs[0]
            )
            down = _producer_through_cast(expert_output, _BLOCK_MATMUL)
            projections = _trace_block_expert(down) if down is not None else None
            if projections is None:
                continue
            if self.routing_weights is None:
                self.routing_weights = routing_weights
            elif self.routing_weights is not routing_weights:
                continue
            self.experts[expert_id] = projections
            self.contributions.append(contribution.outputs[0])

    def _resolve_k(self) -> None:
        shape = self.selected_experts.shape
        if shape is not None and len(shape) >= 1 and isinstance(shape[-1], int):
            self.k = shape[-1]
            return
        producer = self.selected_experts.producer()
        if producer is not None and producer.op_type == "TopK" and len(producer.inputs) > 1:
            self.k = _scalar_int(producer.inputs[1])

    @property
    def has_native_experts(self) -> bool:
        return bool(self.experts)

    def check_fusable(self) -> None:
        if self.routing_weights is None or self.routed_out is None:
            raise _UnfusableBlockMoEError("could not trace the router mask or weighted-sum accumulation")
        missing = sorted(self.declared_ids - set(self.experts))
        if missing:
            raise _UnfusableBlockMoEError(
                f"experts {missing} did not trace as native BlockQuantizedMatMul MLPs sharing the router"
            )
        expert_ids = sorted(self.experts)
        if expert_ids != list(range(len(expert_ids))):
            raise _UnfusableBlockMoEError(f"routed expert ids are not contiguous 0..E-1: {expert_ids}")
        if self.k is None or self.k < 1:
            raise _UnfusableBlockMoEError("could not determine the router top-k")


def _projection_format(node: ir.Node, role: str) -> str:
    attribute = node.attributes.get("format")
    if attribute is None:
        raise _UnfusableBlockMoEError(f"{role} projection {node.name!r} has no 'format' attribute")
    fmt = attribute.value
    if fmt not in _NATIVE_BLOCK_FORMATS:
        raise _UnfusableBlockMoEError(f"{role} projection uses unknown native format {fmt!r}")
    return fmt


def _uniform_format(nodes: list[ir.Node], role: str) -> str:
    formats = {_projection_format(node, role) for node in nodes}
    if len(formats) != 1:
        raise _UnfusableBlockMoEError(f"{role} projections use mixed native formats {sorted(formats)}")
    return next(iter(formats))


def _require_no_bias(nodes: list[ir.Node], role: str) -> None:
    for node in nodes:
        if len(node.inputs) > 2 and node.inputs[2] is not None:
            raise _UnfusableBlockMoEError(f"{role} projection {node.name!r} carries an unsupported bias")


def _stack_block_weights(nodes: list[ir.Node], out_features: int, role: str) -> np.ndarray:
    arrays = []
    for node in nodes:
        weight = _array(node.inputs[1], block_moe=True)
        if weight.ndim != 3 or weight.shape[0] != out_features:
            raise _UnfusableBlockMoEError(
                f"{role} projection weight has shape {weight.shape}, expected [{out_features}, n_blocks, block_bytes]"
            )
        arrays.append(weight)
    shapes = {weight.shape for weight in arrays}
    if len(shapes) != 1:
        raise _UnfusableBlockMoEError(f"{role} projections have ragged packed shapes {sorted(shapes)}")
    return np.stack(arrays, axis=0)


def _weight_out_features(node: ir.Node) -> int:
    return int(_array(node.inputs[1], block_moe=True).shape[0])


def _flat_last_dim_shape(source: ir.Value, prefix: str, negative_one: ir.Value) -> tuple[list[ir.Node], ir.Value]:
    last_dimension = ir.node(
        "Shape",
        inputs=[source],
        attributes={"start": -1},
        num_outputs=1,
        name=f"{prefix}.last_dim",
    )
    flat_shape = ir.node(
        "Concat",
        inputs=[negative_one, last_dimension.outputs[0]],
        attributes={"axis": 0},
        num_outputs=1,
        name=f"{prefix}.flat_shape",
    )
    return [last_dimension, flat_shape], flat_shape.outputs[0]


def _build_block_routing(
    graph: ir.Graph, layer: _DenseBlockMoELayer, experts: int, prefix: str
) -> tuple[list[ir.Node], ir.Value, ir.Value]:
    negative_one = _make_initializer(graph, f"{prefix}.neg_one", np.array([-1], dtype=np.int64), ir.DataType.INT64)
    expert_dimension = _make_initializer(
        graph, f"{prefix}.experts", np.array([experts], dtype=np.int64), ir.DataType.INT64
    )
    zero = _make_initializer(graph, f"{prefix}.zero", np.array(0.0, dtype=np.float32), ir.DataType.FLOAT)
    one = _make_initializer(graph, f"{prefix}.one", np.array(1.0, dtype=np.float32), ir.DataType.FLOAT)
    nodes: list[ir.Node] = []

    selected_shape_nodes, selected_flat_shape = _flat_last_dim_shape(
        layer.selected_experts, f"{prefix}.sel", negative_one
    )
    nodes.extend(selected_shape_nodes)
    selected_2d = ir.node(
        "Reshape",
        inputs=[layer.selected_experts, selected_flat_shape],
        num_outputs=1,
        name=f"{prefix}.sel2d",
    )
    nodes.append(selected_2d)

    weight_shape_nodes, weight_flat_shape = _flat_last_dim_shape(layer.routing_weights, f"{prefix}.rw", negative_one)
    nodes.extend(weight_shape_nodes)
    weights_2d = ir.node(
        "Reshape",
        inputs=[layer.routing_weights, weight_flat_shape],
        num_outputs=1,
        name=f"{prefix}.rw2d",
    )
    nodes.append(weights_2d)
    if layer.routing_weights.dtype == ir.DataType.FLOAT:
        routing_weights = weights_2d.outputs[0]
    else:
        cast = ir.node(
            "Cast",
            inputs=[weights_2d.outputs[0]],
            attributes={"to": ir.DataType.FLOAT.value},
            num_outputs=1,
            name=f"{prefix}.rw_float",
        )
        nodes.append(cast)
        routing_weights = cast.outputs[0]

    rows = ir.node(
        "Shape",
        inputs=[selected_2d.outputs[0]],
        attributes={"start": 0, "end": 1},
        num_outputs=1,
        name=f"{prefix}.rows",
    )
    dense_shape = ir.node(
        "Concat",
        inputs=[rows.outputs[0], expert_dimension],
        attributes={"axis": 0},
        num_outputs=1,
        name=f"{prefix}.dense_shape",
    )
    zeros = ir.node("Expand", inputs=[zero, dense_shape.outputs[0]], num_outputs=1, name=f"{prefix}.zeros")
    selected_shape = ir.node("Shape", inputs=[selected_2d.outputs[0]], num_outputs=1, name=f"{prefix}.sel_kshape")
    ones = ir.node("Expand", inputs=[one, selected_shape.outputs[0]], num_outputs=1, name=f"{prefix}.ones")
    nodes.extend([rows, dense_shape, zeros, selected_shape, ones])

    logits = ir.node(
        "ScatterElements",
        inputs=[zeros.outputs[0], selected_2d.outputs[0], ones.outputs[0]],
        attributes={"axis": 1},
        num_outputs=1,
        name=f"{prefix}.router_logits",
    )
    logits.outputs[0].name = f"{prefix}.router_logits"
    weights = ir.node(
        "ScatterElements",
        inputs=[zeros.outputs[0], selected_2d.outputs[0], routing_weights],
        attributes={"axis": 1},
        num_outputs=1,
        name=f"{prefix}.router_weights",
    )
    weights.outputs[0].name = f"{prefix}.router_weights"
    nodes.extend([logits, weights])
    return nodes, logits.outputs[0], weights.outputs[0]


@dataclasses.dataclass
class _BlockFusionPlan:
    layer: _DenseBlockMoELayer
    prefix: str
    moe_input: ir.Value
    fc1_weights: np.ndarray
    fc2_weights: np.ndarray
    fc3_weights: np.ndarray | None
    attributes: dict[str, object]
    experts: int


def _plan_block_layer(
    layer: _DenseBlockMoELayer, index: int, *, allow_per_projection_layout_v2: bool
) -> _BlockFusionPlan:
    expert_ids = sorted(layer.experts)
    gate_nodes = [layer.experts[expert_id].gate for expert_id in expert_ids]
    up_nodes = [layer.experts[expert_id].up for expert_id in expert_ids]
    down_nodes = [layer.experts[expert_id].down for expert_id in expert_ids]

    _require_no_bias(gate_nodes, "gate")
    _require_no_bias(up_nodes, "up")
    _require_no_bias(down_nodes, "down")
    gate_format = _uniform_format(gate_nodes, "gate")
    up_format = _uniform_format(up_nodes, "up")
    down_format = _uniform_format(down_nodes, "down")

    intermediate_size = _weight_out_features(gate_nodes[0])
    hidden_size = _weight_out_features(down_nodes[0])
    if _weight_out_features(up_nodes[0]) != intermediate_size:
        raise _UnfusableBlockMoEError("gate and up projections have different output widths")

    gate_weights = _stack_block_weights(gate_nodes, intermediate_size, "gate")
    up_weights = _stack_block_weights(up_nodes, intermediate_size, "up")
    down_weights = _stack_block_weights(down_nodes, hidden_size, "down")

    fc3_weights = None
    projection_formats = {"fc2": down_format}
    if gate_format == up_format:
        swiglu_fusion = 2
        fc1_weights = np.concatenate([gate_weights, up_weights], axis=1)
        projection_formats["fc1"] = gate_format
    else:
        swiglu_fusion = 0
        fc1_weights = gate_weights
        fc3_weights = up_weights
        projection_formats["fc1"] = gate_format
        projection_formats["fc3"] = up_format

    attributes: dict[str, object] = {
        "k": layer.k,
        "activation_type": "swiglu",
        "normalize_routing_weights": 0,
        "swiglu_fusion": swiglu_fusion,
    }
    formats = set(projection_formats.values())
    if len(formats) == 1:
        attributes["format"] = next(iter(formats))
    else:
        if not allow_per_projection_layout_v2:
            raise _UnfusableBlockMoEError(
                "mixed per-projection native formats require block_layout_version=2, "
                "which is not enabled for production graph surgery"
            )
        attributes.update(
            {
                "block_layout_version": 2,
                "format": projection_formats["fc1"],
                "fc1_format": projection_formats["fc1"],
                "fc2_format": projection_formats["fc2"],
            }
        )
        if "fc3" in projection_formats:
            attributes["fc3_format"] = projection_formats["fc3"]

    return _BlockFusionPlan(
        layer=layer,
        prefix=f"bqmoe.layer{index}",
        moe_input=gate_nodes[0].inputs[0],
        fc1_weights=fc1_weights,
        fc2_weights=down_weights,
        fc3_weights=fc3_weights,
        attributes=attributes,
        experts=len(expert_ids),
    )


def _emit_block_layer(graph: ir.Graph, plan: _BlockFusionPlan) -> None:
    fc1 = _make_initializer(graph, f"{plan.prefix}.fc1_experts_weights", plan.fc1_weights, ir.DataType.UINT8)
    fc2 = _make_initializer(graph, f"{plan.prefix}.fc2_experts_weights", plan.fc2_weights, ir.DataType.UINT8)
    fc3 = (
        _make_initializer(graph, f"{plan.prefix}.fc3_experts_weights", plan.fc3_weights, ir.DataType.UINT8)
        if plan.fc3_weights is not None
        else None
    )
    routing_nodes, router_logits, router_weights = _build_block_routing(graph, plan.layer, plan.experts, plan.prefix)
    moe = ir.node(
        _BLOCK_MOE,
        inputs=[plan.moe_input, router_logits, fc1, None, fc2, None, fc3, None, router_weights],
        attributes=plan.attributes,
        domain=_NXRT_DOMAIN,
        num_outputs=1,
        name=f"{plan.prefix}.bqmoe",
    )
    moe.outputs[0].name = f"{plan.prefix}.bqmoe_output"
    moe.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    moe.outputs[0].shape = plan.moe_input.shape
    new_nodes = [*routing_nodes, moe]

    routed_dtype = plan.layer.routed_out.dtype
    if routed_dtype not in (None, ir.DataType.FLOAT):
        cast = ir.node(
            "Cast",
            inputs=[moe.outputs[0]],
            attributes={"to": routed_dtype.value},
            num_outputs=1,
            name=f"{plan.prefix}.bqmoe_cast",
        )
        cast.outputs[0].name = f"{plan.prefix}.bqmoe_output_cast"
        cast.outputs[0].type = ir.TensorType(routed_dtype)
        cast.outputs[0].shape = plan.layer.routed_out.shape
        new_nodes.append(cast)
        final_output = cast.outputs[0]
    else:
        final_output = moe.outputs[0]

    graph.insert_after(plan.layer.routed_out.producer(), new_nodes)
    if plan.layer.routed_out in graph.outputs:
        final_output.name = plan.layer.routed_out.name
    plan.layer.routed_out.replace_all_uses_with(final_output, replace_graph_outputs=True)
    graph.opset_imports[_NXRT_DOMAIN] = 1


def _fail_block_moe_closed(
    layer: _DenseBlockMoELayer, reason: _UnfusableBlockMoEError, *, allow_dense_moe: bool
) -> None:
    detail = (
        f"routed native-block MoE at {layer.selected_experts.name!r} cannot be expressed as one sparse "
        f"pkg.nxrt::BlockQuantizedMoE node: {reason}"
    )
    if allow_dense_moe:
        logger.warning(
            "allow_dense_moe: %s. Keeping the dense BlockQuantizedMatMul fallback; every expert executes.",
            detail,
        )
        return
    raise MoEGraphSurgeryError(
        f"Sparse-MoE graph surgery blocker: {detail}. The surgery fails closed instead of emitting or preserving "
        "a dense-all-expert performance path. Set allow_dense_moe=True only to retain the runnable dense fallback."
    ) from reason


def _fuse_block_quantized_moe(model: ir.Model, *, allow_dense_moe: bool, allow_per_projection_layout_v2: bool) -> int:
    graph = model.graph
    plans: list[_BlockFusionPlan] = []
    for index, layer in enumerate(_DenseBlockMoELayer(value) for value in _find_moe_selectors(graph)):
        if not layer.has_native_experts:
            continue
        try:
            layer.check_fusable()
            plans.append(
                _plan_block_layer(
                    layer,
                    index,
                    allow_per_projection_layout_v2=allow_per_projection_layout_v2,
                )
            )
        except _UnfusableBlockMoEError as reason:
            _fail_block_moe_closed(layer, reason, allow_dense_moe=allow_dense_moe)

    for plan in plans:
        _emit_block_layer(graph, plan)
    if plans:
        _remove_dead_nodes(graph)
        graph.sort()
    return len(plans)


class FuseBlockQuantizedMoE(Surgeon):
    """Fuse eligible dense native-block expert storms into pkg.nxrt::BlockQuantizedMoE."""

    def __init__(self, allow_dense_moe: bool = False, _allow_perproj_v2_schema: bool = False) -> None:
        self.allow_dense_moe = allow_dense_moe
        self._allow_perproj_v2_schema = _allow_perproj_v2_schema

    def call_ir(self, model: ir.Model) -> ir.Model:
        _fuse_block_quantized_moe(
            model,
            allow_dense_moe=self.allow_dense_moe,
            allow_per_projection_layout_v2=self._allow_perproj_v2_schema,
        )
        return model


__all__ = ["FuseBlockQuantizedMoE", "FuseDenseMoEToQMoE", "MoEGraphSurgeryError"]
