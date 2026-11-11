# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import onnx
import onnx_ir as ir

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import (
    VISION_PIPELINE_KEY,
    get_external_data_config,
    ir_model_to_olive_model,
    update_vision_pipeline_genai_config,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam


def _graph_input_dependencies(value: ir.Value, seen: set[ir.Value] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    if value in seen:
        return set()
    seen.add(value)

    if value.is_graph_input():
        return {value.name}

    producer = value.producer()
    if producer is None:
        return set()

    dependencies: set[str] = set()
    for node_input in producer.inputs:
        if node_input is not None:
            dependencies.update(_graph_input_dependencies(node_input, seen))
    return dependencies


def _ancestor_nodes(outputs: Iterable[ir.Value], stop_values: set[ir.Value]) -> set[ir.Node]:
    selected: set[ir.Node] = set()
    pending = list(outputs)
    while pending:
        value = pending.pop()
        if value in stop_values or value.is_graph_input() or value.is_initializer():
            continue

        producer = value.producer()
        if producer is None or producer in selected:
            continue

        selected.add(producer)
        pending.extend(node_input for node_input in producer.inputs if node_input is not None)
    return selected


def _find_value(graph: ir.Graph, name: str) -> ir.Value:
    for value in graph.inputs:
        if value.name == name:
            return value
    for value in graph.outputs:
        if value.name == name:
            return value
    initializer = graph.initializers.get(name)
    if initializer is not None:
        return initializer
    for node in graph:
        for value in (*node.inputs, *node.outputs):
            if value is not None and value.name == name:
                return value
    raise ValueError(f"value {name!r} was not found in graph {graph.name!r}")


def _infer_qdq_value_metadata(value: ir.Value) -> None:
    if value.shape is not None and value.type is not None:
        return

    producer = value.producer()
    if value.type is None and producer is not None:
        if producer.op_type == "QuantizeLinear":
            zero_point = producer.inputs[2] if len(producer.inputs) > 2 else None
            value.type = (
                zero_point.type
                if zero_point is not None and zero_point.type is not None
                else ir.TensorType(ir.DataType.UINT8)
            )
        elif producer.op_type == "DequantizeLinear":
            scale = producer.inputs[1]
            if scale is not None and scale.type is not None:
                value.type = scale.type

    current = value
    while current.producer() is not None and current.producer().op_type in {"QuantizeLinear", "DequantizeLinear"}:
        producer = current.producer()
        data_input = producer.inputs[0]
        if data_input is None:
            break
        if value.shape is None and data_input.shape is not None:
            value.shape = data_input.shape
        current = data_input

    if value.shape is None or value.type is None:
        raise ValueError(f"could not infer complete type/shape metadata for boundary value {value.name!r}")


def _required_graph_inputs(nodes: set[ir.Node], graph_inputs: Iterable[ir.Value]) -> list[str]:
    used = {
        node_input.name
        for node in nodes
        for node_input in node.inputs
        if node_input is not None and node_input.is_graph_input()
    }
    return [value.name for value in graph_inputs if value.name in used]


def _referenced_value_names(graph: ir.Graph) -> set[str]:
    referenced = {value.name for node in graph for value in node.inputs if value is not None}
    referenced.update(value.name for value in graph.outputs)
    for node in graph:
        for attr in node.attributes.values():
            if not isinstance(attr, ir.Attr):
                continue
            if attr.type == ir.AttributeType.GRAPH:
                referenced.update(_referenced_value_names(attr.value))
            elif attr.type == ir.AttributeType.GRAPHS:
                for subgraph in attr.value:
                    referenced.update(_referenced_value_names(subgraph))
    return referenced


def _select_components(
    model: ir.Model, pooler_marker: str, primary_input: str
) -> tuple[set[int], set[int], str, list[str], list[str]]:
    graph = model.graph
    nodes = list(graph)
    pooler_nodes = {node for node in nodes if node.name is not None and pooler_marker in node.name}
    if not pooler_nodes:
        raise ValueError(f"no nodes contain pooler marker {pooler_marker!r}")

    pooler_inputs = {
        value
        for node in pooler_nodes
        for value in node.inputs
        if value is not None and not value.is_initializer() and value.producer() not in pooler_nodes
    }
    boundary_values = [
        value
        for value in pooler_inputs
        if not value.is_graph_input() and primary_input in _graph_input_dependencies(value)
    ]
    if len(boundary_values) != 1:
        names = sorted(value.name for value in boundary_values)
        raise ValueError(f"expected one boundary value depending on {primary_input!r}, found {names}")

    boundary = boundary_values[0]
    _infer_qdq_value_metadata(boundary)
    encoder_nodes = _ancestor_nodes([boundary], set())
    pooler_component_nodes = _ancestor_nodes(graph.outputs, {boundary})
    node_indices = {node: index for index, node in enumerate(nodes)}

    encoder_inputs = _required_graph_inputs(encoder_nodes, graph.inputs)
    pooler_inputs = _required_graph_inputs(pooler_component_nodes, graph.inputs)
    pooler_inputs.append(boundary.name)

    return (
        {node_indices[node] for node in encoder_nodes},
        {node_indices[node] for node in pooler_component_nodes},
        boundary.name,
        encoder_inputs,
        pooler_inputs,
    )


def _prune_component(
    source_model: ir.Model,
    selected_indices: set[int],
    input_names: list[str],
    output_names: list[str],
    boundary_original_name: str,
    boundary_name: str,
    component_name: str,
) -> ir.Model:
    model = source_model.clone(deep_copy=False)
    graph = model.graph
    nodes = list(graph)

    graph.inputs.clear()
    graph.outputs.clear()
    graph.remove([node for index, node in enumerate(nodes) if index not in selected_indices], safe=False)

    boundary = _find_value(graph, boundary_original_name)
    _infer_qdq_value_metadata(boundary)

    boundary_input: ir.Value | None = None
    if boundary_original_name in input_names:
        boundary_input = ir.Value(name=boundary_name, shape=boundary.shape, type=boundary.type)
        for node in graph:
            for index, node_input in enumerate(node.inputs):
                if node_input is boundary:
                    node.replace_input_with(index, boundary_input)
    else:
        boundary.name = boundary_name

    inputs = [boundary_input if name == boundary_original_name else _find_value(graph, name) for name in input_names]
    outputs = [boundary if name == boundary_original_name else _find_value(graph, name) for name in output_names]

    graph.inputs.extend(inputs)
    graph.outputs.extend(outputs)

    used_initializers = _referenced_value_names(graph)
    for initializer_name in list(graph.initializers):
        if initializer_name not in used_initializers:
            graph.initializers.pop(initializer_name)

    graph.sort()
    model.metadata_props["gemma_vision_split_component"] = component_name
    model.metadata_props["gemma_vision_split_boundary"] = boundary_name
    return model


class SplitVisionPooler(Pass):
    """Split a vision encoder from its dynamic pooler/projector at the feature tensor."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "pooler_marker": PassConfigParam(
                type_=str, default_value="pooler", description="Substring in pooler node names."
            ),
            "primary_input": PassConfigParam(
                type_=str, default_value="pixel_values", description="Input used to identify the encoder boundary."
            ),
            "boundary_name": PassConfigParam(
                type_=str, default_value="vision_features", description="Name of the tensor joining the components."
            ),
            "stage_session_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description="ORT GenAI session options keyed by vision pipeline stage name.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        for name in ("pooler_marker", "primary_input", "boundary_name"):
            if not getattr(config, name):
                raise ValueError(f"{name} cannot be empty")

        source_model = ir.load(model.model_path)
        encoder_indices, pooler_indices, boundary, encoder_inputs, pooler_inputs = _select_components(
            source_model, config.pooler_marker, config.primary_input
        )
        components = [
            _prune_component(
                source_model, encoder_indices, encoder_inputs, [boundary], boundary, config.boundary_name, "encoder"
            ),
            _prune_component(
                source_model,
                pooler_indices,
                pooler_inputs,
                [value.name for value in source_model.graph.outputs],
                boundary,
                config.boundary_name,
                "pooler_projector",
            ),
        ]
        names = ["encoder", "pooler_projector"]
        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers = []
        for name, component in zip(names, components):
            path = output_dir / f"{name}.onnx"
            external_data_config = config.model_dump()
            if external_data_config["external_data_name"]:
                external_data_name = Path(external_data_config["external_data_name"])
                external_data_config["external_data_name"] = str(
                    external_data_name.with_name(f"{name}.{external_data_name.name}")
                )
            handler = ir_model_to_olive_model(component, path, external_data_config)
            if component.ir_version <= onnx.IR_VERSION:
                onnx.checker.check_model(str(path))
            handlers.append(handler)

        vision_pipeline = {
            "vision_encoder": "encoder",
            "vision_pooler_projector": "pooler_projector",
        }
        model_attributes = deepcopy(model.model_attributes) or {}
        model_attributes[VISION_PIPELINE_KEY] = vision_pipeline
        return update_vision_pipeline_genai_config(
            CompositeModelHandler(
                handlers,
                names,
                model_path=output_dir,
                model_attributes=model_attributes,
            ),
            stage_session_options=config.stage_session_options,
        )
