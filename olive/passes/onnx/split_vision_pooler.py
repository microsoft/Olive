# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from collections.abc import Iterable
from pathlib import Path

import onnx
import onnx_ir as ir

from olive.common.config_utils import ParamCategory
from olive.common.utils import hardlink_copy_dir
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.config.model_config import ModelConfig
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

_GENAI_PROVIDERS = {
    ExecutionProvider.QNNExecutionProvider: "QNN",
    ExecutionProvider.OpenVINOExecutionProvider: "OpenVINO",
    ExecutionProvider.VitisAIExecutionProvider: "VitisAI",
}


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

    current = value
    while current.producer() is not None and current.producer().op_type in {"QuantizeLinear", "DequantizeLinear"}:
        producer = current.producer()
        data_input = producer.inputs[0]
        if data_input is None:
            break
        if value.shape is None and data_input.shape is not None:
            value.shape = data_input.shape
        if value.type is None:
            if producer.op_type == "DequantizeLinear":
                scale = producer.inputs[1]
                if scale is not None and scale.type is not None:
                    value.type = scale.type
            elif len(producer.inputs) > 2:
                zero_point = producer.inputs[2]
                if zero_point is not None and zero_point.type is not None:
                    value.type = zero_point.type
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

    used_initializers = {
        value.name for node in graph for value in node.inputs if value is not None and value.is_initializer()
    }
    for initializer_name in list(graph.initializers):
        if initializer_name not in used_initializers:
            graph.initializers.pop(initializer_name)

    graph.sort()
    model.metadata_props["gemma_vision_split_component"] = component_name
    model.metadata_props["gemma_vision_split_boundary"] = boundary_name
    return model


def _rebase_package_paths(value, source_root: Path, output_root: Path):
    if isinstance(value, dict):
        return {key: _rebase_package_paths(item, source_root, output_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_package_paths(item, source_root, output_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(source_root):
            return str(output_root / path.relative_to(source_root))
    return value


def _package_vision_split(
    source_dir: Path,
    output_dir: Path,
    components: list[ir.Model],
    names: list[str],
    config: type[BasePassConfig],
    provider: str,
) -> CompositeModelHandler:
    genai_path = source_dir / "genai_config.json"
    metadata_path = source_dir / "model_config.json"
    if not genai_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Mobius package must contain genai_config.json and model_config.json: {source_dir}")
    if output_dir.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("Split output directory cannot be inside the source Mobius package")

    with genai_path.open() as f:
        genai = json.load(f)
    with metadata_path.open() as f:
        metadata = json.load(f)
    vision = genai.get("model", {}).get("vision")
    if not isinstance(vision, dict) or not isinstance(vision.get("filename"), str):
        raise ValueError("Mobius genai_config.json must contain model.vision.filename")
    source_model_config = metadata.get("config", {})
    if metadata.get("type", "").lower() != "compositemodel" or not (
        source_model_config.get("model_attributes") or {}
    ).get("no_flatten"):
        raise ValueError("Mobius model_config.json must describe a no_flatten CompositeModel")

    original_vision = Path(vision["filename"])
    if original_vision.is_absolute() or ".." in original_vision.parts or not (source_dir / original_vision).is_file():
        raise ValueError(f"Invalid or missing Mobius vision model: {original_vision}")
    if "vision_encoder" not in source_model_config.get("model_component_names", []):
        raise ValueError("Mobius model_config.json has no vision_encoder component")
    old_root = Path(source_model_config["model_path"])
    if not old_root.is_absolute():
        raise ValueError("Mobius model_config.json must contain an absolute model_path")

    hardlink_copy_dir(source_dir, output_dir)
    vision_dir = output_dir / original_vision.parent
    for name, component in zip(names, components):
        filename = f"{original_vision.stem}_{name}.onnx"
        path = vision_dir / filename
        external_path = path.with_name(f"{filename}.data")
        for existing in (path, external_path):
            if existing.exists():
                existing.unlink()
        external_config = config.model_dump()
        external_config["external_data_name"] = external_path.name
        ir_model_to_olive_model(component, path, external_config)
        if component.ir_version <= onnx.IR_VERSION:
            onnx.checker.check_model(str(path))

    encoder_io = components[0].graph
    projector_io = components[1].graph
    encoder_inputs = [value.name for value in encoder_io.inputs]
    encoder_outputs = [value.name for value in encoder_io.outputs]
    projector_inputs = [value.name for value in projector_io.inputs]
    projector_outputs = [value.name for value in projector_io.outputs]
    if len(encoder_outputs) != 1 or projector_inputs.count(encoder_outputs[0]) != 1:
        raise ValueError("Split vision components must have one matching encoder-to-projector boundary")
    boundary = encoder_outputs[0]
    pipeline = vision.setdefault("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("model.vision.pipeline must be an object")
    encoder = pipeline.setdefault("encoder", {})
    projector = pipeline.setdefault("projector", {})
    if not isinstance(encoder, dict) or not isinstance(projector, dict):
        raise ValueError("model.vision.pipeline entries must be objects")
    encoder.update(
        filename=(original_vision.parent / f"{original_vision.stem}_encoder.onnx").as_posix(),
        inputs=encoder_inputs,
        outputs=encoder_outputs,
    )
    projector.update(
        filename=(original_vision.parent / f"{original_vision.stem}_pooler_projector.onnx").as_posix(),
        inputs=[boundary, *(name for name in projector_inputs if name != boundary)],
        outputs=projector_outputs,
        run_on_cpu=config.projector_on_cpu,
    )
    session_options = vision.setdefault("session_options", {})
    if not isinstance(session_options, dict):
        raise ValueError("model.vision.session_options must be an object")
    session_options["log_id"] = f"onnxruntime-genai-{provider.lower()}"
    session_options["provider_options"] = [{provider: config.provider_options or {}}]

    packaged_genai = output_dir / "genai_config.json"
    packaged_genai.unlink()
    packaged_genai.write_text(json.dumps(genai, indent=4) + "\n")

    rebased = _rebase_package_paths(metadata, old_root, output_dir)
    packaged_metadata = output_dir / "model_config.json"
    packaged_metadata.unlink()
    packaged_metadata.write_text(json.dumps(rebased, indent=4) + "\n")
    result = ModelConfig.model_validate(rebased).create_model()
    if not isinstance(result, CompositeModelHandler):
        raise ValueError("Mobius model_config.json did not create a CompositeModelHandler")
    return result


class SplitVisionPooler(Pass):
    """Split a vision encoder from its pooler, optionally packaging the complete Mobius export."""

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

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
            "source_package_dir": PassConfigParam(
                type_=str,
                default_value=None,
                category=ParamCategory.PATH,
                description="Optional Mobius export directory to copy and update with the split vision models.",
            ),
            "provider_options": PassConfigParam(
                type_=dict, default_value=None, description="Vision provider options in the packaged GenAI config."
            ),
            "projector_on_cpu": PassConfigParam(
                type_=bool, default_value=True, description="Run the packaged vision projector on CPU."
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        for name in ("pooler_marker", "primary_input", "boundary_name"):
            if not getattr(config, name):
                raise ValueError(f"{name} cannot be empty")

        source_dir = Path(config.source_package_dir).resolve() if config.source_package_dir else None
        provider = None
        if source_dir:
            if not source_dir.is_dir():
                raise ValueError(f"Mobius package directory does not exist: {source_dir}")
            provider = _GENAI_PROVIDERS.get(self.accelerator_spec.execution_provider)
            if provider is None:
                raise ValueError(f"Unsupported vision pipeline provider: {self.accelerator_spec.execution_provider}")
        elif config.provider_options is not None:
            raise ValueError("provider_options requires source_package_dir")

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
        if source_dir:
            return _package_vision_split(source_dir, output_dir, components, names, config, provider)
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers = []
        for name, component in zip(names, components):
            path = output_dir / f"{name}.onnx"
            handler = ir_model_to_olive_model(component, path, config)
            if component.ir_version <= onnx.IR_VERSION:
                onnx.checker.check_model(str(path))
            handlers.append(handler)
        return CompositeModelHandler(handlers, names, model_path=output_dir)
