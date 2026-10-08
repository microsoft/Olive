# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnx_ir as ir
from onnx_ir import serde
from onnx_ir.passes.common import TopologicalSortPass

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    copy_context_bin_files,
    get_context_bin_file_names,
    get_external_data_config,
    model_proto_to_file,
    process_llm_pipeline,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ComposeOnnxModels(Pass):
    """Compose multiple ONNX models into a single model.

    This pass chains multiple ONNX models together by itertively connecting the output of the preceding model to the
    input of the next model. The final inputs and outputs are the set of all inputs and outputs of the models excluding
    those used to connect the models together.

    It also handles llm_pipeline models:
    - embeddings: the embeddings model is saved as is
    - context: the context model is composed of all models in the context group
    - iterator: the iterator model is composed of all models in the iterator group
    - lm_head: the lm_head model is saved as is

    Consumed outputs are preserved when named by the ORT GenAI decoder output contract
    or explicitly listed in ``preserved_outputs`` (literal names or ``%d`` templates).
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_external_data_config(),
            "preserved_outputs": PassConfigParam(
                type_=list[str],
                default_value=[],
                description=(
                    "Outputs to keep even when consumed by another component. Accepts literal names and %d templates."
                    " Added to outputs declared in genai_config.json, if present."
                ),
            ),
        }

    def _run_for_config(
        self,
        model: CompositeModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        assert isinstance(model, CompositeModelHandler), "ComposeOnnxModels pass only supports CompositeModelHandler"
        assert all(isinstance(m, ONNXModelHandler) for m in model.model_components), (
            "All components must be ONNXModelHandler"
        )

        state_output_patterns = self._state_output_patterns(model)
        state_output_patterns.extend(self._compile_output_patterns(config.preserved_outputs))
        external_config = {key: value for key, value in config.model_dump().items() if key != "preserved_outputs"}

        if pipeline := (model.model_attributes or {}).get("llm_pipeline"):
            output_model_path = Path(output_model_path).with_suffix("")

            def process_context_iterator(component_models, llm_pipeline, output_dir):
                new_groups = {
                    "context": {},
                    "iterator": {},
                }

                # compose the context and iterator models
                composed_suffix = (
                    "_ctx"
                    if get_context_bin_file_names(component_models[llm_pipeline["context"][0]].model_path)
                    else ""
                )
                saved_cb_files = {}
                for group_name in ["context", "iterator"]:
                    composed_name = f"{group_name}{composed_suffix}"
                    new_groups[group_name][composed_name] = self._get_composed_model(
                        [component_models[component_name].model_path for component_name in llm_pipeline[group_name]],
                        output_dir / f"{composed_name}.onnx",
                        external_config=external_config,
                        saved_cb_files=saved_cb_files,
                        as_model_dir=True,
                        state_output_patterns=state_output_patterns,
                    )

                return new_groups

            return process_llm_pipeline(model, pipeline, process_context_iterator, output_model_path)

        return self._get_composed_model(
            [component.model_path for component in model.model_components],
            resolve_onnx_path(output_model_path),
            external_config=external_config,
            state_output_patterns=state_output_patterns,
        )

    @staticmethod
    def _state_output_patterns(model: CompositeModelHandler) -> list[re.Pattern]:
        """Return patterns matching the values the decoder contract declares as state outputs.

        Composition treats any component output that a later component reads as an internal
        connection and drops it from the composed graph. That is right for an activation handed
        from one chunk to the next, but wrong for a KV cache tensor. A model that shares one
        layer's cache with later layers (Gemma 4 feeds layers 13 and 14 into layers 15 through
        34) produces a ``present.*`` value that is both cache state the runtime has to read back
        and an input to a later chunk. Dropping it leaves those cache slots frozen at whatever
        the runtime first supplied, which is invisible during prefill, because the value is
        recomputed and consumed within the same run, and corrupts every step after it.

        The genai config names these values explicitly, as literal names or templates such as
        ``present.%d.key``. Without a readable contract, use the pass's ``preserved_outputs`` option.
        """
        for file_path in (model.model_attributes or {}).get("additional_files") or []:
            if Path(file_path).name != "genai_config.json":
                continue
            try:
                with open(file_path) as f:
                    outputs = json.load(f)["model"]["decoder"]["outputs"]
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Could not read decoder outputs from %s. State outputs may be dropped.", file_path)
                return []
            return ComposeOnnxModels._compile_output_patterns(
                [value for value in outputs.values() if isinstance(value, str)]
            )
        return []

    @staticmethod
    def _compile_output_patterns(names: list[str]) -> list[re.Pattern]:
        return [re.compile("^" + r"\d+".join(re.escape(part) for part in name.split("%d")) + "$") for name in names]

    @staticmethod
    def _get_composed_model(
        onnx_model_paths: list,
        output_model_path: Union[str, Path],
        external_config: dict,
        saved_cb_files: Optional[dict] = None,
        as_model_dir: bool = False,
        state_output_patterns: Optional[list[re.Pattern]] = None,
    ) -> ONNXModelHandler:
        """Compose multiple ONNX models into a single model.

        :param onnx_model_paths: List of ONNX model paths.
        :param output_model_path: Path to save the composed ONNX model.
        :param external_config: Configuration for external data.
        :param saved_cb_files: Dictionary of saved context binary files.
        :param as_model_dir: Use model parent directory as output model_path.
        :param state_output_patterns: Patterns for outputs that must be kept even when a later
            component consumes them, because they carry state the runtime reads back rather than
            an internal connection. See :meth:`_state_output_patterns`.
        :return: Composed ONNX model.
        """

        def shape_list(value: ir.Value):
            if value.shape is None:
                return None
            return [dim.value if isinstance(dim, ir.SymbolicDim) else dim for dim in value.shape]

        ir_models = []
        for path in onnx_model_paths:
            ir_model = ir.load(path)
            TopologicalSortPass()(ir_model)
            ir_models.append(ir_model)

        seen_inputs = set()
        seen_outputs = set()
        for ir_model in ir_models:
            graph_inputs = {value.name for value in ir_model.graph.inputs}
            graph_outputs = {value.name for value in ir_model.graph.outputs}
            # avoid circular connection, model_2 output cannot be model_1 input
            assert graph_outputs.isdisjoint(seen_inputs), (
                f"Output names {graph_outputs.intersection(seen_inputs)} are already used as input names."
            )
            # avoid reused output name
            assert graph_outputs.isdisjoint(seen_outputs), (
                f"Output names {graph_outputs.intersection(seen_outputs)} are already used as output names."
            )

            # update seen inputs and outputs
            seen_inputs.update(graph_inputs)
            seen_outputs.update(graph_outputs)

        # will only keep the unused outputs
        # inputs will be automatically taken care of during compose
        final_outputs = seen_outputs - seen_inputs

        # ... except for values the decoder contract declares as state. Those are read back by
        # the runtime, so they stay graph outputs even though a later component also consumes
        # them. Without this, a model that shares a KV cache across layers silently loses the
        # shared entries and produces garbage after its first token.
        if state_output_patterns:
            preserved = {
                name
                for name in seen_outputs - final_outputs
                if any(pattern.match(name) for pattern in state_output_patterns)
            }
            if preserved:
                logger.debug("Keeping state outputs consumed by later components: %s", sorted(preserved))
                final_outputs |= preserved

        # compose by relinking values across models by name
        composed_values: dict[str, ir.Value] = {}
        composed_inputs: list[ir.Value] = []
        composed_input_names: set[str] = set()
        composed_initializers: dict[str, ir.Value] = {}
        composed_nodes: list[ir.Node] = []
        composed_node_names: dict[str, ir.Node] = {}
        composed_output_names: list[str] = []
        produced_names: set[str] = set()
        reserved_value_names = {
            value.name
            for model in ir_models
            for value in (
                *model.graph.inputs,
                *model.graph.initializers.values(),
                *model.graph.outputs,
                *(output for node in model.graph for output in node.outputs),
            )
            if value.name
        }
        reserved_node_names = {node.name for model in ir_models for node in model.graph if node.name}

        def get_value(name: str) -> ir.Value:
            value = composed_values.get(name)
            if value is None:
                value = ir.Value(name=name)
                composed_values[name] = value
            return value

        def make_unique(name: str, taken: set[str]) -> str:
            candidate = f"{name}_composed{model_idx}"
            suffix = 0
            while candidate in taken:
                suffix += 1
                candidate = f"{name}_composed{model_idx}_{suffix}"
            taken.add(candidate)
            return candidate

        def epcontext_io_names(graph: ir.Graph) -> set[str]:
            """Return the names of every value an EPContext node in ``graph`` reads or writes.

            An EPContext node's input and output names are a contract with the compiled context
            binary it points at: the runtime deserializes the binary, reads the tensor names baked
            into it, and looks each one up among the names on the ONNX node. Renaming any of them
            makes that lookup fail during session creation, so they must be left untouched.
            """
            names = set()
            for node in graph:
                if node.op_type != "EPContext":
                    continue
                for value in (*node.inputs, *node.outputs):
                    if value is not None and value.name:
                        names.add(value.name)
            return names

        for model_idx, ir_model in enumerate(ir_models):  # noqa: B007  # used by make_unique closure
            graph = ir_model.graph
            # Names generated by an execution provider (e.g. the QNN EP's "<tensor>_q_token_<N>"
            # bridging nodes) are only unique within a single compilation. Independently compiled
            # component models can therefore reuse a name for a genuinely different node. Such nodes
            # are renamed, and this map rewires the rest of the component onto the new names.
            local_renames: dict[str, str] = {}

            # Graph outputs form the wiring contract between components and must keep their names.
            graph_output_names = {out.name for out in graph.outputs if out.name}
            # ... and so do the names an EPContext node reads or writes.
            protected_names = epcontext_io_names(graph)
            previous_value_names = set(composed_values)
            graph_input_names = {value.name for value in graph.inputs}

            for inp in graph.inputs:
                name = inp.name
                # An earlier component's initializer already supplies this value, so consume it
                # instead of also declaring a graph input of the same name, which would leave one
                # value both constant and externally fed.
                if name in composed_input_names or name in produced_names or name in composed_initializers:
                    # already a graph input, an initializer, or an internal connection from a
                    # previous model
                    existing = composed_values[name]
                    assert shape_list(inp) == shape_list(existing), f"Input shape mismatch: {name}"
                    assert inp.dtype == existing.dtype, f"Input dtype mismatch: {name}"
                    continue

                # will add to the composed graph inputs
                value = get_value(name)
                value.type = inp.type
                value.shape = inp.shape
                composed_inputs.append(value)
                composed_input_names.add(name)

            for init in graph.initializers.values():
                name = init.name
                is_overridable = name in graph_input_names
                different_default_contract = name in composed_initializers and (
                    is_overridable != (name in composed_input_names)
                )
                if name in composed_initializers and not different_default_contract:
                    np.testing.assert_array_equal(
                        init.const_value.numpy(),
                        composed_initializers[name].const_value.numpy(),
                        err_msg=f"Initializer mismatch: {name}",
                    )
                    continue

                # ONNX values share a single namespace, so an initializer collides with any
                # existing composed value, not only with an earlier initializer. Reusing the name
                # would attach this constant to an earlier component's graph input or node output
                # rather than introducing a new value. Rename and rewire, mirroring the node output
                # handling below.
                is_own_input_default = name in graph_input_names and name not in previous_value_names
                if (
                    name in produced_names
                    or (name in composed_input_names and not is_own_input_default)
                    or different_default_contract
                ):
                    if name in graph_output_names:
                        raise ValueError(
                            f"Cannot compose the given models: '{name}' is an initializer of one component"
                            " and a graph output of another. Graph output names are the wiring contract"
                            " between components and cannot be renamed, so the two values cannot coexist"
                            " in one graph. Make this name unique across components."
                        )
                    if name in protected_names:
                        raise ValueError(
                            f"Cannot compose the given models: '{name}' is an initializer of one component"
                            " and is read or written by an EPContext node. EPContext input and"
                            " output names are a contract with the compiled context binary and cannot be"
                            " renamed, so the two values cannot coexist in one graph. Make this name unique"
                            " across components before generating the context binaries."
                        )
                    new_name = local_renames.setdefault(name, make_unique(name, reserved_value_names))
                    logger.debug("Renaming colliding initializer %s to %s", name, new_name)
                    name = new_name

                value = get_value(name)
                value.const_value = init.const_value
                value.type = init.type if init.type is not None else ir.TensorType(init.const_value.dtype)
                value.shape = init.shape if init.shape is not None else ir.Shape(init.const_value.shape)
                composed_initializers[name] = value
                if init.name in graph_input_names and name not in composed_input_names:
                    # A renamed overridable initializer still needs its matching public input.
                    composed_inputs.append(value)
                    composed_input_names.add(name)

            for node in graph:
                name = node.name
                # ONNX node names are optional, so only dedup nodes that carry a non-empty name.
                # Unnamed nodes always get added to avoid different unnamed nodes colliding on the
                # same "" / None key (which could trigger false "Node mismatch" assertions).
                if name and name in composed_node_names:
                    # there might be some dq nodes for initializers that are common between models
                    # since split model keeps dq with the consumer op.
                    # The comparison is against the node as written in this component, before
                    # ``local_renames`` is applied to it. If an earlier collision renamed one of
                    # this node's values, the node is no longer equivalent to the one already
                    # composed even though it still serializes identically, so dropping it would
                    # wire this component onto the earlier component's value.
                    remapped = any(
                        value is not None and value.name in local_renames for value in (*node.inputs, *node.outputs)
                    )
                    if (
                        not remapped
                        and serde.serialize_node(composed_node_names[name]).SerializeToString()
                        == serde.serialize_node(node).SerializeToString()
                    ):
                        continue
                    # Same name but different node: rename this one. Its outputs are only renamed
                    # if they actually collide, which the loop below takes care of.
                    new_name = make_unique(name, reserved_node_names)
                    logger.debug("Renaming duplicate node %s to %s", name, new_name)
                    name = new_name

                # Distinct nodes can still collide on output names: an EP numbers the values it
                # invents (e.g. the unused optional outputs of SkipSimplifiedLayerNormalization)
                # per compilation, so independently compiled components restart the same "val_N"
                # sequence. Rename only the outputs that actually collide, and rewire the rest of
                # this component onto them. Renaming a node's other outputs as well would be enough
                # to break an EPContext node downstream of them.
                for output in node.outputs:
                    out_name = output.name
                    # ONNX values share a single namespace, so a node output collides with any
                    # existing composed value, not only with an earlier node output: an earlier
                    # component's graph input or initializer of the same name would otherwise keep
                    # that name while this node claims it, leaving the graph non SSA.
                    if not out_name or not (
                        out_name in produced_names
                        or out_name in composed_input_names
                        or out_name in composed_initializers
                    ):
                        continue
                    if out_name in graph_output_names:
                        raise ValueError(
                            f"Cannot compose the given models: '{out_name}' is produced by more than one"
                            " component and is a graph output of one of them. Graph output names are the"
                            " wiring contract between components and cannot be renamed, so the two values"
                            " cannot coexist in one graph. Make this name unique across components."
                        )
                    if out_name in protected_names:
                        raise ValueError(
                            f"Cannot compose the given models: '{out_name}' is produced by more than one"
                            " component and is read or written by an EPContext node. EPContext input and"
                            " output names are a contract with the compiled context binary and cannot be"
                            " renamed, so the two values cannot coexist in one graph. Make this name unique"
                            " across components before generating the context binaries."
                        )
                    local_renames.setdefault(out_name, make_unique(out_name, reserved_value_names))

                new_inputs = [
                    get_value(local_renames.get(inp.name, inp.name)) if inp is not None else None for inp in node.inputs
                ]
                new_node = ir.Node(
                    node.domain,
                    node.op_type,
                    inputs=new_inputs,
                    attributes=list(node.attributes.values()),
                    overload=node.overload,
                    num_outputs=len(node.outputs),
                    version=node.version,
                    name=name,
                )
                for old_output, new_output in zip(node.outputs, new_node.outputs):
                    out_name = old_output.name
                    if not out_name:
                        continue
                    out_name = local_renames.get(out_name, out_name)
                    new_output.name = out_name
                    new_output.type = old_output.type
                    new_output.shape = old_output.shape
                    composed_values[out_name] = new_output
                    produced_names.add(out_name)
                composed_nodes.append(new_node)
                if name:
                    composed_node_names[name] = new_node

            for out in graph.outputs:
                name = local_renames.get(out.name, out.name)
                value = composed_values.get(name)
                if value is None:
                    value = get_value(name)
                    value.type = out.type
                    value.shape = out.shape
                if name not in composed_output_names:
                    composed_output_names.append(name)

        # merge opset imports across all component models, taking the max version per domain
        composed_opset_imports: dict[str, int] = {}
        for ir_model in ir_models:
            for domain, version in ir_model.graph.opset_imports.items():
                if domain not in composed_opset_imports or version > composed_opset_imports[domain]:
                    composed_opset_imports[domain] = version

        composed_graph = ir.Graph(
            inputs=composed_inputs,
            outputs=[composed_values[name] for name in composed_output_names if name in final_outputs],
            nodes=composed_nodes,
            initializers=list(composed_initializers.values()),
            name=ir_models[0].graph.name,
            opset_imports=composed_opset_imports,
        )
        composed_model = ir.Model(
            composed_graph,
            ir_version=ir_models[0].ir_version,
            producer_name="olive",
        )
        TopologicalSortPass()(composed_model)

        # save the composed model
        output_model_path = Path(output_model_path)
        has_external_data = model_proto_to_file(ir.to_proto(composed_model), output_model_path, **external_config)

        # copy over context binary files if any
        saved_cb_files = saved_cb_files if saved_cb_files is not None else {}
        for path in onnx_model_paths:
            has_external_data |= copy_context_bin_files(path, output_model_path.parent, saved_cb_files=saved_cb_files)

        return ONNXModelHandler(
            output_model_path.parent if (has_external_data or as_model_dir) else output_model_path,
            onnx_file_name=output_model_path.name if (has_external_data or as_model_dir) else None,
        )
