# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
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
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return get_external_data_config()

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
                        external_config=config.model_dump(),
                        saved_cb_files=saved_cb_files,
                        as_model_dir=True,
                    )

                return new_groups

            return process_llm_pipeline(model, pipeline, process_context_iterator, output_model_path)

        return self._get_composed_model(
            [component.model_path for component in model.model_components],
            resolve_onnx_path(output_model_path),
            external_config=config.model_dump(),
        )

    @staticmethod
    def _get_composed_model(
        onnx_model_paths: list,
        output_model_path: Union[str, Path],
        external_config: dict,
        saved_cb_files: Optional[dict] = None,
        as_model_dir: bool = False,
    ) -> ONNXModelHandler:
        """Compose multiple ONNX models into a single model.

        :param onnx_model_paths: List of ONNX model paths.
        :param output_model_path: Path to save the composed ONNX model.
        :param external_config: Configuration for external data.
        :param saved_cb_files: Dictionary of saved context binary files.
        :param as_model_dir: Use model parent directory as output model_path.
        :return: Composed ONNX model.
        """

        def shape_list(value: ir.Value):
            if value.shape is None:
                return None
            return [dim.value if isinstance(dim, ir.SymbolicDim) else dim for dim in value.shape]

        def dtype_of(value: ir.Value):
            return value.type.dtype if value.type is not None else None

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

        # compose by relinking values across models by name
        composed_values: dict[str, ir.Value] = {}
        composed_inputs: list[ir.Value] = []
        composed_input_names: set[str] = set()
        composed_initializers: dict[str, ir.Value] = {}
        composed_nodes: list[ir.Node] = []
        composed_node_names: dict[str, ir.Node] = {}
        composed_output_names: list[str] = []
        produced_names: set[str] = set()

        def get_value(name: str) -> ir.Value:
            value = composed_values.get(name)
            if value is None:
                value = ir.Value(name=name)
                composed_values[name] = value
            return value

        for ir_model in ir_models:
            graph = ir_model.graph

            for inp in graph.inputs:
                name = inp.name
                if name in composed_input_names or name in produced_names:
                    # already a graph input or an internal connection from a previous model
                    existing = composed_values[name]
                    assert shape_list(inp) == shape_list(existing), f"Input shape mismatch: {name}"
                    assert dtype_of(inp) == dtype_of(existing), f"Input dtype mismatch: {name}"
                    continue

                # will add to the composed graph inputs
                value = get_value(name)
                value.type = inp.type
                value.shape = inp.shape
                composed_inputs.append(value)
                composed_input_names.add(name)

            for init in graph.initializers.values():
                name = init.name
                if name in composed_initializers:
                    np.testing.assert_array_equal(
                        init.const_value.numpy(),
                        composed_initializers[name].const_value.numpy(),
                        err_msg=f"Initializer mismatch: {name}",
                    )
                    continue

                value = get_value(name)
                value.const_value = init.const_value
                value.type = init.type if init.type is not None else ir.TensorType(init.const_value.dtype)
                value.shape = init.shape if init.shape is not None else ir.Shape(init.const_value.shape)
                composed_initializers[name] = value

            for node in graph:
                name = node.name
                if name in composed_node_names:
                    # there might be some dq nodes for initializers that are common between models
                    # since split model keeps dq with the consumer op
                    assert (
                        serde.serialize_node(composed_node_names[name]).SerializeToString()
                        == serde.serialize_node(node).SerializeToString()
                    ), f"Node mismatch: {name}"
                    continue

                new_inputs = [get_value(inp.name) if inp is not None else None for inp in node.inputs]
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
                    new_output.name = out_name
                    new_output.type = old_output.type
                    new_output.shape = old_output.shape
                    composed_values[out_name] = new_output
                    produced_names.add(out_name)
                composed_nodes.append(new_node)
                composed_node_names[name] = new_node

            for out in graph.outputs:
                name = out.name
                value = composed_values.get(name)
                if value is None:
                    value = get_value(name)
                    value.type = out.type
                    value.shape = out.shape
                if name not in composed_output_names:
                    composed_output_names.append(name)

        composed_graph = ir.Graph(
            inputs=composed_inputs,
            outputs=[composed_values[name] for name in composed_output_names if name in final_outputs],
            nodes=composed_nodes,
            initializers=list(composed_initializers.values()),
            name=ir_models[0].graph.name,
            opset_imports=dict(ir_models[0].graph.opset_imports),
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
