# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnx_ir as ir
from onnx_ir.passes.common import IdentityEliminationPass, TopologicalSortPass

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class SplitModel(Pass):
    """Split an ONNX model into multiple smaller sub-models based on predefined assignments."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "split_assignments": PassConfigParam(
                type_=Union[dict[str, int], str],
                default_value=None,
                description=(
                    "Set split assignments in the format of name1=0;name2=1 etc."
                    " Overwrite the one from CaptureSplitInfo pass."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> CompositeModelHandler:
        model_proto = model.load_model()

        split_assignments = config.split_assignments
        if split_assignments is None:
            for metadata_prop in model_proto.metadata_props:
                if metadata_prop.key == "split_assignments":
                    split_assignments = metadata_prop.value
                    break
        # TODO(jambayk): Should we allow split assignments in the model attributes too?
        if not split_assignments:
            raise ValueError("No split assignments found in the model metadata")

        if isinstance(split_assignments, str):
            split_assignments = {
                key: int(value) for key, value in (assignment.split("=") for assignment in split_assignments.split(";"))
            }

        # TODO(jambayk): Make this more generic, for now only assume transformers layers are split
        # so depth of namespace is same for all split assignments
        num_splits = len(np.unique(list(split_assignments.values())))

        # create an ir model for the graph, won't split nested graphs
        ir_model = ir.from_proto(model_proto)
        IdentityEliminationPass()(ir_model)
        TopologicalSortPass()(ir_model)
        graph = ir_model.graph

        input_values = set(graph.inputs)
        initializer_names = set(graph.initializers)
        output_values = set(graph.outputs)

        # go through the nodes in topological order (main graph only)
        node_order = list(graph)
        node_assignments = {}
        constant_nodes = set()
        dq_nodes = set()
        for node in node_order:
            op_type = node.op_type

            # will handle constant nodes laters
            if op_type == "Constant":
                constant_nodes.add(node)
                continue

            # need to reassign dq nodes to child later
            if op_type == "DequantizeLinear":
                dq_nodes.add(node)

            # get the assignment for the node
            # will always assign Q,DQ to the parent split, this way node split only originates from base op
            # Q,DQ will behave like pass through nodes
            # Q always stays with the parent base op, DQ reassigned later to all children
            split_id = (
                self.get_assignment(node.name, split_assignments)
                if op_type not in {"QuantizeLinear", "DequantizeLinear"}
                else None
            )

            # if not assigned: assign to parent split if any
            # if assigned: check the parent ids also, the child should not be lower than the parent
            # this can happen in case the pytorch model did not order the modules correctly
            # phi3: o_proj in module comes before qkv_proj but o_proj is a child of qkv_proj
            parent_splits = [node_assignments[parent] for parent in node.predecessors() if parent in node_assignments]

            if split_id is None and not parent_splits:
                # will assign later
                # either before the first split or outside the splits
                continue

            node_assignments[node] = max([*([split_id] if split_id is not None else []), *parent_splits])

        # go in reverse order to assign the remaining nodes
        for node in node_order[::-1]:
            # constants cannot be children of node
            if node in node_assignments or node in constant_nodes:
                continue

            # before the splits - assign to the closest child split
            # outside the splits - assign to 0
            child_splits = [node_assignments[child] for child in node.successors() if child in node_assignments]

            node_assignments[node] = min(child_splits) if child_splits else 0

        # change all assignments to list for consistency
        for node, split_id in node_assignments.items():
            if not isinstance(split_id, list):
                node_assignments[node] = [split_id]

        # handle constant and DQ nodes, add a copy of each to all child splits
        # do DQ first since it could be a child of a constant node
        for node in list(dq_nodes) + list(constant_nodes):
            splits = set()
            for consumer in node.successors():
                splits.update(node_assignments[consumer])
            if splits:
                # else condition could be when DQ node goes directly to output
                node_assignments[node] = list(splits)

        # empty graphs for each split
        split_graphs = [ir.Graph([], [], nodes=[], name=graph.name) for _ in range(num_splits)]
        # value maps to relink values within each split by name
        value_maps: list[dict[str, ir.Value]] = [{} for _ in range(num_splits)]

        # add the nodes to the split graphs
        # keep track of missing value info for inputs/outputs to the split graphs
        missing_vi = defaultdict(list)
        for node in node_order:
            split_ids = node_assignments.get(node)
            if split_ids is None:
                continue

            for idx in split_ids:
                split_graph = split_graphs[idx]
                value_map = value_maps[idx]

                # add the inputs to the nodes if not already present
                new_inputs = []
                for inp in node.inputs:
                    if inp is None:
                        # optional input left as None
                        new_inputs.append(None)
                        continue

                    name = inp.name
                    # already added
                    if name in value_map:
                        new_inputs.append(value_map[name])
                        continue

                    is_graph_input = inp in input_values
                    is_initializer = name in initializer_names
                    if is_graph_input or is_initializer:
                        # main graph inputs and/or initializers
                        new_value = ir.Value(
                            name=name,
                            type=inp.type,
                            shape=inp.shape,
                            const_value=inp.const_value if is_initializer else None,
                        )
                        if is_graph_input:
                            split_graph.inputs.append(new_value)
                        if is_initializer:
                            split_graph.register_initializer(new_value)
                    else:
                        # cross split inputs
                        new_value = ir.Value(name=name, type=inp.type, shape=inp.shape)
                        split_graph.inputs.append(new_value)
                        if inp.type is None:
                            # missing value info
                            missing_vi[name].append(idx)

                    value_map[name] = new_value
                    new_inputs.append(new_value)

                # add a copy of the node to the split graph
                new_node = ir.Node(
                    node.domain,
                    node.op_type,
                    inputs=new_inputs,
                    attributes=list(node.attributes.values()),
                    overload=node.overload,
                    num_outputs=len(node.outputs),
                    version=node.version,
                    name=node.name,
                )
                split_graph.append(new_node)

                # process the node outputs
                for old_output, new_output in zip(node.outputs, new_node.outputs):
                    if not old_output.name:
                        # optional output left as ""
                        continue

                    new_output.name = old_output.name
                    new_output.type = old_output.type
                    new_output.shape = old_output.shape
                    value_map[old_output.name] = new_output

                    # mark as output if model_output or any consumer is not in the splits
                    is_output = old_output in output_values
                    for use in old_output.uses():
                        if is_output:
                            break
                        if set(node_assignments.get(use.node, [])) - set(split_ids):
                            is_output = True
                    if is_output:
                        split_graph.outputs.append(new_output)
                        if new_output.type is None:
                            # missing value info
                            missing_vi[old_output.name].append(idx)

        if missing_vi:
            logger.debug("Missing value info for some io. Using onnxruntime shape inference to infer them.")
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

            # should we just use the same model proto? might modify dynamic shapes of existing value infos
            # if this becomes an issue replace with a newly loaded model proto
            shape_inferred_proto = SymbolicShapeInference.infer_shapes(model_proto, auto_merge=True)
            inferred_model = ir.from_proto(shape_inferred_proto)
            vi_map = ir.convenience.create_value_mapping(inferred_model.graph)

            for name, split_ids in missing_vi.items():
                inferred_value = vi_map.get(name)
                if inferred_value is None or inferred_value.type is None:
                    raise ValueError(f"Missing value info for io {name} for splits {split_ids}")
                for idx in split_ids:
                    target = value_maps[idx].get(name)
                    if target is not None:
                        target.type = inferred_value.type
                        target.shape = inferred_value.shape

        component_models = []
        component_names = []
        output_model_dir = Path(output_model_path).with_suffix("")
        output_model_dir.mkdir(parents=True, exist_ok=True)
        # will save the split models directly in the output dir
        for i, split_graph in enumerate(split_graphs):
            if len(split_graph) == 0:
                # no nodes got assigned to this split
                logger.debug("Skipping empty split %d", i)
                continue
            split_graph.opset_imports.update(graph.opset_imports)
            split_model = ir.Model(
                split_graph,
                ir_version=ir_model.ir_version,
                producer_name="olive",
                metadata_props=dict(ir_model.metadata_props),
            )
            TopologicalSortPass()(split_model)
            split_name = f"split_{i}"
            split_path = resolve_onnx_path(output_model_dir, f"{split_name}.onnx")
            component_models.append(
                model_proto_to_olive_model(ir.to_proto(split_model), split_path, config, force_model_dir=True)
            )
            component_names.append(split_name)

        return CompositeModelHandler(component_models, component_names, model_path=output_model_dir)

    def get_assignment(self, node_name: str, split_assignments: dict[str, int]) -> Optional[int]:
        if not node_name:
            return None
        name_components = node_name.replace("/", ".").lstrip(".").split(".")
        while name_components:
            if ".".join(name_components) in split_assignments:
                return split_assignments[".".join(name_components)]
            name_components.pop()
        return None
