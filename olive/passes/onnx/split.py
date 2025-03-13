# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Type, Union

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class SplitModel(Pass):
    """Split an ONNX model into multiple smaller sub-models based on predefined assignments."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "split_assignments": PassConfigParam(
                type_=Union[Dict[str, int], str],
                default_value=None,
                description=(
                    "Set split assignments in the format of name1=0;name2=1 etc."
                    " Overwrite the one from CaptureSplitInfo pass."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
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

        # create a dag for the model, won't split nested graphs
        dag = OnnxDAG(model_proto, only_main_graph=True)
        dag.remove_identity_nodes()
        # empy dags for each split
        split_proto = onnx.ModelProto(
            ir_version=model_proto.ir_version,
            opset_import=model_proto.opset_import,
            producer_name="olive",
            graph=onnx.GraphProto(name=model_proto.graph.name),
        )
        split_dags = [OnnxDAG(deepcopy(split_proto)) for _ in range(num_splits)]

        # go through the nodes in topological order
        node_order = dag.topological_sort()
        node_assignments = {}
        constant_nodes = set()
        dq_nodes = set()
        for node_name in node_order:
            op_type = dag.get_node_op_type(node_name)

            # will handle constant nodes laters
            if op_type == "Constant":
                constant_nodes.add(node_name)
                continue

            # need to reassign dq nodes to child later
            if op_type == "DequantizeLinear":
                dq_nodes.add(node_name)

            # get the assignment for the node
            # will always assign Q,DQ to the parent split, this way node split only originates from base op
            # Q,DQ will behave like pass through nodes
            # Q always stays with the parent base op, DQ reassigned later to all children
            split_id = (
                self.get_assignment(node_name, split_assignments)
                if op_type not in {"QuantizeLinear", "DequantizeLinear"}
                else None
            )

            # if not assigned: assign to parent split if any
            # if assigned: check the parent ids also, the child should not be lower than the parent
            # this can happen in case the pytorch model did not order the modules correctly
            # phi3: o_proj in module comes before qkv_proj but o_proj is a child of qkv_proj
            parent_splits = [
                node_assignments[parent_name]
                for parent_name in dag.get_parents(node_name)
                if parent_name in node_assignments
            ]

            if split_id is None and not parent_splits:
                # will assign later
                # either before the first split or outside the splits
                continue

            node_assignments[node_name] = max([*([split_id] if split_id is not None else []), *parent_splits])

        # go in reverse order to assign the remaining nodes
        for node_name in node_order[::-1]:
            # constants cannot be children of node
            if node_name in node_assignments or node_name in constant_nodes:
                continue

            # before the splits - assign to the closest child split
            # outside the splits - assign to 0
            child_splits = [
                node_assignments[child_name]
                for child_name in dag.get_consumers(node_name)
                if child_name in node_assignments
            ]

            node_assignments[node_name] = min(child_splits) if child_splits else 0

        # change all assignments to list for consistency
        for node_name, split_id in node_assignments.items():
            if not isinstance(split_id, list):
                node_assignments[node_name] = [split_id]

        # handle constant and DQ nodes, add a copy of each to all child splits
        # do DQ first since it could be a child of a constant node
        for node_name in list(dq_nodes) + list(constant_nodes):
            splits = set()
            for consumer in dag.get_consumers(node_name):
                splits.update(node_assignments[consumer])
            if splits:
                # else condition could be when DQ node goes directly to output
                node_assignments[node_name] = list(splits)

        # add the nodes to the split dags
        # keep track of missing value info for inputs to the split dags
        missing_vi = defaultdict(list)
        for node_name in node_order:
            split_ids = node_assignments.get(node_name)
            if split_ids is None:
                continue

            for idx in split_ids:
                split_dag = split_dags[idx]
                # add the inputs to the nodes if not already present
                for input_name in dag.get_node_inputs(node_name):
                    if not input_name:
                        # optional input left as ""
                        continue
                    # already added
                    if split_dag.is_io(input_name):
                        continue

                    io = dag.get_io(input_name)

                    # main graph inputs and/or initializers
                    if dag.is_input(input_name) or dag.is_initializer(input_name):
                        if dag.is_input(input_name):
                            split_dags[idx].add_input(io.proto[0], 0, True)
                        if dag.is_initializer(input_name):
                            split_dags[idx].add_initializer(io.proto[-1], 0, True)
                        continue

                    # cross split inputs
                    proto = io.proto[0] if io.proto else None
                    if not proto:
                        # missing value info
                        missing_vi[input_name].append(idx)
                        proto = onnx.helper.make_empty_tensor_value_info(input_name)
                    split_dag.add_input(proto, 0)

                # add the node to the split dag
                split_dag.add_node(dag.get_node_proto(node_name), 0)

                # process the node outputs
                for output_name in dag.get_node_outputs(node_name):
                    if not output_name:
                        # optional output left as ""
                        continue

                    # mark as output if model_output or any consumer is not in the splits
                    is_output = dag.is_output(output_name)
                    for consumer in dag.get_consumers(output_name):
                        if is_output:
                            break
                        if set(node_assignments.get(consumer, [])) - set(split_ids):
                            is_output = True
                    if is_output:
                        split_dag.make_output(output_name)

                    # add vi for the outputs
                    io = dag.get_io(output_name)
                    if io.proto:
                        split_dag.add_value_info(io.proto[0], 0)
                    elif is_output:
                        # missing value info
                        missing_vi[output_name].append(idx)

        if missing_vi:
            logger.debug("Missing value info for some io. Using onnxruntime shape inference to infer them.")
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

            # should we just use the same model proto? might modify dynamic shapes of existing value infos
            # if this becomes an issue replace with a newly loaded model proto
            shape_inferred_proto = SymbolicShapeInference.infer_shapes(model_proto, auto_merge=True)
            shape_inferred_dag = OnnxDAG(shape_inferred_proto, only_main_graph=True)

            for input_name, split_ids in missing_vi.items():
                io = shape_inferred_dag.get_io(input_name)
                if not io.proto:
                    raise ValueError(f"Missing value info for input {input_name} for splits {split_ids}")
                for idx in split_ids:
                    split_dags[idx].add_value_info(io.proto[0], 0, overwrite=True)

        component_models = []
        component_names = []
        for i, split_dag in enumerate(split_dags):
            if not split_dag.get_node_names():
                # no nodes got assigned to this split
                logger.debug("Skipping empty split %d", i)
                continue
            split_name = f"split_{i}"
            split_dir = Path(output_model_path).with_suffix("") / split_name
            split_path = resolve_onnx_path(split_dir, f"{split_name}.onnx")
            split_dag.update()
            component_models.append(model_proto_to_olive_model(split_dag.model, split_path, config))
            component_names.append(split_name)

        return CompositeModelHandler(component_models, component_names)

    def get_assignment(self, node_name: str, split_assignments: Dict[str, int]) -> Optional[int]:
        name_components = node_name.replace("/", ".").lstrip(".").split(".")
        while name_components:
            if ".".join(name_components) in split_assignments:
                return split_assignments[".".join(name_components)]
            name_components.pop()
        return None
