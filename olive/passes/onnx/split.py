# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class SplitModel(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> CompositeModelHandler:
        model_proto = model.load_model()

        split_assignments = None
        for metadata_prop in model_proto.metadata_props:
            if metadata_prop.key == "split_assignments":
                split_assignments = {
                    key: int(value)
                    for key, value in (assignment.split("=") for assignment in metadata_prop.value.split(";"))
                }
                break

        # TODO(jambayk): Make this more generic, for now only assume transformers layers are split
        # so depth of namespace is same for all split assignments
        num_splits = len(np.unique(list(split_assignments.values())))
        namespace_depth = len(next(iter(split_assignments)).split("."))

        # create a dag for the model, won't split nested graphs
        dag = OnnxDAG(model_proto, only_main_graph=True)
        dag.remove_identity_nodes()
        # empy dags for each split
        split_proto = onnx.ModelProto(
            ir_version=model_proto.ir_version,
            opset_import=model_proto.opset_import,
            producer_name="olive",
            graph=onnx.GraphProto(),
        )
        split_dags = [OnnxDAG(deepcopy(split_proto)) for _ in range(num_splits)]

        # go through the nodes in topological order
        node_order = dag.topological_sort()
        node_assignments = {}
        for node_name in node_order:
            # will handle constant nodes laters
            if dag.get_node_op_type(node_name) == "Constant":
                continue

            name_components = node_name.replace("/", ".").lstrip(".").split(".")
            namespace = ".".join(name_components[:namespace_depth])
            if namespace in split_assignments:
                node_assignments[node_name] = split_assignments[namespace]

        # handle nodes that are not in split assignments but are between splits
        # TODO(jambayk): Might have to handle nodes that are between constants/initializers and splits
        # i.e, nodes unreachable from the model inputs. Like DQ->MatMul where DQ was not assigned to any split
        is_between_splits = {}
        for node_name in node_order:
            if node_name in node_assignments or dag.get_node_op_type(node_name) == "Constant":
                continue

            # put the node in the last parent split
            # assumes that the splits are sequential and there are no parallel splits
            parent_splits = [
                node_assignments[parent_name]
                for parent_name in dag.get_parents(node_name, return_special_inputs=False)
                if parent_name in node_assignments
            ]
            if not parent_splits:
                # before the first split
                continue
            split_to_use = max(parent_splits)

            # check if the node is followed by another split
            # if not, is probably after the last split
            stack = [node_name]
            while stack:
                current_node = stack.pop()

                for child_name in dag.get_consumers(current_node, return_special_outputs=False):
                    if child_name in node_assignments or is_between_splits.get(child_name):
                        is_between_splits[current_node] = True
                        break
                    if is_between_splits.get(child_name) is False:
                        continue
                    # remember to come back to this node
                    stack.append(current_node)
                    # visit the child node first
                    stack.append(child_name)
                    break
                else:
                    is_between_splits[current_node] = False

            if is_between_splits[node_name]:
                node_assignments[node_name] = split_to_use

        # handle constant nodes, will add a copy of the constant to each split
        for node_name in node_order:
            if dag.get_node_op_type(node_name) != "Constant":
                continue

            splits = set()
            for consumer in dag.get_consumers(node_name, return_special_outputs=False):
                if consumer in node_assignments:
                    splits.add(node_assignments[consumer])
            if splits:
                node_assignments[node_name] = list(splits)

        # add the nodes to the split dags
        # keep track of missing value info for inputs to the split dags
        missing_vi = {}
        for node_name in node_order:
            split_id = node_assignments.get(node_name)
            if split_id is None:
                continue

            if not isinstance(split_id, list):
                # no need to worry about inputs for list split_id since it's only for constants
                split_dag = split_dags[split_id]
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
                            split_dags[split_id].add_input(io.proto[0], 0, True)
                        if dag.is_initializer(input_name):
                            split_dags[split_id].add_initializer(io.proto[-1], 0, True)
                        continue

                    # cross split inputs
                    proto = io.proto[0] if io.proto else None
                    if not proto:
                        # missing value info
                        missing_vi[input_name] = split_id
                        proto = onnx.helper.make_empty_tensor_value_info(input_name)
                    split_dag.add_input(proto, 0)

                # add the node to the split dag
                split_dag.add_node(dag.get_node_proto(node_name), 0)

                # process the node outputs
                for output_name in dag.get_node_outputs(node_name):
                    if not output_name:
                        # optional output left as ""
                        continue
                    # add vi for the outputs
                    io = dag.get_io(output_name)
                    if io.proto:
                        split_dag.add_value_info(io.proto[0], 0)

                    # mark as output if any consumer is not in the split
                    for consumer in dag.get_consumers(output_name):
                        # print(consumer, split_id, node_assignments.get(consumer))
                        if node_assignments.get(consumer) != split_id:
                            split_dag.make_output(output_name)
                            break
            else:
                # add the constant to each split
                for idx in split_id:
                    split_dags[idx].add_node(dag.get_node_proto(node_name), 0)

        if missing_vi:
            # need to run shape inference on the model and get the missing vi
            raise NotImplementedError

        component_models = []
        component_names = []
        for i, split_dag in enumerate(split_dags):
            split_name = f"split_{i}"
            split_dir = Path(output_model_path).with_suffix("") / split_name
            split_path = resolve_onnx_path(split_dir, f"{split_name}.onnx")
            split_dag.update()
            component_models.append(model_proto_to_olive_model(split_dag.model, split_path, config))
            component_names.append(split_name)

        return CompositeModelHandler(component_models, component_names)
