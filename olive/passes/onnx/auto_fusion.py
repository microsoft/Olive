# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.auto_fusion_utils import DOMAIN, NP_DTYPE_REVERSE_MAP, Builder, Fusion
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.utils import OnnxDAG
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class AutoFusion(Pass):
    """Automatically fuse nodes in an ONNX model using auto-generated custom operators."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "min_occurrence": PassConfigParam(
                type_=int,
                default_value=10,
                description="Minumum number of occurance of a fusion pattern to be considered for fusion.",
            ),
            "constant_overrides": PassConfigParam(
                type_=Dict[str, Dict[str, int]],
                default_value=None,
                description="Override default constants for the custom op builder. Dict of op type to constants.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # get dag
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove useless nodes
        dag.remove_redundant_cast_nodes()

        # get fusable chains
        fusable_chains = defaultdict(list)
        chains = self.get_fusable_chains(dag)
        for node_names, node_types in chains.values():
            max_valid_len, dtype = self.check_shapes_and_types(dag, node_names)
            # only consider chains equal to or longer than 2
            for i in range(2, max_valid_len + 1):
                # for i in range(1, min(2, max_valid_len + 1)):
                fusable_chains[(dtype, tuple(node_types[:i]))].append(node_names[:i])

        # only consider chains that occur more than min_occurrence times
        for node_types in list(fusable_chains):
            if len(fusable_chains[node_types]) < config["min_occurrence"]:
                del fusable_chains[node_types]

        # order chains by occurrence and length
        # Matmul chains are given higher priority
        ordered_chain_types = sorted(
            fusable_chains.keys(),
            key=lambda x: (x[1][0] == "MatMul", len(fusable_chains[x]), len(x[1])),
            reverse=True,
        )
        logger.debug(
            "Fusion candidates: \n%s", "\n".join(f"{k}: {len(fusable_chains[k])}" for k in ordered_chain_types)
        )

        # fuse chains
        fusions = []
        for dtype, chain_type in ordered_chain_types:
            # if chain_type[0] != "MatMul":
            #     continue
            # if chain_type[0] == "MatMul":
            #     continue
            # if chain_type[0] != "Sigmoid":
            #     continue
            # if chain_type != ('MatMul', 'Mul'):
            #     continue
            # if chain_type != ("MatMul", "Add"):
            #     continue
            if chain_type != ("Add", "Sqrt"):
                continue
            fusion = Fusion(dtype, chain_type[0], list(chain_type[1:]))
            num_fused = 0
            for node_names in fusable_chains[(dtype, chain_type)]:
                node_protos = dag.get_node_protos(node_names)
                if not node_protos:
                    continue
                fused_node = fusion.fuse_nodes(node_protos)
                dag.replace_nodes(node_names, fused_node)
                num_fused += 1
            if num_fused > 0:
                fusions.append((fusion, num_fused))
        logger.info(
            "Fusions: \n%s",
            "\n".join(f"{(f.dtype, (f.base_op, *f.fused_ops))}: {num_fused}" for f, num_fused in fusions),
        )
        dag.update()

        # update opset of model
        opset_import = dag.model.opset_import
        has_custom_domain = False
        for opset in opset_import:
            if opset.domain == DOMAIN:
                has_custom_domain = True
        if not has_custom_domain:
            opset_import.extend([onnx.helper.make_opsetid(DOMAIN, 1)])

        custom_op_lib = None
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Building custom op library...")
            builder = Builder([f for f, _ in fusions], temp_dir, constant_overrides=config["constant_overrides"])
            lib_path = builder.build()

            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(lib_path, Path(output_model_path).parent / Path(lib_path).name)
            custom_op_lib = Path(lib_path).name

        # save the model to the output path and return the model
        return model_proto_to_olive_model(dag.model, output_model_path, config, custom_op_lib=custom_op_lib)

    @classmethod
    def _get_fusable_chains_util(
        cls, dag: OnnxDAG, v: str, visited: Set[str], chains: Dict[str, Tuple[List[str], List[str]]]
    ) -> None:
        """Find fusable chains from all nodes reachable from v."""
        stack = [v]

        while stack:
            v = stack.pop()
            visited.add(v)

            for neighbor in dag.connections[v]:
                if neighbor not in visited:
                    stack.append(v)
                    stack.append(neighbor)
                    break
            else:
                node = dag.nodes[v]
                # check if node can be a base op
                # we only consider nodes with a single output
                if not Fusion.is_valid_base_op(node.op_type) or len(dag.connections[v]) != 1:
                    continue

                child = dag.connections[v][0]
                child_node = dag.nodes[child]
                if not Fusion.is_valid_fused_op(child_node.op_type):
                    continue

                if child in chains:
                    chains[v] = ([v, *chains[child][0]], [node.op_type, *chains[child][1]])
                else:
                    chains[v] = ([v, child], [node.op_type, child_node.op_type])

    @classmethod
    def get_fusable_chains(cls, dag: OnnxDAG) -> Dict[str, Tuple[List[str], List[str]]]:
        """Return fusable chains in the graph.

        There will be overlap between chains. For example, A -> B -> C and B -> C will both be returned.
        A -> B -> C and D -> C is also possible. The priority of the chains during fusion will be determined
        by the fusion rules and heuristics.

        :param dag: The ONNX graph.
        :return: A dictionary of fusable chains. Key is the base op and value is a tuple (op_names, op_types).
        """
        chains = {}
        visited = set()
        for v in dag.nodes:
            cls._get_fusable_chains_util(dag, v, visited, chains)
        return chains

    @staticmethod
    def is_broadcastable(a_shape: List[Union[str, int]], b_shape: List[Union[str, int]]) -> bool:
        """Check if two shapes are broadcastable.

        Broadcasting support is currently limited to the following unidirectional constraints:
            - shape of second input must be a suffix of the shape of the first input
            - Only leading 1s are allowed in the shape of the second input
            - Example [2, 3, 4, 5]: [1], [5], [1, 5], [4, 5], ...

        :param a_shape: The shape of the first input.
        :param b_shape: The shape of the second input.
        :return: True if the shapes are broadcastable, False otherwise.
        """
        if len(b_shape) > len(a_shape):
            return False

        leading_ones = True
        mismatched_dims = False
        for a, b in zip(a_shape[-len(b_shape) :], b_shape):
            if leading_ones and b == 1:
                continue
            leading_ones = False
            if a != b:
                mismatched_dims = True
                break

        return not mismatched_dims

    @classmethod
    def check_shapes_and_types(cls, dag: OnnxDAG, node_names: List[str]) -> Tuple[int, str]:
        """Check if the chain is valid for fusion.

        Rules:
            - Date type of the inputs and outputs must be the same
            - Single input nodes are always valid
            - Non-commutative ops must have the previous output as the first input
            - The other input must be broadcastable to the output of the base op
        Assumes each node has at most two inputs and one output.

        :param dag: The ONNX graph.
        :param node_names: The names of the nodes in the chain.
        :return: (max_valid_len, dtype) where max_valid_len is the maximum length of the valid chain
            and dtype is the data type of the inputs and outputs.
        """
        # base node is the first node in the chain
        base = node_names[0]
        # this is np dtype
        dtype = dag.get_input_dtypes(base)[0]
        a_shape = dag.get_output_shapes(base)[0]

        max_valid_len = 0
        for node_idx, name in enumerate(node_names):
            # check if the data type is the same
            if not all(dtype == dt for dt in dag.get_input_dtypes(name) + dag.get_output_dtypes(name)):
                break

            # check if the shapes are broadcastable

            if node_idx == 0:
                op_type = dag.get_op_type(name)
                input_shapes = dag.get_input_shapes(name)

                if op_type == "MatMul" and len(dag.get_input_shapes(name)[1]) != 2:
                    # second input of matmul must be 2D
                    break
                if (
                    op_type != "MatMul"
                    and len(input_shapes) == 2
                    and not cls.is_broadcastable(input_shapes[0], input_shapes[1])
                ):
                    #  Binary elementwise operators need the second input to be broadcastable
                    # TODO(jambayk): Add support for multidimensional broadcasting
                    # or reorder inputs for commutative ops
                    break

                # skip base node
                max_valid_len += 1
                continue

            inputs = dag.nodes[name].inputs
            if len(inputs) == 1:
                # single input nodes are always valid
                max_valid_len += 1
                continue

            if len(inputs) > 2:
                # should not reach since we only consider two input nodes
                break

            prev_output = dag.nodes[node_names[node_idx - 1]].outputs[0]
            if not Fusion.is_commutative_op(dag.nodes[name].op_type) and dag.nodes[name].inputs[0] != prev_output:
                # output is not the first input
                break

            connection_idx = dag.nodes[name].inputs.index(prev_output)
            b_shape = dag.ios[inputs[1 - connection_idx]].shape
            if not cls.is_broadcastable(a_shape, b_shape):
                break

            max_valid_len += 1

        return max_valid_len, NP_DTYPE_REVERSE_MAP[dtype]
