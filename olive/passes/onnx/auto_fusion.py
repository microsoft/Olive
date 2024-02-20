# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.auto_fusion_utils import get_fusion_class
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
        # chains = self.get_fusable_chains(dag)

        # save the model to the output path and return the model
        custom_op_lib = None
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
                fusion_cls = get_fusion_class(v, dag)
                if not fusion_cls or not fusion_cls(v, dag).can_fuse_more():
                    continue

                # keep track of all possible fusion starting from v
                fusions = []

                # try to fuse the node with its child
                child = dag.connections[v][0]
                fusion = fusion_cls(v, dag)
                if fusion.try_fuse_node(child):
                    fusions.append(fusion)

                # try to fuse the node with fusions of its child
                for child_chain in chains.get(child, []):
                    fusion = fusion_cls(v, dag)
                    if fusion.try_concat_fusion(child_chain):
                        fusions.append(fusion)

                if fusions:
                    chains[v] = fusions

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
