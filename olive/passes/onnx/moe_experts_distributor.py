# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import pprint
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy
import onnx
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from onnxruntime.transformers.onnx_model import OnnxModel
from pydantic import validator

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DistributedOnnxModel, ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam

_domain = "com.microsoft"
_json_separators = (",", ": ")
_expert_pattern = [
    "Slice",  # 14
    "Concat",  # 13
    "MatMul",  # 12
    "Cast",  # 11
    "Relu",  # 10
    "MatMul",  # 9
    "Cast",  # 8
    "Slice",  # 7
    "Mul",  # 6
    "Div",  # 5
    "Add",  # 4
    "Gather",  # 3
    "Shape",  # 2
    "Reshape",  # 1
    "Reshape",  # 0
]


def _dump_graph(node: Message, filepath: str):
    with open(filepath, "wt") as strm:
        json.dump(MessageToDict(node), fp=strm, indent=2, sort_keys=True, separators=_json_separators)
        strm.flush()


def _iterate_experts(model: OnnxModel, producers: Dict[str, Message]):
    for node in model.get_nodes_by_op_type(_expert_pattern[0]):
        path = model.match_parent_path(node, _expert_pattern[1:], output_name_to_node=producers)
        if path:
            path.reverse()
            path.append(node)
            yield path


def _create_node_name(model: OnnxModel, op_type: str, prefix_a: str, prefix_b: str):
    prefix = ""
    last_slash = -1
    for i in range(0, min(len(prefix_a), len(prefix_b))):
        if prefix_a[i] == prefix_b[i]:
            prefix += prefix_a[i]
            if prefix_a[i] == "/":
                last_slash = i
        else:
            break

    if last_slash > 0:
        prefix = prefix[: last_slash + 1]

    if not prefix.endswith("/"):
        prefix += "/"
    prefix += op_type

    return model.create_node_name(op_type, prefix)


def _create_ranked_model(
    nodes: Dict[str, Message],
    producers: Dict[str, Message],
    consumers: Dict[str, Message],
    experts: List[List[str]],
    world_size: int,
    num_experts: int,
    rank: int,
):
    experts_per_rank = int(num_experts / world_size)
    start_index = rank * experts_per_rank
    end_index = start_index + experts_per_rank

    for expert in experts:
        div_node = nodes[expert[5]]
        concat_node = nodes[expert[13]]
        reshape_node = nodes[expert[1]]

        div_node_oname = div_node.output[0]
        reshape_node_oname = reshape_node.output[0]

        mul_nodes = [n for n in consumers[div_node_oname] if n.op_type == "Mul"]
        matmul_nodes = [producers[iname] for iname in concat_node.input if producers[iname].op_type == "MatMul"]
        slice_nodes = [n for n in consumers[reshape_node_oname] if n.op_type == "Slice"]

        slice_nodes[start_index].input[1] = slice_nodes[0].input[1]

        for i in range(num_experts):
            if (i < start_index) or (i >= end_index):
                mul_nodes[i].input.remove(div_node_oname)
                concat_node.input.remove(matmul_nodes[i].output[0])

                for iname in list(slice_nodes[i].input):
                    slice_nodes[i].input.remove(iname)


def _insert_commop_nodes(model: OnnxModel, nodes: Dict[str, Message], experts: List[List[str]], world_size: int):
    for expert in experts:
        for i, j in [(0, 1), (-2, -1)]:
            prev_node = nodes[expert[i]]
            next_node = nodes[expert[j]]

            node_name = _create_node_name(model, "AllToAll", prev_node.name, next_node.name)
            output_name = node_name + "_output_0"
            alltoall = onnx.helper.make_node(
                "AllToAll", prev_node.output, [output_name], name=node_name, domain=_domain, group_size=world_size
            )

            model.add_node(alltoall)
            OnnxModel.replace_node_input(next_node, prev_node.output[0], output_name)


def _replace_constant_value(constant: Message, value: int):
    for attr in constant.attribute:
        if attr.name == "value":
            attr.CopyFrom(
                onnx.helper.make_attribute(
                    attr.name, onnx.numpy_helper.from_array(numpy.array([value], numpy.int64)), attr.doc_string
                )
            )
            return


def _fix_shapes(
    model: OnnxModel,
    nodes: Dict[str, Message],
    producers: Dict[str, Message],
    consumers: Dict[str, Message],
    experts: List[List[str]],
    world_size: int,
    num_experts: int,
):
    experts_per_rank = num_experts // world_size

    for expert in experts:
        div_node = nodes[expert[5]]
        div_node_iname = div_node.input[1]
        div_node_oname = div_node.output[0]
        _replace_constant_value(producers[div_node_iname], experts_per_rank)

        add_node = nodes[expert[4]]
        add_node_iname = add_node.input[1]
        _replace_constant_value(producers[add_node_iname], experts_per_rank - 1)

        mul_nodes = [n for n in consumers[div_node_oname] if n.op_type == "Mul"]
        index = 1
        for mul_node in mul_nodes:
            if div_node_oname in mul_node.input:
                _replace_constant_value(producers[mul_node.input[1]], index)
                index += 1

        reshape_node = nodes[expert[1]]
        for parent in model.get_parents(reshape_node, producers):
            if parent.op_type == "Concat":
                concat_iname_0 = parent.input[0]
                _replace_constant_value(producers[concat_iname_0], int(num_experts / experts_per_rank))

                concat_iname_1 = parent.input[1]
                _replace_constant_value(producers[concat_iname_1], experts_per_rank)

                break


def run(world_size: int, input_filepath: str, output_dirpath: str, debug: bool = False):
    basename = Path(input_filepath).stem
    output_dirpath = Path(output_dirpath)

    model = OnnxModel(onnx.load_model(input_filepath))
    producers = model.output_name_to_node()
    consumers = model.input_name_to_nodes()

    if debug:
        OnnxModel.graph_topological_sort(model.model.graph)
        _dump_graph(model.model, str(output_dirpath / f"{basename}.graph"))

    experts = []
    num_experts = None
    for expert in _iterate_experts(model, producers):
        if debug:
            pprint.pprint([n.name for n in expert])

        div_node = expert[5]
        div_node_oname = div_node.output[0]
        if not num_experts:
            num_experts = len(consumers[div_node_oname])
            if (num_experts % world_size) != 0:
                raise f"""Number of expert paths on node "{div_node.name}" should be a multiple of
                        input world-size={world_size} but found {len(consumers[div_node_oname])}"""
        elif len(consumers[div_node_oname]) != num_experts:
            raise f"""Inconsistent number of expert across layers.
                    expected={num_experts}, found={len(consumers[div_node_oname])}"""

        experts.append([n.name for n in expert])

    output_filepaths = []
    for rank in range(0, world_size):
        rank_model = OnnxModel(onnx.load_model(input_filepath))
        producers = rank_model.output_name_to_node()
        consumers = rank_model.input_name_to_nodes()
        nodes = {node.name: node for node in rank_model.nodes()}

        _create_ranked_model(nodes, producers, consumers, experts, world_size, num_experts, rank)
        _insert_commop_nodes(rank_model, nodes, experts, world_size)
        _fix_shapes(rank_model, nodes, producers, consumers, experts, world_size, num_experts)

        rank_model.prune_graph()
        output_filepath = str(output_dirpath / f"{basename}_{rank:02d}.onnx")
        rank_model.save_model_to_file(output_filepath)
        onnx.checker.check_model(rank_model.model)
        output_filepaths.append(output_filepath)

        if debug:
            _dump_graph(rank_model.model, str(output_dirpath / f"{basename}_{rank:02d}.graph"))

    return output_filepaths


def _validate_world_size(v):
    if int(v) < 2:
        raise ValueError("world_size should be >= 2")

    return v


class MoEExpertsDistributor(Pass):
    """
    Split the input model (and insert necessary communication operations)
    to prepare for distributed inferencing.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description=("Number of GPU nodes to distribute the model for. Must be greater than 1."),
            ),
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {"validate_distributor_config": validator("world_size", allow_reuse=True)(_validate_world_size)}

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> DistributedOnnxModel:
        output_filepaths = run(config["world_size"], model.model_path, output_model_path)
        return DistributedOnnxModel(output_filepaths)
